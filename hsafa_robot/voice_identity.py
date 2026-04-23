"""voice_identity.py - Speaker recognition + cross-modal enrollment.

Hooks together :class:`SileroVAD` (utterance source),
:class:`VoiceEmbedder` (ECAPA-TDNN speaker encoder) and
:class:`IdentityGraph` (face+voice+name store) to deliver the two
user-visible voice features:

1. **Off-camera recognition.** When the robot hears your voice and
   can't see a face, it can still say "hi Husam" because it matched
   the incoming utterance to your stored voice bank. Result lives
   on :attr:`EnvView.last_heard_voice_name` and also fires
   ``voice_identified`` on the :class:`EventBus`.

2. **Automatic voice-face linking.** The first few seconds you
   speak while your face is visible, the identity graph
   accumulates voice embeddings attributed to your name. Once
   ~5 clean co-occurrences are gathered, they commit to
   ``data/identity/voices/<name>.npy``. No explicit "enroll my
   voice" call is needed.

All heavy work (embedding, disk I/O, graph updates) runs on a
dedicated worker thread so the VAD callback stays non-blocking.

If :class:`VoiceEmbedder` can't load (no speechbrain / no network),
this module quietly disables itself -- :attr:`VoiceIdentityWorker.enabled`
stays False and the rest of the robot keeps running unchanged.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, List, Optional

import numpy as np

from .audio_vad import SileroVAD, Utterance
from .events import EVT_VOICE_ENROLLED, EVT_VOICE_IDENTIFIED, EventBus
from .identity_graph import IdentityGraph
from .voice_embedder import EMBED_DIM, EXPECTED_SAMPLE_RATE, VoiceEmbedder
from .world_state import WorldStateHolder

log = logging.getLogger(__name__)


# ---- Tunables ------------------------------------------------------------

# How often to try committing buffered voice samples to disk.
COMMIT_EVERY_UTTERANCES = 3
# Minimum number of co-occurrence samples before a voice bank commits.
MIN_COMMIT_SAMPLES = 5
# Cosine similarity threshold for calling a voice "recognised".
# ECAPA on VoxCeleb hits EER ~0.9% at sim ~0.3 with length-normalised
# embeddings; 0.55 is a safe default for a home-robot registration of
# 3-5 speakers.
IDENTIFY_THRESHOLD = 0.55
# Maximum utterances queued before the worker drops the oldest.
MAX_QUEUE_DEPTH = 8


# Supplier returning the currently visible speaker's canonical name,
# or ``None`` if no visible named person is speaking right now. This
# is the bridge to the perception layer -- typically a closure over
# lip_tracker + face_recognizer, wired up in ``main.py``.
VisibleSpeakerSupplier = Callable[[], Optional[str]]


class VoiceIdentityWorker:
    """Background worker that embeds utterances and updates identities."""

    def __init__(
        self,
        *,
        vad: SileroVAD,
        embedder: VoiceEmbedder,
        identity_graph: IdentityGraph,
        world: Optional[WorldStateHolder] = None,
        bus: Optional[EventBus] = None,
        visible_speaker_supplier: Optional[VisibleSpeakerSupplier] = None,
        identify_threshold: float = IDENTIFY_THRESHOLD,
    ) -> None:
        self._vad = vad
        self._embedder = embedder
        self._graph = identity_graph
        self._world = world
        self._bus = bus
        self._visible_speaker = (
            visible_speaker_supplier or (lambda: None)
        )
        self._identify_threshold = float(identify_threshold)

        self._queue: "queue.Queue[Utterance]" = queue.Queue(
            maxsize=MAX_QUEUE_DEPTH,
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._enrolled_utterances = 0
        self.enabled = False

    # ---- lifecycle -------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self.enabled = True
        self._vad.add_utterance_callback(self._on_utterance)
        self._thread = threading.Thread(
            target=self._run, name="voice-identity", daemon=True,
        )
        self._thread.start()
        log.info(
            "VoiceIdentityWorker: started (identify_threshold=%.2f)",
            self._identify_threshold,
        )

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        try:
            self._vad.remove_utterance_callback(self._on_utterance)
        except Exception:
            pass
        # Unblock the worker if it's sleeping on queue.get.
        try:
            self._queue.put_nowait(None)   # type: ignore[arg-type]
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        # Final flush.
        try:
            self._graph.commit_pending_voices(min_samples=MIN_COMMIT_SAMPLES)
        except Exception as e:
            log.warning("VoiceIdentityWorker: final commit failed: %s", e)

    # ---- VAD hook --------------------------------------------------
    def _on_utterance(self, utt: Utterance) -> None:
        """Invoked from the VAD worker thread; hand off to ours."""
        try:
            self._queue.put_nowait(utt)
        except queue.Full:
            # Drop oldest so we always process the freshest utterance.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(utt)
            except queue.Empty:
                pass

    # ---- Worker loop -----------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                utt = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if utt is None:
                break
            try:
                self._process(utt)
            except Exception as e:   # pragma: no cover - defensive
                log.warning("VoiceIdentity: processing failed: %s", e)

    def _process(self, utt: Utterance) -> None:
        emb = self._embedder.embed(
            utt.samples, sample_rate=EXPECTED_SAMPLE_RATE,
        )
        if emb is None or emb.shape != (EMBED_DIM,):
            return

        # 1. Identification: always try to match against stored banks.
        match = self._graph.identify_voice(
            emb, threshold=self._identify_threshold,
        )
        matched_name = match[0] if match else None
        matched_sim = match[1] if match else 0.0

        if self._world is not None:
            self._world.set_last_heard_voice(matched_name, matched_sim)
        if self._bus is not None:
            self._bus.publish(
                EVT_VOICE_IDENTIFIED,
                source="voice",
                name=matched_name,
                similarity=matched_sim,
                duration_s=utt.duration_s,
            )
        if matched_name is not None:
            log.info(
                "VoiceIdentity: heard %s (sim=%.2f, %.1fs)",
                matched_name, matched_sim, utt.duration_s,
            )

        # 2. Cross-modal enrollment: if a named person is currently
        # visible AND speaking, attribute this voice sample to them.
        # We prefer the vision-side identity over the voice match --
        # vision ground-truth trumps voice during enrollment because
        # ECAPA hasn't learned this speaker yet.
        visible_name = self._visible_speaker()
        if visible_name:
            try:
                n_pending = self._graph.stash_voice_sample(
                    visible_name, emb, min_samples=MIN_COMMIT_SAMPLES,
                )
            except Exception as e:
                log.warning("VoiceIdentity: stash failed: %s", e)
                return
            self._enrolled_utterances += 1
            log.debug(
                "VoiceIdentity: stashed voice sample for %s (pending=%d)",
                visible_name, n_pending,
            )
            # Periodically try committing so banks build up in real
            # time, not only at shutdown.
            if self._enrolled_utterances % COMMIT_EVERY_UTTERANCES == 0:
                committed = self._graph.commit_pending_voices(
                    min_samples=MIN_COMMIT_SAMPLES,
                )
                for name in committed:
                    log.info(
                        "VoiceIdentity: committed voice bank for %s", name,
                    )
                    if self._bus is not None:
                        self._bus.publish(
                            EVT_VOICE_ENROLLED,
                            source="voice",
                            name=name,
                        )
