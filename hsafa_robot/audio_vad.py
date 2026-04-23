"""audio_vad.py - Voice Activity Detection via Silero VAD.

Ingests 16 kHz mono float32 samples coming off Reachy's microphone
pipeline and exposes one boolean: *is the mic hearing human speech
right now?* That signal lets the lip-motion tracker stop false-firing
on chewing / laughing / yawning, and it is the strongest evidence
for the "voice_unseen" search behavior in ``natural_gaze.py``.

Why Silero: pure PyTorch, MIT-licensed, ~1 ms per 30 ms chunk on a
single CPU thread, well-calibrated on conversational audio. See
``docs/tech-recommendations.md`` §1.1.

Usage pattern::

    vad = SileroVAD(bus=event_bus, world=world_holder)
    vad.start()
    ...
    # in a mic callback you already have (e.g. Reachy's MediaManager):
    vad.push_samples(chunk_float32)   # 16 kHz mono float32 in [-1, 1]

The tracker is strictly additive: if PyTorch / the Silero model fails
to load we log a warning, leave :attr:`enabled` = False, and the rest
of the robot keeps working. Callers should check :attr:`enabled`
before adjusting downstream logic.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional

import numpy as np

from .events import EVT_AUDIO_SPEECH_ACTIVE, EventBus
from .world_state import WorldStateHolder

log = logging.getLogger(__name__)


# ---- Tunables -------------------------------------------------------------

# Silero ingest chunk size at 16 kHz (256 samples = 16 ms, 512 = 32 ms).
# Silero's recommended chunk is 512 samples for 16 kHz; going smaller
# costs inference accuracy, larger costs latency.
SILERO_CHUNK_SAMPLES = 512
SILERO_SAMPLE_RATE = 16000

# Debounce: speech must be above threshold for N consecutive chunks
# to flip is_active on, and below for M consecutive to flip off.
# Prevents 30 ms blips from false-triggering the GazePolicy's speech
# term.
ACTIVATE_CHUNKS = 3       # ~96 ms sustained above threshold
DEACTIVATE_CHUNKS = 10    # ~320 ms of silence before we say "quiet"

# Probability threshold. Silero returns a [0, 1] speech probability;
# 0.5 is the authors' default, 0.45 is a touch more sensitive without
# producing many false positives in quiet rooms.
SPEECH_THRESHOLD = 0.5

# Utterance capture bounds. Too-short clips don't give the voice
# embedder enough context; too-long ones risk bleeding two speakers
# into one embedding. Targets roughly match how SpeechBrain's
# ECAPA-TDNN recipe was trained.
UTTERANCE_MIN_S = 0.6        # below this we throw the utterance away
UTTERANCE_MAX_S = 6.0        # above this we chunk mid-stream


@dataclass
class Utterance:
    """One completed speech segment ready for speaker embedding.

    ``samples`` is 16 kHz mono float32 in [-1, 1]. ``start_ts`` /
    ``end_ts`` are :func:`time.monotonic` timestamps (close to when
    the waveform arrived on the mic, not when it was finally flushed).
    """
    samples: np.ndarray
    start_ts: float
    end_ts: float

    @property
    def duration_s(self) -> float:
        return self.samples.shape[0] / float(SILERO_SAMPLE_RATE)


# Utterance callbacks are synchronous from the worker thread.
# Subscribers must not block: off-load embedding work to another
# thread themselves if needed.
UtteranceCallback = Callable[[Utterance], None]


class SileroVAD:
    """Streaming Silero-VAD wrapper running on its own worker thread.

    Thread-safe: :meth:`push_samples` can be called from any thread
    (typically the mic callback). The inference worker pulls from an
    internal deque under a lock.

    Utterance capture: the VAD also buffers the raw waveform between
    rising- and falling-edge transitions. When an utterance completes
    (or gets too long and is chunked), every subscriber registered
    via :meth:`add_utterance_callback` receives the full waveform.
    This is the entry point for downstream speaker embedding.
    """

    def __init__(
        self,
        *,
        bus: Optional[EventBus] = None,
        world: Optional[WorldStateHolder] = None,
        threshold: float = SPEECH_THRESHOLD,
        device: str = "cpu",
    ) -> None:
        self._bus = bus
        self._world = world
        self._threshold = float(threshold)
        self._device = device

        self._lock = threading.Lock()
        # Unbounded in principle but we trim to ~2 s at the head every
        # push to avoid runaway memory if the worker stalls.
        self._buf: Deque[np.ndarray] = deque()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._is_active = False
        self._run_active = 0
        self._run_quiet = 0
        self._last_prob = 0.0

        self._model = None       # lazy-loaded Silero net
        self._model_state = None # h0/c0 for stateful inference
        self.enabled = False

        # Utterance capture state. Lives in the worker thread (no
        # locking needed; callbacks are the only external hand-off).
        self._utt_chunks: List[np.ndarray] = []
        self._utt_start_ts: float = 0.0
        self._utt_callbacks: List[UtteranceCallback] = []
        self._cb_lock = threading.Lock()

    # ---- public read-only status ------------------------------------
    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def last_prob(self) -> float:
        return self._last_prob

    # ---- utterance subscription -------------------------------------
    def add_utterance_callback(self, cb: UtteranceCallback) -> None:
        """Subscribe to completed utterances. Called from VAD worker thread."""
        with self._cb_lock:
            self._utt_callbacks.append(cb)

    def remove_utterance_callback(self, cb: UtteranceCallback) -> None:
        with self._cb_lock:
            try:
                self._utt_callbacks.remove(cb)
            except ValueError:
                pass

    # ---- lifecycle --------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        # Load the model lazily on the worker so import cost stays off
        # the main boot path.
        self._thread = threading.Thread(
            target=self._run, name="silero-vad", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # ---- audio ingress ----------------------------------------------
    def push_samples(self, samples: np.ndarray) -> None:
        """Feed 16 kHz mono float32 samples. Non-blocking."""
        if samples is None or samples.size == 0:
            return
        arr = np.ascontiguousarray(samples, dtype=np.float32)
        if arr.ndim > 1:
            # Downmix to mono.
            arr = arr.mean(axis=1 if arr.shape[0] > arr.shape[1] else 0)
        with self._lock:
            self._buf.append(arr)
            # Cap backlog at ~2 seconds.
            total = sum(a.shape[0] for a in self._buf)
            while total > 2 * SILERO_SAMPLE_RATE and self._buf:
                total -= self._buf[0].shape[0]
                self._buf.popleft()

    # ---- worker -----------------------------------------------------
    def _load_model(self) -> bool:
        try:
            import torch  # local import so torch is optional at boot
            # Try package first (pip install silero-vad), then torch hub.
            try:
                from silero_vad import load_silero_vad  # type: ignore
                self._model = load_silero_vad()
            except Exception:
                self._model = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    trust_repo=True,
                )[0]
            self._model.to(self._device).eval()
            self._torch = torch
            self.enabled = True
            log.info("SileroVAD: loaded (device=%s, threshold=%.2f)",
                     self._device, self._threshold)
            return True
        except Exception as e:
            log.warning(
                "SileroVAD: could not load model (%s). "
                "Install torch + silero-vad to enable audio VAD. "
                "Lip-motion speaker detection will run un-gated.", e,
            )
            return False

    def _pop_chunk(self) -> Optional[np.ndarray]:
        """Return exactly SILERO_CHUNK_SAMPLES if available, else None."""
        with self._lock:
            total = sum(a.shape[0] for a in self._buf)
            if total < SILERO_CHUNK_SAMPLES:
                return None
            out = np.empty(SILERO_CHUNK_SAMPLES, dtype=np.float32)
            filled = 0
            while filled < SILERO_CHUNK_SAMPLES:
                head = self._buf[0]
                need = SILERO_CHUNK_SAMPLES - filled
                take = min(need, head.shape[0])
                out[filled:filled + take] = head[:take]
                filled += take
                if take == head.shape[0]:
                    self._buf.popleft()
                else:
                    self._buf[0] = head[take:]
            return out

    def _run(self) -> None:
        if not self._load_model():
            return
        torch = self._torch
        max_samples = int(UTTERANCE_MAX_S * SILERO_SAMPLE_RATE)

        while not self._stop.is_set():
            chunk = self._pop_chunk()
            if chunk is None:
                time.sleep(0.005)
                continue
            try:
                t = torch.from_numpy(chunk)
                with torch.no_grad():
                    prob = float(self._model(t, SILERO_SAMPLE_RATE).item())
            except Exception as e:  # pragma: no cover - defensive
                log.warning("SileroVAD inference failed: %s", e)
                time.sleep(0.05)
                continue

            self._last_prob = prob
            # Utterance buffering: every chunk emitted *while* active
            # gets appended to the current utterance, including the
            # first few that triggered the rising edge.
            if self._is_active or prob >= self._threshold:
                if not self._utt_chunks:
                    self._utt_start_ts = time.monotonic()
                self._utt_chunks.append(chunk.copy())
                # Chunk very long utterances so one monologue doesn't
                # block voice embedding updates.
                if sum(c.shape[0] for c in self._utt_chunks) >= max_samples:
                    self._flush_utterance()

            if prob >= self._threshold:
                self._run_active += 1
                self._run_quiet = 0
                if (not self._is_active) and self._run_active >= ACTIVATE_CHUNKS:
                    self._set_active(True)
            else:
                self._run_quiet += 1
                self._run_active = 0
                if self._is_active and self._run_quiet >= DEACTIVATE_CHUNKS:
                    # Falling edge -> close the utterance.
                    self._flush_utterance()
                    self._set_active(False)

        # On shutdown, drop any pending partial utterance.
        self._utt_chunks.clear()

    def _flush_utterance(self) -> None:
        """Concatenate the active chunks and ship them to subscribers."""
        if not self._utt_chunks:
            return
        samples = np.concatenate(self._utt_chunks, axis=0)
        self._utt_chunks.clear()
        duration = samples.shape[0] / float(SILERO_SAMPLE_RATE)
        if duration < UTTERANCE_MIN_S:
            return
        utt = Utterance(
            samples=samples,
            start_ts=self._utt_start_ts,
            end_ts=time.monotonic(),
        )
        with self._cb_lock:
            subs = list(self._utt_callbacks)
        for cb in subs:
            try:
                cb(utt)
            except Exception as e:   # pragma: no cover - defensive
                log.warning("SileroVAD utterance callback raised: %s", e)

    def _set_active(self, active: bool) -> None:
        if active == self._is_active:
            return
        self._is_active = active
        if self._bus is not None:
            self._bus.publish(
                EVT_AUDIO_SPEECH_ACTIVE, source="audio", active=active,
            )
        if self._world is not None:
            self._world.set_audio_speech_active(active)
        log.debug("SileroVAD: audio_speech_active=%s (prob=%.2f)",
                  active, self._last_prob)
