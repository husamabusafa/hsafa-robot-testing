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
from typing import Deque, Optional

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


class SileroVAD:
    """Streaming Silero-VAD wrapper running on its own worker thread.

    Thread-safe: :meth:`push_samples` can be called from any thread
    (typically the mic callback). The inference worker pulls from an
    internal deque under a lock.
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

    # ---- public read-only status ------------------------------------
    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def last_prob(self) -> float:
        return self._last_prob

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
            if prob >= self._threshold:
                self._run_active += 1
                self._run_quiet = 0
                if (not self._is_active) and self._run_active >= ACTIVATE_CHUNKS:
                    self._set_active(True)
            else:
                self._run_quiet += 1
                self._run_active = 0
                if self._is_active and self._run_quiet >= DEACTIVATE_CHUNKS:
                    self._set_active(False)

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
