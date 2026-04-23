"""voice_embedder.py - Speaker embeddings via SpeechBrain ECAPA-TDNN.

Thin wrapper around
``speechbrain/spkrec-ecapa-voxceleb`` (192-D embedding, trained on
VoxCeleb with additive-angular-margin loss). Given a mono 16 kHz
float32 waveform ``VoiceEmbedder.embed(waveform)`` returns a
L2-normalized numpy vector that can be compared via cosine
similarity against other embeddings stored in
:class:`~hsafa_robot.identity_graph.IdentityGraph`.

Design notes:

* **Lazy load.** The model (~25 MB + speechbrain) is only fetched
  the first time :meth:`embed` is called; importing the module is
  free. Downloads go to ``data/cache/speechbrain_ecapa`` so they
  survive across runs.
* **CPU by default.** Embedding a ~2 s clip takes ~40-80 ms on a
  MacBook Pro M-series and ~200 ms on a Raspberry Pi 5 -- well
  inside conversational latency.
* **Graceful degradation.** If ``speechbrain`` / ``torchaudio`` are
  missing or the download fails, :attr:`enabled` stays False and
  :meth:`embed` returns ``None``; the rest of the robot keeps
  running un-voiced.

This module is intentionally unaware of who the speaker is -- see
:mod:`hsafa_robot.voice_identity` for the glue that runs the
embedder on VAD utterances and ties them into the IdentityGraph.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

EMBED_DIM = 192                 # ECAPA-TDNN output size
EXPECTED_SAMPLE_RATE = 16000    # SpeechBrain recipe trained at 16 kHz
HUGGINGFACE_ID = "speechbrain/spkrec-ecapa-voxceleb"


class VoiceEmbedder:
    """CPU-friendly speaker embedder with a minimal, thread-safe API."""

    def __init__(
        self,
        *,
        cache_dir: Optional[Path] = None,
        device: str = "cpu",
    ) -> None:
        self._cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else (Path.cwd() / "data" / "cache" / "speechbrain_ecapa")
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._device = device
        self._model = None
        self._torch = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._attempted_load = False
        self.enabled = False

    # ---- lazy loading ---------------------------------------------
    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        with self._load_lock:
            if self._model is not None:
                return True
            if self._attempted_load and self._model is None:
                return False
            self._attempted_load = True
            try:
                import torch  # noqa: F401
                # SpeechBrain's import is expensive (pulls yaml, hparams,
                # etc.); do it here so cold boot stays fast.
                try:
                    from speechbrain.inference.speaker import (  # type: ignore
                        EncoderClassifier,
                    )
                except Exception:
                    # Older speechbrain layout.
                    from speechbrain.pretrained import (  # type: ignore
                        EncoderClassifier,
                    )
                self._model = EncoderClassifier.from_hparams(
                    source=HUGGINGFACE_ID,
                    savedir=str(self._cache_dir),
                    run_opts={"device": self._device},
                )
                self._torch = torch
                self.enabled = True
                log.info(
                    "VoiceEmbedder: loaded ECAPA-TDNN on %s (cache=%s)",
                    self._device, self._cache_dir,
                )
                return True
            except Exception as e:
                log.warning(
                    "VoiceEmbedder: could not load ECAPA-TDNN (%s). "
                    "Install `speechbrain` + `torchaudio` to enable "
                    "voice recognition. Voice features disabled.", e,
                )
                self._model = None
                return False

    # ---- public API -----------------------------------------------
    def embed(
        self,
        waveform: np.ndarray,
        *,
        sample_rate: int = EXPECTED_SAMPLE_RATE,
    ) -> Optional[np.ndarray]:
        """Return one L2-normalized ECAPA embedding for ``waveform``.

        ``waveform`` must be mono float32 in [-1, 1]. If the sample
        rate differs from 16 kHz we resample via linear interpolation
        (good enough for speaker ID; librosa/torchaudio are not
        imported here to keep this module hermetic).

        Returns ``None`` if the embedder is disabled, the waveform
        is empty, or inference fails.
        """
        if waveform is None or waveform.size == 0:
            return None
        if not self._ensure_loaded():
            return None
        arr = np.asarray(waveform, dtype=np.float32)
        if arr.ndim == 2:
            # Average channels to mono.
            arr = arr.mean(axis=1 if arr.shape[0] > arr.shape[1] else 0)
        if sample_rate != EXPECTED_SAMPLE_RATE:
            arr = _resample_linear(arr, sample_rate, EXPECTED_SAMPLE_RATE)

        try:
            torch = self._torch
            with self._infer_lock, torch.no_grad():
                t = torch.from_numpy(arr).unsqueeze(0)   # (1, T)
                emb = self._model.encode_batch(t)        # (1, 1, 192)
                vec = emb.squeeze().detach().cpu().numpy().astype(np.float32)
        except Exception as e:
            log.warning("VoiceEmbedder: encode failed (%s)", e)
            return None

        n = float(np.linalg.norm(vec))
        if n <= 1e-8:
            return None
        return (vec / n).astype(np.float32)


def _resample_linear(
    x: np.ndarray,
    src_sr: int,
    dst_sr: int,
) -> np.ndarray:
    """Simple linear-interpolation resampler (dependency-free).

    Fine for speaker embedding over short clips; for music / wideband
    you'd want a proper polyphase filter instead.
    """
    if src_sr == dst_sr or x.size == 0:
        return x.astype(np.float32, copy=False)
    duration = x.shape[0] / float(src_sr)
    dst_n = max(1, int(round(duration * dst_sr)))
    src_t = np.linspace(0.0, duration, num=x.shape[0], endpoint=False)
    dst_t = np.linspace(0.0, duration, num=dst_n, endpoint=False)
    return np.interp(dst_t, src_t, x).astype(np.float32)
