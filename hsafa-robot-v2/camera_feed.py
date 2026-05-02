"""Shared helper for pulling frames from the Reachy Mini camera.

Offers **two backends** so tests can use whichever is appropriate:

1. **DirectCameraFeed** — ``cv2.VideoCapture`` via AVFoundation (macOS)
   like ``main.py`` does. Full control over resolution, fps, and CLAHE
   brightness. Does NOT need the daemon.

2. **DaemonCameraFeed** — ``reachy.media.get_frame()`` through the daemon's
   GStreamer pipeline. Required when the robot's audio/mic/speaker is also
   needed. No sensor-level control, but we can still throttle the consumer
   and apply software gamma.

Both support ``wait_first_frame()`` and ``get_frame()`` with the same
signature, so downstream scripts don't care which backend is active.
"""

from __future__ import annotations

import platform
import sys
import time
from typing import Optional

import cv2
import numpy as np


DEFAULT_TARGET_FPS = 15

# Direct capture defaults (same as main.py)
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_CAMERA_INDEX = 0

# CLAHE parameters copied from main.py
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _clahe_enhance(frame: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb[..., 0] = _CLAHE.apply(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class _BaseFeed:
    """Common throttling + gamma/CLAHE interface."""

    def __init__(self, target_fps: float = DEFAULT_TARGET_FPS) -> None:
        self.target_fps = float(target_fps)
        self.frame_interval = (
            1.0 / self.target_fps if self.target_fps > 0 else 0.0
        )
        self._last_frame_time: float = 0.0
        self._gamma: float = 1.0
        self._lut: Optional[np.ndarray] = None
        self._clahe: bool = False
        self.set_gamma(1.0)

    # ---- brightness controls --------------------------------------------

    def set_gamma(self, gamma: float) -> None:
        gamma = max(0.1, float(gamma))
        self._gamma = gamma
        if abs(gamma - 1.0) < 1e-3:
            self._lut = None
        else:
            inv = 1.0 / gamma
            self._lut = np.array(
                [((i / 255.0) ** inv) * 255.0 for i in range(256)],
                dtype=np.uint8,
            )

    def bump_gamma(self, delta: float) -> float:
        self.set_gamma(self._gamma + delta)
        return self._gamma

    @property
    def gamma(self) -> float:
        return self._gamma

    def set_clahe(self, on: bool) -> None:
        self._clahe = bool(on)

    @property
    def clahe(self) -> bool:
        return self._clahe

    # ---- post-processing ------------------------------------------------

    def _enhance(self, frame: np.ndarray) -> np.ndarray:
        if self._clahe:
            frame = _clahe_enhance(frame)
        if self._lut is not None:
            frame = cv2.LUT(frame, self._lut)
        return frame

    # ---- throttling -----------------------------------------------------

    def _throttle(self) -> None:
        if self.frame_interval <= 0 or self._last_frame_time <= 0:
            return
        now = time.time()
        elapsed = now - self._last_frame_time
        sleep_for = self.frame_interval - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

    def _stamp(self) -> None:
        self._last_frame_time = time.time()


# ---------------------------------------------------------------------------
# Direct OpenCV capture (same pattern as main.py)
# ---------------------------------------------------------------------------

class DirectCameraFeed(_BaseFeed):
    """Grab frames directly via ``cv2.VideoCapture``.

    On macOS this uses the AVFoundation backend. No daemon required.
    Resolution and FPS are set through OpenCV properties; the OS / driver
    decides what it actually honours.
    """

    def __init__(
        self,
        target_fps: float = DEFAULT_TARGET_FPS,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        camera_index: int = DEFAULT_CAMERA_INDEX,
    ) -> None:
        super().__init__(target_fps)
        self.width = int(width)
        self.height = int(height)
        self.index = int(camera_index)
        self._cap: Optional[cv2.VideoCapture] = None
        self._open()

    def _open(self) -> None:
        backend = (
            cv2.CAP_AVFOUNDATION
            if platform.system() == "Darwin"
            else cv2.CAP_ANY
        )
        self._cap = cv2.VideoCapture(self.index, backend)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {self.index}. "
                f"(macOS: grant Camera permission to your terminal app.)"
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Try to set a low fps so the sensor can use longer exposure.
        if self.target_fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        ok, probe = self._cap.read()
        if not ok or probe is None:
            self._cap.release()
            raise RuntimeError(
                f"Camera {self.index} opened but returned no frame."
            )
        h, w = probe.shape[:2]
        print(
            f"[DirectCameraFeed] index={self.index}  {w}x{h}  "
            f"target_fps={self.target_fps}"
        )

    def wait_first_frame(self, timeout_s: float = 20.0) -> Optional[np.ndarray]:
        # Camera is already verified open in __init__.
        ok, frame = self._cap.read()
        if ok and frame is not None:
            self._stamp()
            return self._enhance(frame)
        return None

    def get_frame(self) -> Optional[np.ndarray]:
        self._throttle()
        ok, frame = self._cap.read()
        self._stamp()
        if not ok or frame is None:
            return None
        return self._enhance(frame)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
        return False


# ---------------------------------------------------------------------------
# Daemon-backed capture (reachy.media)
# ---------------------------------------------------------------------------

class DaemonCameraFeed(_BaseFeed):
    """Grab frames through ``reachy.media.get_frame()``.

    Requires the daemon running with media enabled. The daemon owns the
    sensor, so resolution / fps are whatever the daemon pipeline delivers.
    We can only throttle the consumer and apply software gamma.
    """

    def __init__(
        self,
        reachy,
        target_fps: float = DEFAULT_TARGET_FPS,
    ) -> None:
        super().__init__(target_fps)
        self.reachy = reachy

    def wait_first_frame(self, timeout_s: float = 20.0) -> Optional[np.ndarray]:
        start = time.time()
        while True:
            frame = self.reachy.media.get_frame()
            if frame is not None:
                self._stamp()
                return self._enhance(frame)
            if time.time() - start > timeout_s:
                return None
            print("Waiting for camera frame...")
            time.sleep(0.5)

    def get_frame(self) -> Optional[np.ndarray]:
        self._throttle()
        frame = self.reachy.media.get_frame()
        self._stamp()
        if frame is None:
            return None
        return self._enhance(frame)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_camera_feed(
    reachy=None,
    *,
    target_fps: float = DEFAULT_TARGET_FPS,
    prefer_direct: bool = True,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    camera_index: int = DEFAULT_CAMERA_INDEX,
) -> DirectCameraFeed | DaemonCameraFeed:
    """Return a camera feed backend.

    If ``prefer_direct`` is True (default) we try direct OpenCV first.
    On failure we fall back to the daemon (if ``reachy`` is provided).
    If ``reachy`` is None and direct fails, we raise.
    """
    if prefer_direct:
        try:
            return DirectCameraFeed(
                target_fps=target_fps,
                width=width,
                height=height,
                camera_index=camera_index,
            )
        except RuntimeError as exc:
            print(f"[camera_feed] Direct capture failed: {exc}")
            if reachy is None:
                raise
            print("[camera_feed] Falling back to daemon (reachy.media)...")

    if reachy is None:
        raise RuntimeError(
            "No camera backend available. "
            "Provide a ReachyMini instance or ensure OpenCV can open the camera."
        )
    return DaemonCameraFeed(reachy, target_fps=target_fps)
