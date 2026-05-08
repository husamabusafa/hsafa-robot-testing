"""gyro_stabilizer.py - Robust gyro damping for face tracking.

PROBLEM: The P-controller on visual face error causes overshoot and
wiggle because:
- Camera frame is ~30-50ms old (latency)
- Motor response takes time
- By the time head reaches commanded position, next frame shows overshoot
- Controller then commands back, causing oscillation

SIMPLE FIX (this module)
--------------------------
Use gyroscope to detect when the head is ALREADY moving, and dynamically
smooth the visual error:

- Head moving fast (high gyro) → visual error is STALE → smooth MORE
- Head stable (low gyro)       → visual error is FRESH → smooth LESS

This is robust because:
- No integration → no drift, no sign sensitivity
- Wrong gyro sign just means less damping (never runaway)
- Works even with partially calibrated gyros

Algorithm per tick
------------------
1. Read gyro magnitude: ``motion = sqrt(gyro_z^2 + gyro_x^2)``
2. Map motion to dynamic smoothing alpha:
       alpha = base_alpha * clamp(1 - motion/threshold, 0.1, 1.0)
   - Low motion  (0 deg/s)  → alpha = base_alpha   (responsive)
   - High motion (>threshold) → alpha = 0.1 * base_alpha (very smooth)
3. Smooth the raw visual error with this dynamic alpha.
4. Return smoothed error.

The gyro is used as a MOTION-CONFIDENCE indicator, not as a
geometric corrector.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

from .head_gyro import HeadGyro

log = logging.getLogger(__name__)


@dataclass
class StabilizedError:
    err_x: float
    err_y: float
    head_yaw_rate: float   # deg/s (for logging only)
    head_pitch_rate: float  # deg/s
    compensation_applied: bool


class GyroStabilizer:
    """Dynamic visual-error smoothing based on head-motion magnitude.

    Parameters
    ----------
    base_alpha:
        Default EMA smoothing [0,1]. Higher = more responsive.
        Typical 0.4-0.6. This is the value used when head is perfectly
        still.
    motion_threshold_deg_s:
        Gyro magnitude (deg/s) at which we max out the smoothing.
        Typical 30-80. Tune by observing when wiggle disappears.
    """

    def __init__(
        self,
        base_alpha: float = 0.5,
        motion_threshold_deg_s: float = 50.0,
    ) -> None:
        self.gyro = HeadGyro()
        self.base_alpha = float(base_alpha)
        self.motion_threshold = float(motion_threshold_deg_s)

        # Smoothed error state (persists across calls).
        self._err_x_s = 0.0
        self._err_y_s = 0.0
        self._last_face_t = 0.0

        # Timeout: if no face for this long, reset smoothers.
        self._timeout_s = 1.5

    # ---- lifecycle ----------------------------------------------------
    def start(self) -> bool:
        ok = self.gyro.start()
        if ok:
            log.info(
                "GyroStabilizer: started (base_alpha=%.2f "
                "motion_thresh=%.0f deg/s)",
                self.base_alpha, self.motion_threshold,
            )
        return ok

    def stop(self) -> None:
        self.gyro.stop()

    def get_latest(self):
        return self.gyro.get_latest()

    def reset(self) -> None:
        """Clear internal state (e.g. on focus switch)."""
        self._err_x_s = 0.0
        self._err_y_s = 0.0
        self._last_face_t = 0.0

    # ---- core ---------------------------------------------------------
    def compensate_head_motion(
        self, err_x: float, err_y: float, dt: float,
    ) -> StabilizedError:
        """Return smoothed visual error with motion-adaptive alpha.

        Call this only when a fresh visual detection is available
        (have_face=True).
        """
        data = self.gyro.get_latest()
        if data is None:
            # No gyro yet -> pass through with base smoothing.
            self._smooth(err_x, err_y, self.base_alpha)
            return StabilizedError(
                self._err_x_s, self._err_y_s, 0.0, 0.0, False,
            )

        now = time.time()

        # --- Motion magnitude from gyro rates -------------------------
        # We only care about magnitude, not direction, so signs don't
        # matter for correctness (wrong sign = less damping, never
        # runaway).
        yaw_rate = abs(float(data.gyro_z))   # primary horizontal
        pitch_rate = abs(float(data.gyro_x))  # primary vertical (mounting)
        motion = math.sqrt(yaw_rate**2 + pitch_rate**2)

        # --- Dynamic alpha -------------------------------------------
        # motion=0        → factor = 1.0     → alpha = base_alpha
        # motion=threshold → factor = 0.0   → alpha = 0.1 * base_alpha
        factor = max(0.0, 1.0 - motion / self.motion_threshold)
        # Clamp minimum so we never completely freeze.
        factor = max(0.1, factor)
        alpha = self.base_alpha * factor

        # --- Smooth visual error --------------------------------------
        # If face was lost for a while, snap to new detection instead
        # of slow-EMAing from an old value.
        stale = (now - self._last_face_t) > self._timeout_s
        if stale:
            self._err_x_s = err_x
            self._err_y_s = err_y
        else:
            self._smooth(err_x, err_y, alpha)

        self._last_face_t = now

        return StabilizedError(
            err_x=self._err_x_s,
            err_y=self._err_y_s,
            head_yaw_rate=float(data.gyro_z),
            head_pitch_rate=float(data.gyro_x),
            compensation_applied=True,
        )

    def _smooth(self, raw_x: float, raw_y: float, alpha: float) -> None:
        self._err_x_s = (1 - alpha) * self._err_x_s + alpha * raw_x
        self._err_y_s = (1 - alpha) * self._err_y_s + alpha * raw_y

    def get_debug(self) -> dict:
        """Return internal state for HUD / logging."""
        data = self.gyro.get_latest()
        motion = 0.0
        if data is not None:
            motion = math.sqrt(data.gyro_z**2 + data.gyro_x**2)
        factor = max(0.1, 1.0 - motion / self.motion_threshold) if self.motion_threshold else 1.0
        return {
            "err_x_s": self._err_x_s,
            "err_y_s": self._err_y_s,
            "motion": motion,
            "alpha": self.base_alpha * factor,
        }


# ---- quick self-test -----------------------------------------------------

def _cli_test() -> None:
    import sys
    print("GyroStabilizer test - motion-adaptive smoothing.")
    print("Move the head fast -> smoothing increases (less wiggle).")
    print("Keep head still   -> smoothing decreases (more responsive).\n")

    s = GyroStabilizer()
    if not s.start():
        print("Failed to start gyro.")
        sys.exit(1)

    fake_err_x, fake_err_y = 0.5, 0.3

    print(f"{'time':>8}  {'motion':>8}  {'alpha':>6}  "
          f"{'err_x_s':>+7}  {'err_y_s':>+7}")
    try:
        while True:
            stab = s.compensate_head_motion(fake_err_x, fake_err_y, 0.033)
            d = s.get_debug()
            print(
                f"{time.strftime('%H:%M:%S'):>8}  "
                f"{d['motion']:>8.1f}  {d['alpha']:>6.3f}  "
                f"{d['err_x_s']:>+7.3f}  {d['err_y_s']:>+7.3f}",
                end="\r",
            )
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        s.stop()
        print()


if __name__ == "__main__":
    _cli_test()
