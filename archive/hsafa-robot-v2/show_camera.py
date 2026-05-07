"""Show Reachy Mini's camera feed in a window.

Tries **direct OpenCV capture** first (640×480 @ 15 fps, no daemon needed).
If that fails — e.g. the camera is already owned by the daemon — it falls
back to ``reachy.media.get_frame()``.

Brightness
----------
* Direct capture is usually bright enough at 640×480.
* If the room is poorly lit, press ``e`` to toggle CLAHE (copied from
  ``main.py``) or ``+`` / ``-`` to adjust gamma.

Usage
-----
    python hsafa-robot-v2/show_camera.py

Keys
----
    q / Esc     quit
    e           toggle CLAHE brightness boost
    + / =       increase gamma
    - / _       decrease gamma
"""

from __future__ import annotations

import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera_feed import make_camera_feed  # noqa: E402
from reachy_mini import ReachyMini  # noqa: E402


WINDOW_NAME = "Reachy Mini Camera"
FRAME_TIMEOUT_S = 20.0
TARGET_FPS = 15


def main() -> int:
    # Try direct OpenCV first; fall back to daemon if the camera is busy.
    try:
        feed = make_camera_feed(
            reachy=None,
            target_fps=TARGET_FPS,
            prefer_direct=True,
        )
        print("Using direct camera capture (no daemon).")
    except RuntimeError:
        print("Direct capture failed; trying daemon...")
        reachy = ReachyMini()
        feed = make_camera_feed(
            reachy=reachy,
            target_fps=TARGET_FPS,
            prefer_direct=True,
        )

    frame = feed.wait_first_frame(timeout_s=FRAME_TIMEOUT_S)
    if frame is None:
        print(
            f"Timeout: no frame after {FRAME_TIMEOUT_S:.0f}s. "
            f"Is the camera available?",
            file=sys.stderr,
        )
        return 1

    print(
        f"Streaming at target ~{TARGET_FPS} FPS. "
        f"CLAHE={'ON' if feed.clahe else 'OFF'}  gamma={feed.gamma:.2f}. "
        f"Keys: e=toggle CLAHE  +/- gamma  q=quit."
    )
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    last_log = time.time()
    n_frames = 0
    while True:
        frame = feed.get_frame()
        if frame is None:
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            continue

        cv2.imshow(WINDOW_NAME, frame)
        n_frames += 1

        now = time.time()
        if now - last_log >= 2.0:
            fps = n_frames / (now - last_log)
            mean = float(frame.mean())
            fmin = int(frame.min())
            fmax = int(frame.max())
            print(
                f"~{fps:.1f} FPS  |  mean px: {mean:.1f}  "
                f"min/max: {fmin}/{fmax}  shape: {frame.shape}  "
                f"CLAHE={'on' if feed.clahe else 'off'}  gamma={feed.gamma:.2f}"
            )
            last_log = now
            n_frames = 0

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("e"):
            feed.set_clahe(not feed.clahe)
            print(f"CLAHE: {'ON' if feed.clahe else 'OFF'}")
        elif key in (ord("+"), ord("=")):
            g = feed.bump_gamma(+0.1)
            print(f"gamma: {g:.2f}")
        elif key in (ord("-"), ord("_")):
            g = feed.bump_gamma(-0.1)
            print(f"gamma: {g:.2f}")

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    if hasattr(feed, "release"):
        feed.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
