"""Click points on the live camera; track them with Lucas-Kanade.

This is the smallest possible test of the optical-flow module described in
``hsafa-robot-v2/idea.md`` (section 4.1). It does NOT use any tracker core,
fusion, or modules infrastructure -- the goal is to convince yourself that
LK + forward-backward filtering actually follows what you click on.

Prerequisites
-------------
1. Reachy Mini daemon running (it owns the camera):

       ./scripts/daemon.sh start

2. Project venv active so ``reachy_mini`` and ``cv2`` are importable.

Usage
-----
    python hsafa-robot-v2/tests/test_lk_points.py

Controls
--------
    Left click  : add a tracked point at cursor
    c           : clear all points
    f           : toggle forward-backward filter on/off
    t           : toggle motion trails on/off
    q / Esc     : quit

What you should see
-------------------
    * Green dots are healthy tracked points.
    * Red dots are points that just failed the FB filter (about to be dropped).
    * A short trail behind each point shows recent motion.
    * Move your hand / a mug in front of the camera: dots stick to texture.
    * Move the camera: dots stick to the world (no gyro compensation yet,
      so fast head motion will lose them -- that is test_05's problem).
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from camera_feed import make_camera_feed  # noqa: E402


WINDOW_NAME = "LK point tracker"
FRAME_TIMEOUT_S = 20.0

# LK parameters: pyramid level 3 is enough for ~640x480 frames.
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# Forward-backward filter: a point is kept if FB displacement is under this
# many pixels. ~1 px is the standard recommendation.
FB_THRESHOLD_PX = 1.0

# Trail length, in frames.
TRAIL_LEN = 20


class PointTracker:
    """Holds a set of 2D points and runs LK + FB filter each frame."""

    def __init__(self) -> None:
        # (N, 1, 2) float32 -- the shape OpenCV LK expects.
        self.points: np.ndarray = np.zeros((0, 1, 2), dtype=np.float32)
        # Per-point state for visualization.
        self.trails: list[deque] = []
        self.last_failed_mask: np.ndarray = np.zeros((0,), dtype=bool)
        self.prev_gray: np.ndarray | None = None

    def add_point(self, x: float, y: float) -> None:
        new = np.array([[[x, y]]], dtype=np.float32)
        self.points = np.concatenate([self.points, new], axis=0)
        self.trails.append(deque(maxlen=TRAIL_LEN))

    def clear(self) -> None:
        self.points = np.zeros((0, 1, 2), dtype=np.float32)
        self.trails = []
        self.last_failed_mask = np.zeros((0,), dtype=bool)

    def update(self, gray: np.ndarray, use_fb: bool) -> None:
        """Run LK from prev_gray -> gray on all current points."""
        if self.prev_gray is None or len(self.points) == 0:
            self.prev_gray = gray
            return

        p0 = self.points
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, p0, None, **LK_PARAMS
        )

        if p1 is None:
            # Total failure -- drop everything.
            self._drop_all()
            self.prev_gray = gray
            return

        st1 = st1.reshape(-1).astype(bool)

        if use_fb:
            # Backward pass: track p1 back to prev frame, compare to p0.
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(
                gray, self.prev_gray, p1, None, **LK_PARAMS
            )
            if p0r is None:
                fb_ok = np.zeros_like(st1, dtype=bool)
            else:
                st2 = st2.reshape(-1).astype(bool)
                fb_err = np.linalg.norm(
                    p0r.reshape(-1, 2) - p0.reshape(-1, 2), axis=1
                )
                fb_ok = (fb_err < FB_THRESHOLD_PX) & st2
            keep = st1 & fb_ok
        else:
            keep = st1

        # Update trails for every still-existing point (kept or just-failed)
        # before we filter, so the visualization can show the failure flash.
        for i, pt in enumerate(p1.reshape(-1, 2)):
            self.trails[i].append((float(pt[0]), float(pt[1])))

        # Now apply the keep mask.
        self.last_failed_mask = ~keep
        self._apply_mask(p1, keep)

        self.prev_gray = gray

    def _apply_mask(self, p1: np.ndarray, keep: np.ndarray) -> None:
        # Keep the failed-this-frame points around for one extra frame so the
        # user gets a visible red flash, then drop them next frame.
        # Simplest: drop immediately. Visualization uses last_failed_mask
        # which we computed before this call, so the red flash happens on
        # THIS frame's render before the next update.
        new_points = []
        new_trails = []
        for i, k in enumerate(keep):
            if k:
                new_points.append(p1[i])
                new_trails.append(self.trails[i])
        if new_points:
            self.points = np.stack(new_points, axis=0).astype(np.float32)
        else:
            self.points = np.zeros((0, 1, 2), dtype=np.float32)
        self.trails = new_trails

    def _drop_all(self) -> None:
        self.last_failed_mask = np.ones((len(self.points),), dtype=bool)
        self.points = np.zeros((0, 1, 2), dtype=np.float32)
        self.trails = []


def draw_overlay(
    frame: np.ndarray,
    tracker: PointTracker,
    show_trails: bool,
    use_fb: bool,
    fps: float,
) -> None:
    # Trails first (so dots are on top).
    if show_trails:
        for trail in tracker.trails:
            if len(trail) < 2:
                continue
            pts = np.array(trail, dtype=np.int32)
            cv2.polylines(
                frame, [pts], isClosed=False, color=(0, 200, 255), thickness=1
            )

    # Healthy points.
    for pt in tracker.points.reshape(-1, 2):
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), 7, (0, 0, 0), 1)

    # HUD.
    h = frame.shape[0]
    n = len(tracker.points)
    n_failed = int(tracker.last_failed_mask.sum())
    hud = [
        f"points: {n}   fb-dropped last frame: {n_failed}",
        f"fb-filter: {'ON' if use_fb else 'OFF'}   trails: {'ON' if show_trails else 'OFF'}",
        f"fps: {fps:.1f}",
        "click=add  c=clear  f=toggle FB  t=toggle trails  q=quit",
    ]
    for i, line in enumerate(hud):
        y = h - 10 - (len(hud) - 1 - i) * 18
        cv2.putText(
            frame, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            frame, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )


def main() -> int:
    tracker = PointTracker()
    state = {"use_fb": True, "show_trails": True}

    def on_mouse(event, x, y, flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            tracker.add_point(x, y)

    feed = make_camera_feed(target_fps=15, prefer_direct=True)
    print(f"Using {type(feed).__name__}.")

    frame = feed.wait_first_frame(timeout_s=FRAME_TIMEOUT_S)
    if frame is None:
        print(
            f"Timeout: no frame after {FRAME_TIMEOUT_S:.0f}s. "
            f"Is the camera available?",
            file=sys.stderr,
        )
        return 1

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    print("Streaming. Click to add points. Press 'q' to quit.")

    last_t = time.time()
    fps = 0.0

    while True:
        frame = feed.get_frame()
        if frame is None:
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tracker.update(gray, use_fb=state["use_fb"])

        # FPS smoothing.
        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        display = frame.copy()
        draw_overlay(
            display, tracker,
            show_trails=state["show_trails"],
            use_fb=state["use_fb"],
            fps=fps,
        )
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("c"):
            tracker.clear()
        elif key == ord("f"):
            state["use_fb"] = not state["use_fb"]
            print(f"FB filter: {state['use_fb']}")
        elif key == ord("t"):
            state["show_trails"] = not state["show_trails"]

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    if hasattr(feed, "release"):
        feed.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
