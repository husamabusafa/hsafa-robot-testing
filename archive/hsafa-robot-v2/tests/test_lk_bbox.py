"""Drag a bounding box on the live camera; track it with LK + FB filter.

This corresponds to ``test_04_optical_flow.py`` from the design doc
(``hsafa-robot-v2/idea.md`` section 5.2). It implements the OpticalFlowModule
behavior on its own, with no other modules:

    * On bbox-set, run ``cv2.goodFeaturesToTrack`` inside the bbox.
    * Each frame, run ``cv2.calcOpticalFlowPyrLK`` from prev -> current.
    * Apply forward-backward filter (drop points whose round-trip error > 1 px).
    * bbox center  = median of surviving points.
    * bbox scale   = median pairwise distance now / at init.
    * Replenish points with goodFeaturesToTrack when survivors drop below 70%.
    * Confidence   = surviving / initial.

Prerequisites
-------------
1. Reachy Mini daemon running (it owns the camera):
       ./scripts/daemon.sh start
2. Project venv active so ``reachy_mini`` and ``cv2`` are importable.

Usage
-----
    python hsafa-robot-v2/tests/test_lk_bbox.py

Controls
--------
    Click + drag : draw a new bbox (replaces the existing one)
    space        : pause / unpause
    r            : force re-init from current bbox (fresh corners)
    c            : clear bbox / stop tracking
    q / Esc      : quit
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from camera_feed import make_camera_feed  # noqa: E402


WINDOW_NAME = "LK bbox tracker"
FRAME_TIMEOUT_S = 20.0

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

GFTT_PARAMS = dict(
    maxCorners=50,
    qualityLevel=0.01,
    minDistance=5,
    blockSize=7,
)

FB_THRESHOLD_PX = 1.0
EDGE_MARGIN_FRAC = 0.10        # drop corners within 10% of bbox edge
REPLENISH_THRESHOLD = 0.70     # refill when survivors / initial < this

# After FB filter, drop points whose per-frame displacement disagrees with
# the median displacement by more than this many pixels (absolute floor) OR
# more than DISP_MAD_K * MAD (data-driven). The combination keeps a tight
# cloud and prevents background drifters from voting on the bbox.
DISP_OUTLIER_FLOOR_PX = 2.0
DISP_MAD_K = 3.0

# Per-frame scale change is clamped to this range. The bbox can still grow
# or shrink across many frames, but it cannot jump in a single step (which
# is what causes the "bbox grows on every frame" failure mode).
SCALE_PER_FRAME_MIN = 0.97
SCALE_PER_FRAME_MAX = 1.03

# Only replenish points when the cloud is still healthy. Reseeding while
# half the cloud is on background just adds more background corners.
REPLENISH_MIN_CONFIDENCE = 0.5


class BBoxTracker:
    """MEDIANFLOW-style bbox tracker.

    Each frame, points are propagated by LK + FB filter, then:
      1. Displacements are computed against the *previous* frame (not init),
         so errors do not compound from time zero.
      2. The median displacement vector translates the bbox.
      3. Points whose displacement disagrees with the median by more than
         max(DISP_OUTLIER_FLOOR_PX, DISP_MAD_K * MAD) are dropped.
      4. Scale is the median over all per-pair distance ratios
         (|p_i - p_j|_now / |p_i - p_j|_prev). This is unbiased under
         outliers, unlike median(d_now) / median(d_prev).
      5. Per-frame scale is clamped to [SCALE_PER_FRAME_MIN, MAX] so a
         bad frame cannot blow the bbox up in one step.
    """

    def __init__(self) -> None:
        self.bbox: tuple[float, float, float, float] | None = None  # x, y, w, h
        # (N, 1, 2) float32 -- the shape OpenCV LK expects.
        self.points: np.ndarray = np.zeros((0, 1, 2), dtype=np.float32)
        self.n_initial: int = 0
        self.prev_gray: np.ndarray | None = None

    # ---- init / re-init ---------------------------------------------------

    def set_bbox(self, gray: np.ndarray, bbox: tuple[float, float, float, float]) -> None:
        self.bbox = bbox
        self._seed_points(gray)
        self.prev_gray = gray
        self.n_initial = self.points.shape[0]

    def clear(self) -> None:
        self.bbox = None
        self.points = np.zeros((0, 1, 2), dtype=np.float32)
        self.n_initial = 0

    def _seed_points(self, gray: np.ndarray) -> None:
        assert self.bbox is not None
        x, y, w, h = self.bbox
        mx = w * EDGE_MARGIN_FRAC
        my = h * EDGE_MARGIN_FRAC
        x0, y0 = int(round(x + mx)), int(round(y + my))
        x1, y1 = int(round(x + w - mx)), int(round(y + h - my))
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(gray.shape[1], x1); y1 = min(gray.shape[0], y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            self.points = np.zeros((0, 1, 2), dtype=np.float32)
            return
        roi = gray[y0:y1, x0:x1]
        corners = cv2.goodFeaturesToTrack(roi, **GFTT_PARAMS)
        if corners is None:
            self.points = np.zeros((0, 1, 2), dtype=np.float32)
            return
        corners = corners.reshape(-1, 2)
        corners[:, 0] += x0
        corners[:, 1] += y0
        self.points = corners.reshape(-1, 1, 2).astype(np.float32)

    # ---- per-frame --------------------------------------------------------

    def update(self, gray: np.ndarray) -> dict:
        """Returns a dict with bbox, confidence, n_surv, n_init, scale."""
        out = {
            "bbox": self.bbox,
            "confidence": 0.0,
            "n_surv": 0,
            "n_init": self.n_initial,
            "scale": 1.0,
        }
        if self.bbox is None or self.prev_gray is None or len(self.points) == 0:
            self.prev_gray = gray
            return out

        p0 = self.points
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, p0, None, **LK_PARAMS
        )
        if p1 is None:
            self.prev_gray = gray
            return out
        st1 = st1.reshape(-1).astype(bool)

        # Forward-backward filter.
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_gray, p1, None, **LK_PARAMS
        )
        if p0r is None:
            keep = np.zeros_like(st1, dtype=bool)
        else:
            st2 = st2.reshape(-1).astype(bool)
            fb_err = np.linalg.norm(
                p0r.reshape(-1, 2) - p0.reshape(-1, 2), axis=1
            )
            keep = st1 & st2 & (fb_err < FB_THRESHOLD_PX)

        prev_pts = p0[keep].reshape(-1, 2)
        curr_pts = p1[keep].reshape(-1, 2)
        n_surv = curr_pts.shape[0]

        if n_surv == 0:
            self.points = np.zeros((0, 1, 2), dtype=np.float32)
            self.prev_gray = gray
            return out

        # --- Spatial outlier rejection on per-frame displacement. -------
        # A point that disagrees with the cloud's median motion is almost
        # certainly off the object. Drop it. This is what kills the
        # "dot wandered onto the background" failure mode.
        disp = curr_pts - prev_pts                       # (M, 2)
        median_disp = np.median(disp, axis=0)            # (2,)
        disp_err = np.linalg.norm(disp - median_disp, axis=1)  # (M,)
        mad = float(np.median(disp_err))
        # Tolerance: max of an absolute floor and a data-driven MAD term.
        tol = max(DISP_OUTLIER_FLOOR_PX, DISP_MAD_K * mad)
        inlier_mask = disp_err <= tol
        prev_pts = prev_pts[inlier_mask]
        curr_pts = curr_pts[inlier_mask]
        n_inliers = curr_pts.shape[0]

        if n_inliers == 0:
            # Should not happen (median is itself an inlier) but be safe.
            self.points = np.zeros((0, 1, 2), dtype=np.float32)
            self.prev_gray = gray
            out["n_surv"] = 0
            return out

        # --- Translate bbox by median inlier displacement. --------------
        med_dx, med_dy = np.median(curr_pts - prev_pts, axis=0)
        x, y, w, h = self.bbox

        # --- Scale from per-pair distance ratios (MEDIANFLOW). ----------
        scale = 1.0
        if n_inliers >= 2:
            i, j = np.triu_indices(n_inliers, k=1)
            d_now = np.linalg.norm(curr_pts[i] - curr_pts[j], axis=1)
            d_prev = np.linalg.norm(prev_pts[i] - prev_pts[j], axis=1)
            valid = d_prev > 1e-3
            if np.any(valid):
                ratios = d_now[valid] / d_prev[valid]
                scale = float(np.median(ratios))

        # Clamp per-frame scale change.
        scale = float(np.clip(scale, SCALE_PER_FRAME_MIN, SCALE_PER_FRAME_MAX))

        new_cx = x + w / 2.0 + med_dx
        new_cy = y + h / 2.0 + med_dy
        new_w = w * scale
        new_h = h * scale
        self.bbox = (new_cx - new_w / 2.0, new_cy - new_h / 2.0, new_w, new_h)

        out["bbox"] = self.bbox
        out["scale"] = scale
        out["n_surv"] = n_inliers
        out["confidence"] = (
            n_inliers / self.n_initial if self.n_initial > 0 else 0.0
        )

        self.points = curr_pts.reshape(-1, 1, 2).astype(np.float32)

        # --- Replenish if we are running low (and still healthy). -------
        # Only top up when confidence is decent. Reseeding while half the
        # cloud is on background just adds more background corners.
        if (
            self.n_initial > 0
            and out["confidence"] >= REPLENISH_MIN_CONFIDENCE
            and n_inliers / self.n_initial < REPLENISH_THRESHOLD
        ):
            self._replenish(gray)

        self.prev_gray = gray
        return out

    def _replenish(self, gray: np.ndarray) -> None:
        """Top up points inside the current bbox.

        New points just join self.points; they will participate in the
        next frame's translation/scale estimate like any other point.
        """
        assert self.bbox is not None
        x, y, w, h = self.bbox
        mx, my = w * EDGE_MARGIN_FRAC, h * EDGE_MARGIN_FRAC
        x0, y0 = int(round(x + mx)), int(round(y + my))
        x1, y1 = int(round(x + w - mx)), int(round(y + h - my))
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(gray.shape[1], x1); y1 = min(gray.shape[0], y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        # Use a mask so new corners do not overlap existing ones.
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        for pt in self.points.reshape(-1, 2):
            cv2.circle(mask, (int(pt[0]), int(pt[1])), 5, 0, -1)
        new_corners = cv2.goodFeaturesToTrack(gray, mask=mask, **GFTT_PARAMS)
        if new_corners is None:
            return
        new_corners = new_corners.reshape(-1, 1, 2).astype(np.float32)
        self.points = np.concatenate([self.points, new_corners], axis=0)


# ---------------------------------------------------------------------------
# Mouse-driven bbox drawing.
# ---------------------------------------------------------------------------


class BBoxDrawer:
    def __init__(self) -> None:
        self.dragging = False
        self.start: tuple[int, int] | None = None
        self.cur: tuple[int, int] | None = None
        self.pending: tuple[float, float, float, float] | None = None

    def on_mouse(self, event, x, y, flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.cur = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.cur = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            if self.start is None:
                return
            x0, y0 = self.start
            x1, y1 = x, y
            xa, xb = sorted((x0, x1))
            ya, yb = sorted((y0, y1))
            w, h = xb - xa, yb - ya
            if w >= 8 and h >= 8:
                self.pending = (float(xa), float(ya), float(w), float(h))
            self.start = None
            self.cur = None

    def take_pending(self) -> tuple[float, float, float, float] | None:
        b = self.pending
        self.pending = None
        return b

    def draw_live(self, frame: np.ndarray) -> None:
        if self.dragging and self.start and self.cur:
            cv2.rectangle(frame, self.start, self.cur, (255, 255, 0), 2)


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------


def draw_overlay(
    frame: np.ndarray,
    tracker: BBoxTracker,
    info: dict,
    paused: bool,
    fps: float,
) -> None:
    if tracker.bbox is not None:
        x, y, w, h = tracker.bbox
        p0 = (int(round(x)), int(round(y)))
        p1 = (int(round(x + w)), int(round(y + h)))
        conf = info["confidence"]
        # Color from conf: red @ 0, green @ 1.
        color = (
            0,
            int(255 * min(1.0, max(0.0, conf))),
            int(255 * (1.0 - min(1.0, max(0.0, conf)))),
        )
        cv2.rectangle(frame, p0, p1, color, 2)

    for pt in tracker.points.reshape(-1, 2):
        cv2.circle(
            frame, (int(round(pt[0])), int(round(pt[1]))),
            3, (0, 255, 0), -1,
        )

    h_img = frame.shape[0]
    hud = [
        f"surviving: {info['n_surv']}/{info['n_init']}   "
        f"conf: {info['confidence']:.2f}   scale: {info['scale']:.2f}",
        f"fps: {fps:.1f}{'   PAUSED' if paused else ''}",
        "drag=bbox  space=pause  r=reseed  c=clear  q=quit",
    ]
    for i, line in enumerate(hud):
        y = h_img - 10 - (len(hud) - 1 - i) * 18
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop.
# ---------------------------------------------------------------------------


def main() -> int:
    tracker = BBoxTracker()
    drawer = BBoxDrawer()
    paused = False

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
    cv2.setMouseCallback(WINDOW_NAME, drawer.on_mouse)
    print("Streaming. Drag a bbox to start tracking. Press 'q' to quit.")

    last_t = time.time()
    fps = 0.0
    last_info = {"bbox": None, "confidence": 0.0, "n_surv": 0,
                 "n_init": 0, "scale": 1.0}

    while True:
        new_frame = feed.get_frame()
        if new_frame is None:
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            continue

        if not paused:
            frame = new_frame

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            pending = drawer.take_pending()
            if pending is not None:
                tracker.set_bbox(gray, pending)
                last_info = {"bbox": pending, "confidence": 1.0,
                             "n_surv": tracker.n_initial,
                             "n_init": tracker.n_initial, "scale": 1.0}

            if not paused and tracker.bbox is not None:
                last_info = tracker.update(gray)

            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            display = frame.copy()
            draw_overlay(display, tracker, last_info, paused, fps)
            drawer.draw_live(display)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                paused = not paused
            elif key == ord("c"):
                tracker.clear()
                last_info = {"bbox": None, "confidence": 0.0, "n_surv": 0,
                             "n_init": 0, "scale": 1.0}
            elif key == ord("r") and tracker.bbox is not None:
                tracker.set_bbox(gray, tracker.bbox)
                last_info = {"bbox": tracker.bbox, "confidence": 1.0,
                             "n_surv": tracker.n_initial,
                             "n_init": tracker.n_initial, "scale": 1.0}

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
    if hasattr(feed, "release"):
        feed.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
