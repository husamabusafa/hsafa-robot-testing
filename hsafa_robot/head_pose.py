"""head_pose.py - Per-face head orientation via MediaPipe Face Landmarker.

Humans use head orientation as the primary "am I being addressed?"
cue. The gaze policy's strongest score term is ``is_being_addressed
= is_facing_camera AND is_speaking`` (see ``docs/gaze-policy.md`` §2
and ``docs/tech-recommendations.md`` §1.5). This module provides
the ``is_facing_camera`` half.

How it works:

* A background thread pulls the latest camera frame a few times a
  second.
* MediaPipe Face Mesh (optional dep, falls back gracefully) gives us
  468 3D landmarks and a face bbox.
* We solve a small rigid-body PnP with six canonical landmarks
  against the camera's intrinsics to get Euler yaw/pitch/roll.
* Each (yaw, pitch, roll) is attached to the nearest YOLO track by
  :class:`HumanRegistry.set_head_pose`, plus a boolean
  ``is_facing_camera`` based on a simple |yaw| < 20 & |pitch| < 20
  threshold (tunable).

If MediaPipe isn't installed, :attr:`enabled` stays False and no
thread starts. The gaze policy simply won't pick up
``is_being_addressed`` scores, which is fine: it degrades to the
speaker / known / proximity terms.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

from .perception import HumanRegistry, _face_in_body_score

log = logging.getLogger(__name__)


FrameGetter = Callable[[], Optional[np.ndarray]]
Bbox = Tuple[int, int, int, int]
YoloTrackSource = Callable[[], List[Tuple[int, Bbox]]]


# ---- Tunables -------------------------------------------------------------

POLL_HZ = 6.0                           # ~6 Hz is plenty for head pose
FACING_CAMERA_YAW_DEG = 25.0            # |yaw| <= this counts as facing
FACING_CAMERA_PITCH_DEG = 25.0          # |pitch| <= this counts as facing


# 3D model points for 6 canonical face landmarks (arbitrary mm units,
# will cancel out in PnP's solve since the camera matrix is also in
# arbitrary pixel units). Values are the commonly-cited dlib reference.
MODEL_3D_POINTS = np.array([
    (  0.0,    0.0,    0.0),     # nose tip
    (  0.0, -330.0,  -65.0),     # chin
    (-225.0,  170.0, -135.0),    # left eye outer corner
    ( 225.0,  170.0, -135.0),    # right eye outer corner
    (-150.0, -150.0, -125.0),    # left mouth corner
    ( 150.0, -150.0, -125.0),    # right mouth corner
], dtype=np.float64)

# MediaPipe Face Mesh landmark indices (468 landmarks).
# Indices from https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
MP_NOSE_TIP = 1
MP_CHIN = 199
MP_LEFT_EYE_OUTER = 33
MP_RIGHT_EYE_OUTER = 263
MP_LEFT_MOUTH = 61
MP_RIGHT_MOUTH = 291
MP_LANDMARK_IDS = [
    MP_NOSE_TIP, MP_CHIN,
    MP_LEFT_EYE_OUTER, MP_RIGHT_EYE_OUTER,
    MP_LEFT_MOUTH, MP_RIGHT_MOUTH,
]


class HeadPoseTracker:
    """Background thread that stamps registry entries with head orientation."""

    def __init__(
        self,
        *,
        get_frame: FrameGetter,
        yolo_tracks: YoloTrackSource,
        registry: HumanRegistry,
        max_faces: int = 4,
        poll_hz: float = POLL_HZ,
    ) -> None:
        self._get_frame = get_frame
        self._yolo_tracks = yolo_tracks
        self._registry = registry
        self._max_faces = int(max_faces)
        self._period = 1.0 / max(poll_hz, 0.5)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.enabled = False
        self._face_mesh = None

    # ---- lifecycle ----------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="head-pose", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # ---- worker -------------------------------------------------------
    def _load_model(self) -> bool:
        try:
            import mediapipe as mp   # type: ignore
            self._mp = mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self._max_faces,
                refine_landmarks=False,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            self.enabled = True
            log.info("HeadPoseTracker: MediaPipe Face Mesh loaded (max_faces=%d)",
                     self._max_faces)
            return True
        except Exception as e:
            log.warning(
                "HeadPoseTracker: could not load MediaPipe (%s). "
                "Install mediapipe to enable head-pose / is_being_addressed.",
                e,
            )
            return False

    def _run(self) -> None:
        if not self._load_model():
            return
        import cv2
        while not self._stop.is_set():
            tick_start = time.monotonic()
            frame = self._get_frame()
            if frame is None:
                time.sleep(self._period)
                continue
            try:
                self._process_frame(frame, cv2)
            except Exception as e:  # pragma: no cover
                log.warning("HeadPoseTracker tick error: %s", e)
            remaining = self._period - (time.monotonic() - tick_start)
            if remaining > 0:
                self._stop.wait(remaining)

    def _process_frame(self, frame: np.ndarray, cv2) -> None:
        h, w = frame.shape[:2]
        # MediaPipe wants RGB, no BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return

        # Intrinsics: pinhole with focal = image width, principal at center.
        # Standard trick when we don't have calibration data.
        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0.0, center[0]],
             [0.0, focal_length, center[1]],
             [0.0, 0.0, 1.0]], dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        yolo = self._yolo_tracks() or []
        now = time.monotonic()

        for landmarks in result.multi_face_landmarks:
            pts = landmarks.landmark
            xs, ys = [], []
            image_points = []
            for idx in MP_LANDMARK_IDS:
                lm = pts[idx]
                image_points.append((lm.x * w, lm.y * h))
            image_points = np.array(image_points, dtype=np.float64)

            # Bbox across all 468 landmarks -- tight enough for matching.
            all_xs = [pts[i].x * w for i in range(0, len(pts), 10)]
            all_ys = [pts[i].y * h for i in range(0, len(pts), 10)]
            bbox = (
                int(min(all_xs)), int(min(all_ys)),
                int(max(all_xs)), int(max(all_ys)),
            )

            ok, rvec, tvec = cv2.solvePnP(
                MODEL_3D_POINTS, image_points,
                camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                continue

            yaw_deg, pitch_deg, roll_deg = _rvec_to_euler(rvec, cv2)
            is_facing = (
                abs(yaw_deg) <= FACING_CAMERA_YAW_DEG
                and abs(pitch_deg) <= FACING_CAMERA_PITCH_DEG
            )

            # Attach to the nearest YOLO body.
            best_tid, best_score = -1, 0.0
            for tid, body in yolo:
                s = _face_in_body_score(bbox, body)
                if s > best_score:
                    best_score = s
                    best_tid = tid
            if best_score >= 0.4 and best_tid != -1:
                self._registry.set_head_pose(
                    best_tid,
                    yaw_deg=yaw_deg,
                    pitch_deg=pitch_deg,
                    roll_deg=roll_deg,
                    is_facing_camera=is_facing,
                    now=now,
                )


# ---- Math helper ----------------------------------------------------------

def _rvec_to_euler(rvec: np.ndarray, cv2) -> Tuple[float, float, float]:
    """Convert a rotation vector to yaw/pitch/roll Euler degrees.

    Uses the ZYX extrinsic convention which matches how we interpret
    "yaw left/right, pitch up/down, roll tilt" for a face camera.
    """
    rmat, _ = cv2.Rodrigues(rvec)
    # Decompose into ZYX Euler angles.
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(-rmat[2, 0], sy)
        yaw = math.atan2(rmat[1, 0], rmat[0, 0])
        roll = math.atan2(rmat[2, 1], rmat[2, 2])
    else:
        pitch = math.atan2(-rmat[2, 0], sy)
        yaw = 0.0
        roll = math.atan2(-rmat[1, 2], rmat[1, 1])
    # Convention: +yaw = face turns to the viewer's right (image +x).
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)
