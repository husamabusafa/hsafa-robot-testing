"""test_object_held.py — Hand-only "holding" detection.

Uses only MediaPipe hand landmarks to decide if the hand is:
  * OPEN  — fingers extended (not holding)
  * EMPTY GRASP — fist with nothing in it
  * HOLDING — fingers wrapped around something (object keeps them apart)

No YOLO / object detector needed.  Press 'q' to quit.
"""
from __future__ import annotations

import math
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe constants
# ---------------------------------------------------------------------------
LM_WRIST = 0
LM_THUMB_TIP = 4
LM_INDEX_TIP = 8
LM_INDEX_PIP = 6
LM_MIDDLE_MCP = 9
LM_MIDDLE_TIP = 12
LM_MIDDLE_PIP = 10
LM_RING_TIP = 16
LM_RING_PIP = 14
LM_PINKY_TIP = 20
LM_PINKY_PIP = 18

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
_SHOWING_Y_MAX = 0.55
_SHOWING_MIN_AREA_FRAC = 0.03
_SHOWING_MAX_MOTION = 0.12
# Max normalized fingertip-to-wrist distance in a compact (empty) fist.
_COMPACT_FIST_MAX_SPAN = 0.80


def _ensure_hand_model() -> str:
    path = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
    if path.exists():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    print(f"Downloading hand landmarker model to {path} ...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception:
        import subprocess
        subprocess.run(
            ["curl", "-fsSL", "-o", str(path), url],
            check=True,
        )
    print("Done.")
    return str(path)


def _finger_extended(tip, pip, wrist) -> bool:
    tip_d = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
    pip_d = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
    return tip_d > pip_d * 1.05


def _is_grasping(lms) -> bool:
    wrist = lms[LM_WRIST]
    index = _finger_extended(lms[LM_INDEX_TIP], lms[LM_INDEX_PIP], wrist)
    middle = _finger_extended(lms[LM_MIDDLE_TIP], lms[LM_MIDDLE_PIP], wrist)
    ring = _finger_extended(lms[LM_RING_TIP], lms[LM_RING_PIP], wrist)
    pinky = _finger_extended(lms[LM_PINKY_TIP], lms[LM_PINKY_PIP], wrist)
    extended = sum((index, middle, ring, pinky))
    return extended < 3


def _is_compact_fist(lms) -> bool:
    """True when fingertips are all collapsed close to the wrist
    (empty fist, nothing inside keeping them apart).
    """
    wrist = lms[LM_WRIST]
    middle_mcp = lms[LM_MIDDLE_MCP]
    hand_scale = math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y)
    if hand_scale < 0.001:
        return False

    tips = [LM_THUMB_TIP, LM_INDEX_TIP, LM_MIDDLE_TIP, LM_RING_TIP, LM_PINKY_TIP]
    # Measure how far the *farthest* fingertip is from the wrist,
    # relative to the palm size.
    max_span = max(
        math.hypot(lms[t].x - wrist.x, lms[t].y - wrist.y) / hand_scale
        for t in tips
    )
    return max_span < _COMPACT_FIST_MAX_SPAN


def _is_showing(wrist, hand_bbox, w, h, history: Deque) -> bool:
    if wrist.y > _SHOWING_Y_MAX:
        return False
    x1, y1, x2, y2 = hand_bbox
    hand_area = max(0, x2 - x1) * max(0, y2 - y1)
    frame_area = max(1, w * h)
    if hand_area / frame_area < _SHOWING_MIN_AREA_FRAC:
        return False
    if len(history) >= 4:
        recent = [x for _, x in list(history)[-4:]]
        if max(recent) - min(recent) > _SHOWING_MAX_MOTION:
            return False
    return True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> int:
    from reachy_mini import ReachyMini  # type: ignore
    print("Opening Reachy camera ...")
    with ReachyMini(automatic_body_yaw=False) as reachy:
        probe = None
        for _ in range(40):
            probe = reachy.media.get_frame()
            if probe is not None:
                break
            time.sleep(0.1)
        if probe is None:
            print("Reachy camera: no frame from daemon", file=sys.stderr)
            return 1

        h, w = probe.shape[:2]
        print(f"Camera: {w}x{h}  Press 'q' to quit")

        try:
            import mediapipe as mp  # type: ignore
            from mediapipe.tasks.python.vision import (
                HandLandmarker, HandLandmarkerOptions, RunningMode,
            )
            from mediapipe.tasks.python.core.base_options import BaseOptions
        except ImportError as e:
            print(f"Could not load mediapipe tasks: {e}", file=sys.stderr)
            return 1

        model_path = _ensure_hand_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        landmarker = HandLandmarker.create_from_options(options)

        hand_histories: dict = {}
        frame_ts_ms = 0

        while True:
            frame = reachy.media.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            frame = frame.copy()

            frame_ts_ms += int(1000.0 / 30.0)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, frame_ts_ms)

            # ---- Hand processing ----
            now = time.monotonic()
            num_hands = 0
            holding_count = 0

            if result.hand_landmarks:
                num_hands = len(result.hand_landmarks)
                for i, hand_lms in enumerate(result.hand_landmarks):
                    label = "right"
                    if result.handedness and i < len(result.handedness):
                        cat = result.handedness[i][0]
                        label = (cat.category_name or "right").lower()

                    xs = [lm.x * w for lm in hand_lms]
                    ys = [lm.y * h for lm in hand_lms]
                    hand_bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

                    cv2.rectangle(
                        frame,
                        (hand_bbox[0], hand_bbox[1]),
                        (hand_bbox[2], hand_bbox[3]),
                        (0, 255, 0), 2,
                    )
                    for lm in hand_lms:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

                    wrist = hand_lms[LM_WRIST]
                    history = hand_histories.setdefault(label, deque(maxlen=20))
                    history.append((now, wrist.x))
                    cutoff = now - 1.5
                    while history and history[0][0] < cutoff:
                        history.popleft()

                    is_showing = _is_showing(wrist, hand_bbox, w, h, history)
                    is_grasp = _is_grasping(hand_lms)
                    is_compact = _is_compact_fist(hand_lms)
                    is_holding = is_showing and is_grasp and not is_compact

                    # Debug text near hand
                    tx, ty = hand_bbox[0], max(20, hand_bbox[1] - 5)
                    if is_holding:
                        cv2.putText(
                            frame, "HOLDING", (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                        )
                        holding_count += 1
                    elif is_grasp and is_compact:
                        cv2.putText(
                            frame, "EMPTY GRASP", (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1,
                        )
                    elif is_grasp:
                        cv2.putText(
                            frame, "GRASP", (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                        )

                    # Debug: show max fingertip span ratio
                    middle_mcp = hand_lms[LM_MIDDLE_MCP]
                    hand_scale = math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y)
                    if hand_scale > 0.001:
                        tips = [LM_THUMB_TIP, LM_INDEX_TIP, LM_MIDDLE_TIP, LM_RING_TIP, LM_PINKY_TIP]
                        max_span = max(
                            math.hypot(hand_lms[t].x - wrist.x, hand_lms[t].y - wrist.y) / hand_scale
                            for t in tips
                        )
                        cv2.putText(
                            frame, f"span={max_span:.2f}",
                            (hand_bbox[0], hand_bbox[3] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
                        )

            # ---- HUD ----
            hud = f"Hands: {num_hands}  |  Holding: {holding_count}"
            cv2.putText(
                frame, hud, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )
            cv2.putText(
                frame, "Raise hand + grip = HOLDING  |  Empty fist = EMPTY GRASP",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
            )

            cv2.imshow("test_object_held", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
