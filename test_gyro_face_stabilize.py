"""Gyro-stabilized face tracking test.

Combines YOLOv8-Pose face detection with BNO055 gyro data to keep the
face bounding box and tracking point stable when the robot head rotates.

Visuals:
  - RED box + point: raw YOLO detection (no gyro compensation)
  - GREEN box + point: gyro-stabilized (compensated for head rotation)
  - Blue crosshair: image center reference
  - Orange crosshair: accumulated gyro heading (drift indicator)

Controls:
  - 'q' or ESC: quit
  - 'p': lock anchor on current face (BNO055 heading/roll/pitch)
  - 'r': reset anchor
  - 'a': cycle axis mapping
  - 's': toggle stabilization on/off
"""

import argparse
import cv2
import math
import sys
import time
import numpy as np

from hsafa_robot.head_gyro import HeadGyro
from hsafa_robot.tracker import (
    CascadeTracker,
    YOLO_CONF,
    YOLO_IMGSZ,
    ensure_pose_model,
    pick_device,
)


AXIS_MAPPINGS = [
    ("z", "x", -1, +1, "yaw=Z- pitch=X+"),
    ("z", "x", -1, -1, "yaw=Z- pitch=X-"),
    ("z", "y", -1, +1, "yaw=Z- pitch=Y+"),
    ("z", "y", -1, -1, "yaw=Z- pitch=Y-"),
    ("z", "y", +1, +1, "yaw=Z+ pitch=Y+"),
    ("z", "y", +1, -1, "yaw=Z+ pitch=Y-"),
    ("y", "x", -1, +1, "yaw=Y- pitch=X+"),
    ("x", "y", -1, +1, "yaw=X- pitch=Y+"),
]


def open_camera(index: int = 0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--reachy-camera", action="store_true")
    parser.add_argument("--no-gyro", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("GYRO-STABILIZED FACE TRACKING TEST")
    print("=" * 60)
    print("  q/ESC=quit  p=lock  r=reset  a=axis  s=toggle stab")
    print()

    cap = None
    reachy = None
    reachy_ctx = None
    get_frame = None

    if args.reachy_camera:
        print("Connecting to Reachy daemon camera...")
        try:
            from reachy_mini import ReachyMini
            reachy_ctx = ReachyMini(automatic_body_yaw=False, host="localhost")
            reachy = reachy_ctx.__enter__()
        except Exception as e:
            print(f"Failed to connect to Reachy: {e}")
            return 1
        print("Waiting for camera frame...")
        probe = None
        for _ in range(40):
            probe = reachy.media.get_frame()
            if probe is not None:
                break
            time.sleep(0.1)
        if probe is None:
            print("Reachy camera: no frame from daemon")
            reachy_ctx.__exit__(None, None, None)
            return 1
        print(f"Reachy camera ready: {probe.shape[1]}x{probe.shape[0]}")
        get_frame = lambda: reachy.media.get_frame()
    else:
        cap = open_camera(args.camera)
        if cap is None:
            print("Failed to open camera")
            return 1
        def _read():
            ok, frame = cap.read()
            return frame if ok else None
        get_frame = _read

    model_path = ensure_pose_model()
    device = pick_device()
    print(f"YOLOv8-Pose on {device.upper()} (imgsz={YOLO_IMGSZ}) ...")
    tracker = CascadeTracker(model_path, device, YOLO_IMGSZ, YOLO_CONF)

    probe = get_frame()
    if probe is None:
        if cap:
            cap.release()
        if reachy_ctx:
            reachy_ctx.__exit__(None, None, None)
        return 1
    h, w = probe.shape[:2]
    tracker.warmup(h, w)
    tracker.start()

    gyro = None
    if not args.no_gyro:
        gyro = HeadGyro()
        if not gyro.start():
            print("Gyro failed - YOLO-only mode")
            gyro = None

    mapping_idx = 0
    stabilization_on = True
    H_FOV_DEG = 90.0
    V_FOV_DEG = 55.0
    VERT_SIGN = -1.0
    ROLL_SIGN = +1.0

    # Anchor: (face_cx, face_cy, heading, roll, pitch) set on 'p' or auto on first detection
    anchor = None

    last_t = time.time()

    print("Ready. Green=stabilized  Red=raw  p=lock anchor")
    print()

    try:
        while True:
            frame = get_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            now = time.time()
            dt = now - last_t
            last_t = now

            tracker.submit(frame)
            det = tracker.get()

            gyro_data = gyro.get_latest() if gyro else None
            yaw_axis, pitch_axis, yaw_sign, pitch_sign, label = AXIS_MAPPINGS[mapping_idx]

            # Auto-anchor: if we have a face detection but no anchor, lock it now
            if anchor is None and det is not None and det.tier != "none" and gyro_data is not None:
                x1, y1, x2, y2, tx, ty = det.bbox_px
                anchor = (tx, ty, gyro_data.heading, gyro_data.roll, gyro_data.pitch)
                print(f"Auto-anchor at face=({tx},{ty}) "
                      f"heading={gyro_data.heading:.1f} "
                      f"roll={gyro_data.roll:.1f} pitch={gyro_data.pitch:.1f}")

            # --- Compute stabilized position from anchor ---
            stab_dx = 0.0
            stab_dy = 0.0
            if anchor is not None and gyro_data is not None and stabilization_on:
                a_cx, a_cy, a_head, a_roll, a_pitch = anchor

                # Yaw delta (handle 360 wrap)
                d_yaw = (gyro_data.heading - a_head + 540.0) % 360.0 - 180.0
                # Vertical delta (BNO055 roll = head nod axis)
                d_vert = gyro_data.roll - a_roll
                if d_vert > 180.0:
                    d_vert -= 360.0
                elif d_vert < -180.0:
                    d_vert += 360.0
                # Camera-roll delta (BNO055 pitch = head sideways tilt)
                d_camroll = gyro_data.pitch - a_pitch
                if d_camroll > 180.0:
                    d_camroll -= 360.0
                elif d_camroll < -180.0:
                    d_camroll += 360.0

                px_per_deg_x = w / H_FOV_DEG
                px_per_deg_y = h / V_FOV_DEG

                ox = d_yaw * px_per_deg_x
                oy = VERT_SIGN * d_vert * px_per_deg_y

                # Rotate by camera roll
                roll_rad = math.radians(ROLL_SIGN * d_camroll)
                cos_r = math.cos(roll_rad)
                sin_r = math.sin(roll_rad)
                stab_dx = ox * cos_r - oy * sin_r
                stab_dy = ox * sin_r + oy * cos_r

            # --- Draw center reference (blue) ---
            cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (255, 0, 0), 2)
            cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 6, (255, 0, 0), 2)

            # --- YOLO detection overlay ---
            if det is not None and det.tier != "none":
                x1, y1, x2, y2, tx, ty = det.bbox_px

                # Raw (RED)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(frame, (tx, ty), 5, (0, 0, 255), -1)
                cv2.putText(frame, "RAW", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Stabilized (GREEN) - using BNO055 anchor projection
                if anchor is not None and stabilization_on:
                    a_cx, a_cy = anchor[0], anchor[1]
                    bw = x2 - x1
                    bh = y2 - y1
                    stx = int(a_cx + stab_dx)
                    sty = int(a_cy + stab_dy)
                    sx1 = stx - bw // 2
                    sy1 = sty - bh // 2
                    sx2 = stx + bw // 2
                    sy2 = sty + bh // 2
                    sx1 = max(0, min(w - 1, sx1))
                    sy1 = max(0, min(h - 1, sy1))
                    sx2 = max(0, min(w - 1, sx2))
                    sy2 = max(0, min(h - 1, sy2))
                    stx = max(0, min(w - 1, stx))
                    sty = max(0, min(h - 1, sty))

                    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
                    cv2.circle(frame, (stx, sty), 6, (0, 255, 0), -1)
                    cv2.putText(frame, "STABILIZED", (sx1, sy2 + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.line(frame, (tx, ty), (stx, sty), (0, 255, 0), 1, cv2.LINE_AA)

            # --- HUD ---
            def put(txt, row, color=(255, 255, 255)):
                y = 25 + row * 22
                cv2.putText(frame, txt, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, txt, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

            put(f"Mapping: {label}", 0, (0, 255, 255))
            stab_status = "ON" if stabilization_on else "OFF"
            put(f"Stabilization: {stab_status}", 1, (0, 255, 0) if stabilization_on else (0, 0, 255))
            if gyro_data:
                put(f"Heading: {gyro_data.heading:+7.1f} deg", 2, (200, 255, 200))
                put(f"Roll:    {gyro_data.roll:+7.1f} deg", 3, (200, 255, 200))
                put(f"Pitch:   {gyro_data.pitch:+7.1f} deg", 4, (200, 255, 200))
                if anchor:
                    put(f"Anchor: h={anchor[2]:+.0f} r={anchor[3]:+.0f} p={anchor[4]:+.0f}", 5, (255, 200, 100))
                    put(f"Delta:  dyaw={gyro_data.heading - anchor[2]:+.1f} dvert={gyro_data.roll - anchor[3]:+.1f}", 6, (100, 200, 255))
            else:
                put("NO GYRO DATA", 2, (0, 0, 255))

            cv2.imshow("Gyro-Stabilized Face Tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("p"):
                if det is not None and det.tier != "none" and gyro_data is not None:
                    x1, y1, x2, y2, tx, ty = det.bbox_px
                    anchor = (tx, ty, gyro_data.heading, gyro_data.roll, gyro_data.pitch)
                    print(f"Anchor locked at face=({tx},{ty}) "
                          f"heading={gyro_data.heading:.1f} "
                          f"roll={gyro_data.roll:.1f} pitch={gyro_data.pitch:.1f}")
                else:
                    print("Cannot lock: no face or gyro data")
            elif key == ord("r"):
                anchor = None
                print("Anchor reset")
            elif key == ord("a"):
                mapping_idx = (mapping_idx + 1) % len(AXIS_MAPPINGS)
                anchor = None
                print(f"Mapping -> {AXIS_MAPPINGS[mapping_idx][4]}")
            elif key == ord("s"):
                stabilization_on = not stabilization_on
                print(f"Stabilization: {'ON' if stabilization_on else 'OFF'}")

    except KeyboardInterrupt:
        pass
    finally:
        if gyro:
            try:
                gyro.stop()
            except Exception:
                pass
        tracker.stop()
        tracker.join(timeout=1.0)
        if cap is not None:
            cap.release()
        if reachy_ctx is not None:
            try:
                reachy_ctx.__exit__(None, None, None)
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
