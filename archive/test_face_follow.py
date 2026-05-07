"""Isolated face-follow test with gyro stabilization on/off toggle.

Uses MOTION-ADAPTIVE smoothing (not rate integration):
- Head moving fast  → smooth MORE (trust visual error less)
- Head stable       → smooth LESS (responsive)

Controls
--------
q / ESC : quit
g       : toggle gyro stabilization on/off
+ / -   : adjust motion threshold (higher = more smoothing)
[ / ]   : adjust base alpha (higher = more responsive)
r       : recenter the head (cmd to 0,0)
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import cv2
import numpy as np

from hsafa_robot.gyro_stabilizer import GyroStabilizer
from hsafa_robot.tracker import (
    CascadeTracker,
    YOLO_CONF,
    YOLO_IMGSZ,
    ensure_pose_model,
    pick_device,
)
from hsafa_robot.robot_control import head_pose
from reachy_mini import ReachyMini


# ---- P-controller tuning ----------------------------------------------------
KP_YAW = 0.6
KP_PITCH = 0.4
STEP_SCALE = 0.2
ERR_ALPHA = 0.6
CMD_ALPHA = 0.4

YAW_SIGN_REACHY = -1.0
PITCH_SIGN_REACHY = +1.0
YAW_LIMIT = math.radians(60)
PITCH_LIMIT = math.radians(30)
DEADZONE = 0.03


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reachy-camera", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-gyro", action="store_true",
                    help="Start with gyro stabilization OFF.")
    ap.add_argument("--motion-threshold", type=float, default=40.0,
                    help="Gyro magnitude (deg/s) where smoothing maxes out.")
    ap.add_argument("--base-alpha", type=float, default=0.5,
                    help="Default EMA alpha when head is still [0-1].")
    args = ap.parse_args()

    print("=" * 60)
    print("FACE FOLLOW TEST (gyro motion-adaptive smoothing)")
    print("=" * 60)
    print(f"motion_threshold={args.motion_threshold:.0f} "
          f"base_alpha={args.base_alpha:.2f}")
    print()

    # ---- Reachy ----------------------------------------------------
    print("Connecting to Reachy...")
    reachy_ctx = ReachyMini(automatic_body_yaw=False, host="localhost")
    reachy = reachy_ctx.__enter__()

    # ---- Camera ----------------------------------------------------
    if args.reachy_camera:
        print("Using Reachy daemon camera.")
        probe = None
        for _ in range(40):
            probe = reachy.media.get_frame()
            if probe is not None:
                break
            time.sleep(0.1)
        if probe is None:
            print("No frame from Reachy camera.")
            reachy_ctx.__exit__(None, None, None)
            return 1
        get_frame = lambda: reachy.media.get_frame()  # noqa: E731
        frame_h, frame_w = probe.shape[:2]
    else:
        print(f"Using local camera index {args.camera}.")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("Could not open camera.")
            reachy_ctx.__exit__(None, None, None)
            return 1
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        def _read():
            ok, f = cap.read()
            return f if ok else None
        get_frame = _read
        ok, probe = cap.read()
        if not ok:
            return 1
        frame_h, frame_w = probe.shape[:2]

    print(f"Camera: {frame_w}x{frame_h}")

    # ---- Face tracker ---------------------------------------------
    print("Loading YOLO model...")
    model_path = ensure_pose_model()
    device = pick_device()
    tracker = CascadeTracker(model_path, device, YOLO_IMGSZ, YOLO_CONF)
    tracker.warmup(frame_h, frame_w)
    tracker.start()

    # ---- Gyro -----------------------------------------------------
    print("Starting gyro...")
    gyro = GyroStabilizer(
        base_alpha=args.base_alpha,
        motion_threshold_deg_s=args.motion_threshold,
    )
    gyro_ok = gyro.start()
    print(f"Gyro: {'OK' if gyro_ok else 'FAILED'}")

    # ---- State ----------------------------------------------------
    use_gyro = (not args.no_gyro) and gyro_ok

    cmd_yaw = 0.0
    cmd_pitch = 0.0
    sent_yaw = 0.0
    sent_pitch = 0.0
    err_x_s = 0.0
    err_y_s = 0.0
    last_t = time.time()

    last_log = 0.0
    yaw_history = []

    print()
    print("Press 'g' to toggle gyro. Press 'q' to quit.")
    print("Press '+'/'-' to adjust motion threshold.")
    print("Press '['/']' to adjust base alpha.")
    print()

    try:
        while True:
            frame = get_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            now = time.time()
            dt = max(1e-3, now - last_t)
            last_t = now

            tracker.submit(frame)
            det = tracker.get()

            have_face = det is not None and (now - det.timestamp) < 0.6
            raw_err_x = raw_err_y = 0.0
            res_err_x = res_err_y = 0.0

            if have_face:
                raw_err_x = det.err_x
                raw_err_y = det.err_y

                if use_gyro and gyro_ok:
                    stab = gyro.compensate_head_motion(raw_err_x, raw_err_y, dt)
                    res_err_x, res_err_y = stab.err_x, stab.err_y
                else:
                    # Simple EMA when gyro off.
                    err_x_s = (1 - ERR_ALPHA) * err_x_s + ERR_ALPHA * raw_err_x
                    err_y_s = (1 - ERR_ALPHA) * err_y_s + ERR_ALPHA * raw_err_y
                    res_err_x, res_err_y = err_x_s, err_y_s

            # ---- P-controller ----------------------------------
            if not (use_gyro and gyro_ok):
                # When gyro is on, the stabilizer already smooths.
                # When off, we use the EMA above.
                pass
            else:
                err_x_s = res_err_x
                err_y_s = res_err_y

            if have_face:
                if abs(err_x_s) > DEADZONE:
                    cmd_yaw += YAW_SIGN_REACHY * KP_YAW * err_x_s * STEP_SCALE
                if abs(err_y_s) > DEADZONE:
                    cmd_pitch += PITCH_SIGN_REACHY * KP_PITCH * err_y_s * STEP_SCALE

            cmd_yaw = max(-YAW_LIMIT, min(YAW_LIMIT, cmd_yaw))
            cmd_pitch = max(-PITCH_LIMIT, min(PITCH_LIMIT, cmd_pitch))

            sent_yaw = (1 - CMD_ALPHA) * sent_yaw + CMD_ALPHA * cmd_yaw
            sent_pitch = (1 - CMD_ALPHA) * sent_pitch + CMD_ALPHA * cmd_pitch

            try:
                reachy.set_target(
                    head=head_pose(yaw=sent_yaw, pitch=sent_pitch),
                    body_yaw=0.0,
                    antennas=[0.0, 0.0],
                )
            except Exception as e:
                print(f"set_target failed: {e}")

            # ---- Wiggle metric ---------------------------------
            yaw_history.append(math.degrees(sent_yaw))
            if len(yaw_history) > 60:
                yaw_history = yaw_history[-60:]
            wiggle = float(np.std(yaw_history)) if yaw_history else 0.0

            # ---- Heartbeat log ---------------------------------
            if now - last_log > 0.5:
                last_log = now
                tag = "GYRO" if use_gyro else "RAW "
                face = "FACE" if have_face else "----"
                dbg = gyro.get_debug() if use_gyro else {}
                print(
                    f"[{tag}] {face} raw=({raw_err_x:+.2f},{raw_err_y:+.2f}) "
                    f"res=({res_err_x:+.2f},{res_err_y:+.2f}) "
                    f"yaw={math.degrees(sent_yaw):+6.1f} "
                    f"pitch={math.degrees(sent_pitch):+6.1f} "
                    f"wiggle={wiggle:.2f}deg "
                    f"mot={dbg.get('motion',0):.1f} "
                    f"alpha={dbg.get('alpha',ERR_ALPHA):.2f}",
                )

            # ---- Display --------------------------------------
            display = cv2.flip(frame, 1).copy()
            h, w = display.shape[:2]
            cx, cy = w // 2, h // 2

            # Center crosshair (green)
            cv2.drawMarker(display, (cx, cy), (0, 255, 0),
                           cv2.MARKER_CROSS, 30, 2)

            # Face bbox (red)
            if det is not None and det.bbox_px is not None:
                x1, y1, x2, y2, _, _ = det.bbox_px
                mx1 = w - 1 - x2
                mx2 = w - 1 - x1
                cv2.rectangle(display, (mx1, y1), (mx2, y2), (0, 0, 255), 2)
                cv2.putText(display, "FACE", (mx1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # HUD
            def put(line, text, color=(255, 255, 255)):
                y = 22 + line * 22
                cv2.putText(display, text, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3,
                            cv2.LINE_AA)
                cv2.putText(display, text, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1,
                            cv2.LINE_AA)

            mode_color = (0, 255, 0) if use_gyro else (0, 0, 255)
            put(0, f"GYRO: {'ON' if use_gyro else 'OFF'} (press g)",
                mode_color)
            put(1, f"raw err  : ({raw_err_x:+.3f}, {raw_err_y:+.3f})")
            put(2, f"res err  : ({res_err_x:+.3f}, {res_err_y:+.3f})",
                (100, 255, 255))
            put(3, f"cmd yaw/pitch: "
                   f"{math.degrees(sent_yaw):+6.1f} / "
                   f"{math.degrees(sent_pitch):+6.1f}")
            if use_gyro:
                dbg = gyro.get_debug()
                put(4, f"motion={dbg.get('motion',0):.1f} deg/s  "
                       f"alpha={dbg.get('alpha',0):.2f}", (200, 255, 200))
            put(5, f"wiggle (std yaw, last 2s): {wiggle:.2f} deg",
                (255, 255, 100))
            put(6, f"threshold={gyro.motion_threshold:.0f} "
                   f"base_alpha={gyro.base_alpha:.2f} "
                   f'(press +/- [/])', (255, 200, 200))

            cv2.imshow("Face Follow Test", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("g"):
                use_gyro = not use_gyro and gyro_ok
                if not use_gyro:
                    gyro.reset()
                yaw_history = []
                print(f"\n>>> GYRO {'ON' if use_gyro else 'OFF'} <<<\n")
            elif key == ord("+") or key == ord("="):
                gyro.motion_threshold += 5
                print(f"Motion threshold -> {gyro.motion_threshold:.0f}")
            elif key == ord("-") or key == ord("_"):
                gyro.motion_threshold = max(5, gyro.motion_threshold - 5)
                print(f"Motion threshold -> {gyro.motion_threshold:.0f}")
            elif key == ord("["):
                gyro.base_alpha = max(0.05, gyro.base_alpha - 0.05)
                print(f"Base alpha -> {gyro.base_alpha:.2f}")
            elif key == ord("]"):
                gyro.base_alpha = min(1.0, gyro.base_alpha + 0.05)
                print(f"Base alpha -> {gyro.base_alpha:.2f}")
            elif key == ord("r"):
                cmd_yaw = 0.0
                cmd_pitch = 0.0
                sent_yaw = 0.0
                sent_pitch = 0.0
                err_x_s = 0.0
                err_y_s = 0.0
                gyro.reset()
                yaw_history = []
                print("Recentered")

    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        try:
            reachy.goto_target(head=head_pose(),
                               duration=0.6, body_yaw=0.0,
                               antennas=[0.0, 0.0])
        except Exception:
            pass
        try:
            tracker.stop()
        except Exception:
            pass
        try:
            gyro.stop()
        except Exception:
            pass
        if not args.reachy_camera:
            cap.release()
        try:
            reachy_ctx.__exit__(None, None, None)
        except Exception:
            pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
