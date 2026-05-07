"""Face-follow using BNO055 fused angles (like test_gyro_visual.py world point).

Approach
--------
1. Read BNO055 fused euler: heading=yaw, roll=nod(pitch).
2. Convert face visual error to a WORLD direction:
       world_yaw   = heading + err_x * (HFOV/2)
       world_pitch = roll*VERT_SIGN + err_y * (VFOV/2)
3. EMA-smooth the world target (so it does not jitter with head motion).
4. Residual error = (world_target - current_head_pose) back to [-1,1].

This is the SAME math that keeps the world-fixed blue dot stable in
test_gyro_visual.py when you press 'p'.

Controls
--------
q/ESC  quit
g      toggle gyro compensation on/off
[ / ]  decrease / increase EMA alpha (more smoothing vs more responsive)
r      recenter head
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import cv2
import numpy as np

from hsafa_robot.head_gyro import HeadGyro
from hsafa_robot.tracker import (
    CascadeTracker,
    YOLO_CONF,
    YOLO_IMGSZ,
    ensure_pose_model,
    pick_device,
)
from hsafa_robot.robot_control import head_pose
from reachy_mini import ReachyMini


# ---- Calibrated from test_gyro_visual.py ----------------------------
HFOV = 90.0
VFOV = 55.0
VERT_SIGN = -1.0   # roll decreases when head pitches down

# ---- P-controller ----------------------------------------------------
KP_YAW = 0.6
KP_PITCH = 0.4
STEP_SCALE = 0.2
CMD_ALPHA = 0.4      # motor command smoothing
YAW_LIMIT = math.radians(60)
PITCH_LIMIT = math.radians(30)
DEADZONE = 0.03


def _wrap180(deg: float) -> float:
    return ((deg + 540.0) % 360.0) - 180.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reachy-camera", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-gyro", action="store_true",
                    help="Start with compensation OFF.")
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="World-target EMA alpha [0-1]. Lower = smoother.")
    args = ap.parse_args()

    print("=" * 60)
    print("FACE FOLLOW  –  world-target via BNO055 fused angles")
    print("=" * 60)
    print(f"HFOV={HFOV}  VFOV={VFOV}  VERT_SIGN={VERT_SIGN:+.0f}  alpha={args.alpha}")
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

    # ---- Gyro (BNO055) -------------------------------------------
    print("Starting gyro...")
    gyro = HeadGyro()
    gyro_ok = gyro.start()
    print(f"Gyro: {'OK' if gyro_ok else 'FAILED'}")

    # ---- State ----------------------------------------------------
    use_gyro = (not args.no_gyro) and gyro_ok
    alpha = args.alpha

    # Rate-integrated head pose (degrees).  Smoother than fused roll.
    # Signs from test_gyro_visual.py default mapping ("z","x",-1,+1).
    YAW_RATE_SIGN = -1.0
    PITCH_RATE_SIGN = +1.0
    head_yaw = 0.0
    head_pitch = 0.0

    # World-target state (degrees).
    world_yaw: float | None = None
    world_pitch: float | None = None

    # Motor state.
    cmd_yaw = 0.0
    cmd_pitch = 0.0
    sent_yaw = 0.0
    sent_pitch = 0.0
    last_t = time.time()

    # Wiggle metric.
    yaw_history = []
    last_log = 0.0

    print()
    print("Press 'g' to toggle. '[' / ']' for alpha. 'q' to quit.")
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

            data = gyro.get_latest() if gyro_ok else None

            # ---- Integrate head pose from gyro rates -----------
            if data is not None:
                head_yaw += float(data.gyro_z) * YAW_RATE_SIGN * dt
                head_pitch += float(data.gyro_x) * PITCH_RATE_SIGN * dt
                # Gentle decay to prevent unbounded drift.
                head_yaw *= 0.999
                head_pitch *= 0.999

            if have_face and data is not None:
                raw_err_x = det.err_x
                raw_err_y = det.err_y

                # ---- Visual error -> angular offset ----
                vis_yaw = raw_err_x * (HFOV / 2.0)
                vis_pitch = raw_err_y * (VFOV / 2.0)

                # ---- Current measurement of world direction ----
                meas_yaw = head_yaw + vis_yaw
                meas_pitch = head_pitch + vis_pitch

                if world_yaw is None:
                    world_yaw = meas_yaw
                    world_pitch = meas_pitch
                else:
                    dyaw = meas_yaw - world_yaw
                    dpitch = meas_pitch - world_pitch
                    world_yaw += alpha * dyaw
                    world_pitch += alpha * dpitch

                # ---- Residual = target minus current head ----
                res_err_x = (world_yaw - head_yaw) / (HFOV / 2.0)
                res_err_y = (world_pitch - head_pitch) / (VFOV / 2.0)
            else:
                # No face → let target drift toward current head.
                if world_yaw is not None and data is not None:
                    world_yaw = head_yaw
                    world_pitch = head_pitch

            # ---- P-controller ----------------------------------
            if abs(res_err_x) > DEADZONE:
                cmd_yaw += -1.0 * KP_YAW * res_err_x * STEP_SCALE
            if abs(res_err_y) > DEADZONE:
                cmd_pitch += +1.0 * KP_PITCH * res_err_y * STEP_SCALE

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
                wy = world_yaw or 0
                wp = world_pitch or 0
                print(
                    f"[{tag}] {face} raw=({raw_err_x:+.2f},{raw_err_y:+.2f}) "
                    f"res=({res_err_x:+.2f},{res_err_y:+.2f}) "
                    f"yaw={math.degrees(sent_yaw):+6.1f} "
                    f"pitch={math.degrees(sent_pitch):+6.1f} "
                    f"wiggle={wiggle:.2f}deg "
                    f"head=({head_yaw:+.1f},{head_pitch:+.1f}) "
                    f"tgt=({wy:+.1f},{wp:+.1f})",
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
            put(0, f"MODE: {'GYRO' if use_gyro else 'RAW'} (press g)",
                mode_color)
            put(1, f"raw err  : ({raw_err_x:+.3f}, {raw_err_y:+.3f})")
            put(2, f"res err  : ({res_err_x:+.3f}, {res_err_y:+.3f})",
                (100, 255, 255))
            put(3, f"cmd yaw/pitch: "
                   f"{math.degrees(sent_yaw):+6.1f} / "
                   f"{math.degrees(sent_pitch):+6.1f}")
            if data is not None:
                put(4, f"head y/p int: {head_yaw:+7.1f} / {head_pitch:+7.1f}",
                    (200, 255, 200))
            if world_yaw is not None:
                put(5, f"tgt h/r*sgn : {world_yaw:+7.1f} / {world_pitch:+7.1f}",
                    (255, 200, 100))
            put(6, f"wiggle (std, 2s): {wiggle:.2f} deg", (255, 255, 100))
            put(7, f"alpha={alpha:.2f} ySign={YAW_RATE_SIGN:+.0f} pSign={PITCH_RATE_SIGN:+.0f}",
                (255, 200, 200))

            cv2.imshow("Face Follow – World Target", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("g"):
                use_gyro = not use_gyro and gyro_ok
                if not use_gyro:
                    world_yaw = None
                    world_pitch = None
                yaw_history = []
                print(f"\n>>> {'GYRO' if use_gyro else 'RAW'} <<<\n")
            elif key == ord("["):
                alpha = max(0.05, alpha - 0.05)
                print(f"alpha -> {alpha:.2f}")
            elif key == ord("]"):
                alpha = min(1.0, alpha + 0.05)
                print(f"alpha -> {alpha:.2f}")
            elif key == ord("1"):
                YAW_RATE_SIGN = -YAW_RATE_SIGN
                head_yaw = 0.0
                world_yaw = None
                print(f"Yaw rate sign -> {YAW_RATE_SIGN:+.0f}")
            elif key == ord("2"):
                PITCH_RATE_SIGN = -PITCH_RATE_SIGN
                head_pitch = 0.0
                world_pitch = None
                print(f"Pitch rate sign -> {PITCH_RATE_SIGN:+.0f}")
            elif key == ord("r"):
                cmd_yaw = 0.0
                cmd_pitch = 0.0
                sent_yaw = 0.0
                sent_pitch = 0.0
                head_yaw = 0.0
                head_pitch = 0.0
                world_yaw = None
                world_pitch = None
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
