"""Visual gyro test - camera feed with gyro overlay.

Shows:
- Live camera feed
- Real-time gyro X/Y/Z values
- Heading/Roll/Pitch (BNO055 fused euler angles)
- A moving crosshair representing where head "points" by integrating gyro
- The image center as a fixed reference

Use this to verify:
1. Gyro axes are not reversed
2. Yaw rotation moves the marker correctly
3. Pitch rotation moves the marker correctly

Controls:
- 'q' or ESC: quit
- 'r': reset crosshair to center
- 'a': switch axis mapping (try different gyro->screen axis combos)
- 'p': place a WORLD-FIXED point at current center (stays put when robot rotates)
- 'c': clear the world-fixed point
"""

import argparse
import cv2
import time
import sys
import math
import numpy as np

from hsafa_robot.head_gyro import HeadGyro


def open_camera(index: int = 0):
    """Open camera at the given index."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


# Different axis mappings to test
AXIS_MAPPINGS = [
    # (yaw_axis, pitch_axis, yaw_sign, pitch_sign, label)
    # Default: yaw flipped (matches user's mounting)
    ("z", "x", -1, +1, "yaw=Z- pitch=X+"),
    ("z", "x", -1, -1, "yaw=Z- pitch=X-"),
    ("z", "y", -1, +1, "yaw=Z- pitch=Y+"),
    ("z", "y", -1, -1, "yaw=Z- pitch=Y-"),
    ("z", "y", +1, +1, "yaw=Z+ pitch=Y+"),
    ("z", "y", +1, -1, "yaw=Z+ pitch=Y-"),
    ("y", "x", -1, +1, "yaw=Y- pitch=X+"),
    ("x", "y", -1, +1, "yaw=X- pitch=Y+"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reachy-camera", action="store_true",
                        help="Use the Reachy daemon camera instead of local OpenCV.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Local camera index (when not using --reachy-camera).")
    args = parser.parse_args()

    print("=" * 60)
    print("GYRO VISUAL TEST")
    print("=" * 60)
    print()
    print("Controls:")
    print("  q/ESC - quit")
    print("  r     - reset crosshair to center")
    print("  a     - cycle axis mapping (test if reversed)")
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
            print(f"❌ Failed to connect to Reachy: {e}")
            return 1

        # Wait for first frame
        print("Waiting for camera frame...")
        probe = None
        for _ in range(40):
            probe = reachy.media.get_frame()
            if probe is not None:
                break
            time.sleep(0.1)
        if probe is None:
            print("❌ Reachy camera: no frame from daemon")
            reachy_ctx.__exit__(None, None, None)
            return 1
        print(f"✅ Reachy camera ready: {probe.shape[1]}x{probe.shape[0]}")
        get_frame = lambda: reachy.media.get_frame()  # noqa: E731
    else:
        print("Opening camera...")
        cap = open_camera(args.camera)
        if cap is None:
            print("❌ Failed to open camera")
            return 1

        def _read():
            ok, frame = cap.read()
            return frame if ok else None
        get_frame = _read

    # Start gyro reader
    print("Starting gyro reader...")
    gyro = HeadGyro()
    if not gyro.start():
        print("❌ Failed to start gyro")
        if cap:
            cap.release()
        if reachy_ctx:
            reachy_ctx.__exit__(None, None, None)
        return 1

    print("✅ Ready. Move the head and watch the orange crosshair move.")
    print("   The orange crosshair should follow the head's rotation direction.")
    print("   The green '+' is the fixed image center.")
    print()

    # Crosshair position (accumulated from gyro)
    cross_x = 0.0  # offset from center in pixels
    cross_y = 0.0
    last_t = time.time()

    # Pixels per degree (sensitivity for visualization)
    PX_PER_DEG = 5.0

    # Decay factor (so it drifts back to center if not moving)
    DECAY = 0.98

    # Axis mapping
    mapping_idx = 0

    # World-fixed anchor: BNO055 (heading_deg, pitch_deg) at the moment
    # the user pressed 'p'. Re-projected each frame as the head moves.
    anchor = None  # tuple (heading_deg, pitch_deg)

    # Camera FOV (deg). Reachy daemon camera is a wide lens
    # (~1920x1080). Adjust with '+' / '-' keys at runtime if needed.
    H_FOV_DEG = 90.0
    V_FOV_DEG = 55.0
    # Vertical sign for the world-fixed point (BNO055 roll axis).
    VERT_SIGN = -1.0
    # Camera-roll sign (BNO055 pitch axis = head sideways tilt).
    ROLL_SIGN = +1.0

    try:
        while True:
            frame = get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Mirror so it feels natural
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2

            now = time.time()
            dt = now - last_t
            last_t = now

            # Get gyro data
            data = gyro.get_latest()

            yaw_axis, pitch_axis, yaw_sign, pitch_sign, label = AXIS_MAPPINGS[mapping_idx]

            if data is not None:
                # Get rates from configured axes
                yaw_rate = getattr(data, f"gyro_{yaw_axis}") * yaw_sign
                pitch_rate = getattr(data, f"gyro_{pitch_axis}") * pitch_sign

                # Integrate (degrees)
                cross_x += yaw_rate * dt * PX_PER_DEG
                cross_y += pitch_rate * dt * PX_PER_DEG

                # Decay slightly so it drifts back if no motion
                cross_x *= DECAY
                cross_y *= DECAY

                # Clamp inside frame
                cross_x = max(-cx + 20, min(cx - 20, cross_x))
                cross_y = max(-cy + 20, min(cy - 20, cross_y))

            # Draw image center (green +)
            cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
            cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), 2)
            cv2.putText(frame, "CENTER", (cx + 25, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # ---- World-fixed anchor point (blue) -----------------------
            # Project the saved (heading, vertical) back into screen
            # space using current head orientation.
            #
            # NOTE on BNO055 mounting in Reachy head: the chip's "pitch"
            # euler axis is NOT aligned with the head's nodding axis;
            # the head's nod actually rotates the chip's ROLL axis.
            # So we use `data.roll` for vertical tracking.
            if anchor is not None and data is not None:
                a_head, a_vert, a_camroll = anchor
                # Yaw delta accounting for 360->0 wrap.
                d_yaw = (data.heading - a_head + 540.0) % 360.0 - 180.0
                d_vert = data.roll - a_vert
                if d_vert > 180.0:
                    d_vert -= 360.0
                elif d_vert < -180.0:
                    d_vert += 360.0
                # Camera-roll delta (head tilting sideways).
                d_camroll = data.pitch - a_camroll
                if d_camroll > 180.0:
                    d_camroll -= 360.0
                elif d_camroll < -180.0:
                    d_camroll += 360.0

                px_per_deg_x = w / H_FOV_DEG
                px_per_deg_y = h / V_FOV_DEG
                # Compute base offset from camera center (yaw + vert).
                ox = d_yaw * px_per_deg_x
                oy = VERT_SIGN * d_vert * px_per_deg_y
                # Rotate offset around camera center by camera-roll.
                # When head tilts right, world points appear rotated
                # left in the image (so apply negative angle).
                roll_rad = math.radians(ROLL_SIGN * d_camroll)
                cos_r = math.cos(roll_rad)
                sin_r = math.sin(roll_rad)
                rx = ox * cos_r - oy * sin_r
                ry = ox * sin_r + oy * cos_r
                ax = int(cx + rx)
                ay = int(cy + ry)

                # Only draw if inside the frame.
                if 0 <= ax < w and 0 <= ay < h:
                    cv2.circle(frame, (ax, ay), 14, (255, 100, 0), 2)
                    cv2.circle(frame, (ax, ay), 4, (255, 100, 0), -1)
                    cv2.putText(frame, "WORLD", (ax + 18, ay + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 100, 0), 1, cv2.LINE_AA)
                else:
                    # Off-screen: draw arrow at edge pointing toward it.
                    arrow_x = max(15, min(w - 15, ax))
                    arrow_y = max(15, min(h - 15, ay))
                    cv2.arrowedLine(frame, (cx, cy),
                                    (arrow_x, arrow_y),
                                    (255, 100, 0), 2, tipLength=0.1)
                    cv2.putText(frame, "WORLD (off)", (arrow_x - 50, arrow_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 100, 0), 1, cv2.LINE_AA)

            # Draw gyro crosshair (orange)
            mx = int(cx + cross_x)
            my = int(cy + cross_y)
            cv2.line(frame, (mx - 15, my), (mx + 15, my), (0, 165, 255), 3)
            cv2.line(frame, (mx, my - 15), (mx, my + 15), (0, 165, 255), 3)
            cv2.circle(frame, (mx, my), 8, (0, 165, 255), 2)

            # Line from center to crosshair
            cv2.line(frame, (cx, cy), (mx, my), (0, 165, 255), 1, cv2.LINE_AA)

            # Draw text overlay
            y0 = 25
            line_h = 22

            def put(label, value, color=(255, 255, 255), idx=0):
                cv2.putText(frame, f"{label}: {value}",
                            (10, y0 + idx * line_h),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3,
                            cv2.LINE_AA)
                cv2.putText(frame, f"{label}: {value}",
                            (10, y0 + idx * line_h),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1,
                            cv2.LINE_AA)

            if data is not None:
                put("Mapping", label, (0, 255, 255), 0)
                put("Gyro X (deg/s)", f"{data.gyro_x:+7.2f}",
                    (255, 200, 100), 1)
                put("Gyro Y (deg/s)", f"{data.gyro_y:+7.2f}",
                    (255, 200, 100), 2)
                put("Gyro Z (deg/s)", f"{data.gyro_z:+7.2f}",
                    (255, 200, 100), 3)
                put("Heading (deg)", f"{data.heading:+7.1f}",
                    (200, 255, 200), 4)
                put("Roll    (deg)", f"{data.roll:+7.1f}",
                    (200, 255, 200), 5)
                put("Pitch   (deg)", f"{data.pitch:+7.1f}",
                    (200, 255, 200), 6)
                cal = f"S{data.cal_sys} G{data.cal_gyro} A{data.cal_acc} M{data.cal_mag}"
                put("Cal", cal, (255, 255, 100), 7)
                put("Offset (px)", f"x={int(cross_x):+4d} y={int(cross_y):+4d}",
                    (100, 200, 255), 8)
            else:
                put("WAITING FOR GYRO DATA...", "", (0, 0, 255), 0)

            # Bottom hint
            anchor_hint = " | press 'p' to place WORLD point" if anchor is None else " | 'c' clears WORLD point"
            hint = "Move head LEFT -> orange should move LEFT" + anchor_hint
            cv2.putText(frame, hint, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, hint, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)

            cv2.imshow("Gyro Visual Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("r"):
                cross_x = 0.0
                cross_y = 0.0
                print("Reset crosshair to center")
            elif key == ord("a"):
                mapping_idx = (mapping_idx + 1) % len(AXIS_MAPPINGS)
                cross_x = 0.0
                cross_y = 0.0
                print(f"Switched mapping -> {AXIS_MAPPINGS[mapping_idx][4]}")
            elif key == ord("p"):
                if data is not None:
                    anchor = (data.heading, data.roll, data.pitch)
                    print(f"World point placed at heading={data.heading:.1f} "
                          f"vert={data.roll:.1f} camroll={data.pitch:.1f}")
                else:
                    print("Cannot place point: no gyro data yet")
            elif key == ord("c"):
                anchor = None
                print("World point cleared")
            elif key == ord("v"):
                VERT_SIGN = -VERT_SIGN
                print(f"Vertical sign flipped -> {VERT_SIGN:+.0f}")
            elif key == ord("t"):
                ROLL_SIGN = -ROLL_SIGN
                print(f"Roll sign flipped -> {ROLL_SIGN:+.0f}")
            elif key == ord("+") or key == ord("="):
                H_FOV_DEG += 5
                V_FOV_DEG += 3
                print(f"FOV: H={H_FOV_DEG} V={V_FOV_DEG}")
            elif key == ord("-") or key == ord("_"):
                H_FOV_DEG = max(20, H_FOV_DEG - 5)
                V_FOV_DEG = max(15, V_FOV_DEG - 3)
                print(f"FOV: H={H_FOV_DEG} V={V_FOV_DEG}")

    except KeyboardInterrupt:
        pass
    finally:
        try:
            gyro.stop()
        except Exception:
            pass
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
