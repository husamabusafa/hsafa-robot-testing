"""
SAM 3 -> Reachy Mini head follow
================================

Minimal wiring between the SAM 3 native video predictor and the real
Reachy Mini head motors.  No body yaw, no antennas, no animations.  One
thread runs perception, one thread runs the preview/control loop, and
every tick at ~30 Hz we do:

    bbox center - frame center -> normalized error (-1 .. +1)
    EMA smooth the error
    P-controller updates head yaw/pitch (radians) with a small step
    clamp to the head workspace
    reachy.set_target(head=4x4_pose)

SAM 3 on MPS runs at roughly 2-3 Hz; the control loop still runs at
30 Hz because perception and motor control are deliberately decoupled.

Run:

    # real robot, daemon on localhost:
    .venv/bin/python3 examples/sam3_reachy_follow.py --concept person

    # no robot - preview + control math only, no commands sent:
    .venv/bin/python3 examples/sam3_reachy_follow.py --no-reachy

    # spawn a simulated daemon in-process:
    .venv/bin/python3 examples/sam3_reachy_follow.py --sim

Controls (UI):

    Type a concept and press "Lock on" (or Enter).
    "Release" stops following; the head decays back to center.
    "Home" snaps the head to neutral immediately.
    Close the window or Ctrl-C to exit cleanly (head goes home).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from sam3_native_tracker import (
    BBox,
    FollowState,
    Sam3NativeFollower,
    discover_cameras,
    pick_default_camera,
)

STATE_COLORS = {
    FollowState.IDLE: "gray",
    FollowState.LOCKING: "orange",
    FollowState.TRACKING: "lime green",
    FollowState.LOST: "red",
}


# ---------------------------------------------------------------------------
# Camera intrinsics.  Only the horizontal FOV is a real knob; vertical is
# derived per-frame from the aspect ratio.  MEASURE this on the actual
# Reachy Mini camera (see docs / Phase 5 tuning) — the default is a
# generic webcam guess.  Wrong FOV → head over-/under-shoots when the
# body pans.
# ---------------------------------------------------------------------------
HORIZONTAL_FOV_RAD = math.radians(66)

# ---------------------------------------------------------------------------
# Control tuning — world-frame controller.
#
# The head is commanded toward a world-frame target yaw/pitch.  Each SAM
# update reports where the target is relative to the camera; we convert
# that to an angular error, add the current head angle, and EMA that
# value into the world target.  The command = world target, clamped to
# the workspace.  Because the world target is independent of where the
# head is currently pointing, head motion doesn't create a feedback loop.
# ---------------------------------------------------------------------------
WORLD_ALPHA = 0.25                   # EMA factor when adopting a new world target
DEADZONE_RAD = math.radians(1.5)     # don't fight sub-1.5° errors
RECENTER_DECAY = 0.97                # per-tick decay when no target (slow)

# Sign conventions for Reachy Mini with the raw, un-mirrored camera:
#   image +x = right, image +y = down
#   head  +yaw = turn left, +pitch = look down
# So a person to the right (err_x > 0) needs yaw to go negative to center.
YAW_SIGN = -1.0
PITCH_SIGN = +1.0

# Hardware workspace limits (radians).
YAW_LIMIT = math.radians(60)
PITCH_LIMIT = math.radians(30)

# How long to coast on a stale bbox before treating it as "no target".
COAST_S = 0.8


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Reachy gateway.  Thin wrapper so we can run with --no-reachy and see the
# control math in the UI without actually commanding any hardware.
# ---------------------------------------------------------------------------
class _NullReachy:
    """Stand-in when --no-reachy is set.  Accepts the same calls, does nothing."""

    def set_target(self, head=None, body_yaw=None, antennas=None):  # noqa: D401
        return

    def close(self):
        return


def _open_reachy(enable: bool, sim: bool):
    """Return (reachy_like_obj, label).  Never raises - errors degrade to null."""
    if not enable:
        return _NullReachy(), "disabled (--no-reachy)"
    try:
        from reachy_mini import ReachyMini  # type: ignore
    except ImportError as e:
        print(f"[reachy] reachy_mini not importable: {e}; running in display-only")
        return _NullReachy(), f"ImportError: {e}"

    try:
        if sim:
            rm = ReachyMini(spawn_daemon=True, use_sim=True, automatic_body_yaw=False)
            return rm, "simulated (spawn_daemon + use_sim)"
        # Default: expect a daemon already running on localhost.
        rm = ReachyMini(automatic_body_yaw=False)
        return rm, f"connected to {rm.host}:{rm.port}"
    except Exception as e:
        print(f"[reachy] connect failed: {e}; running in display-only")
        return _NullReachy(), f"connect error: {e}"


def _head_pose_rpy(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Build a 4x4 head pose from radians.  We import the SDK helper lazily so
    --no-reachy runs even if reachy_mini isn't available."""
    try:
        from reachy_mini.utils import create_head_pose  # type: ignore

        return create_head_pose(
            roll=roll_rad, pitch=pitch_rad, yaw=yaw_rad, degrees=False
        )
    except ImportError:
        # Fallback implementation - matches reachy_mini.utils.create_head_pose
        # behavior (xyz Euler order), avoids a hard dependency when
        # --no-reachy is used without the SDK installed.
        from scipy.spatial.transform import Rotation as R  # type: ignore

        pose = np.eye(4)
        pose[:3, :3] = R.from_euler(
            "xyz", [roll_rad, pitch_rad, yaw_rad], degrees=False
        ).as_matrix()
        return pose


# ---------------------------------------------------------------------------
# World-frame head controller.
#
# The bbox center is converted to an *angular* target in the world frame
# (independent of where the head is currently pointing).  The commanded
# head angle then approaches that world target every tick via an EMA.
# When the head turns, the world target doesn't move — only the
# camera-frame pixel position does.  That cancels cleanly, which is the
# ego-motion compensation.  No separate `fresh` gate is needed: whether
# SAM updates at 2 Hz or 4 Hz, the world target is whatever SAM last
# reported, and the command smoothly approaches it at the 30 Hz tick
# rate.
# ---------------------------------------------------------------------------
@dataclass
class ControlSnap:
    err_yaw_rad: float       # angular error in the camera frame
    err_pitch_rad: float
    cmd_yaw: float           # commanded head yaw (world frame, radians)
    cmd_pitch: float
    have_target: bool


class HeadFollower:
    def __init__(self) -> None:
        self._target_yaw_world = 0.0    # where we want to look (world frame)
        self._target_pitch_world = 0.0
        self._cmd_yaw = 0.0             # what we'll send to the motor
        self._cmd_pitch = 0.0

    def reset(self) -> None:
        self._target_yaw_world = 0.0
        self._target_pitch_world = 0.0
        self._cmd_yaw = 0.0
        self._cmd_pitch = 0.0

    def step(
        self,
        bbox: Optional[BBox],
        frame_hw: tuple[int, int],
        have_target: bool,
    ) -> ControlSnap:
        H, W = frame_hw
        v_fov = HORIZONTAL_FOV_RAD * (H / W)

        err_yaw_cam = 0.0
        err_pitch_cam = 0.0

        if have_target and bbox is not None:
            # 1. Where is the target in camera-frame angles?
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            err_x_px = cx - W / 2.0
            err_y_px = cy - H / 2.0
            err_yaw_cam = err_x_px / W * HORIZONTAL_FOV_RAD
            err_pitch_cam = err_y_px / H * v_fov

            # 2. Target in WORLD frame = current head angle + target offset.
            #    This is the ego-motion comp, in one line.
            new_target_yaw = self._cmd_yaw + YAW_SIGN * err_yaw_cam
            new_target_pitch = self._cmd_pitch + PITCH_SIGN * err_pitch_cam

            # 3. Deadzone — don't chase tiny errors.
            if (
                abs(err_yaw_cam) > DEADZONE_RAD
                or abs(err_pitch_cam) > DEADZONE_RAD
            ):
                self._target_yaw_world = (
                    (1 - WORLD_ALPHA) * self._target_yaw_world
                    + WORLD_ALPHA * new_target_yaw
                )
                self._target_pitch_world = (
                    (1 - WORLD_ALPHA) * self._target_pitch_world
                    + WORLD_ALPHA * new_target_pitch
                )
        else:
            # No target — decay back to center slowly.
            self._target_yaw_world *= RECENTER_DECAY
            self._target_pitch_world *= RECENTER_DECAY

        # 4. Clamp to workspace.
        self._target_yaw_world = _clamp(
            self._target_yaw_world, -YAW_LIMIT, YAW_LIMIT
        )
        self._target_pitch_world = _clamp(
            self._target_pitch_world, -PITCH_LIMIT, PITCH_LIMIT
        )

        # 5. Commanded angle = world target.  The EMA above already
        #    provides smooth approach, so no second smoothing is needed.
        self._cmd_yaw = self._target_yaw_world
        self._cmd_pitch = self._target_pitch_world

        return ControlSnap(
            err_yaw_rad=err_yaw_cam,
            err_pitch_rad=err_pitch_cam,
            cmd_yaw=self._cmd_yaw,
            cmd_pitch=self._cmd_pitch,
            have_target=have_target,
        )


# ---------------------------------------------------------------------------
# Tk app - camera preview + lock controls + head-angle readout.
# ---------------------------------------------------------------------------
class App:
    CAM_W, CAM_H = 640, 480
    # macOS AVFoundation + some cameras (Reachy Mini, iPhone) ignore the
    # requested frame size and hand back a ~2.5k-wide stream. Preview looks
    # terrible at that resolution, so we downscale the *display* copy only
    # (SAM and the control math still use the full-res frame).
    DISPLAY_MAX_W = 720
    CAP_BACKEND = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY

    def __init__(
        self,
        root: tk.Tk,
        follower: Sam3NativeFollower,
        follower_label: str,
        controller: HeadFollower,
        reachy,
        reachy_label: str,
        initial_concept: str = "person",
        prefer_reachy_camera: bool = True,
    ):
        self.root = root
        self.follower = follower
        self.follower_label = follower_label
        self.controller = controller
        self.reachy = reachy
        self.reachy_label = reachy_label
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._drive_enabled = tk.BooleanVar(value=True)

        self.cameras = discover_cameras()
        print(f"[camera] Detected: {self.cameras}")
        self.current_cam_index = self._pick_camera(prefer_reachy_camera)

        self._build_ui(initial_concept)
        self.cap: Optional[cv2.VideoCapture] = None
        self._open_camera(self.current_cam_index)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(0, self._tick)

    # -- camera helpers -------------------------------------------------
    def _pick_camera(self, prefer_reachy: bool) -> int:
        if prefer_reachy:
            for idx, label in self.cameras:
                if "reachy" in label.lower():
                    return idx
        return pick_default_camera(self.cameras)

    def _open_camera(self, index: int) -> bool:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        cap = cv2.VideoCapture(index, self.CAP_BACKEND)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAM_H)
        if not cap.isOpened():
            self._set_status(f"ERROR: could not open camera {index}", "red")
            self.cap = None
            return False
        self.cap = cap
        self.current_cam_index = index
        return True

    # -- UI -------------------------------------------------------------
    def _build_ui(self, initial_concept: str) -> None:
        self.root.title(f"{self.follower_label} -> Reachy Mini head follow")
        self.root.minsize(760, 680)

        top = ttk.Frame(self.root)
        top.pack(fill="x", side="top", padx=8, pady=(8, 4))
        ttk.Label(top, text="Camera:").pack(side="left")
        cam_values = [label for _i, label in self.cameras] or ["(none detected)"]
        self.cam_combo = ttk.Combobox(
            top, values=cam_values, state="readonly", width=34
        )
        # Pre-select the one we opened
        pre = next(
            (lbl for i, lbl in self.cameras if i == self.current_cam_index),
            cam_values[0],
        )
        self.cam_combo.set(pre)
        self.cam_combo.pack(side="left", padx=(6, 6))
        self.cam_combo.bind("<<ComboboxSelected>>", self._on_camera_change)

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", side="top", padx=8, pady=(0, 4))
        ttk.Label(controls, text="Concept:").pack(side="left")
        self.entry = ttk.Entry(controls, width=26)
        self.entry.pack(side="left", padx=(6, 6))
        self.entry.insert(0, initial_concept)
        self.entry.bind("<Return>", lambda _e: self._on_lock())

        ttk.Button(controls, text="Lock on", command=self._on_lock).pack(side="left")
        ttk.Button(controls, text="Release", command=self._on_release).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(controls, text="Home", command=self._on_home).pack(
            side="left", padx=(6, 0)
        )
        ttk.Checkbutton(
            controls, text="Drive Reachy", variable=self._drive_enabled
        ).pack(side="left", padx=(12, 0))

        self.status_label = ttk.Label(self.root, text="Idle.", foreground="black")
        self.status_label.pack(fill="x", side="top", padx=8, pady=(0, 4))

        self.stats_label = ttk.Label(
            self.root,
            text=f"Reachy: {self.reachy_label}",
            foreground="black",
            font=("TkFixedFont", 11),
        )
        self.stats_label.pack(fill="x", side="top", padx=8, pady=(0, 8))

        self.video_label = ttk.Label(self.root)
        self.video_label.pack(side="top", padx=8, pady=8)

    def _set_status(self, text: str, color: str = "black") -> None:
        self.status_label.config(text=text, foreground=color)

    def _on_camera_change(self, _event=None) -> None:
        label = self.cam_combo.get()
        for idx, lbl in self.cameras:
            if lbl == label:
                self._open_camera(idx)
                return

    def _on_lock(self) -> None:
        concept = self.entry.get().strip()
        if not concept:
            self._set_status("Type a concept first.", "orange")
            return
        self.controller.reset()
        self.follower.start_following(concept)
        self._set_status(f"Locking on '{concept}' ...", "blue")

    def _on_release(self) -> None:
        self.follower.stop_following()
        self._set_status("Released - head will decay to center.", "black")

    def _on_home(self) -> None:
        """Hard-snap the head to neutral.  Useful after a bad tune."""
        self.controller.reset()
        try:
            self.reachy.set_target(head=_head_pose_rpy(0.0, 0.0, 0.0))
        except Exception as e:
            print(f"[reachy] home failed: {e}")

    # -- main tick ------------------------------------------------------
    def _tick(self) -> None:
        if self.cap is None:
            self.root.after(100, self._tick)
            return
        ok, frame = self.cap.read()
        if not ok:
            self.root.after(33, self._tick)
            return

        H, W = frame.shape[:2]
        self.follower.push_frame(frame)
        f_state, bbox, age = self.follower.get_current_bbox()

        # have_target: we have a recent bbox and the follower is in an
        # active state.  LOST keeps the last bbox but shouldn't steer the
        # head any more - only decay.
        have_target = (
            bbox is not None
            and f_state == FollowState.TRACKING
            and age < COAST_S
        )

        snap = self.controller.step(bbox, (H, W), have_target)

        # Send to Reachy (or the null gateway when --no-reachy).
        if self._drive_enabled.get():
            try:
                self.reachy.set_target(
                    head=_head_pose_rpy(0.0, snap.cmd_pitch, snap.cmd_yaw)
                )
            except Exception as e:
                # Don't let a transient hiccup kill the tick loop.
                print(f"[reachy] set_target failed: {e}")

        # -- draw -------------------------------------------------------
        # Downscale just for the preview; the SAM follower already has the
        # full-resolution frame and its bbox is in native pixels.
        if W > self.DISPLAY_MAX_W:
            scale = self.DISPLAY_MAX_W / float(W)
            disp_w = int(round(W * scale))
            disp_h = int(round(H * scale))
            display = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            display = frame.copy() if bbox is not None else frame
            disp_w, disp_h = W, H

        if bbox is not None and f_state in (FollowState.TRACKING, FollowState.LOST):
            if scale != 1.0 and display is frame:
                display = cv2.resize(
                    frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA
                )
            x1 = int(round(bbox[0] * scale))
            y1 = int(round(bbox[1] * scale))
            x2 = int(round(bbox[2] * scale))
            y2 = int(round(bbox[3] * scale))
            color = (0, 255, 0) if f_state == FollowState.TRACKING else (0, 0, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            # Target centroid (what the controller tracks)
            cv2.drawMarker(
                display, (cx, cy), (0, 255, 255),
                cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA,
            )
            # Frame center (what we're servoing to)
            cv2.drawMarker(
                display, (disp_w // 2, disp_h // 2), (255, 255, 255),
                cv2.MARKER_TILTED_CROSS, 12, 1, cv2.LINE_AA,
            )
            cv2.line(
                display, (disp_w // 2, disp_h // 2), (cx, cy),
                (0, 255, 255), 1, cv2.LINE_AA,
            )

        self._set_status(
            f"State: {f_state.name}  age={age:.2f}s",
            STATE_COLORS.get(f_state, "black"),
        )

        st = self.follower.get_stats()
        med_ms = st.get("median_ms", 0.0)
        fps = (1000.0 / med_ms) if med_ms else 0.0
        tracker_bit = f"obj_id={st.get('obj_id')}"
        self.stats_label.config(
            text=(
                f"Reachy: {self.reachy_label}  |  "
                f"{self.follower_label} {med_ms:.0f} ms ({fps:.1f} fps) "
                f"{tracker_bit}  |  "
                f"err=({math.degrees(snap.err_yaw_rad):+5.1f}°, "
                f"{math.degrees(snap.err_pitch_rad):+5.1f}°)  "
                f"cmd yaw={math.degrees(snap.cmd_yaw):+5.1f}° "
                f"pitch={math.degrees(snap.cmd_pitch):+5.1f}°  "
                f"drive={'on' if self._drive_enabled.get() else 'off'}"
            )
        )

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.configure(image=self._photo)
        self.root.after(33, self._tick)

    def _on_close(self) -> None:
        """Best-effort cleanup: stop follower, home the head, drop the camera."""
        try:
            self.follower.stop_following()
            self.follower.close()
        except Exception:
            pass
        try:
            self.reachy.set_target(head=_head_pose_rpy(0.0, 0.0, 0.0))
            time.sleep(0.1)  # let the daemon process the last command
        except Exception:
            pass
        try:
            if hasattr(self.reachy, "close"):
                self.reachy.close()
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--concept",
        default="person",
        help="Initial concept to track (editable in the UI).",
    )
    ap.add_argument(
        "--no-reachy",
        action="store_true",
        help="Display-only mode.  Runs the control math but sends no commands.",
    )
    ap.add_argument(
        "--sim",
        action="store_true",
        help="Spawn a simulated Reachy daemon in-process (for dry runs).",
    )
    ap.add_argument(
        "--no-reachy-camera",
        action="store_true",
        help="Don't auto-pick the Reachy Mini camera (use laptop cam instead).",
    )
    args = ap.parse_args()

    print("[startup] Building follower ...")
    follower = Sam3NativeFollower()
    follower_label = "SAM 3 native"
    controller = HeadFollower()

    print("[startup] Opening Reachy gateway ...")
    reachy, reachy_label = _open_reachy(enable=not args.no_reachy, sim=args.sim)
    print(f"[startup] Reachy: {reachy_label}")

    root = tk.Tk()
    _app = App(
        root,
        follower,
        follower_label,
        controller,
        reachy,
        reachy_label=reachy_label,
        initial_concept=args.concept,
        prefer_reachy_camera=not args.no_reachy_camera,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
