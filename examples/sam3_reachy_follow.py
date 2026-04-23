"""
SAM 3 -> Reachy Mini head follow
================================

Minimal wiring between the three SAM 3 followers in this folder and the
real Reachy Mini head motors.  No body yaw, no antennas, no animations,
no natural-gaze planner.  One thread runs perception, one thread runs the
preview/control loop, and every tick at ~30 Hz we do:

    bbox center - frame center -> normalized error (-1 .. +1)
    EMA smooth the error
    P-controller updates head yaw/pitch (radians) with a small step
    clamp to the head workspace
    reachy.set_target(head=4x4_pose)

SAM 3 on MPS runs at roughly 2-3 Hz; the control loop still runs at 30 Hz
because perception and motor control are deliberately decoupled.  Between
SAM updates the last bbox stays valid, and as the head turns, the camera
on top of it sees the person drift toward the frame center - classic
visual servoing, slow detection plus fast closed-loop control.

Which perception layer drives the motors is selected with `--tracker`:

    native   SAM 3.1 native video predictor (no external SOT).  Default.
             Simple, slowest (~2.5 Hz), lowest appearance-based drift.
    vit      SAM 3.1 detector + OpenCV TrackerVit (~30 Hz between SAM
             re-grounds every 0.8 s).  Smooth bbox, higher effective rate.
    csrt     SAM 3.1 detector + OpenCV CSRT (same cadence, classic SOT).

Run:

    # real robot, daemon on localhost, native tracker (default):
    .venv/bin/python3 examples/sam3_reachy_follow.py --concept person

    # with the ViT hybrid (often feels best for head-follow):
    .venv/bin/python3 examples/sam3_reachy_follow.py --tracker vit

    # no robot - preview + control math only, no commands sent:
    .venv/bin/python3 examples/sam3_reachy_follow.py --no-reachy --tracker vit

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
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

# Only the camera helpers are shared / imported at module top. The follower
# itself is built lazily in `build_follower()` based on the --tracker flag
# so loading SAM 3 weights (several GB) only happens once, for the chosen
# variant.
from sam3_native_tracker import (
    BBox,
    discover_cameras,
    pick_default_camera,
)

# Mapping between the FollowState enum names every follower exposes and
# user-visible colors.  We do NOT import FollowState directly - each tracker
# module defines its own enum class with the same names, and we compare by
# `.name` so the App code is tracker-agnostic.
STATE_NAMES = ("IDLE", "LOCKING", "TRACKING", "LOST")
STATE_COLORS = {
    "IDLE": "gray",
    "LOCKING": "orange",
    "TRACKING": "lime green",
    "LOST": "red",
}


class FollowerProtocol:
    """Typing-only protocol so App.__init__ can be tracker-agnostic.  Both
    `sam3_native_tracker.Sam3NativeFollower` and `sam3_tracker.Sam3Follower`
    already satisfy this shape."""

    def start_following(self, concept: str) -> None: ...
    def stop_following(self) -> None: ...
    def push_frame(self, frame_bgr: np.ndarray) -> None: ...
    def get_current_bbox(self): ...
    def get_stats(self) -> dict: ...
    def close(self) -> None: ...


def build_follower(kind: str) -> tuple[FollowerProtocol, str]:
    """Build the follower for the requested tracker kind.

    Returns (follower, human_readable_label).
    """
    kind = kind.lower()
    if kind == "native":
        from sam3_native_tracker import Sam3NativeFollower

        return Sam3NativeFollower(), "SAM 3 native (no external SOT)"

    if kind == "vit":
        from sam3_tracker import Sam3Follower, Sam3Segmenter
        from sam3_vit_tracker import make_vit_factory

        segmenter = Sam3Segmenter()
        vit_factory = make_vit_factory()
        follower = Sam3Follower(
            segmenter, tracker_factory=vit_factory, tracker_name="ViT"
        )
        return follower, "SAM 3 + ViT tracker"

    if kind == "csrt":
        from sam3_tracker import Sam3Follower, Sam3Segmenter

        segmenter = Sam3Segmenter()
        follower = Sam3Follower(segmenter)  # default factory = CSRT
        return follower, "SAM 3 + CSRT tracker"

    raise ValueError(f"unknown --tracker {kind!r}; expected native|vit|csrt")


# ---------------------------------------------------------------------------
# Control tuning.  Defaults match hsafa_robot/robot_control.py so the "feel"
# is consistent with the rest of the codebase.  These are conservative - if
# the head looks lazy, bump KP_* up.  If it oscillates, bump CMD_ALPHA down.
# ---------------------------------------------------------------------------
KP_YAW = 0.6          # radians of command per unit of normalized error
KP_PITCH = 0.4
STEP_SCALE = 0.2      # fraction of the KP*err step applied per tick

# Sign conventions for Reachy Mini with the raw, un-mirrored camera:
#   image +x = right, image +y = down
#   head  +yaw = turn left, +pitch = look down
# So a person to the right (err_x > 0) needs yaw to go negative to center.
YAW_SIGN = -1.0
PITCH_SIGN = +1.0

# Hardware workspace limits (radians).
YAW_LIMIT = math.radians(60)
PITCH_LIMIT = math.radians(30)

# EMA smoothing.  Higher alpha = snappier response, lower = smoother.
ERR_ALPHA = 0.6       # smooths noisy bbox centers before feeding the P-ctrl
CMD_ALPHA = 0.4       # smooths the command we actually send to the motors

# Small dead-band so we don't hunt on sub-pixel error.
DEADZONE = 0.03

# How long to coast on a stale bbox before decaying back to center.
COAST_S = 0.8
RECENTER_DECAY = 0.95 # per-tick multiplier when no target


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
# The controller itself.  One public method: `step(bbox, frame_hw, have_target)`
# returns (yaw, pitch) rad so the caller can both render them and send them.
# ---------------------------------------------------------------------------
@dataclass
class ControlSnap:
    err_x: float
    err_y: float
    cmd_yaw: float
    cmd_pitch: float
    have_target: bool


class HeadFollower:
    def __init__(self) -> None:
        self._err_x_s = 0.0
        self._err_y_s = 0.0
        self._cmd_yaw = 0.0
        self._cmd_pitch = 0.0
        self._sent_yaw = 0.0
        self._sent_pitch = 0.0

    def reset(self) -> None:
        self._err_x_s = 0.0
        self._err_y_s = 0.0
        self._cmd_yaw = 0.0
        self._cmd_pitch = 0.0
        self._sent_yaw = 0.0
        self._sent_pitch = 0.0

    def step(
        self,
        bbox: Optional[BBox],
        frame_hw: tuple[int, int],
        have_target: bool,
        fresh: bool,
    ) -> ControlSnap:
        """One control tick.

        Parameters
        ----------
        bbox, frame_hw, have_target
            Latest target state from the follower.
        fresh
            True only when the follower produced a NEW bbox since the last
            call. SAM 3 runs at ~2.5 Hz; this tick loop runs at 30 Hz. If we
            integrated the P-step every tick, the same error would compound
            12x per real sample and the head would overshoot badly. Instead
            we only integrate on fresh evidence. Between updates we hold
            `_cmd_*` steady and let the CMD_ALPHA smoothing carry the motors
            the rest of the way.
        """
        H, W = frame_hw
        err_x = err_y = 0.0
        if bbox is not None:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            # Normalized error: 0 = centered, +/-1 = edge of frame.
            err_x = float((cx / max(1, W) - 0.5) * 2.0)
            err_y = float((cy / max(1, H) - 0.5) * 2.0)

        if have_target and fresh:
            # New SAM sample -> one proportional nudge.
            self._err_x_s = (1 - ERR_ALPHA) * self._err_x_s + ERR_ALPHA * err_x
            self._err_y_s = (1 - ERR_ALPHA) * self._err_y_s + ERR_ALPHA * err_y

            if abs(self._err_x_s) > DEADZONE:
                self._cmd_yaw += YAW_SIGN * KP_YAW * self._err_x_s * STEP_SCALE
            if abs(self._err_y_s) > DEADZONE:
                self._cmd_pitch += PITCH_SIGN * KP_PITCH * self._err_y_s * STEP_SCALE
        elif not have_target:
            # Tracker lost / released: decay toward center slowly.
            self._err_x_s *= RECENTER_DECAY
            self._err_y_s *= RECENTER_DECAY
            self._cmd_yaw *= RECENTER_DECAY
            self._cmd_pitch *= RECENTER_DECAY
        # else: have_target && not fresh -> hold. The command stays put.

        self._cmd_yaw = _clamp(self._cmd_yaw, -YAW_LIMIT, YAW_LIMIT)
        self._cmd_pitch = _clamp(self._cmd_pitch, -PITCH_LIMIT, PITCH_LIMIT)

        # The outgoing smoothing runs every tick. That's what turns the
        # discrete per-sample nudges into continuous motor motion.
        self._sent_yaw = (1 - CMD_ALPHA) * self._sent_yaw + CMD_ALPHA * self._cmd_yaw
        self._sent_pitch = (
            (1 - CMD_ALPHA) * self._sent_pitch + CMD_ALPHA * self._cmd_pitch
        )

        return ControlSnap(
            err_x=self._err_x_s,
            err_y=self._err_y_s,
            cmd_yaw=self._sent_yaw,
            cmd_pitch=self._sent_pitch,
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
        follower: FollowerProtocol,
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
        # Timestamp of the most recent SAM update we've already "consumed"
        # with a P-step. On each tick we compare against the follower's
        # current update timestamp (derived from `age`); only when it has
        # advanced do we consider the sample fresh and integrate.
        self._last_consumed_update_ts: float = 0.0
        self._drive_enabled = tk.BooleanVar(value=True)
        # Lock to avoid races between camera tick and controller reset
        self._reset_lock = threading.Lock()

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
        with self._reset_lock:
            self.controller.reset()
            self._last_consumed_update_ts = 0.0
        self.follower.start_following(concept)
        self._set_status(f"Locking on '{concept}' ...", "blue")

    def _on_release(self) -> None:
        self.follower.stop_following()
        with self._reset_lock:
            self._last_consumed_update_ts = 0.0
        self._set_status("Released - head will decay to center.", "black")

    def _on_home(self) -> None:
        """Hard-snap the head to neutral.  Useful after a bad tune."""
        with self._reset_lock:
            self.controller.reset()
            self._last_consumed_update_ts = 0.0
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
        now = time.time()

        # Derive the timestamp of the follower's latest bbox from `age`.
        # `age` only shrinks when a new sample arrives, so a larger
        # `update_ts` than the one we last consumed means "fresh data".
        if bbox is not None and age != float("inf"):
            update_ts = now - age
        else:
            update_ts = 0.0

        # have_target: we have a recent bbox and the follower is in an
        # active state.  LOST keeps the last bbox but shouldn't steer the
        # head any more - only decay.  We compare state by .name so this
        # App works with either follower's FollowState enum.
        state_name = f_state.name
        have_target = (
            bbox is not None
            and state_name == "TRACKING"
            and age < COAST_S
        )

        with self._reset_lock:
            fresh = update_ts > self._last_consumed_update_ts + 1e-4
            if fresh:
                self._last_consumed_update_ts = update_ts
            snap = self.controller.step(bbox, (H, W), have_target, fresh=fresh)

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

        if bbox is not None and state_name in ("TRACKING", "LOST"):
            if scale != 1.0 and display is frame:
                display = cv2.resize(
                    frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA
                )
            x1 = int(round(bbox[0] * scale))
            y1 = int(round(bbox[1] * scale))
            x2 = int(round(bbox[2] * scale))
            y2 = int(round(bbox[3] * scale))
            color = (0, 255, 0) if state_name == "TRACKING" else (0, 0, 255)
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
            f"State: {state_name}  age={age:.2f}s",
            STATE_COLORS.get(state_name, "black"),
        )

        st = self.follower.get_stats()
        med_ms = st.get("median_ms", 0.0)
        fps = (1000.0 / med_ms) if med_ms else 0.0
        # `obj_id` is native-follower specific; `score` is CSRT/ViT specific.
        # Show whichever the active follower provides.
        if "obj_id" in st:
            tracker_bit = f"obj_id={st.get('obj_id')}"
        else:
            score = st.get("score", 0.0)
            tracker_bit = f"score={score:.2f}"
        self.stats_label.config(
            text=(
                f"Reachy: {self.reachy_label}  |  "
                f"{self.follower_label} {med_ms:.0f} ms ({fps:.1f} fps) "
                f"{tracker_bit}  |  "
                f"err=({snap.err_x:+.2f}, {snap.err_y:+.2f})  "
                f"cmd yaw={math.degrees(snap.cmd_yaw):+5.1f}deg "
                f"pitch={math.degrees(snap.cmd_pitch):+5.1f}deg  "
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
    ap.add_argument(
        "--tracker",
        choices=("native", "vit", "csrt"),
        default="native",
        help=(
            "Which perception layer to use. 'native' = SAM 3 video tracker, "
            "'vit' = SAM 3 + OpenCV TrackerVit, 'csrt' = SAM 3 + CSRT."
        ),
    )
    args = ap.parse_args()

    print(f"[startup] Building follower: --tracker {args.tracker} ...")
    follower, follower_label = build_follower(args.tracker)
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
