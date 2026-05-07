"""Simple Reachy head follow — SAM 3.1 + camera-motion compensation + direct follow.

SAM runs slowly (~2-5 Hz).  Between detections we estimate how much the
camera itself moved (head turning) by tracking background features across the
full image.  We then shift the last-known target centre by that amount so the
lock stays on a stationary object even while the head moves.
Direct pixel → angle mapping.  No world frame, no velocity predictor.
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from ultralytics import SAM

sys.path.insert(0, os.path.expanduser("~/Reachy/Software/reachy_sdk"))
sys.path.insert(0, os.path.expanduser("~/Reachy/Software/reachy_mini"))
from reachy_mini import ReachyMini  # type: ignore

FOV_DEG = 66.0
TICK_MS = 33
YAW_LIMIT_DEG = 60.0
PITCH_LIMIT_DEG = 40.0


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()


def _head_pose_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    pose = np.eye(4)
    pose[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_matrix()
    return pose


class KltTracker:
    """Track an object region with KLT optical flow.

    When SAM gives a bbox we seed features inside it.
    Every camera frame we forward-track those features.
    Median of tracked feature positions = object centre.
    This runs at ~30 Hz (1-2 ms/frame), much faster than SAM.
    Handles both object motion and camera motion automatically.
    """

    def __init__(self):
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_kp: Optional[np.ndarray] = None
        self.cx: float = 0.0
        self.cy: float = 0.0
        self.lost: bool = True

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_kp = None
        self.cx = 0.0
        self.cy = 0.0
        self.lost = True

    def init(self, gray: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = bbox
        margin = 4
        roi = gray[y1 + margin:y2 - margin, x1 + margin:x2 - margin]
        if roi.size == 0:
            self.lost = True
            return
        kp = cv2.goodFeaturesToTrack(roi, maxCorners=60, qualityLevel=0.01, minDistance=5)
        if kp is not None and len(kp) >= 5:
            kp[:, 0, 0] += x1 + margin
            kp[:, 0, 1] += y1 + margin
            self.prev_kp = kp
        self.cx = float((x1 + x2) / 2)
        self.cy = float((y1 + y2) / 2)
        self.prev_gray = gray.copy()
        self.lost = False

    def update(self, gray: np.ndarray) -> Tuple[float, float, bool]:
        """Return (cx, cy, is_lost)."""
        if self.lost or self.prev_gray is None or self.prev_kp is None:
            return self.cx, self.cy, True

        next_kp, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_kp, None,
            winSize=(21, 21), maxLevel=3
        )
        if next_kp is None:
            return self.cx, self.cy, True

        mask = status.reshape(-1) == 1
        if mask.sum() < 5:
            return self.cx, self.cy, True

        good = next_kp[mask]
        self.cx = float(np.median(good[:, 0, 0]))
        self.cy = float(np.median(good[:, 0, 1]))

        self.prev_gray = gray.copy()
        self.prev_kp = good.reshape(-1, 1, 2)
        self.lost = False
        return self.cx, self.cy, False


@dataclass
class SimpleSnap:
    cmd_yaw: float
    cmd_pitch: float
    have_target: bool


class SimpleController:
    def __init__(self):
        self.tracker = KltTracker()
        self.have_target: bool = False

    def reset(self) -> None:
        self.tracker.reset()
        self.have_target = False

    def step(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]],
             W: int, H: int) -> SimpleSnap:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if bbox is not None:
            self.tracker.init(gray, bbox)
            self.have_target = not self.tracker.lost

        if self.have_target:
            cx, cy, lost = self.tracker.update(gray)
            self.have_target = not lost
            if self.have_target:
                err_x = cx - W / 2.0
                err_y = cy - H / 2.0
                v_fov = FOV_DEG * (H / max(1, W))
                yaw = -(err_x / W) * FOV_DEG
                pitch = (err_y / H) * v_fov
                yaw = max(-YAW_LIMIT_DEG, min(YAW_LIMIT_DEG, yaw))
                pitch = max(-PITCH_LIMIT_DEG, min(PITCH_LIMIT_DEG, pitch))
                return SimpleSnap(
                    cmd_yaw=math.radians(yaw),
                    cmd_pitch=math.radians(pitch),
                    have_target=True,
                )

        return SimpleSnap(cmd_yaw=0.0, cmd_pitch=0.0, have_target=False)


# ---------------------------------------------------------------------------
# Fake video dataset for Ultralytics' SAM 3 video predictor.
# ---------------------------------------------------------------------------
class _FakeVideoDataset:
    mode = "video"
    bs = 1

    def __init__(self, frame_cap: int = 1_000_000) -> None:
        self.frames = frame_cap
        self.frame = 0


class SamBox:
    """SAM 3.1 text-based semantic detector — same approach as sam3-test2.py."""

    def __init__(self, concept: str, device: str):
        self.concept = concept
        self.device = device
        weights = "checkpoints/sam3.1_multiplex.pt"
        imgsz = 448
        print(f"[sam3] device={device} weights={weights} imgsz={imgsz}")

        from ultralytics.models.sam import SAM3VideoSemanticPredictor

        overrides = dict(
            model=weights,
            task="segment",
            mode="predict",
            device=device,
            half=(device == "cuda"),
            imgsz=imgsz,
            conf=0.25,
            verbose=False,
            save=False,
        )
        self.predictor = SAM3VideoSemanticPredictor(overrides=overrides)
        self.predictor.setup_model(model=None, verbose=False)
        # Warm-up: setup source + dataset (does NOT run inference here)
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self.predictor.setup_source(dummy)
        self.predictor.dataset = _FakeVideoDataset()
        print("[sam3] ready")

        self.box: Optional[Tuple[int, int, int, int]] = None
        self.ts = 0.0
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, daemon=True, name="sam")
        self._thread.start()

    def push_frame(self, frame: np.ndarray) -> None:
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            pass

    def _run(self) -> None:
        inf_ctx = torch.inference_mode()
        inf_ctx.__enter__()
        try:
            self.predictor.run_callbacks("on_predict_start")
            frame_index = 0
            while True:
                frame = self._q.get()
                try:
                    frame_index += 1
                    self.predictor.dataset.frame = frame_index
                    self.predictor.batch = (["camera"], [frame], [""])
                    im = self.predictor.preprocess([frame])
                    # First frame needs text prompt; subsequent frames reuse
                    if "text_ids" not in self.predictor.inference_state:
                        preds = self.predictor.inference(im, text=[self.concept])
                    else:
                        preds = self.predictor.inference(im)
                    results = self.predictor.postprocess(preds, im, [frame])
                    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                        self.box = tuple(boxes[np.argmax(areas)].astype(int))
                        self.ts = time.time()
                except Exception as e:
                    import traceback
                    print(f"[sam] error: {type(e).__name__}: {e}")
                    traceback.print_exc()
        finally:
            inf_ctx.__exit__(None, None, None)

    def get(self, max_age: float = 1.0) -> Optional[Tuple[int, int, int, int]]:
        if self.box is None or (time.time() - self.ts) > max_age:
            return None
        return self.box


def list_cameras_macos() -> list[str]:
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPCameraDataType"], text=True, timeout=5
        )
    except Exception:
        return []
    names: list[str] = []
    for raw in out.splitlines():
        line = raw.strip()
        if not line.endswith(":"):
            continue
        name = line.rstrip(":").strip()
        if not name or name.lower() == "camera":
            continue
        names.append(name)
    return names


def discover_cameras(max_probe: int = 5) -> list[tuple[int, str]]:
    names = list_cameras_macos() if sys.platform == "darwin" else []
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    found: list[tuple[int, str]] = []
    for idx in range(max_probe):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            label = names[idx] if idx < len(names) else f"Camera {idx}"
            found.append((idx, f"{idx}: {label}"))
        cap.release()
    return found


def _pick_camera(cams: list[tuple[int, str]], prefer_reachy: bool) -> int:
    if prefer_reachy:
        for idx, label in cams:
            if "reachy" in label.lower():
                return idx
    for idx, label in cams:
        if "reachy" not in label.lower():
            return idx
    return cams[0][0] if cams else 0


class App:
    CAM_W, CAM_H = 640, 480
    DISPLAY_MAX_W = 720
    CAP_BACKEND = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY

    def __init__(
        self,
        root: tk.Tk,
        detector: SamBox,
        controller: SimpleController,
        reachy,
        reachy_label: str,
        initial_concept: str,
        prefer_reachy_camera: bool,
    ):
        self.root = root
        self.detector = detector
        self.controller = controller
        self.reachy = reachy
        self.reachy_label = reachy_label
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._drive_enabled = tk.BooleanVar(value=True)
        self._last_have_target = False
        self._tick_counter = 0

        self.cameras = discover_cameras()
        print(f"[camera] detected: {self.cameras}")
        self.current_cam_index = _pick_camera(self.cameras, prefer_reachy_camera)

        self._build_ui(initial_concept)
        self.cap: Optional[cv2.VideoCapture] = None
        # Try preferred camera first, fall back to any other detected camera.
        if not self._open_camera(self.current_cam_index):
            for idx, _label in self.cameras:
                if idx != self.current_cam_index and self._open_camera(idx):
                    self.current_cam_index = idx
                    self.cam_combo.set(self.cameras[idx][1] if idx < len(self.cameras) else f"Camera {idx}")
                    break

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(0, self._tick)

    def _open_camera(self, index: int) -> bool:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        cap = cv2.VideoCapture(index, self.CAP_BACKEND)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAM_H)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            self._set_status(f"ERROR: could not open camera {index}", "red")
            self.cap = None
            return False
        # Verify camera can actually deliver frames (macOS sometimes reports
        # a camera is open but read() hangs or returns nothing).
        ok, test_frame = cap.read()
        if not ok or test_frame is None or test_frame.size == 0:
            cap.release()
            self._set_status(f"ERROR: camera {index} opened but no frames", "red")
            self.cap = None
            return False
        self.cap = cap
        self.current_cam_index = index
        print(f"[camera] opened index={index} resolution={test_frame.shape[1]}x{test_frame.shape[0]}")
        return True

    def _build_ui(self, initial_concept: str) -> None:
        self.root.title("Simple Ego Reachy — SAM 3.1 + optical flow")
        self.root.minsize(760, 680)

        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(top, text="Camera:").pack(side="left")
        cam_values = [label for _i, label in self.cameras] or ["(none detected)"]
        self.cam_combo = ttk.Combobox(top, values=cam_values, state="readonly", width=34)
        pre = next(
            (lbl for i, lbl in self.cameras if i == self.current_cam_index),
            cam_values[0],
        )
        self.cam_combo.set(pre)
        self.cam_combo.pack(side="left", padx=(6, 6))
        self.cam_combo.bind("<<ComboboxSelected>>", self._on_camera_change)

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=8, pady=(0, 4))
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
        self.status_label.pack(fill="x", padx=8, pady=(0, 4))

        self.stats_label = ttk.Label(
            self.root,
            text=f"Reachy: {self.reachy_label}",
            font=("TkFixedFont", 11),
        )
        self.stats_label.pack(fill="x", padx=8, pady=(0, 8))

        self.video_label = ttk.Label(self.root)
        self.video_label.pack(padx=8, pady=8)

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
        self._set_status(f"Locking on '{concept}' …", "blue")

    def _on_release(self) -> None:
        self.controller.reset()
        self._set_status("Released — head will decay to neutral.", "black")

    def _on_home(self) -> None:
        self.controller.reset()
        try:
            self.reachy.set_target(head=_head_pose_rpy(0.0, 0.0, 0.0))
        except Exception as e:
            print(f"[reachy] home failed: {e}")

    def _tick(self) -> None:
        if self.cap is None:
            self.root.after(100, self._tick)
            return
        ok, frame = self.cap.read()
        if not ok:
            self.root.after(TICK_MS, self._tick)
            return

        if frame.shape[0] > 720:
            scale = 720 / frame.shape[0]
            new_w = int(frame.shape[1] * scale)
            frame = cv2.resize(frame, (new_w, 720), interpolation=cv2.INTER_AREA)

        H, W = frame.shape[:2]
        self.detector.push_frame(frame)
        bbox = self.detector.get()
        snap = self.controller.step(frame, bbox, W, H)

        if self._drive_enabled.get():
            try:
                self.reachy.set_target(
                    head=_head_pose_rpy(0.0, snap.cmd_pitch, snap.cmd_yaw)
                )
            except Exception as e:
                print(f"[reachy] set_target failed: {e}")

        self._tick_counter += 1
        do_display = (self._tick_counter % 2) == 0
        if not do_display:
            have_target = snap.have_target
            if have_target != self._last_have_target:
                self._set_status(
                    f"Tracking: {have_target}",
                    "lime green" if have_target else "red",
                )
                self._last_have_target = have_target
            self.root.after(TICK_MS, self._tick)
            return

        if W > self.DISPLAY_MAX_W:
            scale = self.DISPLAY_MAX_W / float(W)
            disp_w = int(round(W * scale))
            disp_h = int(round(H * scale))
            display = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            disp_w, disp_h = W, H
            display = frame.copy() if bbox is not None else frame

        # Draw tracked target position (updated every frame by KLT)
        if self.controller.have_target or bbox is not None:
            tx = self.controller.tracker.cx
            ty = self.controller.tracker.cy
            if bbox is not None:
                # SAM bbox in original frame (green = reference)
                x1, y1, x2, y2 = bbox
                x1 = int(round(x1 * scale))
                y1 = int(round(y1 * scale))
                x2 = int(round(x2 * scale))
                y2 = int(round(y2 * scale))
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Compensated centre (red) — stays on object even when head turns
            cv2.circle(display,
                       (int(round(tx * scale)), int(round(ty * scale))),
                       6, (0, 0, 255), -1, cv2.LINE_AA)
            # Image centre (white cross)
            cv2.drawMarker(display, (disp_w // 2, disp_h // 2), (255, 255, 255),
                           cv2.MARKER_TILTED_CROSS, 12, 1, cv2.LINE_AA)
            cv2.line(display, (disp_w // 2, disp_h // 2),
                     (int(round(tx * scale)), int(round(ty * scale))),
                     (0, 255, 255), 1, cv2.LINE_AA)

        have_target = snap.have_target
        if have_target != self._last_have_target:
            self._set_status(
                f"Tracking: {have_target}",
                "lime green" if have_target else "red",
            )
            self._last_have_target = have_target

        self.stats_label.config(
            text=(
                f"Reachy: {self.reachy_label}  |  "
                f"SAM 3.1 (simple ego)  |  "
                f"cmd yaw={math.degrees(snap.cmd_yaw):+5.1f}° "
                f"pitch={math.degrees(snap.cmd_pitch):+5.1f}°  "
                f"drive={'on' if self._drive_enabled.get() else 'off'}"
            )
        )

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.configure(image=self._photo)
        self.root.after(TICK_MS, self._tick)

    def _on_close(self) -> None:
        try:
            self.reachy.set_target(head=_head_pose_rpy(0.0, 0.0, 0.0))
            time.sleep(0.1)
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concept", default="person")
    ap.add_argument("--no-reachy", action="store_true")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--no-reachy-camera", action="store_true")
    args = ap.parse_args()

    print(f"[startup] device={DEVICE} fov={FOV_DEG:.1f}°")

    detector = SamBox(args.concept, DEVICE)
    controller = SimpleController()

    reachy = None
    reachy_label = "disabled"
    if not args.no_reachy:
        try:
            if args.sim:
                reachy = ReachyMini(spawn_daemon=True, use_sim=True, automatic_body_yaw=False)
                reachy_label = "simulated"
            else:
                reachy = ReachyMini(automatic_body_yaw=False)
                reachy_label = f"connected to {getattr(reachy, 'host', '?')}:{getattr(reachy, 'port', '?')}"
            print(f"[reachy] {reachy_label}")
        except Exception as e:
            print(f"[reachy] connect failed: {e}")
            reachy = None
            reachy_label = f"error: {e}"

    root = tk.Tk()
    App(
        root,
        detector=detector,
        controller=controller,
        reachy=reachy if reachy is not None else _head_pose_rpy(0, 0, 0),
        reachy_label=reachy_label,
        initial_concept=args.concept,
        prefer_reachy_camera=not args.no_reachy_camera,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
