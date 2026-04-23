"""
SAM 3 / 3.1 live test (Ultralytics) — continuous mode
-----------------------------------------------------

Type any concept ("person", "red cup", "glasses, bus, person"), press
"Start Live". The camera keeps streaming while SAM 3 runs in a background
loop and the freshest masks are drawn over every frame. Press "Stop" to
halt the loop; the video keeps playing either way.

Speed knobs (in priority order):
    - imgsz=448 instead of SAM's default 1024 (~5x fewer visual tokens).
    - fp16 on MPS/CUDA (half=True in overrides).
    - ndarray frames fed directly to set_image (no JPEG round-trip).
    - Warmup call on startup so the first Live iteration isn't a cold start.

Requirements (install once):
    pip install -U "ultralytics>=8.3.237"
    pip uninstall -y clip
    pip install git+https://github.com/ultralytics/CLIP.git

Weights (gated):
    Either model works. SAM 3.1 is preferred if both are on disk.
      - SAM 3:   https://huggingface.co/facebook/sam3        -> checkpoints/sam3.pt
      - SAM 3.1: https://huggingface.co/facebook/sam3.1      -> checkpoints/sam3.1_multiplex.pt
    Override with SAM3_WEIGHTS=/abs/path/to/whatever.pt

Optional:
    SAM3_IMGSZ=448   (default). Try 336 (14*24) for max speed at accuracy cost,
                     or 644 for near-default SAM 3 quality.

Tested on: macOS / Apple Silicon. Expect ~1-2 s per iteration on MPS with
these tweaks (down from ~3-6 s at defaults). CPU is much slower.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass, field
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk


# ---------------------------------------------------------------------------
# Device selection (MPS on Mac, CUDA if available, else CPU)
# ---------------------------------------------------------------------------
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()
print(f"[startup] Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# SAM 3 wrapper (Ultralytics SAM3SemanticPredictor)
# ---------------------------------------------------------------------------
@dataclass
class Sam3Detection:
    mask: np.ndarray        # (H, W) bool
    bbox: tuple[int, int, int, int]
    score: float
    label: str


class Sam3Segmenter:
    """Thin wrapper around Ultralytics SAM3SemanticPredictor. Also loads SAM 3.1."""

    # Pick the best weights present on disk unless SAM3_WEIGHTS is set.
    # SAM 3.1 ("multiplex") is preferred when available.
    CANDIDATE_WEIGHTS = (
        "checkpoints/sam3.1_multiplex.pt",
        "checkpoints/sam3.pt",
    )

    def __init__(self, weights_path: Optional[str] = None, conf: float = 0.25):
        if weights_path is None:
            env = os.getenv("SAM3_WEIGHTS")
            if env:
                weights_path = env
            else:
                weights_path = next(
                    (p for p in self.CANDIDATE_WEIGHTS if os.path.isfile(p)),
                    self.CANDIDATE_WEIGHTS[0],
                )
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"SAM 3 weights not found at {weights_path!r}.\n"
                "  1. Request access:\n"
                "       https://huggingface.co/facebook/sam3    (original SAM 3)\n"
                "       https://huggingface.co/facebook/sam3.1  (SAM 3.1)\n"
                "  2. Download the .pt file and place it under checkpoints/\n"
                "     (or set SAM3_WEIGHTS=/abs/path/to/sam3.pt)"
            )

        print(f"[sam3] Loading weights from {weights_path}...")
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
        except ImportError as e:
            raise ImportError(
                "Ultralytics is not installed or too old. Install with:\n"
                "    pip install -U 'ultralytics>=8.3.237'"
            ) from e

        # Speed tuning:
        #   - half=True on MPS/CUDA cuts memory + time ~2x with negligible
        #     accuracy loss for SAM 3.
        #   - imgsz=448 (a multiple of the encoder stride 14, so no rounding)
        #     cuts visual tokens to ~1024, roughly halving encoder time vs 644.
        imgsz = int(os.getenv("SAM3_IMGSZ", "448"))
        overrides = dict(
            conf=conf,
            task="segment",
            mode="predict",
            model=weights_path,
            half=(DEVICE in ("cuda", "mps")),
            imgsz=imgsz,
            device=DEVICE,
            verbose=False,
            save=False,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        # Lazy-probed: does this Ultralytics version accept ndarray in set_image?
        self._set_image_accepts_ndarray: Optional[bool] = None
        # Rolling latency window for avg/p95 reporting
        self._latencies: deque = deque(maxlen=50)
        print(f"[sam3] Ready (imgsz={imgsz}, half={overrides['half']}).")
        # Report actual parameter dtype so we can confirm fp16 is really active.
        try:
            mdl = getattr(self.predictor, "model", None)
            if mdl is not None:
                dtype = next(mdl.parameters()).dtype
                print(f"[sam3] Predictor parameter dtype: {dtype}")
        except Exception:
            pass
        self._warmup()

    def _warmup(self) -> None:
        """First inference JIT-compiles kernels; burn one cycle upfront."""
        print("[sam3] Warming up...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            self.segment(dummy, ["warmup"])
            print("[sam3] Warm.")
        except Exception as e:
            print(f"[sam3] Warmup skipped: {e}")

    def segment(self, frame_bgr: np.ndarray, concepts: list[str]) -> list[Sam3Detection]:
        """Return every instance in the frame matching any of the given text concepts."""
        if not concepts:
            return []

        t_set = time.time()
        self._set_image(frame_bgr)
        t_inf = time.time()
        results = self.predictor(text=concepts)
        t_end = time.time()
        total_ms = (t_end - t_set) * 1000
        print(
            f"[sam3] set_image: {(t_inf - t_set) * 1000:.0f} ms, "
            f"infer: {(t_end - t_inf) * 1000:.0f} ms, "
            f"concepts={concepts}"
        )

        # Rolling latency report every 10 calls.
        self._latencies.append(total_ms)
        if len(self._latencies) >= 10 and len(self._latencies) % 10 == 0:
            arr = np.asarray(self._latencies, dtype=np.float32)
            p95 = float(np.percentile(arr, 95))
            print(
                f"[sam3] rolling latency over last {len(arr)}: "
                f"avg={arr.mean():.0f} ms, p50={float(np.median(arr)):.0f} ms, "
                f"p95={p95:.0f} ms, min={arr.min():.0f} ms, max={arr.max():.0f} ms"
            )

        return self._to_detections(results, frame_bgr.shape[:2], concepts)

    def _set_image(self, frame_bgr: np.ndarray) -> None:
        """Feed a frame to the predictor. Prefer in-memory ndarray (~100 ms faster)."""
        if self._set_image_accepts_ndarray is not False:
            try:
                self.predictor.set_image(frame_bgr)
                self._set_image_accepts_ndarray = True
                return
            except Exception:
                self._set_image_accepts_ndarray = False
        # Fallback: disk round-trip for older ultralytics releases
        tmp_path = "/tmp/_sam3_frame.jpg"
        cv2.imwrite(tmp_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.predictor.set_image(tmp_path)

    @staticmethod
    def _to_detections(
        results, frame_hw: tuple[int, int], concepts: list[str]
    ) -> list[Sam3Detection]:
        """Normalize Ultralytics results into a flat list of detections."""
        out: list[Sam3Detection] = []
        if not results:
            return out

        # Ultralytics returns a list of Results; take the first (single image).
        r = results[0]
        masks = getattr(r, "masks", None)
        boxes = getattr(r, "boxes", None)
        if masks is None or boxes is None:
            return out

        mask_arr = masks.data.detach().cpu().numpy()       # (N, H, W) float/bool
        xyxy = boxes.xyxy.detach().cpu().numpy()           # (N, 4)
        conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else None
        cls = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else None
        names = getattr(r, "names", None) or {}
        H, W = frame_hw

        for i in range(len(mask_arr)):
            m = mask_arr[i]
            # Masks may come back at a different resolution; resize to frame HxW.
            # INTER_LINEAR (vs NEAREST) smooths the staircased edges you'd get
            # when upscaling a 448-res mask back to 640.
            if m.shape != (H, W):
                m = cv2.resize(
                    m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR
                )
            mask_bool = m > 0.5
            x1, y1, x2, y2 = xyxy[i].astype(int).tolist()
            score = float(conf[i]) if conf is not None else 1.0
            if cls is not None and cls[i] in names:
                label = str(names[int(cls[i])])
            elif cls is not None and 0 <= int(cls[i]) < len(concepts):
                label = concepts[int(cls[i])]
            else:
                label = concepts[0] if concepts else "?"
            out.append(Sam3Detection(mask=mask_bool, bbox=(x1, y1, x2, y2), score=score, label=label))
        return out


# ---------------------------------------------------------------------------
# Camera discovery (same logic used in new-teck-test.py)
# ---------------------------------------------------------------------------
def list_cameras_macos() -> list[str]:
    """Return camera names from `system_profiler`, in AVFoundation order."""
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
    """Return [(index, display_name), ...] for cameras that actually open."""
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


def pick_default_camera(cams: list[tuple[int, str]]) -> int:
    """Prefer a non-Reachy camera (the user's computer webcam)."""
    for idx, label in cams:
        if "reachy" not in label.lower():
            return idx
    return cams[0][0] if cams else 0


# ---------------------------------------------------------------------------
# Mask overlay utilities
# ---------------------------------------------------------------------------
_PALETTE_BGR = [
    (255, 56, 56),   (255, 159, 56),  (255, 239, 56), (56, 255, 110),
    (56, 255, 230),  (56, 169, 255),  (123, 56, 255), (222, 56, 255),
    (255, 56, 172),  (180, 180, 180),
]


def color_for(idx: int) -> tuple[int, int, int]:
    return _PALETTE_BGR[idx % len(_PALETTE_BGR)]


def overlay_detections(
    frame_bgr: np.ndarray, detections: list[Sam3Detection], alpha: float = 0.45
) -> np.ndarray:
    """Blend masks + draw boxes/labels onto a copy of the frame."""
    if not detections:
        return frame_bgr.copy()

    out = frame_bgr.copy()
    H, W = out.shape[:2]

    # Colour per unique label
    label_to_color: dict[str, tuple[int, int, int]] = {}
    for i, det in enumerate(detections):
        if det.label not in label_to_color:
            label_to_color[det.label] = color_for(len(label_to_color))

    # Blend masks
    overlay = out.copy()
    for det in detections:
        c = label_to_color[det.label]
        overlay[det.mask] = c
    out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0)

    # Draw boxes + labels
    for det in detections:
        c = label_to_color[det.label]
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        tag = f"{det.label} {det.score:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), c, -1)
        cv2.putText(
            out, tag, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

    return out


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
@dataclass
class AppState:
    concepts: list[str] = field(default_factory=list)
    latest_frame: Optional[np.ndarray] = None          # latest raw frame from cv2
    last_detections: list = field(default_factory=list)  # list[Sam3Detection]
    last_latency_ms: float = 0.0
    status: str = "Idle — type concepts, press Start Live."
    live: bool = False


class App:
    CAM_W, CAM_H = 640, 480
    CAP_BACKEND = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY

    def __init__(self, root: tk.Tk, segmenter: Sam3Segmenter):
        self.root = root
        self.segmenter = segmenter
        self.state = AppState()

        self.cameras = discover_cameras()
        print(f"[camera] Detected: {self.cameras}")
        self.current_cam_index = pick_default_camera(self.cameras)

        self._build_ui()

        self.cap: Optional[cv2.VideoCapture] = None
        self._open_camera(self.current_cam_index)

        # Persistent worker that runs segment() in a loop while state.live is True.
        self._worker_stop = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._live_worker, daemon=True
        )
        self._worker_thread.start()

        self.root.after(0, self._tick)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- camera -----------------------------------------------------------
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
            self._set_status(f"ERROR: Could not open camera {index}.", "red")
            self.cap = None
            return False
        self.cap = cap
        self.current_cam_index = index
        self._set_status(f"Camera {index} ready.", "black")
        return True

    # ---- UI ---------------------------------------------------------------
    def _build_ui(self):
        self.root.title("SAM 3 Test")
        self.root.minsize(720, 640)

        # Row 1: camera picker
        cam_row = ttk.Frame(self.root)
        cam_row.pack(fill="x", side="top", padx=8, pady=(8, 4))

        ttk.Label(cam_row, text="Camera:").pack(side="left")
        cam_values = [label for _i, label in self.cameras] or ["(none detected)"]
        self.cam_combo = ttk.Combobox(
            cam_row, values=cam_values, state="readonly", width=36
        )
        if self.cameras:
            default_label = next(
                (label for idx, label in self.cameras if idx == self.current_cam_index),
                cam_values[0],
            )
            self.cam_combo.set(default_label)
        else:
            self.cam_combo.set(cam_values[0])
        self.cam_combo.pack(side="left", padx=(6, 6))
        self.cam_combo.bind("<<ComboboxSelected>>", self._on_camera_change)

        # Row 2: concepts entry + buttons
        controls = ttk.Frame(self.root)
        controls.pack(fill="x", side="top", padx=8, pady=(0, 4))

        ttk.Label(controls, text="Concepts (comma-separated):").pack(side="left")
        self.entry = ttk.Entry(controls, width=30)
        self.entry.pack(side="left", padx=(6, 6))
        self.entry.insert(0, "person")
        self.entry.bind("<Return>", lambda _e: self._toggle_live())

        self.toggle_button = ttk.Button(
            controls, text="Start Live", command=self._toggle_live
        )
        self.toggle_button.pack(side="left")

        # Row 3: status
        self.status_label = ttk.Label(self.root, text=self.state.status, foreground="black")
        self.status_label.pack(fill="x", side="top", padx=8, pady=(0, 8))

        # Row 4: video
        self.video_label = ttk.Label(self.root)
        self.video_label.pack(side="top", padx=8, pady=8)

    def _on_camera_change(self, _event=None):
        label = self.cam_combo.get()
        for idx, lbl in self.cameras:
            if lbl == label:
                self.state.last_detections = []
                self._open_camera(idx)
                return

    def _set_status(self, text: str, color: str = "black"):
        self.state.status = text
        self.status_label.config(text=text, foreground=color)

    # ---- buttons ----------------------------------------------------------
    def _toggle_live(self) -> None:
        if self.state.live:
            self.state.live = False
            self.toggle_button.config(text="Start Live")
            self._set_status("Stopped. Camera still live.", "black")
            return

        raw = self.entry.get().strip()
        if not raw:
            self._set_status("Type one or more concepts first.", "orange")
            return
        concepts = [c.strip() for c in raw.split(",") if c.strip()]
        if not concepts:
            self._set_status("Type one or more concepts first.", "orange")
            return

        self.state.concepts = concepts
        self.state.last_detections = []
        self.state.live = True
        self.toggle_button.config(text="Stop")
        self._set_status(f"Live segmenting: {concepts} ...", "blue")

    # ---- worker -----------------------------------------------------------
    def _live_worker(self) -> None:
        """Run SAM 3 in a tight loop while self.state.live is True."""
        while not self._worker_stop.is_set():
            if (
                not self.state.live
                or not self.state.concepts
                or self.state.latest_frame is None
            ):
                time.sleep(0.05)
                continue

            # Snapshot the frame reference AND its pixels. The camera thread
            # reassigns state.latest_frame every ~33 ms, and capture buffers
            # can be recycled — a .copy() here is cheap and removes the race.
            frame = self.state.latest_frame
            if frame is None:
                time.sleep(0.05)
                continue
            frame = frame.copy()
            concepts = list(self.state.concepts)
            t0 = time.time()
            try:
                detections = self.segmenter.segment(frame, concepts)
            except Exception as e:
                msg = str(e)
                print(f"[sam3] Error: {msg}")
                self.state.live = False
                self.root.after(0, lambda m=msg: self._set_status(
                    f"SAM 3 error: {m}", "red"
                ))
                self.root.after(0, lambda: self.toggle_button.config(text="Start Live"))
                continue

            latency_ms = (time.time() - t0) * 1000
            self.state.last_detections = detections
            self.state.last_latency_ms = latency_ms
            n = len(detections)
            fps = 1000.0 / max(1.0, latency_ms)
            self.root.after(
                0,
                lambda n=n, ms=latency_ms, fps=fps, concepts=concepts:
                    self._set_status(
                        f"{n} instance(s) for {concepts}  |  "
                        f"{ms:.0f} ms ({fps:.1f} FPS)",
                        "green" if n else "orange",
                    ),
            )

    # ---- render loop ------------------------------------------------------
    def _tick(self):
        if self.cap is None:
            self.root.after(100, self._tick)
            return
        ok, frame = self.cap.read()
        if not ok:
            self.root.after(33, self._tick)
            return

        self.state.latest_frame = frame
        # Draw the freshest detections on the newest live frame. Masks will
        # lag behind fast motion (that's fine — SAM 3 runs ~1–2x/sec).
        dets = self.state.last_detections if self.state.live else []
        display = overlay_detections(frame, dets) if dets else frame

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(img)
        self.video_label.configure(image=self._photo)

        self.root.after(33, self._tick)

    def _on_close(self):
        self.state.live = False
        self._worker_stop.set()
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("[startup] Initializing SAM 3 segmenter.")
    try:
        segmenter = Sam3Segmenter()
    except (FileNotFoundError, ImportError) as e:
        print(f"\n[startup] {e}\n")
        sys.exit(1)

    root = tk.Tk()
    App(root, segmenter)
    root.mainloop()


if __name__ == "__main__":
    main()
