"""
Florence-2 live test (MIT-licensed, ungated)
--------------------------------------------

Type any concepts (comma-separated, e.g. "person, glasses, cup"), press
"Segment". The current webcam frame is sent to Microsoft Florence-2 in two
stages:

    1. <CAPTION_TO_PHRASE_GROUNDING> -> (bbox, label) for every instance.
    2. <REGION_TO_SEGMENTATION> per bbox -> polygon -> filled mask.

The overlay shows boxes + masks just like the SAM 3 test. Florence-2 is
fully local (~500 MB download on first run) and requires no access
request, unlike facebook/sam3.

Requirements (install once):
    pip install -U transformers pillow einops timm
    # torch, opencv-python, numpy are already pulled in by ultralytics.

Optional:
    FLORENCE2_MODEL=microsoft/Florence-2-base  (default, ~230M params)
    FLORENCE2_MODEL=microsoft/Florence-2-large (~770M, more accurate, slower)

Tested on: macOS / Apple Silicon. Runs on MPS, falls back to CPU.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()
print(f"[startup] Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Florence-2 wrapper
# ---------------------------------------------------------------------------
@dataclass
class Florence2Detection:
    mask: np.ndarray        # (H, W) bool
    bbox: tuple[int, int, int, int]
    label: str


class Florence2Segmenter:
    """Two-stage text->masks pipeline using Microsoft Florence-2."""

    DEFAULT_MODEL = "microsoft/Florence-2-base"
    TASK_GROUND = "<CAPTION_TO_PHRASE_GROUNDING>"
    TASK_SEG = "<REGION_TO_SEGMENTATION>"

    def __init__(self, model_id: Optional[str] = None):
        model_id = model_id or os.getenv("FLORENCE2_MODEL", self.DEFAULT_MODEL)
        print(f"[florence2] Loading {model_id} (first run downloads ~500 MB)...")

        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "transformers is not installed. Install with:\n"
                "    pip install -U transformers einops timm"
            ) from e

        # fp16 on CUDA, fp32 elsewhere (MPS can be flaky with fp16 for Florence-2)
        dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(DEVICE).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model_id = model_id
        print("[florence2] Ready.")

    @torch.inference_mode()
    def _run(self, task: str, text_payload: str, pil_image: Image.Image) -> dict:
        """Run one Florence-2 call and post-process its output."""
        prompt = task + text_payload
        inputs = self.processor(
            text=prompt, images=pil_image, return_tensors="pt"
        ).to(DEVICE, self.model.dtype if hasattr(self.model, "dtype") else torch.float32)

        # pixel_values must match model dtype; input_ids must stay long.
        pixel_values = inputs["pixel_values"].to(
            self.model.parameters().__next__().dtype
        )
        generated = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=pixel_values,
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )
        raw = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
        return self.processor.post_process_generation(
            raw, task=task, image_size=(pil_image.width, pil_image.height)
        )

    def segment(
        self, frame_bgr: np.ndarray, concepts: list[str]
    ) -> list[Florence2Detection]:
        """Return every instance in the frame matching any of the given text concepts."""
        if not concepts:
            return []

        pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        H, W = frame_bgr.shape[:2]

        # Stage 1: grounding. Florence expects period-separated phrases.
        phrase = ". ".join(c.strip(" .,") for c in concepts if c.strip())
        if not phrase.endswith("."):
            phrase += "."

        t0 = time.time()
        ground = self._run(self.TASK_GROUND, phrase, pil)
        t_ground = (time.time() - t0) * 1000
        bboxes = ground.get(self.TASK_GROUND, {}).get("bboxes", []) or []
        labels = ground.get(self.TASK_GROUND, {}).get("labels", []) or []
        print(
            f"[florence2] Grounding: {t_ground:.0f} ms, "
            f"{len(bboxes)} instance(s) for {concepts}"
        )
        if not bboxes:
            return []

        # Stage 2: per-bbox segmentation. Each call returns one polygon.
        detections: list[Florence2Detection] = []
        t1 = time.time()
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
            y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            # Florence-2 region prompt uses quantized 0..999 coords.
            loc = self._loc_prompt(x1, y1, x2, y2, W, H)
            seg = self._run(self.TASK_SEG, loc, pil)
            polys = seg.get(self.TASK_SEG, {}).get("polygons", []) or []
            mask = self._polys_to_mask(polys, H, W)
            if mask.sum() == 0:
                # Fall back to bbox-filled rectangle so we still show something.
                mask = np.zeros((H, W), dtype=bool)
                mask[y1:y2, x1:x2] = True
            detections.append(
                Florence2Detection(
                    mask=mask, bbox=(x1, y1, x2, y2), label=str(label) or concepts[0]
                )
            )
        t_seg = (time.time() - t1) * 1000
        print(f"[florence2] Segmentation: {t_seg:.0f} ms for {len(detections)} region(s)")
        return detections

    @staticmethod
    def _loc_prompt(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> str:
        def q(v: int, dim: int) -> int:
            return max(0, min(999, int(round(v * 999 / max(1, dim - 1)))))
        return (
            f"<loc_{q(x1, W)}><loc_{q(y1, H)}>"
            f"<loc_{q(x2, W)}><loc_{q(y2, H)}>"
        )

    @staticmethod
    def _polys_to_mask(polys, H: int, W: int) -> np.ndarray:
        """Convert Florence-2's nested polygon output to a boolean mask."""
        canvas = np.zeros((H, W), dtype=np.uint8)
        # polys is list[list[list[float]]]: per-object -> per-contour -> flat [x,y,x,y,...]
        for obj in polys or []:
            for contour in obj or []:
                if not contour:
                    continue
                pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
                if pts.shape[0] < 3:
                    continue
                pts_int = np.round(pts).astype(np.int32)
                cv2.fillPoly(canvas, [pts_int], 255)
        return canvas > 127


# ---------------------------------------------------------------------------
# Camera discovery (same logic used in the other tests)
# ---------------------------------------------------------------------------
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


def pick_default_camera(cams: list[tuple[int, str]]) -> int:
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
    frame_bgr: np.ndarray, detections: list[Florence2Detection], alpha: float = 0.45
) -> np.ndarray:
    if not detections:
        return frame_bgr.copy()

    out = frame_bgr.copy()

    label_to_color: dict[str, tuple[int, int, int]] = {}
    for det in detections:
        if det.label not in label_to_color:
            label_to_color[det.label] = color_for(len(label_to_color))

    overlay = out.copy()
    for det in detections:
        overlay[det.mask] = label_to_color[det.label]
    out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0)

    for det in detections:
        c = label_to_color[det.label]
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        tag = det.label
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
    last_overlay: Optional[np.ndarray] = None
    latest_frame: Optional[np.ndarray] = None
    status: str = "Idle - type concepts, press Segment."
    busy: bool = False


class App:
    CAM_W, CAM_H = 640, 480
    CAP_BACKEND = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY

    def __init__(self, root: tk.Tk, segmenter: Florence2Segmenter):
        self.root = root
        self.segmenter = segmenter
        self.state = AppState()

        self.cameras = discover_cameras()
        print(f"[camera] Detected: {self.cameras}")
        self.current_cam_index = pick_default_camera(self.cameras)

        self._build_ui()

        self.cap: Optional[cv2.VideoCapture] = None
        self._open_camera(self.current_cam_index)

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
        self.root.title("Florence-2 Test")
        self.root.minsize(720, 640)

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

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", side="top", padx=8, pady=(0, 4))

        ttk.Label(controls, text="Concepts (comma-separated):").pack(side="left")
        self.entry = ttk.Entry(controls, width=30)
        self.entry.pack(side="left", padx=(6, 6))
        self.entry.insert(0, "person")
        self.entry.bind("<Return>", lambda _e: self._on_segment())

        self.seg_button = ttk.Button(controls, text="Segment", command=self._on_segment)
        self.seg_button.pack(side="left")

        self.clear_button = ttk.Button(controls, text="Clear", command=self._on_clear)
        self.clear_button.pack(side="left", padx=(6, 0))

        self.status_label = ttk.Label(self.root, text=self.state.status, foreground="black")
        self.status_label.pack(fill="x", side="top", padx=8, pady=(0, 8))

        self.video_label = ttk.Label(self.root)
        self.video_label.pack(side="top", padx=8, pady=8)

    def _on_camera_change(self, _event=None):
        label = self.cam_combo.get()
        for idx, lbl in self.cameras:
            if lbl == label:
                self.state.last_overlay = None
                self._open_camera(idx)
                return

    def _set_status(self, text: str, color: str = "black"):
        self.state.status = text
        self.status_label.config(text=text, foreground=color)

    # ---- buttons ----------------------------------------------------------
    def _on_segment(self):
        if self.state.busy:
            self._set_status("Busy - wait for the current segmentation to finish.", "orange")
            return
        if self.state.latest_frame is None:
            self._set_status("No camera frame yet.", "orange")
            return

        raw = self.entry.get().strip()
        if not raw:
            self._set_status("Type one or more concepts first.", "orange")
            return
        concepts = [c.strip() for c in raw.split(",") if c.strip()]
        if not concepts:
            self._set_status("Type one or more concepts first.", "orange")
            return

        frame = self.state.latest_frame.copy()
        self.state.concepts = concepts
        self.state.busy = True
        self._set_status(f"Running Florence-2 on: {concepts} ...", "blue")
        threading.Thread(
            target=self._run_segmentation, args=(frame, concepts), daemon=True
        ).start()

    def _on_clear(self):
        self.state.last_overlay = None
        self._set_status("Cleared. Live view restored.", "black")

    # ---- worker -----------------------------------------------------------
    def _run_segmentation(self, frame_bgr: np.ndarray, concepts: list[str]):
        try:
            detections = self.segmenter.segment(frame_bgr, concepts)
        except Exception as e:
            msg = str(e)
            print(f"[florence2] Error: {msg}")
            self.root.after(0, lambda m=msg: self._set_status(f"Florence-2 error: {m}", "red"))
            self.state.busy = False
            return

        overlay = overlay_detections(frame_bgr, detections)
        n = len(detections)
        self.state.last_overlay = overlay
        self.state.busy = False
        self.root.after(
            0,
            lambda n=n, concepts=concepts: self._set_status(
                f"{n} instance(s) found for {concepts}. Press Clear for live view.",
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
        display = self.state.last_overlay if self.state.last_overlay is not None else frame

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(img)
        self.video_label.configure(image=self._photo)

        self.root.after(33, self._tick)

    def _on_close(self):
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
    print("[startup] Initializing Florence-2 segmenter.")
    try:
        segmenter = Florence2Segmenter()
    except (ImportError, OSError) as e:
        print(f"\n[startup] {e}\n")
        sys.exit(1)

    root = tk.Tk()
    App(root, segmenter)
    root.mainloop()


if __name__ == "__main__":
    main()
