"""
OpenRouter Qwen-VL + SAM 2 live test
------------------------------------

Type what to look for (e.g. "human", "cat", "red cup"), press "Lock on".
Qwen2.5-VL-72B (hosted on OpenRouter) grounds the description to a bbox.
SAM 2 (local, on MPS) tracks that box frame-by-frame at ~30 FPS.

Requirements (install once):
    pip install torch torchvision torchaudio
    pip install openai python-dotenv
    pip install sam2            # Meta's SAM 2 package (real-time fork)
    pip install opencv-python pillow numpy

Env vars (put in `.env` at repo root):
    OPENROUTER_API_KEY=sk-or-...
    # optional override
    OPENROUTER_VL_MODEL=qwen/qwen2.5-vl-72b-instruct:nitro

Tested on: macOS with Apple Silicon (MPS). SAM 2 runs locally; Qwen runs
remotely on OpenRouter.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Device selection (MPS on Mac, CUDA if available, else CPU)
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
# Qwen-VL wrapper: "find this thing in this image, return a bbox"
# Uses OpenRouter's OpenAI-compatible API. No local weights needed.
# ---------------------------------------------------------------------------
def _first_complete_json(text: str) -> Optional[str]:
    """Return the first balanced {...} substring, or None if not yet complete."""
    depth = 0
    in_str = False
    esc = False
    start = -1
    for i, c in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start:i + 1]
    return None


class QwenGrounder:
    """Turns a text description + an image into a bounding box via OpenRouter."""

    DEFAULT_MODEL = "qwen/qwen3-vl-8b-instruct"
    BASE_URL = "https://openrouter.ai/api/v1"
    MAX_SIDE = 448  # downscale long side before upload to cut prefill cost
    MAX_OUTPUT_TOKENS = 64

    def __init__(self):
        from openai import OpenAI

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Put it in `.env` at the repo root."
            )
        self.model_id = os.getenv("OPENROUTER_VL_MODEL", self.DEFAULT_MODEL)
        print(f"[qwen] Using OpenRouter model: {self.model_id}")
        self.client = OpenAI(base_url=self.BASE_URL, api_key=api_key)
        print("[qwen] Ready.")

    def find(self, frame_bgr: np.ndarray, description: str) -> Optional[tuple[int, int, int, int]]:
        """Return (x1, y1, x2, y2) in ORIGINAL-frame pixel coords, or None."""
        orig_h, orig_w = frame_bgr.shape[:2]

        # Downsize so the longest side is MAX_SIDE. Cuts Qwen visual tokens
        # ~4x for a 640x480 source (from ~370 tokens to ~100).
        scale = min(1.0, self.MAX_SIDE / max(orig_h, orig_w))
        if scale < 1.0:
            sent_w = int(round(orig_w * scale))
            sent_h = int(round(orig_h * scale))
            sent = cv2.resize(frame_bgr, (sent_w, sent_h), interpolation=cv2.INTER_AREA)
        else:
            sent_w, sent_h = orig_w, orig_h
            sent = frame_bgr

        # Encode downsized frame as JPEG + base64 data URL for the API
        ok, jpeg = cv2.imencode(".jpg", sent, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            print("[qwen] Failed to JPEG-encode frame.")
            return None
        b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64}"

        prompt = (
            f"Locate the {description} in the image. "
            f"Return ONLY a JSON object of the form "
            f'{{"bbox_2d": [x1, y1, x2, y2], "label": "{description}"}} '
            f"using absolute pixel coordinates in an image that is "
            f"{sent_w} pixels wide and {sent_h} pixels tall. "
            f'If the {description} is not visible, return {{"bbox_2d": null}}.'
        )

        t0 = time.time()
        try:
            stream = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                max_tokens=self.MAX_OUTPUT_TOKENS,
                temperature=0.0,
                stream=True,
            )
        except Exception as e:
            print(f"[qwen] OpenRouter error: {e}")
            raise

        # Accumulate streamed deltas and stop as soon as the first balanced
        # JSON object closes — we don't need any trailing fences/commentary.
        buf: list[str] = []
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content or ""
            except (AttributeError, IndexError):
                delta = ""
            if not delta:
                continue
            buf.append(delta)
            if _first_complete_json("".join(buf)) is not None:
                break
        try:
            stream.close()
        except Exception:
            pass

        elapsed = (time.time() - t0) * 1000
        print(f"[qwen] Inference: {elapsed:.0f} ms")

        output_text = "".join(buf).strip()
        print(f"[qwen] Raw output: {output_text!r}")

        bbox = self._parse_bbox(output_text, sent_w, sent_h)
        if bbox is None or scale >= 1.0:
            return bbox
        # Rescale from downsized coords back to the original frame
        x1, y1, x2, y2 = bbox
        inv = 1.0 / scale
        return (
            max(0, min(orig_w - 1, int(round(x1 * inv)))),
            max(0, min(orig_h - 1, int(round(y1 * inv)))),
            max(0, min(orig_w - 1, int(round(x2 * inv)))),
            max(0, min(orig_h - 1, int(round(y2 * inv)))),
        )

    @staticmethod
    def _parse_bbox(text: str, w: int, h: int) -> Optional[tuple[int, int, int, int]]:
        """Best-effort JSON parse. Clamps to image bounds."""
        # Pull the first JSON-looking object out of the text
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

        bbox = data.get("bbox_2d") or data.get("bbox") or data.get("box")
        if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        # Some Qwen variants output normalized [0..1000] — detect and rescale.
        if max(x1, y1, x2, y2) <= 1000 and max(w, h) > 1000:
            x1 = int(x1 * w / 1000); x2 = int(x2 * w / 1000)
            y1 = int(y1 * h / 1000); y2 = int(y2 * h / 1000)

        # Clamp & sanity-check
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# SAM 2 wrapper: initialize from a box, then update mask every frame
# ---------------------------------------------------------------------------
class Sam2Tracker:
    """Per-frame appearance tracker using SAM 2's camera predictor."""

    MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
    CHECKPOINT = "checkpoints/sam2.1_hiera_small.pt"

    def __init__(self):
        print("[sam2] Loading SAM 2 (first run downloads ~180 MB)...")
        # SAM 2 has a dedicated streaming-video predictor ("camera predictor")
        # that is built for exactly this use case.
        from sam2.build_sam import build_sam2_camera_predictor

        self.predictor = build_sam2_camera_predictor(
            self.MODEL_CFG, self.CHECKPOINT, device=DEVICE
        )
        self.active = False
        print("[sam2] Ready.")

    def start(self, frame_rgb: np.ndarray, bbox: tuple[int, int, int, int]):
        """(Re)initialize tracking with a new box."""
        # Reset any prior state, then load the first frame and inject the box.
        self.predictor.load_first_frame(frame_rgb)
        x1, y1, x2, y2 = bbox
        box_np = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
        _, _, mask_logits = self.predictor.add_new_prompt(
            frame_idx=0, obj_id=1, bbox=box_np
        )
        self.active = True
        return self._mask_to_bbox(mask_logits)

    def track(self, frame_rgb: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """Run one tracking step. Returns the current bbox, or None if lost."""
        if not self.active:
            return None
        _, mask_logits = self.predictor.track(frame_rgb)
        return self._mask_to_bbox(mask_logits)

    def stop(self):
        self.active = False

    @staticmethod
    def _mask_to_bbox(mask_logits) -> Optional[tuple[int, int, int, int]]:
        """Convert SAM 2's mask logits to an axis-aligned bbox."""
        if mask_logits is None:
            return None
        # mask_logits shape: (N, 1, H, W) tensor; positive = object
        mask = (mask_logits[0, 0] > 0).detach().cpu().numpy()
        ys, xs = np.where(mask)
        if xs.size < 20:  # target is effectively lost
            return None
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


# ---------------------------------------------------------------------------
# Camera discovery
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
# GUI + main loop
# ---------------------------------------------------------------------------
@dataclass
class AppState:
    target_description: str = ""
    bbox: Optional[tuple[int, int, int, int]] = None
    status: str = "Idle — type what to look for, then press Lock on."
    status_color: str = "black"
    # Most recent frame captured by the video thread. Tk reads this each tick.
    latest_frame: Optional[np.ndarray] = None


class App:
    CAM_W, CAM_H = 640, 480
    CAP_BACKEND = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY

    def __init__(self, root: tk.Tk, grounder: QwenGrounder, tracker: Sam2Tracker):
        self.root = root
        self.grounder = grounder
        self.tracker = tracker
        self.state = AppState()

        # Discover cameras BEFORE building the UI so the dropdown has options.
        self.cameras = discover_cameras()
        print(f"[camera] Detected: {self.cameras}")
        self.current_cam_index = pick_default_camera(self.cameras)

        self._build_ui()

        # Open the chosen camera
        self.cap: Optional[cv2.VideoCapture] = None
        self._open_camera(self.current_cam_index)

        # Work queue: GUI thread posts (frame, description) onto it,
        # a worker thread runs Qwen (slow) without freezing the UI.
        self.lock_queue: queue.Queue = queue.Queue(maxsize=1)
        self.worker = threading.Thread(target=self._grounding_worker, daemon=True)
        self.worker.start()

        # Start the render/track loop
        self.root.after(0, self._tick)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _open_camera(self, index: int) -> bool:
        """(Re)open a camera by index. Returns True on success."""
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
        self._set_status(f"Camera {index} ready. Type what to look for, then Lock on.", "black")
        return True

    # ---- UI ------------------------------------------------------------
    def _build_ui(self):
        self.root.title("Qwen-VL + SAM 2 Test")
        self.root.minsize(720, 640)

        # --- Row 1: camera picker (always on top so it's never clipped) ---
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

        # --- Row 2: query + action buttons ---
        controls = ttk.Frame(self.root)
        controls.pack(fill="x", side="top", padx=8, pady=(0, 4))

        ttk.Label(controls, text="Look for:").pack(side="left")
        self.entry = ttk.Entry(controls, width=30)
        self.entry.pack(side="left", padx=(6, 6))
        self.entry.insert(0, "human")
        self.entry.bind("<Return>", lambda _e: self._on_lock())

        self.lock_button = ttk.Button(controls, text="Lock on", command=self._on_lock)
        self.lock_button.pack(side="left")

        self.clear_button = ttk.Button(controls, text="Clear", command=self._on_clear)
        self.clear_button.pack(side="left", padx=(6, 0))

        # --- Row 3: status ---
        self.status_label = ttk.Label(self.root, text=self.state.status, foreground="black")
        self.status_label.pack(fill="x", side="top", padx=8, pady=(0, 8))

        # --- Row 4: video (last so it absorbs leftover space) ---
        self.video_label = ttk.Label(self.root)
        self.video_label.pack(side="top", padx=8, pady=8)

    def _on_camera_change(self, _event=None):
        label = self.cam_combo.get()
        for idx, lbl in self.cameras:
            if lbl == label:
                # Reset any active tracking when switching sources
                self.tracker.stop()
                self.state.bbox = None
                self.state.latest_frame = None
                self._open_camera(idx)
                return

    def _set_status(self, text: str, color: str = "black"):
        self.state.status = text
        self.state.status_color = color
        self.status_label.config(text=text, foreground=color)

    # ---- Button handlers ----------------------------------------------
    def _on_lock(self):
        desc = self.entry.get().strip()
        if not desc:
            self._set_status("Type something to look for first.", "orange")
            return
        if self.state.latest_frame is None:
            self._set_status("No camera frame yet.", "orange")
            return

        self.state.target_description = desc
        self._set_status(f"Asking Qwen-VL to find '{desc}'...", "blue")
        # Send a COPY of the frame to the worker so it's not racing with capture
        try:
            self.lock_queue.put_nowait((self.state.latest_frame.copy(), desc))
        except queue.Full:
            # A previous grounding request is still running; skip.
            self._set_status("Already working on a request, try again.", "orange")

    def _on_clear(self):
        self.tracker.stop()
        self.state.bbox = None
        self.state.target_description = ""
        self._set_status("Cleared. Type something and press Lock on.", "black")

    # ---- Background worker: runs Qwen without blocking the UI ---------
    def _grounding_worker(self):
        while True:
            frame, desc = self.lock_queue.get()   # blocks
            try:
                bbox = self.grounder.find(frame, desc)
            except Exception as e:
                msg = str(e)
                print(f"[qwen] Error: {msg}")
                self.root.after(0, lambda m=msg: self._set_status(f"Qwen error: {m}", "red"))
                continue

            if bbox is None:
                self.root.after(0, lambda d=desc:
                    self._set_status(f"Qwen couldn't see '{d}' in the frame.", "red"))
                continue

            # Hand off to SAM 2 (must run in main thread since it owns MPS state cleanly)
            self.root.after(0, lambda f=frame, b=bbox, d=desc: self._init_tracker(f, b, d))

    def _init_tracker(self, frame_bgr: np.ndarray, bbox: tuple[int, int, int, int], desc: str):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            new_bbox = self.tracker.start(frame_rgb, bbox)
        except Exception as e:
            print(f"[sam2] Error: {e}")
            self._set_status(f"SAM 2 error: {e}", "red")
            return
        self.state.bbox = new_bbox or bbox
        self._set_status(f"Locked onto '{desc}'. Tracking...", "green")

    # ---- Main render loop ---------------------------------------------
    def _tick(self):
        if self.cap is None:
            self.root.after(100, self._tick)
            return
        ok, frame = self.cap.read()
        if not ok:
            self.root.after(33, self._tick)
            return

        self.state.latest_frame = frame

        # If SAM 2 is active, update the tracked bbox every frame
        if self.tracker.active:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                new_bbox = self.tracker.track(frame_rgb)
            except Exception as e:
                print(f"[sam2] Track error: {e}")
                new_bbox = None

            if new_bbox is None:
                # Lost the target
                self.tracker.stop()
                self.state.bbox = None
                self._set_status(
                    f"Lost track of '{self.state.target_description}'. Lock on again.",
                    "red",
                )
            else:
                self.state.bbox = new_bbox

        # Draw overlay
        display = frame.copy()
        if self.state.bbox is not None:
            x1, y1, x2, y2 = self.state.bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = self.state.target_description or "target"
            cv2.putText(
                display, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
            )

        # OpenCV BGR -> Tk-friendly RGB PhotoImage
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=photo)
        self.video_label.image = photo  # keep a reference

        self.root.after(15, self._tick)  # ~60 FPS upper bound; camera caps it

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
    print("[startup] Initializing grounding client + local SAM 2.")
    grounder = QwenGrounder()
    tracker = Sam2Tracker()

    root = tk.Tk()
    App(root, grounder, tracker)
    root.mainloop()


if __name__ == "__main__":
    main()