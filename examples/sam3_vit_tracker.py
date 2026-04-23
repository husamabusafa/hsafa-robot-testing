"""
SAM 3 / 3.1 + ViT tracker — the third point of the triangle
============================================================

Two siblings already exist in this folder. This one is the middle option:

    sam3_tracker.py           SAM 3.1 + OpenCV CSRT (classic SOT)
    sam3_vit_tracker.py       SAM 3.1 + OpenCV TrackerVit (this file)
    sam3_native_tracker.py    SAM 3.1 native video tracker (no external SOT)

They all expose the same concept — "lock on a text concept, follow one
instance" — and the same Tk UI, so you can A/B them on the same hardware
and the same camera to see which combination feels best on Reachy Mini.

ViT vs CSRT in one paragraph:
    CSRT is a 2017 correlation filter (spatial regularization). Very
    robust on rigid objects and fast on CPU (~20–30 FPS at 640×480), but
    its appearance model struggles with scale jumps, deformation and
    partial occlusion.
    TrackerVit (OpenCV Model Zoo, 2023) is a distilled Vision Transformer
    SOT model. It handles appearance changes and partial occlusion better,
    typically at ~30–60 FPS on CPU with a lightweight ~700 KB ONNX net.

The only delta vs `sam3_tracker.py` is the tracker factory:
    Sam3Follower(segmenter, tracker_factory=_create_vit_tracker(onnx_path),
                 tracker_name="ViT")

Weights:
    Download the ONNX once:
        curl -L -o checkpoints/vit_tracker.onnx \\
          https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx
    Or set VIT_TRACKER_ONNX=/path/to/model.onnx to point at your own copy.
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from typing import Optional

import cv2

# Pull everything shared from the CSRT variant so the two stay in lockstep.
# This file is intentionally tiny — only the tracker factory is different.
from sam3_tracker import (
    App,
    FollowState,  # noqa: F401 — re-exported for consumers
    Sam3Follower,
    Sam3Segmenter,
)


# ---------------------------------------------------------------------------
# ViT tracker factory
# ---------------------------------------------------------------------------
DEFAULT_VIT_ONNX = "checkpoints/vit_tracker.onnx"


def _resolve_onnx_path(path: Optional[str] = None) -> str:
    if path:
        return path
    env = os.getenv("VIT_TRACKER_ONNX")
    if env:
        return env
    return DEFAULT_VIT_ONNX


def _build_vit_params(onnx_path: str) -> "cv2.TrackerVit_Params":
    """Build a TrackerVit_Params pointing at the ONNX model.

    `backend` / `target` are OpenCV DNN selectors. Defaults (0/0) = CPU,
    which is fine — the ONNX net is ~700 KB and runs at 30+ FPS on an M-series
    CPU. Setting target=cv2.dnn.DNN_TARGET_CPU explicitly silences a noisy
    warning on some OpenCV builds.
    """
    params = cv2.TrackerVit_Params()
    params.net = onnx_path
    # Explicit CPU target (DNN_BACKEND_OPENCV / DNN_TARGET_CPU).
    params.backend = cv2.dnn.DNN_BACKEND_OPENCV
    params.target = cv2.dnn.DNN_TARGET_CPU
    return params


def make_vit_factory(onnx_path: Optional[str] = None):
    """Return a no-arg callable that builds a fresh cv2.TrackerVit each time
    `Sam3Follower._init_tracker` is called.

    OpenCV trackers are single-target and single-init by design — every
    re-ground or re-acquire creates a new instance. This keeps the ONNX
    path resolution outside the hot path.
    """
    resolved = _resolve_onnx_path(onnx_path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"ViT tracker ONNX not found at {resolved!r}.\n"
            "Download with:\n"
            "  curl -L -o checkpoints/vit_tracker.onnx \\\n"
            "    https://github.com/opencv/opencv_zoo/raw/main/models/"
            "object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx\n"
            "or set VIT_TRACKER_ONNX=/path/to/model.onnx"
        )
    params = _build_vit_params(resolved)

    def _factory():
        return cv2.TrackerVit_create(params)

    # Attach the resolved path so the UI can surface it.
    _factory.onnx_path = resolved  # type: ignore[attr-defined]
    return _factory


# ---------------------------------------------------------------------------
# Entry point — identical to sam3_tracker.py, only the follower's tracker
# factory differs. We also swap the window title to avoid confusion when
# both demos are open at once.
# ---------------------------------------------------------------------------
def main():
    print("[startup] Initializing SAM 3 segmenter + ViT tracker follower.")
    try:
        vit_factory = make_vit_factory()
    except FileNotFoundError as e:
        print(f"\n[startup] {e}\n")
        sys.exit(1)
    print(f"[startup] ViT tracker ONNX: {vit_factory.onnx_path}")

    try:
        segmenter = Sam3Segmenter()
    except (FileNotFoundError, ImportError) as e:
        print(f"\n[startup] {e}\n")
        sys.exit(1)

    follower = Sam3Follower(
        segmenter,
        tracker_factory=vit_factory,
        tracker_name="ViT",
    )

    root = tk.Tk()
    app = App(root, follower)
    # Retitle so users can distinguish this from the CSRT demo window.
    root.title("SAM 3 + ViT follower")
    root.mainloop()
    _ = app  # keep reference so GC doesn't close it early


if __name__ == "__main__":
    main()
