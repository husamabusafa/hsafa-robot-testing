"""
Smoke test for SAM 3 native video tracking on Apple Silicon (MPS).

What this answers (no UI, ~20 seconds):

    1. Does `SAM3VideoSemanticPredictor` LOAD on this machine/device?
    2. Does a manual per-frame drive (OpenCV owned camera + predictor fed
       one frame at a time) actually run end-to-end on MPS?
    3. What is the PER-FRAME latency? This decides whether native SAM 3
       video tracking is viable as a drop-in for the CSRT path.

Design note:
    Ultralytics' `stream_inference` pipeline assumes a file-backed video
    dataset (`mode='video'`, `.frame` counter). Live webcams via LoadStreams
    don't have that shape, so `init_state` and `inference()` both fail.
    Rather than fight the dataset layer, this test drives the predictor
    by hand — the same pattern the follower class will use:

        predictor.preprocess([frame])  ->  predictor.inference(im, text=...)
        ->  predictor.postprocess(preds, im, [frame])  ->  Results

    Per-frame cost is what matters; we time that, not `stream_inference`.

Run:
    .venv/bin/python3 examples/sam3_video_smoke.py
    # optional env:
    #   SAM3_WEIGHTS=checkpoints/sam3.1_multiplex.pt
    #   SAM3_IMGSZ=448
    #   SAM3_DEVICE=mps|cpu|cuda
    #   SAM3_CONCEPT="person"
    #   SAM3_CAM=0
    #   SAM3_FRAMES=15
"""

from __future__ import annotations

import os
import statistics
import sys
import time
import traceback

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def pick_device() -> str:
    if os.getenv("SAM3_DEVICE"):
        return os.environ["SAM3_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


CANDIDATE_WEIGHTS = (
    "checkpoints/sam3.1_multiplex.pt",
    "checkpoints/sam3.pt",
)


def pick_weights() -> str:
    env = os.getenv("SAM3_WEIGHTS")
    if env:
        return env
    for p in CANDIDATE_WEIGHTS:
        if os.path.isfile(p):
            return p
    return CANDIDATE_WEIGHTS[0]


# ---------------------------------------------------------------------------
# Minimal fake dataset — just enough for SAM 3 video predictor internals.
# `mode='video'` satisfies the assert in init_state; `frames` sizes the
# per-frame geometric-prompt buffer; `frame` is the 1-based current index
# that `inference()` reads (it does `frame - 1` to get 0-based).
# ---------------------------------------------------------------------------
class _FakeVideoDataset:
    mode = "video"
    bs = 1

    def __init__(self, frame_cap: int = 1_000_000) -> None:
        self.frames = frame_cap
        self.frame = 0


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def main() -> int:
    device = pick_device()
    weights = pick_weights()
    imgsz = int(os.getenv("SAM3_IMGSZ", "448"))
    concept = os.getenv("SAM3_CONCEPT", "person")
    cam = int(os.getenv("SAM3_CAM", "0"))
    n_frames = int(os.getenv("SAM3_FRAMES", "15"))

    print("=" * 70)
    print(" SAM 3 native video tracking — smoke test (manual drive)")
    print("=" * 70)
    print(f"  device   : {device}")
    print(f"  weights  : {weights}")
    print(f"  imgsz    : {imgsz}")
    print(f"  concept  : {concept!r}")
    print(f"  camera   : {cam}")
    print(f"  frames   : {n_frames}")
    print("-" * 70)

    if not os.path.isfile(weights):
        print(f"ERROR: weights not found: {weights}", file=sys.stderr)
        return 2

    # --- 1) Construct predictor + load model ------------------------------
    t0 = time.time()
    try:
        from ultralytics.models.sam import SAM3VideoSemanticPredictor
    except Exception as e:
        print(f"IMPORT FAILED: {e}", file=sys.stderr)
        return 3

    overrides = dict(
        model=weights,
        task="segment",
        mode="predict",
        device=device,
        half=(device in ("cuda", "mps")),
        imgsz=imgsz,
        conf=0.25,
        verbose=False,
        save=False,
    )
    try:
        predictor = SAM3VideoSemanticPredictor(overrides=overrides)
        # Load the underlying model now (setup_model on the base predictor).
        predictor.setup_model(model=None, verbose=False)
        # Run SAM3SemanticPredictor.setup_source with a 1-frame numpy source
        # so imgsz + tracker internals are initialized. We then swap the
        # dataset for our fake one before feeding real webcam frames.
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        predictor.setup_source(dummy)
        predictor.dataset = _FakeVideoDataset()
        # Fire on_predict_start manually (it runs init_state).
        predictor.inference_state = {}  # force re-init on callback
        predictor.run_callbacks("on_predict_start")
    except Exception as e:
        print(f"SETUP FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 4
    construct_ms = (time.time() - t0) * 1000
    print(f"[1/3] constructed + set up in {construct_ms:.0f} ms")

    # --- 2) Open the webcam with OpenCV and drive the predictor -----------
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {cam}", file=sys.stderr)
        return 5

    # Pull one good frame before we start timing.
    for _ in range(5):
        ok, warm = cap.read()
        if ok and warm is not None:
            break
        time.sleep(0.05)
    if not ok or warm is None:
        print("ERROR: no frames from camera", file=sys.stderr)
        cap.release()
        return 5

    print(f"[2/3] feeding {n_frames} frames to predictor (manual drive) ...")
    per_frame_ms: list[float] = []
    first_frame_ms: float = -1.0
    first_det_ids: list = []

    try:
        # The whole loop must run under inference_mode. The predictor's
        # internal `@smart_inference_mode()` on `add_prompt` would otherwise
        # produce tensors that flow back into normal autograd and crash in
        # layer_norm ("Inference tensors cannot be saved for backward").
        inf_ctx = torch.inference_mode()
        inf_ctx.__enter__()
        for i in range(n_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"    [f{i}] no frame from camera; skipping", file=sys.stderr)
                time.sleep(0.02)
                continue

            t_step = time.time()
            # Advance the fake frame counter (1-based; inference does -1).
            predictor.dataset.frame = i + 1

            # Standard ultralytics batch is (paths, im0s, s). `add_prompt`
            # and `postprocess` both read `self.batch`, so we must set it.
            im0s = [frame]
            predictor.batch = (["camera"], im0s, [""])
            im = predictor.preprocess(im0s)

            # First frame carries the text prompt; subsequent frames don't.
            if i == 0:
                preds = predictor.inference(im, text=[concept])
            else:
                preds = predictor.inference(im)

            results = predictor.postprocess(preds, im, im0s)
            dt_ms = (time.time() - t_step) * 1000

            # Dig out box count + obj_ids (column 4 of the 7-col boxes tensor).
            r = results[0]
            boxes = getattr(r, "boxes", None)
            masks = getattr(r, "masks", None)
            n_boxes = 0 if boxes is None else len(boxes)
            n_masks = 0 if masks is None else len(masks.data)
            ids: list = []
            if boxes is not None and n_boxes > 0:
                data = boxes.data.detach().cpu()
                if data.shape[1] >= 7:
                    ids = data[:, 4].int().tolist()

            tag = "first" if i == 0 else f"f{i}"
            print(
                f"    [{tag}] {dt_ms:6.0f} ms  "
                f"boxes={n_boxes}  masks={n_masks}  ids={ids}"
            )

            if i == 0:
                first_frame_ms = dt_ms
                first_det_ids = ids
            else:
                per_frame_ms.append(dt_ms)
    except Exception as e:
        print(f"INFER RAISED: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        try:
            inf_ctx.__exit__(None, None, None)
        except Exception:
            pass
        cap.release()
        return 6
    finally:
        try:
            inf_ctx.__exit__(None, None, None)
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass

    # --- 3) Verdict -------------------------------------------------------
    print("-" * 70)
    if per_frame_ms:
        med = statistics.median(per_frame_ms)
        mean = statistics.mean(per_frame_ms)
        p95 = (
            sorted(per_frame_ms)[max(0, int(len(per_frame_ms) * 0.95) - 1)]
            if len(per_frame_ms) >= 2
            else per_frame_ms[-1]
        )
        fps = 1000.0 / med if med > 0 else 0.0
        print(
            f"  first-frame   : {first_frame_ms:.0f} ms (one-time backbone + warmup)"
        )
        print(
            f"  steady-state  : median={med:.0f} ms  mean={mean:.0f} ms  "
            f"p95={p95:.0f} ms  ->  {fps:.1f} FPS"
        )
    else:
        print("  NO steady-state frames captured.")

    print("")
    print("Verdict:")
    if not per_frame_ms:
        print("  Could not get multi-frame latency. Re-run with SAM3_FRAMES=5+.")
    else:
        med = statistics.median(per_frame_ms)
        if med <= 200:
            print(
                f"  OK — native SAM 3 video tracking is viable on {device} "
                f"(~{med:.0f} ms/frame). Path B is on the table."
            )
        elif med <= 400:
            print(
                f"  MARGINAL — ~{med:.0f} ms/frame. Usable for ~2 Hz control "
                f"loops, borderline for live UI."
            )
        else:
            print(
                f"  TOO SLOW — ~{med:.0f} ms/frame on {device}. "
                f"Stay on CSRT (Path A), or try SAM3_DEVICE=cpu as a sanity check."
            )
    if first_det_ids:
        print(f"  Identity confirmed: first-frame obj_ids = {first_det_ids}")
    else:
        print(
            f"  Nothing detected for concept {concept!r}. Point the camera "
            f"at a real instance and re-run, or lower conf."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
