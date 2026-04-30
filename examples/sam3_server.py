"""
SAM 3.1 Native Tracker — Server
=================================
HTTP endpoint that accepts image frames and returns SAM 3.1 native
detection + tracking results.  Maintains stateful tracking across calls.

POST /init
    {"concept": "person with red shirt"}
    -> 200  {ok: true}

POST /frame  (multipart/form-data, image=JPEG/PNG bytes)
    -> 200  {
        "state": "TRACKING",
        "bbox": [x1, y1, x2, y2],
        "obj_id": 1,
        "score": 0.92,
        "latency_ms": 340
    }
    -> 200  { "state": "LOST", ... }

GET /status
    -> tracker state + last stats

Run:
    /opt/sam3/bin/python sam3_server.py
"""
from __future__ import annotations

import io
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np
from flask import Flask, jsonify, request

# Add parent so we can import sam3_native_tracker
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from sam3_native_tracker import Sam3NativeFollower, FollowState

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEIGHTS = os.getenv("SAM3_WEIGHTS", "/opt/sam3/checkpoints/sam3.pt")
IMGSZ = int(os.getenv("SAM3_IMGSZ", "448"))
CONF = float(os.getenv("SAM3_CONF", "0.25"))
HOST = os.getenv("SAM3_HOST", "0.0.0.0")
PORT = int(os.getenv("SAM3_PORT", "8080"))

# ---------------------------------------------------------------------------
# Tracker singleton (thread-safe internally)
# ---------------------------------------------------------------------------
app = Flask(__name__)

_tracker: Optional[Sam3NativeFollower] = None


def _get_tracker() -> Sam3NativeFollower:
    global _tracker
    if _tracker is None:
        if not os.path.isfile(WEIGHTS):
            raise RuntimeError(f"SAM3 weights not found: {WEIGHTS}")
        print(f"[server] Loading SAM3 weights: {WEIGHTS}")
        _tracker = Sam3NativeFollower(
            weights_path=WEIGHTS,
            imgsz=IMGSZ,
            conf=CONF,
            max_lost_frames=8,
            debug=os.getenv("SAM3_DEBUG", "").lower() in ("1", "true", "yes"),
        )
        print("[server] Tracker ready.")
    return _tracker


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "sam3-native-tracker-server",
        "weights": WEIGHTS,
        "imgsz": IMGSZ,
        "endpoints": ["/init", "/frame", "/status"],
    })


@app.route("/init", methods=["POST"])
def init_tracker():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        concept = payload.get("concept", "").strip()
        if not concept:
            return jsonify({"ok": False, "error": "missing 'concept'"}), 400

        tracker = _get_tracker()
        tracker.stop_following()
        tracker.start_following(concept)
        return jsonify({"ok": True, "concept": concept})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/frame", methods=["POST"])
def frame():
    try:
        if "image" not in request.files:
            return jsonify({"ok": False, "error": "missing 'image' file"}), 400

        file = request.files["image"]
        img_bytes = file.read()
        if not img_bytes:
            return jsonify({"ok": False, "error": "empty image"}), 400

        # Decode image
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return jsonify({"ok": False, "error": "cv2 decode failed"}), 400

        tracker = _get_tracker()
        tracker.push_frame(frame_bgr)

        # Give worker a moment (SAM 3 is ~1-3 FPS, but server call should
        # return latest result immediately; caller is expected to pace ~300ms).
        time.sleep(0.02)

        state, bbox, age = tracker.get_current_bbox()
        mask = tracker.get_current_mask()
        stats = tracker.get_stats()

        mask_b64: Optional[str] = None
        if mask is not None:
            import base64
            _, buf = cv2.imencode(".png", mask.astype(np.uint8) * 255)
            mask_b64 = base64.b64encode(buf).decode("utf-8")

        return jsonify({
            "ok": True,
            "state": state.name,
            "bbox": list(bbox) if bbox else None,
            "obj_id": stats.get("obj_id"),
            "score": tracker._current_score if hasattr(tracker, "_current_score") else None,
            "latency_median_ms": stats.get("median_ms"),
            "age_s": age,
            "mask_png_b64": mask_b64,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/stop", methods=["POST"])
def stop():
    try:
        tracker = _get_tracker()
        tracker.stop_following()
        return jsonify({"ok": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/status", methods=["GET"])
def status():
    try:
        tracker = _get_tracker()
        state, bbox, age = tracker.get_current_bbox()
        stats = tracker.get_stats()
        return jsonify({
            "state": state.name,
            "bbox": list(bbox) if bbox else None,
            "obj_id": stats.get("obj_id"),
            "latency_median_ms": stats.get("median_ms"),
            "age_s": age,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Pre-warm tracker on startup so first request is faster
    try:
        _get_tracker()
    except Exception as e:
        print(f"[server] Warm-up failed: {e}")
        traceback.print_exc()

    print(f"[server] Starting on http://{HOST}:{PORT}")
    # threaded=True so /frame and /status can overlap safely
    app.run(host=HOST, port=PORT, threaded=True)
