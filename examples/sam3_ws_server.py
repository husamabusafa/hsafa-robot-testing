"""
SAM 3.1 WebSocket Server — lower latency than HTTP

Receives frames via WebSocket, runs SAM 3.1 tracking, returns results.

Client protocol:
1. Send JSON: {"action": "init", "concept": "person"}
2. Send binary: JPEG frame bytes
3. Receive JSON: {"state": "...", "bbox": [...], "latency_ms": ...}
"""
from __future__ import annotations

import io
import os
import sys
import time
import traceback
import base64

import cv2
import numpy as np
from websocket_server import WebsocketServer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from sam3_native_tracker import Sam3NativeFollower

# Config
WEIGHTS = os.getenv("SAM3_WEIGHTS", "/opt/sam3/checkpoints/sam3.pt")
IMGSZ = int(os.getenv("SAM3_IMGSZ", "320"))
CONF = float(os.getenv("SAM3_CONF", "0.25"))
HOST = os.getenv("SAM3_HOST", "0.0.0.0")
PORT = int(os.getenv("SAM3_PORT_WS", "9000"))

# Tracker
_tracker: Sam3NativeFollower | None = None


def get_tracker() -> Sam3NativeFollower:
    global _tracker
    if _tracker is None:
        if not os.path.isfile(WEIGHTS):
            raise RuntimeError(f"SAM3 weights not found: {WEIGHTS}")
        print(f"[ws-server] Loading SAM3 weights: {WEIGHTS}")
        _tracker = Sam3NativeFollower(
            weights_path=WEIGHTS,
            imgsz=IMGSZ,
            conf=CONF,
            max_lost_frames=8,
            debug=False,
        )
        print("[ws-server] Tracker ready.")
    return _tracker


def on_new_client(client, server):
    print(f"[ws-server] Client connected: {client['id']}")


def on_client_left(client, server):
    print(f"[ws-server] Client disconnected: {client['id']}")


def on_message(client, server, message):
    try:
        # Try JSON first (init command)
        try:
            data = eval(message)  # Simple JSON parse
            if isinstance(data, dict) and data.get("action") == "init":
                concept = data.get("concept", "").strip()
                if concept:
                    tracker = get_tracker()
                    tracker.stop_following()
                    tracker.start_following(concept)
                    server.send_message(client, '{"ok": true, "action": "init"}')
                    print(f"[ws-server] Initialized with concept: {concept}")
                return
        except:
            pass  # Not JSON, assume binary frame

        # Binary frame data
        nparr = np.frombuffer(message, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            server.send_message(client, '{"ok": false, "error": "decode failed"}')
            return

        tracker = get_tracker()
        tracker.push_frame(frame)
        time.sleep(0.02)  # Small yield for worker

        state, bbox, age = tracker.get_current_bbox()
        stats = tracker.get_stats()

        result = {
            "ok": True,
            "state": state.name,
            "bbox": list(bbox) if bbox else None,
            "obj_id": stats.get("obj_id"),
            "latency_ms": stats.get("median_ms"),
            "age_s": age,
        }
        server.send_message(client, str(result))
    except Exception as e:
        traceback.print_exc()
        server.send_message(client, f'{{"ok": false, "error": "{str(e)}"}}')


if __name__ == "__main__":
    print(f"[ws-server] Starting WebSocket server on {HOST}:{PORT}")
    server = WebsocketServer(host=HOST, port=PORT)
    server.set_fn_new_client(on_new_client)
    server.set_fn_client_left(on_client_left)
    server.set_fn_message_received(on_message)
    
    # Pre-warm tracker
    try:
        get_tracker()
    except Exception as e:
        print(f"[ws-server] Warm-up failed: {e}")
    
    server.run_forever()
