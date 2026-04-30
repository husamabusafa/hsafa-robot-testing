"""
Real camera test for SAM 3.1 remote tracker.

Captures from local camera, sends frames to remote server (localhost:8080),
and displays tracking results overlaid on the video.

Usage:
    python sam3_camera_test.py --concept "person"
"""
from __future__ import annotations

import argparse
import json
import time
from io import BytesIO

import cv2
import numpy as np
import requests

URL = "http://localhost:8080"


def draw_result(frame, result):
    """Draw bounding box and info on frame."""
    state = result.get("state", "IDLE")
    bbox = result.get("bbox")
    obj_id = result.get("obj_id")
    score = result.get("score")

    # State color
    color = {
        "IDLE": (128, 128, 128),
        "LOCKING": (0, 255, 255),
        "TRACKING": (0, 255, 0),
        "LOST": (0, 0, 255),
    }.get(state, (128, 128, 128))

    # Draw bbox
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID={obj_id} s={score:.2f}" if obj_id is not None else f"s={score:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw state text
    cv2.putText(frame, f"State: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    latency = result.get("latency_median_ms")
    if latency:
        cv2.putText(frame, f"Latency: {latency:.0f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", default="person", help="Concept to track")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    args = parser.parse_args()

    # Initialize tracker on server
    print(f"Initializing tracker with concept: {args.concept}")
    r = requests.post(f"{URL}/init", json={"concept": args.concept})
    print(f"Init response: {r.json()}")

    # Open camera
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"Failed to open camera {args.camera}")
        return

    print("Camera opened. Press 'q' to quit.")

    frame_count = 0
    last_send_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # Send as fast as possible (server will queue if overloaded)
        now = time.time()
        if now - last_send_time > 0.1:  # ~10 FPS max, but server SAM runs at ~1-3 FPS
            try:
                ok, buf = cv2.imencode(".jpg", frame)
                assert ok
                files = {"image": ("frame.jpg", BytesIO(buf), "image/jpeg")}
                r = requests.post(f"{URL}/frame", files=files, timeout=5.0)
                if r.status_code == 200:
                    result = r.json()
                    frame = draw_result(frame, result)
                else:
                    print(f"Error: {r.status_code} {r.text}")
            except Exception as e:
                print(f"Request error: {e}")
            last_send_time = now

        cv2.imshow("SAM 3.1 Remote Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Stop tracker
    requests.post(f"{URL}/stop")
    print("Tracker stopped.")


if __name__ == "__main__":
    import sys
    main()
