"""
Quick test client for the SAM 3.1 remote tracker server.

Usage:
    python sam3_test_client.py --concept "person" --frames 5

Pushes random synthetic frames to http://localhost:8080 and prints the
JSON tracking response for each frame.
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


def send_init(concept: str):
    r = requests.post(f"{URL}/init", json={"concept": concept})
    print("init ->", r.status_code, r.json())


def send_frame(img_bgr: np.ndarray):
    ok, buf = cv2.imencode(".jpg", img_bgr)
    assert ok
    files = {"image": ("frame.jpg", BytesIO(buf), "image/jpeg")}
    t0 = time.time()
    r = requests.post(f"{URL}/frame", files=files)
    dt = (time.time() - t0) * 1000
    data = r.json()
    print(f"frame -> {r.status_code}  lat={dt:.0f}ms  {json.dumps(data, indent=2)}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", default="person")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    print(f"Testing SAM 3.1 server at {URL}")
    print(f"Concept: {args.concept}")
    send_init(args.concept)

    for i in range(args.frames):
        # Synthetic frame with a bright rectangle so there is *something*
        # in the image for SAM to look at.
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        x1, y1 = 200 + i * 20, 150
        x2, y2 = x1 + 120, y1 + 180
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), -1)
        cv2.putText(img, "test", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        send_frame(img)
        if i < args.frames - 1:
            time.sleep(args.delay)

    # Final status
    r = requests.get(f"{URL}/status")
    print("status ->", r.status_code, r.json())


if __name__ == "__main__":
    main()
