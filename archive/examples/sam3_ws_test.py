"""
WebSocket test client for SAM 3.1 tracker.

Measures latency compared to HTTP.
"""
import time
import numpy as np
import cv2
import websocket

URL = "ws://localhost:9000"

def test_latency():
    print(f"Connecting to {URL}...")
    ws = websocket.create_connection(URL, timeout=10)
    print("Connected")

    # Init
    ws.send('{"action": "init", "concept": "person"}')
    resp = ws.recv()
    print(f"Init: {resp}")

    # Test frames
    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        x, y = 200 + (i*20)%300, 150
        cv2.rectangle(frame, (x, y), (x+100, y+150), (0, 200, 0), -1)
        ok, buf = cv2.imencode(".jpg", frame)
        assert ok

        t0 = time.time()
        ws.send(buf.tobytes())
        resp = ws.recv()
        dt = (time.time() - t0) * 1000
        print(f"Frame {i}: {dt:.0f}ms -> {resp}")

    ws.close()

if __name__ == "__main__":
    test_latency()
