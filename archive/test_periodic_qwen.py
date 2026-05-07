#!/usr/bin/env python3
"""
test_periodic_qwen.py - Simple periodic Qwen detection with EMA smoothing.

Key idea:
- Run Qwen every 500ms to get fresh bbox
- Smooth between detections using EMA (Exponential Moving Average)
- No optical flow, just smooth interpolation

Usage:
    python test_periodic_qwen.py [--camera INDEX] [--interval 500] [--target "person's face"]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import requests

# --- Configuration ---------------------------------------------------------

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
QWEN_MODEL = "qwen/qwen2.5-vl-72b-instruct"

# Smoothing parameters
EMA_ALPHA = 0.3  # Lower = smoother but more lag (0.3 = good balance)
MIN_CONFIDENCE = 0.3  # Ignore detections below this confidence


# --- Detection Result ------------------------------------------------------

@dataclass
class DetectionResult:
    """Bounding box with confidence."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    timestamp: float


# --- EMA Smoother ----------------------------------------------------------

class EMASmoother:
    """Exponential Moving Average smoother for bbox."""
    
    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = alpha
        self.smooth_bbox: Optional[Tuple[float, float, float, float]] = None
        self.last_detection_time: float = 0
        self.frames_without_detection: int = 0
    
    def update(self, detection: Optional[DetectionResult]) -> Optional[Tuple[float, float, float, float]]:
        """Update with new detection, return smoothed bbox."""
        if detection is None or detection.confidence < MIN_CONFIDENCE:
            self.frames_without_detection += 1
            # If no detection for too long, gradually fade out
            if self.frames_without_detection > 10 and self.smooth_bbox is not None:
                # Decay the bbox (shrink towards center)
                cx = (self.smooth_bbox[0] + self.smooth_bbox[2]) / 2
                cy = (self.smooth_bbox[1] + self.smooth_bbox[3]) / 2
                w = (self.smooth_bbox[2] - self.smooth_bbox[0]) * 0.95
                h = (self.smooth_bbox[3] - self.smooth_bbox[1]) * 0.95
                self.smooth_bbox = (cx - w/2, cy - h/2, cx + w/2, cy + h/2)
            return self.smooth_bbox
        
        self.frames_without_detection = 0
        
        if self.smooth_bbox is None:
            # First detection - initialize
            self.smooth_bbox = (detection.x1, detection.y1, detection.x2, detection.y2)
        else:
            # EMA update
            sx1, sy1, sx2, sy2 = self.smooth_bbox
            self.smooth_bbox = (
                self._ema(sx1, detection.x1),
                self._ema(sy1, detection.y1),
                self._ema(sx2, detection.x2),
                self._ema(sy2, detection.y2),
            )
        
        self.last_detection_time = detection.timestamp
        return self.smooth_bbox
    
    def _ema(self, current: float, target: float) -> float:
        """Exponential moving average: current + alpha * (target - current)."""
        return current + self.alpha * (target - current)
    
    def predict_linear(self, dt: float, vx: float = 0, vy: float = 0) -> Optional[Tuple[float, float, float, float]]:
        """Simple linear prediction for smooth motion between detections."""
        if self.smooth_bbox is None:
            return None
        x1, y1, x2, y2 = self.smooth_bbox
        # Simple velocity-based prediction
        return (x1 + vx * dt, y1 + vy * dt, x2 + vx * dt, y2 + vy * dt)


# --- Qwen API --------------------------------------------------------------

def encode_frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode BGR frame to JPEG base64."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


def qwen_detect(frame: np.ndarray, target: str, api_key: str = "") -> Optional[DetectionResult]:
    """Call Qwen-VL to detect target in frame."""
    key = api_key or OPENROUTER_API_KEY
    if not key:
        print("ERROR: No OPENROUTER_API_KEY set")
        return None
    
    base64_image = encode_frame_to_base64(frame)
    
    prompt = (
        f"Find the '{target}' in this image. "
        "Return ONLY a JSON object with keys: "
        "'found' (boolean), 'x1', 'y1', 'x2', 'y2' (all floats 0-1 normalized), "
        "and 'confidence' (0-1). If not found, return {'found': false}. "
        "No markdown, just the JSON."
    )
    
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": QWEN_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 200,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        
        content = data["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines)
        
        result = json.loads(content)
        
        if not result.get("found", False):
            return None
        
        h, w = frame.shape[:2]
        return DetectionResult(
            x1=result["x1"] * w,
            y1=result["y1"] * h,
            x2=result["x2"] * w,
            y2=result["y2"] * h,
            confidence=result.get("confidence", 1.0),
            timestamp=time.time(),
        )
        
    except Exception as e:
        print(f"Qwen API error: {e}")
        return None


# --- Camera helpers --------------------------------------------------------

def open_camera(index: int) -> Optional[cv2.VideoCapture]:
    """Open camera on macOS with AVFoundation backend."""
    import platform
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


# --- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Periodic Qwen detection with EMA smoothing")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--interval", type=int, default=500, help="Qwen call interval in ms (default: 500)")
    parser.add_argument("--target", type=str, default="person's face", help="Target to detect")
    parser.add_argument("--alpha", type=float, default=EMA_ALPHA, help=f"EMA alpha (default: {EMA_ALPHA})")
    parser.add_argument("--api-key", type=str, default="", help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    args = parser.parse_args()
    
    # Setup
    cap = open_camera(args.camera)
    if cap is None:
        print(f"Failed to open camera {args.camera}")
        return
    
    smoother = EMASmoother(alpha=args.alpha)
    api_key = args.api_key or OPENROUTER_API_KEY
    
    last_qwen_time = 0
    interval_sec = args.interval / 1000.0
    frame_count = 0
    
    print(f"Starting periodic Qwen detection (interval: {args.interval}ms, target: '{args.target}')")
    print("Press 'q' to quit, 'd' to force detection, '+'/'-' to adjust smoothing")
    print(f"EMA alpha: {args.alpha} (lower = smoother)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        frame_count += 1
        
        # Check if it's time to call Qwen
        should_call_qwen = (current_time - last_qwen_time) >= interval_sec
        
        detection: Optional[DetectionResult] = None
        is_qwen_frame = False
        
        if should_call_qwen:
            is_qwen_frame = True
            last_qwen_time = current_time
            detection = qwen_detect(frame, args.target, api_key)
            if detection:
                print(f"Qwen: bbox=({detection.x1:.0f},{detection.y1:.0f})-({detection.x2:.0f},{detection.y2:.0f}) conf={detection.confidence:.2f}")
        
        # Update smoother (works with None detection too)
        smooth_bbox = smoother.update(detection)
        
        # Draw results
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Draw smoothed bbox
        if smooth_bbox is not None:
            x1, y1, x2, y2 = map(int, smooth_bbox)
            color = (0, 255, 0) if not is_qwen_frame else (0, 165, 255)  # Orange on Qwen frame, green otherwise
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(display, (cx, cy), 5, (255, 0, 0), -1)
            
            # Label
            label = f"{args.target}"
            if is_qwen_frame and detection:
                label += f" (Qwen conf: {detection.confidence:.2f})"
            else:
                label += " (smooth)"
            cv2.putText(display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw detection bbox (raw from Qwen) if available
        if detection and is_qwen_frame:
            x1, y1, x2, y2 = int(detection.x1), int(detection.y1), int(detection.x2), int(detection.y2)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red thin line for raw detection
        
        # Status overlay
        status_lines = [
            f"Interval: {args.interval}ms | Alpha: {smoother.alpha:.2f}",
            f"Qwen calls: {frame_count // int(interval_sec * 30 + 1)} | Smoothing: {smoother.frames_without_detection == 0}",
        ]
        if smooth_bbox:
            status_lines.append(f"BBox: ({smooth_bbox[0]:.0f},{smooth_bbox[1]:.0f})-({smooth_bbox[2]:.0f},{smooth_bbox[3]:.0f})")
        
        y_offset = 30
        for line in status_lines:
            cv2.putText(display, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(display, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset += 25
        
        cv2.imshow("Periodic Qwen + EMA Smoothing", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            # Force immediate detection
            last_qwen_time = 0
        elif key == ord('+') or key == ord('='):
            smoother.alpha = min(0.9, smoother.alpha + 0.05)
            print(f"Alpha increased to {smoother.alpha:.2f}")
        elif key == ord('-'):
            smoother.alpha = max(0.05, smoother.alpha - 0.05)
            print(f"Alpha decreased to {smoother.alpha:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
