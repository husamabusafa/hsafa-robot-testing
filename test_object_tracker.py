#!/usr/bin/env python3
"""
test_object_tracker.py - Simple object tracker demo using Qwen-VL + Optical Flow.

Features from new-logic.md:
- Qwen-VL via OpenRouter for initial detection
- Lucas-Kanade optical flow on Shi-Tomasi corners
- Forward-backward error filtering
- Color histogram drift detection
- Simple Kalman predictor for occlusion

Usage:
    python test_object_tracker.py [--camera INDEX] [--reachy-camera]
    
    # With Reachy daemon camera:
    python test_object_tracker.py --reachy-camera
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests

# --- Configuration ---------------------------------------------------------

# OpenRouter API (get key from env or .env file)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
QWEN_MODEL = "qwen/qwen2.5-vl-72b-instruct"  # Or other vision model on OpenRouter

# Tracking parameters (from new-logic.md)
MAX_CORNERS = 80
QUALITY_LEVEL = 0.005
MIN_DISTANCE = 5
BLOCK_SIZE = 7
WIN_SIZE = (21, 21)
PYR_LEVELS = 4
FB_ERROR_THRESH = 8.0  # pixels (8px for ~10px/frame motion at 30fps)
MIN_SURVIVAL_RATIO = 0.15  # More tolerant
REPLENISH_THRESH = 0.4

# Multi-scale pyramid for scale handling
SCALE_PYRAMID_LEVELS = 3
SCALE_FACTORS = [1.0, 0.7, 1.4]  # Original, smaller, larger

# Template matching
TEMPLATE_SIZE = 64
TEMPLATE_MATCH_INTERVAL = 5  # frames
TEMPLATE_SEARCH_MARGIN = 40  # pixels around predicted location

# Color histogram (dropped V channel - more lighting robust)
HSV_BINS_H = 16
HSV_BINS_S = 16
HSV_BINS_V = 0  # Not used - only H+S for lighting invariance
HIST_SIM_THRESH = 0.5  # More tolerant

# Kalman
KALMAN_PROCESS_NOISE = 1e-2  # Lower = trust prediction more
KALMAN_MEASURE_NOISE_BASE = 2.0
KALMAN_MEASURE_NOISE_RANGE = (0.5, 10.0)  # (min, max) based on confidence

# Scale estimation
SCALE_SMOOTHING = 0.1  # EMA alpha for scale
MIN_SCALE = 0.3
MAX_SCALE = 3.0

# Occlusion handling
OCCLUSION_MAX_COAST_FRAMES = 60  # ~2 seconds at 30fps, allows recovery attempts
CONFIDENCE_RECOVERY_THRESH = 0.6

# Confidence hysteresis
CONFIDENCE_DEGRADED_ENTER = 0.4
CONFIDENCE_DEGRADED_EXIT = 0.55
CONFIDENCE_HEALTHY_ENTER = 0.7


# --- Camera helpers (same as main.py) -------------------------------------

def open_camera(index: int) -> "cv2.VideoCapture | None":
    """Open a camera on macOS with the AVFoundation backend at 640x480."""
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


def list_cameras(max_index: int = 6) -> None:
    """Probe camera indices 0..max_index-1 and print what works."""
    import platform
    print("Probing cameras...")
    found = 0
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        if not cap.isOpened():
            print(f"  [{i}] (not available)")
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"  [{i}] OK  {w}x{h}")
            found += 1
        else:
            print(f"  [{i}] opened but no frame")
        cap.release()
    if found == 0:
        print("\nNo cameras found. Grant camera permission to your terminal.")


# --- Qwen-VL via OpenRouter ------------------------------------------------

@dataclass
class DetectionResult:
    """Bounding box from Qwen-VL."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0


def encode_frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode BGR frame to JPEG base64 for API."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


def qwen_detect_object(
    frame: np.ndarray,
    description: str,
    api_key: Optional[str] = None,
) -> Optional[DetectionResult]:
    """Call Qwen-VL via OpenRouter to detect object in frame.
    
    Returns bbox or None if not found.
    """
    key = api_key or OPENROUTER_API_KEY
    if not key:
        print("ERROR: No OPENROUTER_API_KEY set")
        return None
    
    base64_image = encode_frame_to_base64(frame)
    
    # Prompt asking for normalized bbox coordinates
    prompt = (
        f"Find the '{description}' in this image. "
        "Return ONLY a JSON object with keys: "
        "'found' (boolean), 'x1', 'y1', 'x2', 'y2' (all floats 0-1 representing "
        "normalized coordinates), and 'confidence' (0-1). "
        "If not found, return {'found': false}. "
        "No markdown, no explanation, just the JSON."
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
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
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
        
        content = data["choices"][0]["message"]["content"]
        # Extract JSON from possible markdown
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last ``` lines
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines)
        
        result = json.loads(content)
        
        if not result.get("found", False):
            print(f"Qwen: Object '{description}' not found")
            return None
            
        # Convert normalized to pixel coords
        h, w = frame.shape[:2]
        x1 = int(result["x1"] * w)
        y1 = int(result["y1"] * h)
        x2 = int(result["x2"] * w)
        y2 = int(result["y2"] * h)
        conf = result.get("confidence", 1.0)
        
        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        print(f"Qwen detected: ({x1},{y1})-({x2},{y2}) conf={conf:.2f}")
        return DetectionResult(x1, y1, x2, y2, conf)
        
    except Exception as e:
        print(f"Qwen API error: {e}")
        return None


# --- Simple Kalman Predictor -----------------------------------------------

class SimpleKalman:
    """2D position + velocity Kalman filter."""
    
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEASURE_NOISE_BASE
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False
    
    def init(self, x: float, y: float):
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.initialized = True
    
    def predict(self) -> Tuple[float, float]:
        if not self.initialized:
            return 0.0, 0.0
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])
    
    def update(self, x: float, y: float, measure_noise: Optional[float] = None):
        """Update with measurement, optionally with dynamic noise."""
        if not self.initialized:
            self.init(x, y)
        else:
            if measure_noise is not None:
                self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measure_noise
            self.kf.correct(np.array([[x], [y]], dtype=np.float32))


# --- Color Histogram Tracker -----------------------------------------------

class ColorHistTracker:
    """HS histogram for drift detection (H+S only for lighting invariance)."""
    
    def __init__(self):
        self.hist_center = None
        self.hist_full = None
        self.initialized = False
        self.weights = [0.7, 0.3]  # Center vs full bbox
    
    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Compute reference histograms from bbox region (center + full)."""
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return
        
        # Full bbox
        roi_full = frame[y1:y2, x1:x2]
        if roi_full.size == 0:
            return
        
        # Center region (50% of bbox, more stable)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        w, h = (x2 - x1) // 4, (y2 - y1) // 4
        xc1, yc1 = max(0, cx - w), max(0, cy - h)
        xc2, yc2 = min(frame.shape[1], cx + w), min(frame.shape[0], cy + h)
        roi_center = frame[yc1:yc2, xc1:xc2]
        
        hsv_full = cv2.cvtColor(roi_full, cv2.COLOR_BGR2HSV)
        hsv_center = cv2.cvtColor(roi_center, cv2.COLOR_BGR2HSV) if roi_center.size > 0 else hsv_full
        
        # Tighter mask: exclude shadows (V<40), desaturated (S<40), highlights (V>240)
        mask_full = cv2.inRange(hsv_full, (0, 40, 40), (180, 255, 240))
        mask_center = cv2.inRange(hsv_center, (0, 40, 40), (180, 255, 240))
        
        # Only H+S channels (drop V for lighting robustness)
        self.hist_full = cv2.calcHist([hsv_full], [0, 1], mask_full,
                                       [HSV_BINS_H, HSV_BINS_S], 
                                       [0, 180, 0, 256])
        self.hist_center = cv2.calcHist([hsv_center], [0, 1], mask_center,
                                        [HSV_BINS_H, HSV_BINS_S],
                                        [0, 180, 0, 256])
        
        cv2.normalize(self.hist_full, self.hist_full, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(self.hist_center, self.hist_center, 0, 1, cv2.NORM_MINMAX)
        self.initialized = True
    
    def compare(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Compare current region to reference histograms."""
        if not self.initialized:
            return 0.0
        
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        roi_full = frame[y1:y2, x1:x2]
        if roi_full.size == 0:
            return 0.0
        
        # Center region
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        w, h = (x2 - x1) // 4, (y2 - y1) // 4
        xc1, yc1 = max(0, cx - w), max(0, cy - h)
        xc2, yc2 = min(frame.shape[1], cx + w), min(frame.shape[0], cy + h)
        roi_center = frame[yc1:yc2, xc1:xc2] if xc2 > xc1 and yc2 > yc1 else roi_full
        
        hsv_full = cv2.cvtColor(roi_full, cv2.COLOR_BGR2HSV)
        hsv_center = cv2.cvtColor(roi_center, cv2.COLOR_BGR2HSV)
        
        mask_full = cv2.inRange(hsv_full, (0, 40, 40), (180, 255, 240))
        mask_center = cv2.inRange(hsv_center, (0, 40, 40), (180, 255, 240))
        
        hist_full = cv2.calcHist([hsv_full], [0, 1], mask_full,
                                  [HSV_BINS_H, HSV_BINS_S],
                                  [0, 180, 0, 256])
        hist_center = cv2.calcHist([hsv_center], [0, 1], mask_center,
                                   [HSV_BINS_H, HSV_BINS_S],
                                   [0, 180, 0, 256])
        
        cv2.normalize(hist_full, hist_full, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_center, hist_center, 0, 1, cv2.NORM_MINMAX)
        
        sim_full = cv2.compareHist(self.hist_full, hist_full, cv2.HISTCMP_CORREL)
        sim_center = cv2.compareHist(self.hist_center, hist_center, cv2.HISTCMP_CORREL)
        
        # Weighted combination (center more important)
        similarity = self.weights[0] * max(0, sim_center) + self.weights[1] * max(0, sim_full)
        return similarity


# --- Optical Flow Tracker (Core from new-logic.md) -----------------------

@dataclass
class TrackState:
    """Current tracking state with scale and occlusion handling."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[float, float]
    points: np.ndarray = field(default_factory=lambda: np.array([]))
    prev_gray: Optional[np.ndarray] = None
    initialized: bool = False
    lost: bool = False
    occluded: bool = False
    
    # Scale tracking
    scale: float = 1.0
    initial_size: Tuple[int, int] = (0, 0)  # w, h at init
    
    # Last known good position (for coasting)
    last_good_center: Optional[Tuple[float, float]] = None
    
    # Metrics
    survival_ratio: float = 1.0
    hist_similarity: float = 1.0
    template_score: float = 1.0
    confidence: float = 1.0
    
    # Frame counters
    frames_since_update: int = 0
    frames_lost: int = 0


class TemplateMatcher:
    """NCC template matching for drift verification (from new-logic.md §4.6)."""
    
    def __init__(self, template_size: int = TEMPLATE_SIZE):
        self.template = None
        self.template_size = template_size
        self.initialized = False
        self.match_frame_count = 0  # Separate counter for matching intervals
        self.update_frame_count = 0  # Separate counter for updates
        self.last_real_score = 1.0  # Cache last real score
    
    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Extract template from center of bbox."""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        w, h = (x2 - x1), (y2 - y1)
        
        # Extract proportional template (e.g., 50% of bbox)
        tw, th = min(self.template_size, w // 2), min(self.template_size, h // 2)
        tx1 = max(0, cx - tw // 2)
        ty1 = max(0, cy - th // 2)
        tx2 = min(frame.shape[1], tx1 + tw)
        ty2 = min(frame.shape[0], ty1 + th)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.template = gray[ty1:ty2, tx1:tx2].copy()
        self.initialized = True
        self.match_frame_count = 0
        self.update_frame_count = 1  # Just updated
        print(f"Template extracted: {self.template.shape[1]}x{self.template.shape[0]}")
    
    def match(self, frame: np.ndarray, predicted_bbox: Tuple[int, int, int, int]) -> Tuple[float, Tuple[int, int]]:
        """Match template in search window around prediction. Returns (score, offset)."""
        if not self.initialized or self.template is None:
            return 1.0, (0, 0)
        
        self.match_frame_count += 1
        if self.match_frame_count % TEMPLATE_MATCH_INTERVAL != 0:
            # Return cached score on skip frames
            return self.last_real_score, (0, 0)
        
        x1, y1, x2, y2 = predicted_bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        th, tw = self.template.shape[:2]
        
        # Define search region
        sx1 = max(0, cx - tw // 2 - TEMPLATE_SEARCH_MARGIN)
        sy1 = max(0, cy - th // 2 - TEMPLATE_SEARCH_MARGIN)
        sx2 = min(gray.shape[1], cx + tw // 2 + TEMPLATE_SEARCH_MARGIN)
        sy2 = min(gray.shape[0], cy + th // 2 + TEMPLATE_SEARCH_MARGIN)
        
        search_roi = gray[sy1:sy2, sx1:sx2]
        if search_roi.shape[0] < th or search_roi.shape[1] < tw:
            self.last_real_score = 0.0
            return 0.0, (0, 0)
        
        # NCC matching
        result = cv2.matchTemplate(search_roi, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        # Convert match location to offset from predicted center
        match_x = sx1 + max_loc[0] + tw // 2
        match_y = sy1 + max_loc[1] + th // 2
        offset_x = match_x - cx
        offset_y = match_y - cy
        
        self.last_real_score = max(0, max_val)
        return self.last_real_score, (offset_x, offset_y)
    
    def update_template(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Update template (call when tracking is healthy)."""
        self.update_frame_count += 1
        if self.update_frame_count % (TEMPLATE_MATCH_INTERVAL * 3) == 0:
            self.init(frame, bbox)


class OpticalFlowTracker:
    """Advanced LK tracker with multi-scale, template matching, and occlusion handling."""
    
    def __init__(self):
        self.state = TrackState(bbox=(0, 0, 0, 0), center=(0, 0))
        self.kalman = SimpleKalman()
        self.color_hist = ColorHistTracker()
        self.template_matcher = TemplateMatcher()
        self.initialized = False
        
        # Scale tracking
        self.current_scale = 1.0
        self.initial_diag = 0.0
        
    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Initialize tracker with first frame and bbox."""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        self.initial_diag = math.sqrt(w*w + h*h)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Single-scale feature detection on original frame (fixed ghost point bug)
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        roi_x1 = max(0, x1 + margin_x)
        roi_y1 = max(0, y1 + margin_y)
        roi_x2 = min(gray.shape[1], x2 - margin_x)
        roi_y2 = min(gray.shape[0], y2 - margin_y)
        
        points = None
        if roi_x2 > roi_x1 and roi_y2 > roi_y1:
            roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
            corners = cv2.goodFeaturesToTrack(
                roi, MAX_CORNERS, 
                QUALITY_LEVEL * 0.5, MIN_DISTANCE, blockSize=BLOCK_SIZE
            )
            if corners is not None:
                corners[:, 0, 0] += roi_x1
                corners[:, 0, 1] += roi_y1
                points = corners
        
        if points is None or len(points) < 5:
            # Fallback: center point
            points = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
        
        self.state = TrackState(
            bbox=bbox,
            center=(cx, cy),
            points=points,
            prev_gray=gray.copy(),
            initialized=True,
            lost=False,
            scale=1.0,
            initial_size=(w, h),
            last_good_center=(cx, cy),
        )
        
        self.kalman.init(cx, cy)
        self.color_hist.init(frame, bbox)
        self.template_matcher.init(frame, bbox)
        self.initialized = True
        
        print(f"Tracker initialized with {len(points)} points")
    
    def _estimate_scale_from_points(self, old_pts: np.ndarray, new_pts: np.ndarray) -> float:
        """Estimate scale change from pairwise point distances."""
        if len(old_pts) < 4 or len(new_pts) < 4:
            return 1.0
        
        n = min(len(old_pts), len(new_pts))
        if n < 4:
            return 1.0
        
        # Use random subset for efficiency
        indices = np.random.choice(n, min(20, n), replace=False)
        
        scale_changes = []
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                old_dist = np.linalg.norm(old_pts[indices[i]] - old_pts[indices[j]])
                new_dist = np.linalg.norm(new_pts[indices[i]] - new_pts[indices[j]])
                if old_dist > 5:  # Avoid division by tiny numbers
                    scale_changes.append(new_dist / old_dist)
        
        if len(scale_changes) == 0:
            return 1.0
        
        # Use median for robustness, then smooth
        median_scale = np.median(scale_changes)
        # EMA smoothing
        self.current_scale = (1 - SCALE_SMOOTHING) * self.current_scale + SCALE_SMOOTHING * median_scale
        # Clamp
        return np.clip(self.current_scale, MIN_SCALE, MAX_SCALE)
    
    def update(self, frame: np.ndarray) -> TrackState:
        """Update tracker with new frame using multi-cue fusion."""
        if not self.initialized:
            return self.state
        
        self.state.frames_since_update += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- Always run optical flow (even when occluded - for recovery) ---
        # Flag to force Kalman if we're in coast mode but still want to check for recovery
        force_kalman = self.state.lost or self.state.occluded
        
        if force_kalman:
            self.state.frames_lost += 1
            if self.state.frames_lost > OCCLUSION_MAX_COAST_FRAMES:
                print(f"LOST: Coasted too long ({self.state.frames_lost} frames)")
                self.state.lost = True
                self.state.occluded = False
                # Even when fully lost, try one more time before giving up
                if self.state.frames_lost > OCCLUSION_MAX_COAST_FRAMES + 30:
                    self.state.prev_gray = gray.copy()
                    return self.state
        
        # --- 1. Optical Flow Tracking ---
        # Replenish early and often (also when occluded - try to find object again)
        if len(self.state.points) < MAX_CORNERS * 0.6:
            self._replenish_points(frame, gray)
        
        # If too few points and we're tracking (not occluded), try to recover by detecting more
        if len(self.state.points) < 6 and not force_kalman:
            print(f"Too few points ({len(self.state.points)}), entering occlusion mode...")
            self.state.occluded = True
            self.state.prev_gray = gray.copy()
            return self.state
        
        # If occluded and have some points, try tracking anyway (might recover)
        if len(self.state.points) < 6:
            self.state.prev_gray = gray.copy()
            return self.state
        
        # Forward LK
        next_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            self.state.prev_gray, gray, self.state.points, None,
            winSize=WIN_SIZE, maxLevel=PYR_LEVELS
        )
        
        if next_pts is None:
            self.state.occluded = True
            # Coast on Kalman
            cx, cy = self.kalman.predict()
            init_w, init_h = self.state.initial_size
            w, h = int(init_w * self.current_scale), int(init_h * self.current_scale)
            self.state.center = (cx, cy)
            self.state.bbox = (int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2))
            self.state.confidence = max(0.1, 0.5 - self.state.frames_lost * 0.02)
            self.state.prev_gray = gray.copy()
            return self.state
        
        # Backward LK for FB check
        back_pts, status_back, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.state.prev_gray, next_pts, None,
            winSize=WIN_SIZE, maxLevel=PYR_LEVELS
        )
        
        # Guard against back_pts being None
        if back_pts is None:
            fb_status = np.zeros(len(next_pts), dtype=bool)
        else:
            fb_errors = np.linalg.norm(self.state.points - back_pts, axis=2).reshape(-1)
            fb_status = fb_errors < FB_ERROR_THRESH
        
        # Combined status: forward succeeded AND backward consistent
        status = status_fwd.reshape(-1).astype(bool) & fb_status
        
        good_new = next_pts[status].reshape(-1, 2)
        good_old = self.state.points[status].reshape(-1, 2)
        
        survival_ratio = len(good_new) / max(len(self.state.points), 1)
        self.state.survival_ratio = survival_ratio
        
        # --- 2. RANSAC Affine Transform (handles rotation!) ---
        affine_ok = False
        M = None
        inliers = None
        
        if len(good_new) >= 6:
            M, inliers = cv2.estimateAffinePartial2D(
                good_old.astype(np.float32), 
                good_new.astype(np.float32),
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0
            )
            if M is not None and inliers is not None:
                # M = [[a, -b, tx], [b, a, ty]] where a = s*cos(θ), b = s*sin(θ)
                # CRITICAL: tx/ty are translation OFFSETS, not absolute positions!
                # Apply transform to OLD CENTER to get NEW CENTER
                old_cx, old_cy = self.state.center
                pts = np.array([[[old_cx, old_cy]]], dtype=np.float32)
                new_pts = cv2.transform(pts, M)
                cx = float(new_pts[0, 0, 0])
                cy = float(new_pts[0, 0, 1])
                
                # Sanity check: reject if center jumped too far
                displacement = math.sqrt((cx - old_cx)**2 + (cy - old_cy)**2)
                max_jump = max(50, 2.0 * self.initial_diag * self.current_scale)
                if displacement > max_jump:
                    affine_ok = False  # Reject bad RANSAC fit, fall back to median
                else:
                    affine_ok = True
                    # Extract scale: sqrt(a² + b²)
                    a, b = M[0, 0], M[0, 1]
                    estimated_scale = math.sqrt(a*a + b*b)
                    # Smooth scale
                    self.current_scale = (1 - SCALE_SMOOTHING) * self.current_scale + SCALE_SMOOTHING * estimated_scale
                    self.current_scale = np.clip(self.current_scale, MIN_SCALE, MAX_SCALE)
        
        if not affine_ok:
            # Fallback to median of points
            if len(good_new) >= 3:
                new_center = np.median(good_new, axis=0)
                cx, cy = float(new_center[0]), float(new_center[1])
                # Fallback scale estimation
                if len(good_new) >= 4:
                    self.current_scale = self._estimate_scale_from_points(good_old, good_new)
            else:
                # Too few points - coast on Kalman
                cx, cy = self.kalman.predict()
                init_w, init_h = self.state.initial_size
                w, h = int(init_w * self.current_scale), int(init_h * self.current_scale)
                self.state.center = (cx, cy)
                self.state.bbox = (int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2))
                self.state.occluded = True
                self.state.confidence = max(0.1, 0.4 - self.state.frames_lost * 0.02)
                self.state.prev_gray = gray.copy()
                return self.state
        
        # --- 3. Apply motion vector outlier rejection (median-based) ---
        if len(good_new) >= 4 and affine_ok:
            # Compute motion vectors
            motion = good_new - good_old
            median_motion = np.median(motion, axis=0)
            # MAD (median absolute deviation)
            mad = np.median(np.abs(motion - median_motion), axis=0) + 1e-6
            # Reject points with motion > 3*MAD from median
            inlier_mask = np.all(np.abs(motion - median_motion) < 3 * mad, axis=1)
            # Combine with RANSAC inliers
            if inliers is not None:
                final_inliers = inliers.reshape(-1).astype(bool) & inlier_mask
            else:
                final_inliers = inlier_mask
            
            good_new_filtered = good_new[final_inliers]
            good_old_filtered = good_old[final_inliers]
            
            if len(good_new_filtered) >= 3:
                good_new = good_new_filtered
                good_old = good_old_filtered
                survival_ratio = len(good_new) / max(len(self.state.points), 1)
        
        # --- 4. Template Matching Verification ---
        init_w, init_h = self.state.initial_size
        pred_w, pred_h = int(init_w * self.current_scale), int(init_h * self.current_scale)
        pred_bbox = (
            int(cx - pred_w/2), int(cy - pred_h/2),
            int(cx + pred_w/2), int(cy + pred_h/2)
        )
        
        template_score, template_offset = self.template_matcher.match(frame, pred_bbox)
        self.state.template_score = template_score
        
        # Fuse optical flow with template if score is good
        if template_score > 0.6:
            alpha = 0.8  # Flow weight
            cx = alpha * cx + (1 - alpha) * (cx + template_offset[0])
            cy = alpha * cy + (1 - alpha) * (cy + template_offset[1])
            self.state.frames_lost = max(0, self.state.frames_lost - 2)  # Strong recovery
        
        # --- 5. Color Histogram Check ---
        new_bbox = (
            int(cx - pred_w/2), int(cy - pred_h/2),
            int(cx + pred_w/2), int(cy + pred_h/2)
        )
        hist_sim = self.color_hist.compare(frame, new_bbox)
        self.state.hist_similarity = hist_sim
        
        # --- 6. Update Kalman with dynamic measurement noise ---
        # High confidence → trust measurement (low noise)
        # Low confidence → trust prediction (high noise)
        flow_conf = min(1.0, survival_ratio * 2)
        combined_conf = 0.5 * flow_conf + 0.25 * template_score + 0.25 * max(0, hist_sim)
        
        noise_min, noise_max = KALMAN_MEASURE_NOISE_RANGE
        measure_noise = noise_max - (noise_max - noise_min) * combined_conf
        self.kalman.update(cx, cy, measure_noise)
        
        # --- 7. Compute confidence with hysteresis logic ---
        conf_smooth = 0.4 * flow_conf + 0.3 * template_score + 0.3 * max(0, hist_sim)
        conf_min = min(flow_conf, template_score, max(0, hist_sim))  # Weakest link
        
        # Penalize if multiple signals are bad
        bad_signals = sum([
            survival_ratio < MIN_SURVIVAL_RATIO,
            template_score < 0.4,
            hist_sim < HIST_SIM_THRESH
        ])
        if bad_signals >= 2:
            conf_smooth *= 0.5
        
        # Apply hysteresis for state transitions
        if self.state.confidence > CONFIDENCE_HEALTHY_ENTER:
            # Was healthy, now check if degrading
            if conf_min < CONFIDENCE_DEGRADED_ENTER:
                self.state.occluded = True
        else:
            # Was degraded, check if recovering
            if conf_smooth > CONFIDENCE_DEGRADED_EXIT and conf_min > 0.35:
                self.state.frames_lost = max(0, self.state.frames_lost - 1)
        
        # Recovery: if we were occluded/lost but now have good flow, recover
        if force_kalman and conf_smooth > 0.5 and bad_signals <= 1:
            if self.state.lost:
                print(f"RECOVERED from lost: confidence={conf_smooth:.2f}")
                self.state.lost = False
            if self.state.occluded:
                print(f"RECOVERED from occluded: confidence={conf_smooth:.2f}")
                self.state.occluded = False
            self.state.frames_lost = 0
        
        # Update state
        self.state.center = (cx, cy)
        self.state.bbox = new_bbox
        self.state.scale = self.current_scale
        self.state.confidence = conf_smooth
        self.state.prev_gray = gray.copy()
        
        # Update last known good position when tracking is healthy
        if conf_smooth > 0.5 and bad_signals == 0 and not self.state.occluded:
            self.state.last_good_center = (cx, cy)
        
        # Reset frames_lost on healthy tracking
        if conf_smooth > 0.65 and bad_signals == 0:
            self.state.frames_lost = 0
        
        # --- 8. Replenish and deduplicate points ---
        if len(good_new) > 0:
            self.state.points = good_new.reshape(-1, 1, 2).astype(np.float32)
        
        # Replenish early
        if len(self.state.points) < MAX_CORNERS * 0.7 or survival_ratio < REPLENISH_THRESH:
            self._replenish_points(frame, gray, new_bbox)
        
        # Update template occasionally on healthy frames
        if conf_smooth > 0.7 and bad_signals == 0:
            self.template_matcher.update_template(frame, new_bbox)
        
        return self.state
    
    def _deduplicate_points(self, points: np.ndarray, min_dist: int = MIN_DISTANCE) -> np.ndarray:
        """Remove duplicate points that are too close to each other."""
        if len(points) == 0:
            return points
        
        pts = points.reshape(-1, 2)
        keep = [0]  # Always keep first point
        
        for i in range(1, len(pts)):
            distances = np.linalg.norm(pts[i] - pts[keep], axis=1)
            if np.all(distances >= min_dist):
                keep.append(i)
        
        return pts[keep].reshape(-1, 1, 2).astype(np.float32)
    
    def _replenish_points(self, frame: np.ndarray, gray: np.ndarray, 
                          bbox: Optional[Tuple[int, int, int, int]] = None):
        """Add new corners with adaptive quality threshold and deduplication."""
        if bbox is None:
            bbox = self.state.bbox
        
        x1, y1, x2, y2 = bbox
        margin = int(min(x2-x1, y2-y1) * 0.15)
        x1 = max(0, x1 + margin)
        y1 = max(0, y1 + margin)
        x2 = min(frame.shape[1], x2 - margin)
        y2 = min(frame.shape[0], y2 - margin)
        
        if x2 <= x1 or y2 <= y1:
            return
        
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return
        
        # Adaptive quality: lower threshold if we really need points
        quality = QUALITY_LEVEL
        needed = MAX_CORNERS - len(self.state.points)
        if needed > 20:
            quality = QUALITY_LEVEL * 0.3
        elif needed > 10:
            quality = QUALITY_LEVEL * 0.5
        
        new_corners = cv2.goodFeaturesToTrack(
            roi, min(needed, 25), quality, MIN_DISTANCE, 
            blockSize=BLOCK_SIZE
        )
        
        if new_corners is not None and len(new_corners) > 0:
            new_corners[:, 0, 0] += x1
            new_corners[:, 0, 1] += y1
            
            # Combine with existing points and deduplicate
            if len(self.state.points) > 0:
                combined = np.vstack([self.state.points, new_corners])
                self.state.points = self._deduplicate_points(combined)
            else:
                self.state.points = self._deduplicate_points(new_corners)
            
            print(f"Replenished: now {len(self.state.points)} points")


# --- Simple UI using OpenCV -----------------------------------------------

class TrackerUI:
    """Simple OpenCV-based UI with text input and buttons."""
    
    def __init__(self, window_name: str = "Object Tracker"):
        self.window_name = window_name
        self.input_text = ""
        self.is_tracking = False
        self.tracker = OpticalFlowTracker()
        self.target_description = ""
        
        # UI state
        self.button_rect = (10, 500, 150, 540)  # x1, y1, x2, y2
        self.input_rect = (170, 500, 620, 540)
        self.help_rect = (10, 545, 620, 580)
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
    
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if button clicked
            bx1, by1, bx2, by2 = self.button_rect
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                self._on_start()
    
    def _on_start(self):
        """Start tracking button pressed."""
        if not self.input_text.strip():
            print("Please enter an object description first")
            return
        
        self.target_description = self.input_text.strip()
        print(f"\n>>> Starting tracking: '{self.target_description}'")
        print(">>> Press 'd' when object is in frame to detect with Qwen-VL")
        self.is_tracking = False  # Will be set after detection
    
    def _draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI overlay on frame."""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Draw button
        bx1, by1, bx2, by2 = self.button_rect
        cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 200, 0), -1)
        cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.putText(display, "START", (bx1 + 20, by2 - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw input box
        ix1, iy1, ix2, iy2 = self.input_rect
        cv2.rectangle(display, (ix1, iy1), (ix2, iy2), (50, 50, 50), -1)
        cv2.rectangle(display, (ix1, iy1), (ix2, iy2), (200, 200, 200), 2)
        
        # Draw input text
        text = self.input_text if self.input_text else "Type object name..."
        color = (200, 200, 200) if not self.input_text else (255, 255, 255)
        cv2.putText(display, text, (ix1 + 10, iy2 - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Draw status/help
        hx1, hy1, hx2, hy2 = self.help_rect
        cv2.rectangle(display, (hx1, hy1), (hx2, hy2), (30, 30, 30), -1)
        
        if self.tracker.initialized:
            if self.tracker.state.lost:
                state_str = f"LOST ({self.tracker.state.frames_lost}f)"
                status_color = (0, 0, 255)
            elif self.tracker.state.occluded:
                state_str = f"OCCLUDED ({self.tracker.state.frames_lost}f)"
                status_color = (0, 165, 255)
            else:
                state_str = f"conf={self.tracker.state.confidence:.2f} pts={len(self.tracker.state.points)}"
                status_color = (0, 255, 0)
            
            status = f"Track: {self.target_description[:15]} | {state_str}"
            
            # Add metrics
            metrics = f"S:{self.tracker.state.scale:.2f} H:{self.tracker.state.hist_similarity:.2f} T:{self.tracker.state.template_score:.2f}"
            cv2.putText(display, metrics, (hx1 + 10, hy1 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        else:
            status = f"Ready: '{self.target_description}' | Press 'd' to detect"
            status_color = (0, 200, 200)
        
        cv2.putText(display, status, (hx1 + 10, hy2 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Draw bbox if tracking
        if self.tracker.initialized:
            x1, y1, x2, y2 = self.tracker.state.bbox
            
            # Color based on state
            if self.tracker.state.lost:
                color = (0, 0, 255)  # Red = lost
            elif self.tracker.state.occluded:
                color = (0, 165, 255)  # Orange = occluded/coasting
            elif self.tracker.state.confidence > 0.7:
                color = (0, 255, 0)  # Green = healthy
            else:
                color = (0, 255, 255)  # Yellow = degraded
            
            thickness = 3 if self.tracker.state.confidence > 0.5 else 2
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            
            # Draw tracked points (only show subset for clarity)
            if len(self.tracker.state.points) > 0:
                pts = self.tracker.state.points.reshape(-1, 2)
                for i, pt in enumerate(pts):
                    if i % 3 == 0:  # Show every 3rd point
                        cv2.circle(display, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
            
            # Draw center with crosshair
            cx, cy = int(self.tracker.state.center[0]), int(self.tracker.state.center[1])
            size = 8
            cv2.line(display, (cx - size, cy), (cx + size, cy), color, 2)
            cv2.line(display, (cx, cy - size), (cx, cy + size), color, 2)
            
            # Draw scale indicator
            scale_text = f"S:{self.tracker.state.scale:.2f}"
            cv2.putText(display, scale_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return display
    
    def run(self, get_frame_func, use_reachy_camera: bool = False):
        """Main loop."""
        print("=" * 60)
        print("Object Tracker Test")
        print("=" * 60)
        print("Controls:")
        print("  Type: Enter object description")
        print("  Click START or press ENTER: Begin detection")
        print("  Press 'd': Detect object with Qwen-VL (when ready)")
        print("  Press 'q' or ESC: Quit")
        print("=" * 60)
        
        while True:
            frame = get_frame_func()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Update tracker if active
            if self.tracker.initialized:
                self.tracker.update(frame)
            
            # Draw UI
            display = self._draw_ui(frame)
            cv2.imshow(self.window_name, display)
            
            # Handle key input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC or q
                break
            elif key == ord('d') and self.target_description and not self.tracker.initialized:
                # Detect with Qwen-VL
                print(f"\nDetecting '{self.target_description}' with Qwen-VL...")
                result = qwen_detect_object(frame, self.target_description)
                if result:
                    print(f"Initializing tracker at bbox: ({result.x1},{result.y1})-({result.x2},{result.y2})")
                    self.tracker.init(frame, (result.x1, result.y1, result.x2, result.y2))
                    self.is_tracking = True
                else:
                    print("Detection failed. Try again.")
            elif key == ord('\r') or key == ord('\n'):  # Enter
                self._on_start()
            elif key == ord('\b') or key == 127:  # Backspace
                self.input_text = self.input_text[:-1]
            elif 32 <= key <= 126:  # Printable ASCII
                self.input_text += chr(key)
        
        cv2.destroyAllWindows()


# --- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test object tracker with Qwen-VL")
    parser.add_argument("--camera", type=int, default=0,
                       help="OpenCV camera index (default: 0)")
    parser.add_argument("--reachy-camera", action="store_true",
                       help="Use Reachy daemon camera instead")
    parser.add_argument("--list-cameras", action="store_true",
                       help="List available cameras and exit")
    parser.add_argument("--api-key", type=str, default=None,
                       help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    args = parser.parse_args()
    
    if args.list_cameras:
        list_cameras()
        return
    
    # Set API key
    global OPENROUTER_API_KEY
    if args.api_key:
        OPENROUTER_API_KEY = args.api_key
    
    if not OPENROUTER_API_KEY:
        print("WARNING: No OPENROUTER_API_KEY set. Detection will not work.")
        print("Set via: export OPENROUTER_API_KEY=your_key")
    
    # Open camera
    cap = None
    reachy = None
    
    if args.reachy_camera:
        print("Using Reachy daemon camera...")
        try:
            from reachy_mini import ReachyMini
            reachy = ReachyMini(automatic_body_yaw=False)
            reachy.connect()
            
            def get_frame():
                return reachy.media.get_frame()
            
            # Wait for first frame
            print("Waiting for camera frame...")
            for _ in range(40):
                frame = get_frame()
                if frame is not None:
                    print(f"Camera ready: {frame.shape[1]}x{frame.shape[0]}")
                    break
                time.sleep(0.1)
            if frame is None:
                print("ERROR: No frame from Reachy camera")
                return
        except Exception as e:
            print(f"ERROR: Could not connect to Reachy: {e}")
            return
    else:
        print(f"Opening camera index {args.camera}...")
        cap = open_camera(args.camera)
        if cap is None:
            print(f"ERROR: Could not open camera {args.camera}")
            print("Run with --list-cameras to see available cameras")
            return
        
        ok, frame = cap.read()
        if not ok:
            print("ERROR: Camera opened but no frames")
            return
        print(f"Camera ready: {frame.shape[1]}x{frame.shape[0]}")
        
        def get_frame():
            ok, frame = cap.read()
            return frame if ok else None
    
    # Run UI
    try:
        ui = TrackerUI()
        ui.run(get_frame, use_reachy_camera=args.reachy_camera)
    finally:
        if cap is not None:
            cap.release()
        if reachy is not None:
            reachy.disconnect()
        print("\nShutdown complete.")


if __name__ == "__main__":
    main()
