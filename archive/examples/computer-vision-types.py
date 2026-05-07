"""
Computer Vision Playground (Python)
====================================
6 classic CV techniques running live on your webcam.
Perfect for learning CV fundamentals before moving to neural nets on Reachy.

SETUP:
    pip install opencv-python numpy

USAGE:
    python cv_playground.py              # webcam mode (camera 0)
    python cv_playground.py --camera 1   # different camera
    python cv_playground.py --image photo.jpg  # test on an image

CONTROLS (inside the window):
    1-6  = switch technique
    + -  = tune main parameter up/down
    s    = save screenshot
    q    = quit

Techniques:
    1. Edges       - Sobel edge detection (object outlines)
    2. Motion      - Frame differencing (detects movement)
    3. Depth Cue   - Sharpness-based pseudo-depth (no AI)
    4. Obstacles   - Left/Center/Right zones with nav commands (Reachy logic!)
    5. Flow        - Optical flow motion vectors
    6. Color Track - HSV color tracking (follow a colored object)
"""

import cv2
import numpy as np
import argparse
import sys
import time


# ═══════════════════════════════════════════════════════════════
#  CV Technique implementations
# ═══════════════════════════════════════════════════════════════

def technique_edges(frame, params, state):
    """01 - Canny edge detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, params['edge_low'], params['edge_high'])
    
    # Colorize edges (green on dark background)
    out = np.zeros_like(frame)
    out[:, :, 1] = edges  # Green channel
    out[:, :, 2] = edges // 2
    out = cv2.addWeighted(out, 1.0, np.full_like(frame, 10), 1.0, 0)
    
    return out


def technique_motion(frame, params, state):
    """02 - Motion detection via frame differencing."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if state.get('prev_gray') is None or state['prev_gray'].shape != gray.shape:
        state['prev_gray'] = gray
        return (frame * 0.3).astype(np.uint8)
    
    diff = cv2.absdiff(state['prev_gray'], gray)
    _, motion_mask = cv2.threshold(diff, params['motion_thresh'], 255, cv2.THRESH_BINARY)
    
    # Dilate to fill in moving regions
    motion_mask = cv2.dilate(motion_mask, None, iterations=2)
    
    # Dim the original frame
    out = (frame * 0.25).astype(np.uint8)
    
    # Highlight motion in magenta
    motion_colored = np.zeros_like(frame)
    motion_colored[:, :, 0] = motion_mask  # Blue
    motion_colored[:, :, 2] = motion_mask  # Red
    out = cv2.add(out, motion_colored)
    
    state['prev_gray'] = gray
    return out


def technique_depth(frame, params, state):
    """03 - Pseudo-depth from sharpness (focus = close, blur = far)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    block = 16
    
    # Compute Laplacian variance per block (sharpness indicator)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_sq = lap * lap
    
    # Box filter = average sharpness in neighborhood
    sharpness = cv2.boxFilter(lap_sq, -1, (block, block))
    
    # Normalize to 0-255
    sharpness = cv2.normalize(sharpness, None, 0, 255, cv2.NORM_MINMAX)
    sharpness = sharpness.astype(np.uint8)
    
    # Apply turbo colormap (hot = close, cold = far)
    depth_color = cv2.applyColorMap(sharpness, cv2.COLORMAP_TURBO)
    return depth_color


def technique_obstacles(frame, params, state):
    """04 - Obstacle zones (Reachy navigation logic!)."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use brightness as depth proxy for this demo
    # In a real robot, you'd use actual depth here (MiDaS output)
    zone_w = w // 3
    zones = {
        'LEFT':   gray[:, :zone_w],
        'CENTER': gray[:, zone_w:zone_w*2],
        'RIGHT':  gray[:, zone_w*2:],
    }
    
    means = {name: np.mean(z) for name, z in zones.items()}
    threshold = params['obstacle_thresh']
    blocked = {name: v > threshold for name, v in means.items()}
    
    # Build colored output
    out = frame.copy()
    for i, (name, is_blocked) in enumerate(blocked.items()):
        x1 = i * zone_w
        x2 = (i + 1) * zone_w if i < 2 else w
        
        overlay = out[:, x1:x2].copy()
        if is_blocked:
            # Red tint for blocked zones
            overlay[:, :, 2] = np.minimum(255, overlay[:, :, 2].astype(int) + 80)
            overlay[:, :, 0] = overlay[:, :, 0] * 0.4
            overlay[:, :, 1] = overlay[:, :, 1] * 0.4
        else:
            # Green tint for clear zones
            overlay[:, :, 1] = np.minimum(255, overlay[:, :, 1].astype(int) + 60)
            overlay[:, :, 0] = overlay[:, :, 0] * 0.5
            overlay[:, :, 2] = overlay[:, :, 2] * 0.4
        out[:, x1:x2] = overlay
    
    # Draw zone dividers
    cv2.line(out, (zone_w, 0), (zone_w, h), (255, 255, 255), 2)
    cv2.line(out, (zone_w * 2, 0), (zone_w * 2, h), (255, 255, 255), 2)
    
    # Draw zone labels
    for i, (name, is_blocked) in enumerate(blocked.items()):
        x_center = i * zone_w + zone_w // 2
        status = "BLOCKED" if is_blocked else "CLEAR"
        color = (60, 60, 255) if is_blocked else (100, 255, 100)
        
        # Background pill
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
        cv2.rectangle(out, (x_center - tw//2 - 8, 10),
                      (x_center + tw//2 + 8, 40), (0, 0, 0), -1)
        cv2.putText(out, name, (x_center - tw//2, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)
        
        (sw, sh), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x_center - sw//2 - 6, 46),
                      (x_center + sw//2 + 6, 68), (0, 0, 0), -1)
        cv2.putText(out, status, (x_center - sw//2, 63),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Decide navigation command (same logic Reachy would use)
    if blocked['CENTER']:
        if not blocked['LEFT']:
            cmd, color = "<- TURN LEFT", (0, 200, 255)
        elif not blocked['RIGHT']:
            cmd, color = "TURN RIGHT ->", (0, 200, 255)
        else:
            cmd, color = "STOP !!!", (60, 60, 255)
    else:
        cmd, color = "GO FORWARD", (100, 255, 100)
    
    # Command banner at bottom
    (cw, ch), _ = cv2.getTextSize(cmd, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
    cv2.rectangle(out, (w//2 - cw//2 - 15, h - 50),
                  (w//2 + cw//2 + 15, h - 15), (0, 0, 0), -1)
    cv2.putText(out, cmd, (w//2 - cw//2, h - 25),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
    
    return out


def technique_flow(frame, params, state):
    """05 - Dense optical flow (Farneback)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if state.get('prev_gray_flow') is None or state['prev_gray_flow'].shape != gray.shape:
        state['prev_gray_flow'] = gray
        return (frame * 0.3).astype(np.uint8)
    
    flow = cv2.calcOpticalFlowFarneback(
        state['prev_gray_flow'], gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0
    )
    
    # Dim background
    out = (frame * 0.3).astype(np.uint8)
    
    # Draw arrows on a grid
    step = params['flow_step']
    h, w = gray.shape
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            fx, fy = flow[y, x]
            mag = np.sqrt(fx * fx + fy * fy)
            if mag > 1.0:
                intensity = min(255, int(mag * 40))
                # Arrow color scales with motion magnitude
                color = (157, 255, intensity)
                end_x = int(x + fx * 3)
                end_y = int(y + fy * 3)
                cv2.arrowedLine(out, (x, y), (end_x, end_y),
                                color, 1, tipLength=0.4)
    
    state['prev_gray_flow'] = gray
    return out


def technique_color(frame, params, state):
    """06 - Color tracking in HSV space."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    target_h = params['color_hue']
    h_range = params['color_range']
    
    # Handle hue wraparound
    if target_h - h_range < 0:
        lower1 = np.array([0, 80, 60])
        upper1 = np.array([target_h + h_range, 255, 255])
        lower2 = np.array([180 + (target_h - h_range), 80, 60])
        upper2 = np.array([180, 255, 255])
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower1, upper1),
            cv2.inRange(hsv, lower2, upper2)
        )
    elif target_h + h_range > 180:
        lower1 = np.array([target_h - h_range, 80, 60])
        upper1 = np.array([180, 255, 255])
        lower2 = np.array([0, 80, 60])
        upper2 = np.array([(target_h + h_range) - 180, 255, 255])
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower1, upper1),
            cv2.inRange(hsv, lower2, upper2)
        )
    else:
        lower = np.array([target_h - h_range, 80, 60])
        upper = np.array([target_h + h_range, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    
    # Clean up with morphology
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=3)
    
    # Build output: color where matched, gray elsewhere
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor((gray * 0.3).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    color_part = cv2.bitwise_and(frame, frame, mask=mask)
    out = cv2.add(gray_bgr, color_part)
    
    # Find and mark the centroid of the tracked blob
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 300:
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Draw crosshair
                cv2.circle(out, (cx, cy), 25, (0, 255, 255), 2)
                cv2.line(out, (cx - 35, cy), (cx - 15, cy), (0, 255, 255), 2)
                cv2.line(out, (cx + 15, cy), (cx + 35, cy), (0, 255, 255), 2)
                cv2.line(out, (cx, cy - 35), (cx, cy - 15), (0, 255, 255), 2)
                cv2.line(out, (cx, cy + 15), (cx, cy + 35), (0, 255, 255), 2)
                
                label = f"TARGET @ ({cx},{cy})"
                cv2.putText(out, label, (cx + 30, cy - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return out


# ═══════════════════════════════════════════════════════════════
#  Main app
# ═══════════════════════════════════════════════════════════════

TECHNIQUES = {
    '1': {
        'name': 'EDGES',
        'func': technique_edges,
        'param_key': 'edge_low',
        'param_label': 'Low threshold',
        'param_min': 10, 'param_max': 200, 'param_step': 10,
    },
    '2': {
        'name': 'MOTION',
        'func': technique_motion,
        'param_key': 'motion_thresh',
        'param_label': 'Sensitivity',
        'param_min': 5, 'param_max': 80, 'param_step': 5,
    },
    '3': {
        'name': 'DEPTH CUE',
        'func': technique_depth,
        'param_key': 'depth_block',
        'param_label': '(no tunable)',
        'param_min': 0, 'param_max': 0, 'param_step': 0,
    },
    '4': {
        'name': 'OBSTACLES',
        'func': technique_obstacles,
        'param_key': 'obstacle_thresh',
        'param_label': 'Threshold',
        'param_min': 50, 'param_max': 220, 'param_step': 10,
    },
    '5': {
        'name': 'OPTICAL FLOW',
        'func': technique_flow,
        'param_key': 'flow_step',
        'param_label': 'Arrow density',
        'param_min': 8, 'param_max': 40, 'param_step': 4,
    },
    '6': {
        'name': 'COLOR TRACK',
        'func': technique_color,
        'param_key': 'color_hue',
        'param_label': 'Target hue',
        'param_min': 0, 'param_max': 180, 'param_step': 5,
    },
}


def add_hud(out, current_tech, params, fps, inference_ms):
    """Overlay FPS / mode info on the frame."""
    h, w = out.shape[:2]
    
    # Top-left info strip
    info = f"[{current_tech}] {TECHNIQUES[current_tech]['name']}"
    cv2.rectangle(out, (0, 0), (300, 28), (0, 0, 0), -1)
    cv2.putText(out, info, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 157), 1)
    
    # Bottom-left stats
    stats = f"FPS:{fps:5.1f}  MS:{inference_ms:5.1f}"
    cv2.rectangle(out, (0, h - 25), (240, h), (0, 0, 0), -1)
    cv2.putText(out, stats, (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    # Top-right param display
    tech = TECHNIQUES[current_tech]
    if tech['param_step'] > 0:
        val = params.get(tech['param_key'], 0)
        param_info = f"{tech['param_label']}: {val}  (+/-)"
        (tw, _), _ = cv2.getTextSize(param_info, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (w - tw - 20, 0), (w, 25), (0, 0, 0), -1)
        cv2.putText(out, param_info, (w - tw - 10, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 204, 0), 1)
    
    # Controls hint
    hint = "1-6:mode  +/-:param  s:save  q:quit"
    (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.rectangle(out, (w - hw - 20, h - 22), (w, h), (0, 0, 0), -1)
    cv2.putText(out, hint, (w - hw - 10, h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 140), 1)
    
    return out


def main():
    parser = argparse.ArgumentParser(description='CV Playground')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--image', type=str, help='Test on a single image')
    args = parser.parse_args()
    
    print("=" * 55)
    print("  COMPUTER VISION PLAYGROUND")
    print("=" * 55)
    print()
    print("Techniques:")
    for k, v in TECHNIQUES.items():
        print(f"  {k}. {v['name']}")
    print()
    print("Controls:")
    print("  1-6 = switch technique")
    print("  + - = tune parameter")
    print("  s   = save screenshot")
    print("  q   = quit")
    print()
    
    # Default parameters
    params = {
        'edge_low': 50,
        'edge_high': 150,
        'motion_thresh': 25,
        'depth_block': 16,
        'obstacle_thresh': 130,
        'flow_step': 16,
        'color_hue': 0,
        'color_range': 20,
    }
    
    state = {}  # For persistent data between frames (prev_gray, etc.)
    current_tech = '1'
    
    # Image mode
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"ERROR: Could not load {args.image}")
            sys.exit(1)
        
        # Resize if too big
        h, w = frame.shape[:2]
        if w > 900:
            scale = 900 / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        
        print("Showing image. Press 1-6 to switch technique, q to quit.")
        
        while True:
            out = TECHNIQUES[current_tech]['func'](frame, params, state)
            combined = np.hstack([frame, out])
            combined = add_hud(combined, current_tech, params, 0, 0)
            cv2.imshow('CV Playground - Left: Input | Right: Output', combined)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif chr(key) in TECHNIQUES:
                current_tech = chr(key)
                state = {}
                print(f"Switched to: {TECHNIQUES[current_tech]['name']}")
            elif key == ord('+') or key == ord('='):
                tech = TECHNIQUES[current_tech]
                if tech['param_step'] > 0:
                    params[tech['param_key']] = min(tech['param_max'],
                        params[tech['param_key']] + tech['param_step'])
            elif key == ord('-'):
                tech = TECHNIQUES[current_tech]
                if tech['param_step'] > 0:
                    params[tech['param_key']] = max(tech['param_min'],
                        params[tech['param_key']] - tech['param_step'])
            elif key == ord('s'):
                fn = f"cv_playground_{int(time.time())}.png"
                cv2.imwrite(fn, combined)
                print(f"Saved: {fn}")
        
        cv2.destroyAllWindows()
        return
    
    # Webcam mode
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {args.camera}")
        print(f"  Try: python {sys.argv[0]} --camera 1")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Camera ready. Starting...\n")
    
    fps_hist = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera feed lost.")
            break
        
        # Mirror for natural interaction (like a mirror)
        frame = cv2.flip(frame, 1)
        
        t0 = time.time()
        out = TECHNIQUES[current_tech]['func'](frame, params, state)
        t1 = time.time()
        
        inference_ms = (t1 - t0) * 1000
        fps = 1.0 / max(t1 - t0, 0.001)
        fps_hist.append(fps)
        if len(fps_hist) > 15:
            fps_hist.pop(0)
        avg_fps = sum(fps_hist) / len(fps_hist)
        
        # Side-by-side view
        combined = np.hstack([frame, out])
        combined = add_hud(combined, current_tech, params, avg_fps, inference_ms)
        
        cv2.imshow('CV Playground - Left: Input | Right: Output', combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 255:  # No key pressed
            continue
        elif chr(key) in TECHNIQUES:
            current_tech = chr(key)
            state = {}  # Reset motion history when switching
            print(f"[{current_tech}] {TECHNIQUES[current_tech]['name']}")
        elif key == ord('+') or key == ord('='):
            tech = TECHNIQUES[current_tech]
            if tech['param_step'] > 0:
                params[tech['param_key']] = min(tech['param_max'],
                    params[tech['param_key']] + tech['param_step'])
                print(f"  {tech['param_label']}: {params[tech['param_key']]}")
        elif key == ord('-'):
            tech = TECHNIQUES[current_tech]
            if tech['param_step'] > 0:
                params[tech['param_key']] = max(tech['param_min'],
                    params[tech['param_key']] - tech['param_step'])
                print(f"  {tech['param_label']}: {params[tech['param_key']]}")
        elif key == ord('s'):
            fn = f"cv_playground_{int(time.time())}.png"
            cv2.imwrite(fn, combined)
            print(f"Saved: {fn}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nGoodbye!")


if __name__ == '__main__':
    main()