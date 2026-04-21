"""main.py - Reachy Mini runtime (face-tracking + Gemini Live voice/vision).

This is the main long-running process that brings Reachy Mini to life:

  * A background :class:`CascadeTracker` finds you on the camera (YOLOv8-Pose
    + ByteTrack + Kalman + motion fallback).
  * :class:`RobotController` drives the head and body to stare at you.
  * :class:`GeminiLiveSession` connects to Gemini Live so the robot can hear
    you, see you, and talk back through Reachy's built-in speaker.

Camera capture is done directly with OpenCV/AVFoundation (raw BGR at
640×480, aggressive auto-exposure) because the WebRTC-encoded stream from
Reachy's daemon was too dark, too laggy, and had unsteady frame pacing
that broke ByteTrack ID continuity. On macOS multiple processes can open
the camera simultaneously so this coexists with Reachy's daemon.

Audio still goes through Reachy's built-in :class:`MediaManager`
(GStreamer), which handles device selection, channel duplication, and
device-rate resampling for us. Gemini's 24 kHz replies are resampled to
16 kHz float32 mono before ``reachy.media.push_audio_sample``.

Usage
-----

1. Put your Gemini API key in a ``.env`` file at the repo root::

       GEMINI_API_KEY=your_key_here

2. Make sure the Reachy daemon is running **without** ``--no-media`` so
   it owns the audio hardware (camera ownership is not required -- we
   open it directly). ``./scripts/daemon.sh restart`` does the right
   thing.

3. Run::

       ./.venv/bin/python main.py

   Press ``q`` in the preview window or ``Ctrl-C`` to quit.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from reachy_mini import ReachyMini

from hsafa_robot.gemini_live import GeminiLiveSession
from hsafa_robot.robot_control import RobotController, head_pose
from hsafa_robot.tracker import (
    CascadeTracker,
    TIER_COLORS,
    YOLO_CONF,
    YOLO_IMGSZ,
    ensure_pose_model,
    pick_device,
)

log = logging.getLogger("hsafa_robot.main")


DEFAULT_SYSTEM_INSTRUCTION = (
    "You are Hsafa, a small expressive desk robot (Reachy Mini). "
    "Keep replies short, warm, and natural - usually one or two sentences. "
    "You can see the person through your camera; if they show you something, "
    "react to it briefly. Do not narrate your actions; just talk like a "
    "friendly companion. If you don't know what to say, ask a small question."
)


# --- Shared frame buffer ---------------------------------------------------

class LatestFrame:
    """Thread-safe holder for the most recent camera frame + a JPEG snapshot.

    ``get_jpeg()`` is the callable handed to the Gemini session as its
    ``frame_source``; it returns the latest encoded frame or ``None``.
    """

    def __init__(self, jpeg_quality: int = 70) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._jpeg_quality = jpeg_quality

    def set(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
        )
        return buf.tobytes() if ok else None


# --- Camera (direct AVFoundation, same as examples/05_face_follow.py) -----

def open_camera(index: int) -> "cv2.VideoCapture | None":
    """Open a camera on macOS with the AVFoundation backend at 640x480."""
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
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
    print("Probing cameras (AVFoundation)...")
    found = 0
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
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
        print("\nNo cameras produced frames. Most likely cause on macOS:")
        print("  Terminal lacks camera permission.")
        print("  Open System Settings -> Privacy & Security -> Camera")
        print("  and enable it for your terminal app (Terminal / iTerm / VSCode).")
        print("  Then fully QUIT and relaunch the terminal and try again.")


# --- Optional CLAHE auto-brightness ---------------------------------------
# Direct AVFoundation capture at 480p usually has good auto-exposure so CLAHE
# is OFF by default. Enable it with ``--enhance`` if a room is poorly lit.
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def enhance_brightness(frame: np.ndarray) -> np.ndarray:
    """Return a brightened copy of ``frame`` (BGR uint8) using CLAHE on luma."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb[..., 0] = _CLAHE.apply(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


# --- Preview overlay -------------------------------------------------------

def draw_overlay(view: np.ndarray, snap, det_bbox) -> None:
    h, w = view.shape[:2]
    color = TIER_COLORS.get(snap.tier, (200, 200, 200))
    if det_bbox is not None:
        x1, y1, x2, y2, dx, dy = det_bbox
        x1m, x2m = w - x2, w - x1
        dxm = w - dx
        cv2.rectangle(view, (x1m, y1), (x2m, y2), color, 2)
        cv2.circle(view, (dxm, dy), 5, color, -1)
    cv2.line(view, (w // 2, 0), (w // 2, h), (80, 80, 80), 1)
    cv2.line(view, (0, h // 2), (w, h // 2), (80, 80, 80), 1)
    mode = "TALKING" if snap.talking else "idle"
    tid = f"#{snap.track_id}" if snap.track_id is not None else "--"
    cv2.putText(
        view,
        f"{snap.tier} {tid}  {mode}  "
        f"yaw={math.degrees(snap.sent_yaw):+.0f}  "
        f"pitch={math.degrees(snap.sent_pitch):+.0f}  "
        f"body={math.degrees(snap.body_yaw):+.0f}  (q to quit)",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
    )


# --- Main ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0,
                        help="OpenCV camera index (default: 0). Use "
                             "--list-cameras to see which indices work.")
    parser.add_argument("--list-cameras", action="store_true",
                        help="Probe camera indices 0..5 and exit.")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable the debug preview window")
    parser.add_argument("--no-body", action="store_true",
                        help="Do NOT rotate the body")
    parser.add_argument("--no-gemini", action="store_true",
                        help="Run tracking only (no voice)")
    parser.add_argument("--voice", default="Puck",
                        help="Gemini prebuilt voice name (default: Puck)")
    parser.add_argument("--model", default=None,
                        help="Gemini Live model name (overrides GEMINI_MODEL "
                             "env var and the built-in default).")
    parser.add_argument("--video-fps", type=float, default=1.0,
                        help="How many camera frames per second to stream "
                             "to Gemini (default: 1.0)")
    parser.add_argument("--enhance", action="store_true",
                        help="Apply CLAHE auto-brightness to camera frames. "
                             "Usually not needed at 480p with AVFoundation; "
                             "enable if the room is poorly lit.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    # --- Tracker --------------------------------------------------------
    model_path = ensure_pose_model()
    device = pick_device()
    log.info("Loading YOLOv8-Pose on %s (imgsz=%d) ...", device.upper(), YOLO_IMGSZ)
    tracker = CascadeTracker(model_path, device, YOLO_IMGSZ, YOLO_CONF)

    # --- Signal handling ------------------------------------------------
    stop = {"flag": False}
    def _sigint(_sig, _frm):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _sigint)

    # --- Shared state ---------------------------------------------------
    latest = LatestFrame()
    gemini: Optional[GeminiLiveSession] = None
    cap: Optional[cv2.VideoCapture] = None

    # --- Camera (direct OpenCV, coexists with daemon on macOS) ---------
    cap = open_camera(args.camera)
    if cap is None:
        print(f"Could not open camera index {args.camera}.", file=sys.stderr)
        print("Tips:", file=sys.stderr)
        print("  * Run `python main.py --list-cameras` to see which "
              "indices work.", file=sys.stderr)
        print("  * If no cameras are listed, grant camera permission to "
              "your terminal:", file=sys.stderr)
        print("      System Settings -> Privacy & Security -> Camera -> "
              "enable for Terminal / iTerm / VSCode,", file=sys.stderr)
        print("      then fully quit and relaunch the terminal.",
              file=sys.stderr)
        sys.exit(1)

    ok, probe = cap.read()
    if not ok:
        print("Camera opened but returned no frame.", file=sys.stderr)
        sys.exit(1)
    frame_h, frame_w = probe.shape[:2]

    enhance = enhance_brightness if args.enhance else (lambda f: f)
    log.info(
        "Camera ready: %dx%d (enhance=%s)",
        frame_w, frame_h, "CLAHE" if args.enhance else "off",
    )
    tracker.warmup(frame_h, frame_w)
    tracker.start()

    # --- Reachy & control loop -----------------------------------------
    log.info("Opening Reachy ... (Ctrl-C or q to quit)")
    try:
        with ReachyMini(automatic_body_yaw=False) as reachy:
            media = reachy.media
            gemini_audio_ok = (
                media is not None and getattr(media, "audio", None) is not None
            )
            if gemini_audio_ok:
                # Start audio capture + playback pipelines so mic/speaker
                # are available for Gemini.
                media.start_recording()
                media.start_playing()
                log.info(
                    "Reachy audio ready: in_sr=%d (%dch) out_sr=%d (%dch)",
                    media.get_input_audio_samplerate(),
                    media.get_input_channels(),
                    media.get_output_audio_samplerate(),
                    media.get_output_channels(),
                )
            elif not args.no_gemini:
                log.warning(
                    "Reachy MediaManager not initialised (daemon in no_media "
                    "mode?). Gemini voice will be DISABLED. Restart the "
                    "daemon without --no-media to enable voice."
                )
                args.no_gemini = True

            # --- Gemini Live ------------------------------------------------
            if not args.no_gemini:
                if not api_key:
                    log.warning("GEMINI_API_KEY not set - running without voice. "
                                "Put it in a .env file or pass --no-gemini to "
                                "silence this warning.")
                else:
                    kwargs = dict(
                        api_key=api_key,
                        voice_name=args.voice,
                        system_instruction=DEFAULT_SYSTEM_INSTRUCTION,
                        frame_source=latest.get_jpeg,
                        video_fps=args.video_fps,
                        mic_source=media.get_audio_sample,
                        speaker_sink=media.push_audio_sample,
                    )
                    model = args.model or os.environ.get("GEMINI_MODEL")
                    if model:
                        kwargs["model"] = model
                    gemini = GeminiLiveSession(**kwargs)
                    gemini.start()

            is_talking_fn = (gemini.is_speaking.is_set
                             if gemini is not None else (lambda: False))

            reachy.goto_target(head=head_pose(), duration=0.8, body_yaw=0.0)
            time.sleep(0.3)

            controller = RobotController(
                reachy, tracker, is_talking_fn, no_body=args.no_body,
            )

            last_log = 0.0
            while not stop["flag"]:
                ok, frame = cap.read()
                if not ok or frame is None:
                    # Camera momentarily starved; retry rather than tear down.
                    time.sleep(0.005)
                    continue
                if args.enhance:
                    frame = enhance(frame)
                latest.set(frame)

                snap = controller.tick(frame)

                det = tracker.get()
                det_bbox = det.bbox_px if det is not None else None

                # Heartbeat
                now = time.time()
                if now - last_log > 0.75:
                    last_log = now
                    tid = f"#{snap.track_id}" if snap.track_id else "--"
                    mode = "TALK" if snap.talking else "idle"
                    log.info(
                        "tier=%-9s %-4s %-4s err=(%+.2f,%+.2f) "
                        "yaw=%+6.1f pitch=%+6.1f body=%+6.1f",
                        snap.tier, tid, mode,
                        snap.err_x, snap.err_y,
                        math.degrees(snap.sent_yaw),
                        math.degrees(snap.sent_pitch),
                        math.degrees(snap.body_yaw),
                    )

                # Preview
                if not args.no_preview:
                    view = cv2.flip(frame, 1)
                    draw_overlay(view, snap, det_bbox)
                    cv2.imshow("hsafa robot", view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            log.info("Stopping, recentering ...")
            try:
                reachy.goto_target(head=head_pose(), duration=0.6,
                                   body_yaw=0.0, antennas=[0.0, 0.0])
            except Exception as e:
                log.warning("recenter failed: %s", e)
    finally:
        if gemini is not None:
            gemini.stop()
        if cap is not None:
            cap.release()
        # ``tracker.join()`` raises if the thread was never started (e.g. the
        # Reachy daemon rejected us before ``tracker.start()`` was reached).
        if tracker.is_alive():
            tracker.stop()
            tracker.join(timeout=1.0)
        if tracker.infer_count:
            avg_ms = tracker.infer_total_ms / tracker.infer_count
            total = sum(tracker.tier_counts.values())
            log.info("Detector: %d inferences, avg %.1f ms (%.1f FPS)",
                     tracker.infer_count, avg_ms, 1000.0 / avg_ms)
            if total:
                parts = [f"{k}={v}({100*v/total:.0f}%)"
                         for k, v in tracker.tier_counts.items() if v]
                log.info("Tier usage: %s", "  ".join(parts))
        if not args.no_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
