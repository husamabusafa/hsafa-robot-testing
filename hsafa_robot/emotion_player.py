"""emotion_player.py — Download and play Hugging Face emotion clips for Reachy Mini.

Uses the official pollen-robotics/reachy-mini-emotions-library dataset.
Each clip is a JSON with timestamped 4x4 head poses + antenna positions.
"""
from __future__ import annotations

import bisect
import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

log = logging.getLogger("emotion_player")

# Map our simple emotion names → HF dataset clip names (best match).
# Every base name points to the best single variant; numbered variants are
# also accepted directly so the LLM can pick a specific clip.
EMOTION_MAP: Dict[str, str] = {
    "neutral":       "",
    "happy":         "cheerful1",
    "sad":           "sad1",
    "angry":         "furious1",
    "surprised":     "surprised1",
    "love":          "loving1",
    "tired":         "tired1",
    "confused":      "confused1",
    "excited":       "enthusiastic1",
    # --- full HF library ---
    "amazed":        "amazed1",
    "anxiety":       "anxiety1",
    "attentive":     "attentive1",
    "boredom":       "boredom1",
    "calming":       "calming1",
    "come":          "come1",
    "contempt":      "contempt1",
    "curious":       "curious1",
    "dance":         "dance1",
    "disgusted":     "disgusted1",
    "displeased":    "displeased1",
    "downcast":      "downcast1",
    "dying":         "dying1",
    "electric":      "electric1",
    "exhausted":     "exhausted1",
    "fear":          "fear1",
    "frustrated":    "frustrated1",
    "furious":       "furious1",
    "go_away":       "go_away1",
    "grateful":      "grateful1",
    "helpful":       "helpful1",
    "impatient":     "impatient1",
    "indifferent":   "indifferent1",
    "inquiring":     "inquiring1",
    "irritated":     "irritated1",
    "laughing":      "laughing1",
    "lonely":        "lonely1",
    "lost":          "lost1",
    "no":            "no1",
    "oops":          "oops1",
    "proud":         "proud1",
    "rage":          "rage1",
    "relief":        "relief1",
    "reprimand":     "reprimand1",
    "resigned":      "resigned1",
    "scared":        "scared1",
    "serenity":      "serenity1",
    "shy":           "shy1",
    "sleep":         "sleep1",
    "success":       "success1",
    "thoughtful":    "thoughtful1",
    "uncertain":     "uncertain1",
    "uncomfortable": "uncomfortable1",
    "understanding": "understanding1",
    "welcoming":     "welcoming1",
    "yes":           "yes1",
    # numbered variants (passthrough)
    "amazed1": "amazed1",
    "amazed2": "amazed2",
    "anxiety1": "anxiety1",
    "attentive1": "attentive1",
    "attentive2": "attentive2",
    "boredom1": "boredom1",
    "boredom2": "boredom2",
    "calming1": "calming1",
    "cheerful1": "cheerful1",
    "come1": "come1",
    "confused1": "confused1",
    "contempt1": "contempt1",
    "curious1": "curious1",
    "dance1": "dance1",
    "dance2": "dance2",
    "dance3": "dance3",
    "disgusted1": "disgusted1",
    "displeased1": "displeased1",
    "displeased2": "displeased2",
    "downcast1": "downcast1",
    "dying1": "dying1",
    "electric1": "electric1",
    "enthusiastic1": "enthusiastic1",
    "enthusiastic2": "enthusiastic2",
    "exhausted1": "exhausted1",
    "fear1": "fear1",
    "frustrated1": "frustrated1",
    "furious1": "furious1",
    "go_away1": "go_away1",
    "grateful1": "grateful1",
    "helpful1": "helpful1",
    "helpful2": "helpful2",
    "impatient1": "impatient1",
    "impatient2": "impatient2",
    "incomprehensible2": "incomprehensible2",
    "indifferent1": "indifferent1",
    "inquiring1": "inquiring1",
    "inquiring2": "inquiring2",
    "inquiring3": "inquiring3",
    "irritated1": "irritated1",
    "irritated2": "irritated2",
    "laughing1": "laughing1",
    "laughing2": "laughing2",
    "lonely1": "lonely1",
    "lost1": "lost1",
    "loving1": "loving1",
    "no1": "no1",
    "no_excited1": "no_excited1",
    "no_sad1": "no_sad1",
    "oops1": "oops1",
    "oops2": "oops2",
    "proud1": "proud1",
    "proud2": "proud2",
    "proud3": "proud3",
    "rage1": "rage1",
    "relief1": "relief1",
    "relief2": "relief2",
    "reprimand1": "reprimand1",
    "reprimand2": "reprimand2",
    "reprimand3": "reprimand3",
    "resigned1": "resigned1",
    "sad1": "sad1",
    "sad2": "sad2",
    "scared1": "scared1",
    "serenity1": "serenity1",
    "shy1": "shy1",
    "sleep1": "sleep1",
    "success1": "success1",
    "success2": "success2",
    "surprised1": "surprised1",
    "surprised2": "surprised2",
    "thoughtful1": "thoughtful1",
    "thoughtful2": "thoughtful2",
    "tired1": "tired1",
    "uncertain1": "uncertain1",
    "uncomfortable1": "uncomfortable1",
    "understanding1": "understanding1",
    "understanding2": "understanding2",
    "welcoming1": "welcoming1",
    "welcoming2": "welcoming2",
    "yes1": "yes1",
    "yes_sad1": "yes_sad1",
}


def _matrix_to_yaw_pitch(matrix_4x4: List[List[float]]) -> tuple[float, float]:
    """Extract yaw (Z) and pitch (Y) in degrees from a 4x4 homogeneous matrix.

    Uses scipy Rotation with 'ZYX' (yaw-pitch-roll) convention.
    """
    rot = np.array(matrix_4x4, dtype=np.float64)[:3, :3]
    r = SciRotation.from_matrix(rot)
    yaw, pitch, _roll = r.as_euler("ZYX", degrees=True)
    return float(yaw), float(pitch)


def _lerp(a: float, b: float, alpha: float) -> float:
    return a + alpha * (b - a)


class EmotionClip:
    """A single parsed emotion clip ready for playback."""

    def __init__(self, name: str, data: dict, sound_path: Optional[Path] = None) -> None:
        self.name = name
        self.description = data.get("description", "")
        self.timestamps: List[float] = data["time"]
        self.trajectory: List[dict] = data["set_target_data"]
        self.sound_path = sound_path

        # Pre-compute yaw/pitch/body_yaw for every keyframe
        self._poses: List[tuple[float, float, float, float, float]] = []  # yaw, pitch, body_yaw, ant_r, ant_l
        for frame in self.trajectory:
            yaw, pitch = _matrix_to_yaw_pitch(frame["head"])
            ants = frame.get("antennas", [0.0, 0.0])
            body_yaw = math.degrees(frame.get("body_yaw", 0.0))
            self._poses.append((yaw, pitch, body_yaw, float(ants[0]), float(ants[1])))

        self.duration = self.timestamps[-1] if self.timestamps else 0.0

    def evaluate(self, t: float) -> tuple[float, float, float, float, float]:
        """Return (yaw_deg, pitch_deg, body_yaw_deg, ant_r, ant_l) at time t."""
        if t <= 0.0 or not self.timestamps:
            return self._poses[0]
        if t >= self.duration:
            return self._poses[-1]

        idx = bisect.bisect_right(self.timestamps, t)
        idx_prev = max(0, idx - 1)
        idx_next = min(len(self.timestamps) - 1, idx)

        t_prev = self.timestamps[idx_prev]
        t_next = self.timestamps[idx_next]
        if t_next == t_prev:
            alpha = 0.0
        else:
            alpha = (t - t_prev) / (t_next - t_prev)

        p0 = self._poses[idx_prev]
        p1 = self._poses[idx_next]
        return (
            _lerp(p0[0], p1[0], alpha),
            _lerp(p0[1], p1[1], alpha),
            _lerp(p0[2], p1[2], alpha),
            _lerp(p0[3], p1[3], alpha),
            _lerp(p0[4], p1[4], alpha),
        )


class EmotionLibrary:
    """Loads and caches emotion clips from the HF dataset."""

    DATASET_NAME = "pollen-robotics/reachy-mini-emotions-library"

    def __init__(self) -> None:
        self._clips: Dict[str, EmotionClip] = {}
        self._local_path: Optional[str] = None
        self._loaded = False
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if not HF_AVAILABLE:
                log.warning("huggingface_hub not available; skipping emotion library.")
                self._loaded = True
                return

            try:
                log.info("Loading emotion library from HuggingFace...")
                self._local_path = snapshot_download(
                    self.DATASET_NAME,
                    repo_type="dataset",
                    local_files_only=False,
                )
                log.info("Dataset cached at %s", self._local_path)

                data_dir = Path(self._local_path)
                for json_path in data_dir.glob("*.json"):
                    name = json_path.stem
                    with open(json_path, "r") as f:
                        data = json.load(f)
                    sound_path = json_path.with_suffix(".wav")
                    self._clips[name] = EmotionClip(
                        name, data,
                        sound_path=sound_path if sound_path.exists() else None,
                    )

                log.info("Loaded %d emotion clips.", len(self._clips))
            except Exception as e:
                log.error("Failed to load emotion library: %s", e)
            self._loaded = True

    def get(self, emotion_name: str) -> Optional[EmotionClip]:
        """Get a clip by our internal emotion name."""
        self._ensure_loaded()
        clip_name = EMOTION_MAP.get(emotion_name, "")
        if not clip_name:
            return None
        return self._clips.get(clip_name)

    def list_available(self) -> List[str]:
        self._ensure_loaded()
        return sorted(self._clips.keys())


class EmotionClipPlayer:
    """Plays an EmotionClip in a background thread, feeding poses to a head."""

    def __init__(self, head) -> None:
        self.head = head
        self.library = EmotionLibrary()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._current_name: Optional[str] = None

    def play(self, emotion_name: str, duration: Optional[float] = None) -> bool:
        """Start playing an emotion clip. Returns True if clip started."""
        clip = self.library.get(emotion_name)
        if clip is None:
            return False

        self.stop()
        self._stop.clear()
        self._current_name = emotion_name
        self._thread = threading.Thread(
            target=self._playback_loop,
            args=(clip, duration),
            daemon=True,
            name=f"emote-{emotion_name}",
        )
        self._thread.start()
        log.info("Playing emotion clip '%s' (%.2fs)", clip.name, clip.duration)
        return True

    def stop(self) -> None:
        """Stop any running clip."""
        if self._thread and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=1.0)
        self._thread = None
        self._current_name = None

    def _playback_loop(self, clip: EmotionClip, duration: Optional[float]) -> None:
        t0 = time.time()
        play_duration = duration if duration is not None else clip.duration
        dt = 1 / 60.0  # ~60 Hz update for smooth motion

        # Start sound in parallel if available
        sound_thread = None
        if clip.sound_path is not None and clip.sound_path.exists():
            sound_thread = threading.Thread(
                target=self._play_sound,
                args=(clip.sound_path,),
                daemon=True,
                name=f"sound-{clip.name}",
            )
            sound_thread.start()

        while not self._stop.is_set():
            elapsed = time.time() - t0
            if elapsed >= play_duration:
                break

            yaw, pitch, body_yaw, ant_r, ant_l = clip.evaluate(elapsed)
            # Use set_pose (direct set_target) so the robot follows the clip
            # trajectory exactly, without extra goto_target smoothing.
            self.head.set_pose(yaw, pitch, body_yaw, ant_r, ant_l)
            time.sleep(dt)

        if sound_thread is not None:
            sound_thread.join(timeout=0.5)
        log.info("Emotion clip '%s' finished.", clip.name)

    def _play_sound(self, path: Path) -> None:
        """Play a WAV file locally."""
        try:
            import subprocess
            import platform
            if platform.system() == "Darwin":
                subprocess.run(["afplay", str(path)], check=False, timeout=30)
            else:
                subprocess.run(["aplay", str(path)], check=False, timeout=30)
        except Exception as e:
            log.warning("Failed to play sound %s: %s", path, e)
