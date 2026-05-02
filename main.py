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
import asyncio
import base64
import json
import logging
import math
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from reachy_mini import ReachyMini

LOOK_AT_MODEL = "qwen/qwen3-vl-8b-instruct"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
from hsafa_robot.audio_vad import SileroVAD
from hsafa_robot.events import (
    EVT_PERSON_LEFT,
    EVT_VOICE_IDENTIFIED,
    EVT_VOICE_UNSEEN,
    EventBus,
)
from hsafa_robot.face_db import FaceDB, canonicalize_name
from hsafa_robot.face_recognizer import FaceRecognizer
from hsafa_robot.focus import FocusManager, FocusSnapshot
from hsafa_robot.gaze_policy import GazePrior
from hsafa_robot.gemini_live import GeminiLiveSession
from hsafa_robot.gestures import GestureTracker
from hsafa_robot.head_pose import HeadPoseTracker
from hsafa_robot.identity_graph import IdentityGraph
from hsafa_robot.lip_motion import LipMotionTracker
from hsafa_robot.robot_control import RobotController, head_pose
from hsafa_robot.tracker import (
    CascadeTracker,
    TIER_COLORS,
    YOLO_CONF,
    YOLO_IMGSZ,
    ensure_pose_model,
    pick_device,
)
from hsafa_robot.voice_embedder import VoiceEmbedder
from hsafa_robot.voice_identity import VoiceIdentityWorker
from hsafa_robot.world_state import WorldStateHolder

log = logging.getLogger("hsafa_robot.main")


DEFAULT_SYSTEM_INSTRUCTION = (
    "You are Hsafa, a small expressive desk robot (Reachy Mini). "
    "Keep replies short, warm, and natural - usually one or two sentences. "
    "You can see the person through your camera; if they show you something, "
    "react to it briefly. Do not narrate your actions; just talk like a "
    "friendly companion. If you don't know what to say, ask a small question. "
    "\n\n"
    "You can remember people's faces, even when several people are in "
    "view at once. "
    "When someone introduces themselves (e.g. says 'I am Husam', "
    "'my name is X', or 'remember me as Y'), call the `enroll_face` tool "
    "with their name, then confirm naturally in your reply. During "
    "enrollment the closest person in frame is the one who gets saved. "
    "When someone asks who they are, who you are looking at, how many "
    "people you see, or whether you recognize someone (e.g. 'who am "
    "I?', 'who is here?', 'do you know me?'), call `identify_person` "
    "FIRST -- it returns every visible person with their position "
    "(`left`, `center`, `right`). Use positions to be specific "
    "(\"Husam on the left and someone I don't recognize on the right\"). "
    "If someone asks whether a SPECIFIC known person is visible right "
    "now (e.g. 'is Husam here?'), call `find_person` with that name "
    "instead of scanning everyone. "
    "If nobody known is recognized, say so and offer to remember them. "
    "Always prefer calling these tools over guessing from the image."
    "\n\n"
    "You can also tell who is currently speaking by watching mouths. "
    "When the user asks who is talking, who just spoke, or 'is that "
    "X talking?', call `who_is_speaking` and use the result. It may "
    "return the name 'unknown' for a visible person you haven't been "
    "introduced to, or no speaker at all if nobody's mouth is moving. "
    "Only people visible to the camera can be detected this way; an "
    "off-camera voice will come back as 'nobody'."
    "\n\n"
    "You can also control who the robot physically looks at:\n"
    "- `focus_on_person(name)` locks the head and body onto that "
    "known person, even if they move across the frame. Use it when "
    "the user says 'look at me', 'focus on Husam', 'keep your eyes "
    "on him'. It also accepts names that aren't visible right now - "
    "the lock arms immediately and engages the moment that person "
    "steps into frame, so 'when Husam comes in, look at him' works "
    "without waiting. The response includes `visible_now` so you "
    "know whether to say 'got it, watching you' or 'okay, I'll look "
    "for Husam'.\n"
    "- `focus_on_speaker()` makes the robot automatically turn "
    "toward whoever is currently talking. Use it for 'look at "
    "whoever is speaking', 'turn to the person talking'.\n"
    "- `clear_focus()` returns to the default behavior (follow the "
    "closest person). Use it for 'look around normally', 'stop "
    "following him', 'relax'.\n"
    "- `set_gaze_mode(mode, name?)` is the lower-level version: "
    "`mode=\"person\"` locks to `name`, `mode=\"normal\"` uses the "
    "scoring engine (default), `mode=\"speaker\"` biases it toward "
    "the current speaker.\n"
    "When a focus call succeeds briefly describe what you are doing "
    '("okay, watching you" / "following the speaker"); if it fails '
    "because the person is not visible, say so and suggest stepping "
    "into view."
    "\n\n"
    "You can also directly control the robot's head angles: "
    "`set_head_angle(yaw_deg, pitch_deg)` moves to specific degrees "
    "(yaw -60..+60 left/right, pitch -30..+30 up/down). "
    "`look_straight()` resets to center. `look_left`, `look_right`, "
    "`look_up`, `look_down` are convenience presets. "
    "`enable_face_follow()` resumes automatic face tracking; "
    "`disable_face_follow()` freezes the head at its current angle. "
    "The robot starts with face following DISABLED, so you must call "
    "`enable_face_follow()` if the user wants you to track them.\n\n"
    "You can also look at specific objects in view: "
    "`look_at(description)` uses computer vision to find the described "
    "object (e.g. 'the red cup', 'my phone'), marks its location with "
    "a bright box on the camera preview, and moves the head to stare "
    "at it. Use it when the user points something out or asks you to "
    "look at a specific item."
    "\n\n"
    "You can also sense hand gestures (wave, point, thumbs_up, "
    "open_palm, fist). When you notice a gesture mentioned in the "
    "scene, react to it naturally (\"hey, hi!\" on a wave, "
    "\"looking at what you're pointing at\" on a point). You can call "
    "`detect_gestures()` to explicitly ask what gestures are "
    "currently visible if the user asks."
    "\n\n"
    "For any quick \"what's happening right now\" question (who's "
    "here, who's addressing you, what gestures are up, what gaze "
    "mode you're in) you can call `describe_scene()` -- it returns "
    "a compact one-line summary of the live WorldState."
    "\n\n"
    "The robot can also recognise people by their voice (via a "
    "learned speaker embedding) even when they are off-camera. "
    "`describe_scene()` surfaces the most recent recognized voice "
    "under `env.last_heard_voice_name`. If you hear a familiar "
    "voice but the person is not in frame, you can acknowledge "
    "them by name (\"hey Husam, I hear you\") and they can ask you "
    "to look for them. Voice banks are built automatically: the "
    "first few times someone who already has an enrolled face "
    "speaks in front of the camera, their voice signature is "
    "captured silently -- no explicit command needed."
)


# --- Data directories ------------------------------------------------------

FACE_DB_DIR = Path(__file__).resolve().parent / "data" / "faces"
IDENTITY_DIR = Path(__file__).resolve().parent / "data" / "identity"


def build_test_tools() -> list:
    """Simple test tools that work even without face recognition."""
    return [
        genai_types.Tool(
            function_declarations=[
                genai_types.FunctionDeclaration(
                    name="get_current_time",
                    description=(
                        "Return the current local time and date. "
                        "Useful when the user asks 'what time is it' or "
                        "wants a timestamp."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={},
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="ping",
                    description=(
                        "A quick health-check tool. Returns 'pong'. "
                        "Use this when the user says 'ping', "
                        "'are you there', or wants to test responsiveness."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={},
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="get_robot_status",
                    description=(
                        "Return basic robot runtime status. "
                        "Use when the user asks 'what is your status', "
                        "'how are you', or similar diagnostic questions."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={},
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="set_head_angle",
                    description=(
                        "Move the robot's head to a specific yaw and pitch "
                        "angle in degrees. yaw=0 looks straight ahead; "
                        "positive yaw turns left, negative turns right. "
                        "pitch=0 is level; positive looks down, negative "
                        "looks up. Range: yaw -60..+60, pitch -30..+30. "
                        "Use when the user says 'look left', 'look right', "
                        "'look up', 'look down', or gives specific angles. "
                        "This disables automatic face following until "
                        "`enable_face_follow` is called."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={
                            "yaw_deg": genai_types.Schema(
                                type=genai_types.Type.NUMBER,
                                description="Head yaw in degrees. 0 = center, positive = left, negative = right. Range -60..60.",
                            ),
                            "pitch_deg": genai_types.Schema(
                                type=genai_types.Type.NUMBER,
                                description="Head pitch in degrees. 0 = level, positive = down, negative = up. Range -30..30.",
                            ),
                        },
                        required=["yaw_deg", "pitch_deg"],
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="look_straight",
                    description=(
                        "Reset the robot's head to center (yaw=0, pitch=0). "
                        "Use when the user says 'look straight', 'center', "
                        "'reset your head', or 'look forward'."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={},
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="look_left",
                    description=(
                        "Turn the robot's head 30 degrees to the left. "
                        "Use when the user says 'look left'."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={},
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="look_right",
                    description=(
                        "Turn the robot's head 30 degrees to the right. "
                        "Use when the user says 'look right'."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={},
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="look_up",
                    description=(
                        "Tilt the robot's head 15 degrees up. "
                        "Use when the user says 'look up'."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={},
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="look_down",
                    description=(
                        "Tilt the robot's head 15 degrees down. "
                        "Use when the user says 'look down'."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={},
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="enable_face_follow",
                    description=(
                        "Enable automatic face tracking. The robot will "
                        "follow the closest visible person. Use when the user "
                        "says 'follow me', 'track my face', or 'look at people'."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={},
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="disable_face_follow",
                    description=(
                        "Disable automatic face tracking. The head will "
                        "stay at its current angle until moved by another "
                        "command. Use when the user says 'stop following', "
                        "'don't follow me', or 'freeze'."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={},
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="look_at",
                    description=(
                        "Look at a specific object in the camera view. "
                        "Describe the object (e.g. 'the red cup', 'my phone', "
                        "'the book on the table') and the robot will use "
                        "computer vision to locate it, move its head to stare "
                        "directly at it, and draw a bright bounding box on the "
                        "preview showing exactly where it is. Use when the user "
                        "says 'look at the X', 'find the Y', 'what is that Z', "
                        "or any request to direct attention to a visible object."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={
                            "description": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description="Description of the object to look at, e.g. 'the red cup on the desk'",
                            ),
                        },
                        required=["description"],
                    ),
                ),
            ],
        ),
    ]


def build_face_tools() -> list:
    """Gemini Live function declarations for face enroll / identify."""
    return [
        genai_types.Tool(
            function_declarations=[
                genai_types.FunctionDeclaration(
                    name="enroll_face",
                    description=(
                        "Remember the face of a person visible to the "
                        "camera under the given name. Capture takes a "
                        "couple of seconds. If several people are in "
                        "frame you can disambiguate with `position` "
                        "('left' / 'center' / 'right') or `who` ('me' "
                        "for whoever is speaking, 'other' for the "
                        "non-speaking newcomer, 'pointed' if the user "
                        "is pointing at them). When the user is "
                        "introducing someone - e.g. 'this is my friend "
                        "Ahmad' - the safest defaults are "
                        "`who='pointed'` if they're pointing, else "
                        "`who='other'`, else ask which position. If no "
                        "hint is given and only one person is unknown, "
                        "that unknown person is enrolled automatically."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={
                            "name": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description=(
                                    "The person's name, e.g. 'Husam'. "
                                    "Will be stored in lowercase."
                                ),
                            ),
                            "position": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description=(
                                    "Optional horizontal hint: 'left', "
                                    "'center', 'right'."
                                ),
                            ),
                            "who": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description=(
                                    "Optional semantic hint: 'me' (the "
                                    "current speaker), 'other' (the "
                                    "non-speaking person - typical for "
                                    "introductions), 'pointed' (the "
                                    "person the user is pointing at)."
                                ),
                            ),
                        },
                        required=["name"],
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="identify_person",
                    description=(
                        "Recognize EVERY person currently visible in the "
                        "camera, not just the closest one. Returns a list "
                        "of people with their names (or 'unknown') and "
                        "where they are in the frame ('left', 'center', "
                        "'right'), sorted largest-face-first. Call this "
                        "when the user asks 'who am I', 'who do you see', "
                        "'do you know me', or similar."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={},
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="find_person",
                    description=(
                        "Check whether a specific known person is "
                        "currently visible to the camera, even if they "
                        "are off-center or another person is also in "
                        "frame. Returns their position when found."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={
                            "name": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description=(
                                    "The known person's name to look "
                                    "for, e.g. 'Husam'."
                                ),
                            ),
                        },
                        required=["name"],
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="list_known_people",
                    description=(
                        "List the names of everyone whose face is already "
                        "enrolled in memory. Useful when the user asks who "
                        "you know or remember."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={},
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="who_is_speaking",
                    description=(
                        "Return the name (or 'unknown') of the person whose "
                        "mouth is currently moving, i.e. who is talking "
                        "right now. Only works for people visible to the "
                        "camera. Returns no speaker when nobody's mouth "
                        "is moving, which usually means an off-camera "
                        "voice."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={},
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="focus_on_person",
                    description=(
                        "Lock the robot's head (and body) onto a named "
                        "person so it keeps following them as they "
                        "move, instead of looking at whoever is "
                        "closest. Person must be currently visible; "
                        "fails with not_visible otherwise."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={
                            "name": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description="Known person's name.",
                            ),
                        },
                        required=["name"],
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="focus_on_speaker",
                    description=(
                        "Switch to speaker-tracking mode: the robot "
                        "turns toward whoever is currently speaking "
                        "(detected by lip motion). Stays in this mode "
                        "until `clear_focus` or another focus call."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={},
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="clear_focus",
                    description=(
                        "Return to default focus behavior (follow the "
                        "closest / most prominent person). Use when "
                        "the user says 'look around normally', 'stop "
                        "following', 'relax'."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={},
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="set_gaze_mode",
                    description=(
                        "Directly set the robot's gaze mode. "
                        "`mode=\"person\"` with a `name` locks onto "
                        "that person (silently falls back to normal "
                        "scoring if they leave frame). "
                        "`mode=\"normal\"` runs the default scoring "
                        "engine. `mode=\"speaker\"` is syntactic "
                        "sugar for normal + a speaker-bias prior."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={
                            "mode": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description=(
                                    "One of 'normal', 'person', "
                                    "'speaker'."
                                ),
                            ),
                            "name": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description=(
                                    "Required when mode='person'."
                                ),
                            ),
                        },
                        required=["mode"],
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="describe_scene",
                    description=(
                        "Return a compact summary of the current "
                        "world state: everyone visible, their "
                        "direction, whether they are speaking, "
                        "whether they are facing the camera, any "
                        "active gestures, and what the robot is "
                        "currently focused on. Call this when the "
                        "user asks 'what do you see?', 'who's here?', "
                        "'what's going on?', or any similar "
                        "situational question."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={},
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="detect_gestures",
                    description=(
                        "Return the list of hand gestures currently "
                        "visible, attributed to each person. "
                        "Gestures include wave, point, thumbs_up, "
                        "open_palm, fist."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={},
                    ),
                ),
            ],
        ),
    ]


def _resolve_enroll_target(
    recognizer: FaceRecognizer,
    lip_tracker: Optional[LipMotionTracker],
    gesture_tracker: "Optional[GestureTracker]",
    get_frame: Any,
    position_hint: Optional[str],
    who_hint: Optional[str],
) -> Dict[str, Any]:
    """Choose which visible face to enroll.

    Priority (highest wins):

    1. Explicit ``position`` hint from Gemini - pick the single face
       at that side of the frame.
    2. ``who='pointed'`` OR a fresh PointHint visible in the scene -
       pick the face inside the pointed-at body bbox.
    3. ``who='other'`` (the introducee / non-speaker) when the
       current speaker is identifiable via lip-motion - pick the
       non-speaking face.
    4. Exactly one unknown face visible - pick it. This is the most
       common "this is my friend Ahmad" case.
    5. Only one face visible at all - pick it (old behaviour).
    6. Otherwise return ``reason='ambiguous'`` with a candidates
       list so Gemini can ask the user to clarify.

    Returns a dict with either ``ok=True`` plus the target bbox +
    human-readable reason, or ``ok=False`` with a reason and
    candidates so the caller can relay to Gemini.
    """
    frame = get_frame()
    if frame is None:
        return {"ok": False, "reason": "no_frame"}
    matches = recognizer.identify_all_in_frame(frame)
    if not matches:
        return {"ok": False, "reason": "no_face_visible"}

    candidates = [
        {
            "position": m.position,
            "name": m.name or "unknown",
            "bbox": list(m.bbox),
        }
        for m in matches
    ]

    # ---- 1. Position hint ---------------------------------------------
    if position_hint in ("left", "center", "right"):
        picks = [m for m in matches if m.position == position_hint]
        if len(picks) == 1:
            p = picks[0]
            return {
                "ok": True, "bbox": list(p.bbox),
                "position": p.position,
                "reason": f"position={position_hint}",
            }
        if not picks:
            return {
                "ok": False, "reason": "position_empty",
                "position": position_hint, "candidates": candidates,
            }
        # Multiple faces at that side -- still ambiguous; fall through.

    # ---- 2. Pointing ---------------------------------------------------
    point_hint = None
    if gesture_tracker is not None:
        point_hint = gesture_tracker.get_point_hint(max_age_s=2.0)
    want_pointed = (who_hint == "pointed") or (
        who_hint is None and point_hint is not None
        and point_hint.pointed_at_bbox is not None
    )
    if want_pointed:
        if point_hint is None or point_hint.pointed_at_bbox is None:
            if who_hint == "pointed":
                return {
                    "ok": False, "reason": "no_pointing_detected",
                    "candidates": candidates,
                }
        else:
            body_bbox = point_hint.pointed_at_bbox
            # Pick the face whose bbox is best contained inside the
            # pointed-at body bbox.
            best = None
            best_score = 0.0
            for m in matches:
                s = _bbox_containment(m.bbox, body_bbox)
                if s > best_score:
                    best_score = s
                    best = m
            if best is not None and best_score >= 0.5:
                return {
                    "ok": True, "bbox": list(best.bbox),
                    "position": best.position, "reason": "pointing",
                }

    # ---- 3. "other" / non-speaker --------------------------------------
    if who_hint == "other" and lip_tracker is not None:
        speech_snap = lip_tracker.snapshot()
        speaking_bboxes = [
            tuple(s.bbox) for s in speech_snap if s.is_speaking
        ]
        if speaking_bboxes:
            non_speakers = [
                m for m in matches
                if not any(
                    _bbox_iou_simple(m.bbox, sb) > 0.3 for sb in speaking_bboxes
                )
            ]
            if len(non_speakers) == 1:
                p = non_speakers[0]
                return {
                    "ok": True, "bbox": list(p.bbox),
                    "position": p.position, "reason": "non_speaker",
                }

    # ---- 4. Unknown-only -----------------------------------------------
    unknown = [m for m in matches if m.name is None]
    if len(unknown) == 1:
        p = unknown[0]
        return {
            "ok": True, "bbox": list(p.bbox),
            "position": p.position, "reason": "only_unknown",
        }

    # ---- 5. Single face ------------------------------------------------
    if len(matches) == 1:
        p = matches[0]
        return {
            "ok": True, "bbox": list(p.bbox),
            "position": p.position, "reason": "only_face",
        }

    # ---- 6. Ambiguous --------------------------------------------------
    return {
        "ok": False, "reason": "ambiguous",
        "candidates": candidates,
    }


def _bbox_containment(inner, outer) -> float:
    """Fraction of ``inner`` bbox that falls inside ``outer`` bbox."""
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    ax1, ay1 = max(ix1, ox1), max(iy1, oy1)
    ax2, ay2 = min(ix2, ox2), min(iy2, oy2)
    if ax2 <= ax1 or ay2 <= ay1:
        return 0.0
    inter = (ax2 - ax1) * (ay2 - ay1)
    inner_area = max(1, (ix2 - ix1) * (iy2 - iy1))
    return inter / inner_area


def _bbox_iou_simple(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1, a_area + b_area - inter)


def _look_at_object(
    api_key: str, jpeg_bytes: bytes, description: str,
    frame_w: int, frame_h: int,
) -> dict:
    """Ask Qwen3-VL on OpenRouter to locate an object.

    Returns:
        {"found": bool, "nx": float, "ny": float,
         "bbox_norm": [xmin, ymin, xmax, ymax] (0..1),
         "confidence": str, "label": str,
         "error": str|None}
    """
    try:
        from openai import OpenAI

        or_key = os.getenv("OPENROUTER_API_KEY", api_key)
        if not or_key:
            return {"found": False, "error": "OPENROUTER_API_KEY not set"}
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=or_key)

        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64}"

        prompt = (
            f"Locate the {description} in the image. "
            f"Return ONLY a JSON object of the form "
            f'{{"bbox_2d": [x1, y1, x2, y2], "label": "{description}"}} '
            f"using absolute pixel coordinates in an image that is "
            f"{frame_w} pixels wide and {frame_h} pixels tall. "
            f'If the {description} is not visible, return '
            f'{{"bbox_2d": null}}.'
        )

        response = client.chat.completions.create(
            model=LOOK_AT_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=128,
            temperature=0.0,
        )
        text = response.choices[0].message.content or ""

        # Extract first JSON object from the response
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if not match:
            return {"found": False, "error": "no JSON in model response"}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"found": False, "error": "invalid JSON in model response"}

        bbox = data.get("bbox_2d") or data.get("bbox") or data.get("box")
        if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return {"found": False, "error": "object not visible"}

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]

        # Some Qwen variants output normalized [0..1000] — detect and rescale
        if max(x1, y1, x2, y2) <= 1000 and max(frame_w, frame_h) > 1000:
            x1 = int(x1 * frame_w / 1000)
            x2 = int(x2 * frame_w / 1000)
            y1 = int(y1 * frame_h / 1000)
            y2 = int(y2 * frame_h / 1000)

        # Clamp & sanity-check
        x1 = max(0, min(frame_w - 1, x1))
        x2 = max(0, min(frame_w - 1, x2))
        y1 = max(0, min(frame_h - 1, y1))
        y2 = max(0, min(frame_h - 1, y2))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return {"found": False, "error": "bbox too small"}

        # Convert to normalized 0..1
        nx = (x1 + x2) / 2.0 / max(1, frame_w)
        ny = (y1 + y2) / 2.0 / max(1, frame_h)
        return {
            "found": True,
            "nx": nx,
            "ny": ny,
            "bbox_norm": [
                x1 / max(1, frame_w), y1 / max(1, frame_h),
                x2 / max(1, frame_w), y2 / max(1, frame_h),
            ],
            "confidence": "high",
            "label": data.get("label", description),
        }
    except Exception as e:
        return {"found": False, "error": str(e)}


def make_tool_handler(
    recognizer: FaceRecognizer,
    latest: "LatestFrame",
    lip_tracker: Optional[LipMotionTracker] = None,
    focus_manager: Optional[FocusManager] = None,
    cascade_tracker: Optional[CascadeTracker] = None,
    world: Optional[WorldStateHolder] = None,
    identity_graph: Optional[IdentityGraph] = None,
    gesture_tracker: "Optional[GestureTracker]" = None,
    controller: Optional["RobotController"] = None,
    api_key: Optional[str] = None,
    frame_w: int = 640,
    frame_h: int = 480,
):
    """Build the async tool handler closure Gemini Live will call."""

    async def handler(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "enroll_face":
            person = str(args.get("name", "")).strip()
            if not person:
                return {"ok": False, "error": "name is required"}
            position_hint = (args.get("position") or "").strip().lower() or None
            who_hint = (args.get("who") or "").strip().lower() or None

            # Resolve WHICH face to enroll from the current frame.
            pick = await asyncio.to_thread(
                _resolve_enroll_target,
                recognizer,
                lip_tracker,
                gesture_tracker,
                latest.get_frame,
                position_hint,
                who_hint,
            )
            if not pick.get("ok"):
                return pick

            target_bbox = tuple(pick["bbox"])
            count = await asyncio.to_thread(
                recognizer.enroll,
                person,
                latest.get_frame,
                target_bbox=target_bbox,
            )
            if count == 0:
                return {
                    "ok": False,
                    "name": person,
                    "reason": "target_lost",
                    "picked": pick.get("reason"),
                }
            # Mirror into the identity graph so voice enrollment
            # can link to the same person later.
            if identity_graph is not None:
                try:
                    identity_graph.record_face_enrollment(person, count)
                except Exception as e:  # pragma: no cover
                    log.warning("identity_graph enrollment hook failed: %s", e)
            return {
                "ok": True,
                "name": person,
                "samples_captured": count,
                "picked": pick.get("reason"),
                "position": pick.get("position"),
            }

        if name == "identify_person":
            matches = await asyncio.to_thread(
                recognizer.identify_all, latest.get_frame,
            )
            if not matches:
                return {"ok": False, "reason": "no_face_visible"}
            # Drop bulky bbox info for Gemini; keep the semantic bits.
            people = [
                {
                    "name": m.name or "unknown",
                    "similarity": round(m.similarity, 3),
                    "position": m.position,
                }
                for m in matches
            ]
            known = [p for p in people if p["name"] != "unknown"]
            return {
                "ok": True,
                "count": len(people),
                "known_count": len(known),
                "people": people,
            }

        if name == "find_person":
            target = str(args.get("name", "")).strip()
            if not target:
                return {"ok": False, "error": "name is required"}
            match = await asyncio.to_thread(
                recognizer.find, latest.get_frame, target,
            )
            if match is None:
                return {"ok": True, "found": False, "name": target}
            return {
                "ok": True,
                "found": True,
                "name": match.name,
                "position": match.position,
                "similarity": round(match.similarity, 3),
            }

        if name == "list_known_people":
            return {"ok": True, "names": recognizer.db.list_names()}

        if name == "who_is_speaking":
            if lip_tracker is None:
                return {
                    "ok": False,
                    "error": "lip-motion tracker not running",
                }
            snap = lip_tracker.snapshot()
            if not snap:
                return {
                    "ok": True,
                    "is_anyone_speaking": False,
                    "reason": "no_face_visible",
                    "speaker": None,
                    "faces": [],
                }
            speaker = snap[0] if snap[0].is_speaking else None
            return {
                "ok": True,
                "is_anyone_speaking": speaker is not None,
                "speaker": speaker.to_dict() if speaker else None,
                "faces": [c.to_dict() for c in snap],
            }

        if name == "focus_on_person":
            if focus_manager is None or cascade_tracker is None:
                return {"ok": False, "error": "focus subsystem disabled"}
            person = str(args.get("name", "")).strip()
            if not person:
                return {"ok": False, "error": "name is required"}
            canonical = canonicalize_name(person)
            if canonical not in recognizer.db.list_names():
                return {
                    "ok": False,
                    "reason": "unknown_name",
                    "name": person,
                    "known_names": recognizer.db.list_names(),
                }
            # Always ARM the mode -- even if the person isn't visible
            # yet. The GazePolicy silently falls through to normal
            # scoring while the target is offscreen; the instant they
            # step into frame the lock engages automatically. This
            # matches "when Husam walks in, look at him" semantics.
            focus_manager.set_mode_person(canonical)

            # Opportunistic: if they ARE visible right now, engage
            # immediately instead of waiting for the next lip-tracker
            # tick.
            matches = await asyncio.to_thread(
                recognizer.identify_all, latest.get_frame,
            )
            face = next((m for m in matches if m.name == canonical), None)
            yolo_id: Optional[int] = None
            position: Optional[str] = None
            if face is not None:
                yolo_tracks = cascade_tracker.get_all_tracks()
                yolo_id = focus_manager.try_focus_by_face_match(
                    face, yolo_tracks,
                )
                position = face.position
            return {
                "ok": True,
                "mode": "person",
                "name": canonical,
                "visible_now": face is not None,
                "position": position,
                "yolo_track_id": yolo_id,
                "engaged_now": yolo_id is not None,
            }

        if name == "focus_on_speaker":
            if focus_manager is None:
                return {"ok": False, "error": "focus subsystem disabled"}
            focus_manager.set_mode_speaker()
            return {"ok": True, "mode": "speaker"}

        if name == "clear_focus":
            if focus_manager is None:
                return {"ok": False, "error": "focus subsystem disabled"}
            focus_manager.set_mode_auto()
            return {"ok": True, "mode": "auto"}

        if name == "set_gaze_mode":
            if focus_manager is None:
                return {"ok": False, "error": "focus subsystem disabled"}
            mode = str(args.get("mode", "")).strip().lower()
            if mode in ("normal", "auto"):
                focus_manager.set_mode_normal()
                return {"ok": True, "mode": "normal"}
            if mode == "speaker":
                focus_manager.set_mode_speaker()
                return {"ok": True, "mode": "speaker"}
            if mode == "person":
                target = str(args.get("name", "")).strip()
                if not target:
                    return {
                        "ok": False,
                        "error": "name is required when mode='person'",
                    }
                canonical = canonicalize_name(target)
                if canonical not in recognizer.db.list_names():
                    return {
                        "ok": False,
                        "reason": "unknown_name",
                        "name": target,
                        "known_names": recognizer.db.list_names(),
                    }
                focus_manager.set_mode_person(canonical)
                return {
                    "ok": True,
                    "mode": "person",
                    "name": canonical,
                }
            return {
                "ok": False,
                "error": (
                    f"unknown mode {mode!r}; expected "
                    f"'normal' / 'person' / 'speaker'"
                ),
            }

        if name == "get_current_time":
            return {"ok": True, "time": time.strftime("%H:%M:%S"), "date": time.strftime("%Y-%m-%d")}

        if name == "ping":
            return {"ok": True, "pong": True}

        if name == "get_robot_status":
            return {
                "ok": True,
                "face_recognition": recognizer is not None,
                "lip_tracker": lip_tracker is not None,
                "focus_manager": focus_manager is not None,
            }

        def _clamp_angle(v: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, v))

        if name == "set_head_angle":
            if controller is None:
                return {"ok": False, "error": "robot controller not available"}
            yaw = float(args.get("yaw_deg", 0))
            pitch = float(args.get("pitch_deg", 0))
            yaw = _clamp_angle(yaw, -60, 60)
            pitch = _clamp_angle(pitch, -30, 30)
            controller.set_manual_target(yaw_deg=yaw, pitch_deg=pitch)
            return {"ok": True, "yaw_deg": yaw, "pitch_deg": pitch, "mode": "manual"}

        if name == "look_straight":
            if controller is None:
                return {"ok": False, "error": "robot controller not available"}
            controller.set_manual_target(yaw_deg=0, pitch_deg=0)
            return {"ok": True, "yaw_deg": 0, "pitch_deg": 0, "mode": "manual"}

        if name == "look_left":
            if controller is None:
                return {"ok": False, "error": "robot controller not available"}
            controller.set_manual_target(yaw_deg=30, pitch_deg=0)
            return {"ok": True, "yaw_deg": 30, "pitch_deg": 0, "mode": "manual"}

        if name == "look_right":
            if controller is None:
                return {"ok": False, "error": "robot controller not available"}
            controller.set_manual_target(yaw_deg=-30, pitch_deg=0)
            return {"ok": True, "yaw_deg": -30, "pitch_deg": 0, "mode": "manual"}

        if name == "look_up":
            if controller is None:
                return {"ok": False, "error": "robot controller not available"}
            controller.set_manual_target(yaw_deg=0, pitch_deg=-15)
            return {"ok": True, "yaw_deg": 0, "pitch_deg": -15, "mode": "manual"}

        if name == "look_down":
            if controller is None:
                return {"ok": False, "error": "robot controller not available"}
            controller.set_manual_target(yaw_deg=0, pitch_deg=15)
            return {"ok": True, "yaw_deg": 0, "pitch_deg": 15, "mode": "manual"}

        if name == "enable_face_follow":
            if controller is None:
                return {"ok": False, "error": "robot controller not available"}
            controller.clear_manual()
            return {"ok": True, "mode": "auto_follow"}

        if name == "disable_face_follow":
            if controller is None:
                return {"ok": False, "error": "robot controller not available"}
            controller.set_manual_target(
                yaw_deg=math.degrees(controller.snapshot.sent_yaw),
                pitch_deg=math.degrees(controller.snapshot.sent_pitch),
            )
            return {"ok": True, "mode": "manual"}

        if name == "describe_scene":
            if world is None:
                return {"ok": False, "error": "world-state disabled"}
            snap = world.snapshot()
            return {
                "ok": True,
                "brief": snap.brief_text(),
                "humans": [h.to_dict() for h in snap.humans],
                "robot": {
                    "gaze_mode": snap.robot.gaze_mode,
                    "gaze_state": snap.robot.gaze_state,
                    "current_target_name": snap.robot.current_target_name,
                    "current_target_track_id": snap.robot.current_target_track_id,
                },
                "env": {
                    "audio_speech_active": snap.env.audio_speech_active,
                    "last_heard_voice_name": snap.env.last_heard_voice_name,
                    "last_heard_voice_similarity": round(
                        snap.env.last_heard_voice_similarity, 3,
                    ),
                    "last_heard_voice_age_s": (
                        None
                        if snap.env.last_heard_voice_ts == 0.0
                        else round(
                            time.monotonic() - snap.env.last_heard_voice_ts, 2,
                        )
                    ),
                },
            }

        if name == "detect_gestures":
            if world is None:
                return {"ok": False, "error": "world-state disabled"}
            snap = world.snapshot()
            seen = []
            for h in snap.humans:
                if not h.active_gestures:
                    continue
                seen.append({
                    "name": h.name or "unknown",
                    "position": h.direction,
                    "gestures": list(h.active_gestures),
                })
            return {
                "ok": True,
                "count": len(seen),
                "people": seen,
            }

        if name == "look_at":
            if controller is None:
                return {"ok": False, "error": "robot controller not available"}
            if not api_key:
                return {"ok": False, "error": "API key not available for vision query"}
            description = str(args.get("description", "")).strip()
            if not description:
                return {"ok": False, "error": "description is required"}

            jpeg = latest.get_mirrored_jpeg()
            if jpeg is None:
                return {"ok": False, "error": "no camera frame available"}

            # Fire-and-forget: reply instantly so Gemini Live isn't blocked,
            # then run the slow vision query in the background.
            async def _look_at_task() -> None:
                try:
                    result = await asyncio.to_thread(
                        _look_at_object, api_key, jpeg, description,
                        frame_w, frame_h,
                    )
                    if not result.get("found"):
                        log.info("look_at background: not found: %s",
                                 result.get("error"))
                        return
                    nx = result["nx"]
                    ny = result["ny"]
                    _set_look_at_marker(
                        nx, ny, description,
                        bbox_norm=result.get("bbox_norm"),
                    )
                    if controller is not None:
                        yaw_deg = (nx - 0.5) * 120.0
                        pitch_deg = (ny - 0.5) * 60.0
                        yaw_deg = max(-60.0, min(60.0, yaw_deg))
                        pitch_deg = max(-30.0, min(30.0, pitch_deg))
                        controller.set_manual_target(
                            yaw_deg=yaw_deg, pitch_deg=pitch_deg,
                        )
                except Exception:
                    log.exception("look_at background task failed")

            asyncio.create_task(_look_at_task())
            return {
                "ok": True,
                "status": "searching",
                "description": description,
            }

        return {"ok": False, "error": f"unknown tool: {name}"}

    return handler


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

    def get_mirrored_jpeg(self) -> Optional[bytes]:
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        mirrored = cv2.flip(frame, 1)
        ok, buf = cv2.imencode(
            ".jpg", mirrored,
            [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
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


# --- Look-at marker (shared between tool handler and overlay) ---------------
_LOOK_AT_LOCK = threading.Lock()
_LOOK_AT_MARKER: dict = {}  # x, y, description, timestamp


def _set_look_at_marker(
    x: float, y: float, description: str,
    bbox_norm: Optional[list] = None,
) -> None:
    with _LOOK_AT_LOCK:
        _LOOK_AT_MARKER["nx"] = float(x)
        _LOOK_AT_MARKER["ny"] = float(y)
        _LOOK_AT_MARKER["description"] = description
        _LOOK_AT_MARKER["bbox_norm"] = bbox_norm
        _LOOK_AT_MARKER["timestamp"] = time.monotonic()


def _get_look_at_marker(max_age_s: float = 5.0) -> Optional[dict]:
    with _LOOK_AT_LOCK:
        ts = _LOOK_AT_MARKER.get("timestamp", 0)
        if time.monotonic() - ts > max_age_s:
            return None
        return dict(_LOOK_AT_MARKER)


# --- Preview overlay -------------------------------------------------------

def draw_overlay(
    view: np.ndarray,
    snap,
    det_bbox,
    face_snap: Optional[list] = None,
    focus_snap: Optional[FocusSnapshot] = None,
) -> None:
    """Draw the preview overlay.

    Besides the original "currently-tracked body" box, this draws every
    detected face with its name (green = known, amber = unknown), marks
    who's currently speaking, and adds an extra highlight on the face
    the focus manager is locked onto.

    Note: the preview is mirror-flipped (``cv2.flip(frame, 1)``) before
    we get here, so every x coordinate must be flipped to ``w - x``.
    """
    h, w = view.shape[:2]

    # ---- 1. Per-face boxes + names ------------------------------------
    if face_snap:
        focused_tid = focus_snap.locked_id if focus_snap else None
        focused_name = focus_snap.focused_name if focus_snap else None
        for cand in face_snap:
            x1, y1, x2, y2 = cand.bbox
            # Mirror.
            mx1, mx2 = w - x2, w - x1
            name = cand.name or "unknown"
            known = cand.name is not None
            speaking = cand.is_speaking

            # Color convention:
            #   green  = known person
            #   amber  = unknown / not recognized
            #   cyan   = currently speaking (overrides)
            if speaking:
                color = (255, 220, 0)           # cyan-ish in BGR
            elif known:
                color = (80, 220, 80)           # green
            else:
                color = (0, 180, 240)           # amber

            # A focused face deserves a fatter, brighter outline.
            is_focused = (
                (focused_tid is not None and cand.track_id == focused_tid)
                or (focused_name is not None and cand.name == focused_name)
            )
            thickness = 3 if is_focused else 1

            cv2.rectangle(view, (mx1, y1), (mx2, y2), color, thickness)

            label = name
            if speaking:
                label += "  [speaking]"
            if is_focused:
                label += "  [FOCUS]"
            # Label background for readability.
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
            )
            ly1 = max(0, y1 - th - 6)
            cv2.rectangle(
                view, (mx1, ly1), (mx1 + tw + 6, ly1 + th + 6),
                color, -1,
            )
            cv2.putText(
                view, label, (mx1 + 3, ly1 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1,
                cv2.LINE_AA,
            )

    # ---- 2. Current tracked-body bbox + target point ------------------
    color = TIER_COLORS.get(snap.tier, (200, 200, 200))
    if det_bbox is not None:
        x1, y1, x2, y2, dx, dy = det_bbox
        x1m, x2m = w - x2, w - x1
        dxm = w - dx
        cv2.rectangle(view, (x1m, y1), (x2m, y2), color, 2)
        cv2.circle(view, (dxm, dy), 5, color, -1)

    # ---- 3. Crosshair + status HUD ------------------------------------
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

    # ---- 4. Focus mode banner -----------------------------------------
    if focus_snap is not None and focus_snap.mode != "auto":
        banner = f"FOCUS: {focus_snap.mode}"
        if focus_snap.focused_name:
            banner += f" -> {focus_snap.focused_name}"
        elif focus_snap.target_name:
            banner += f" -> {focus_snap.target_name} (not visible)"
        (tw, th), _ = cv2.getTextSize(
            banner, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2,
        )
        cv2.rectangle(
            view, (w - tw - 20, 10), (w - 5, 20 + th + 5),
            (40, 40, 40), -1,
        )
        cv2.putText(
            view, banner, (w - tw - 12, 20 + th),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2,
            cv2.LINE_AA,
        )

    # ---- 5. Look-at object marker -----------------------------------------
    lam = _get_look_at_marker(max_age_s=5.0)
    if lam is not None:
        nx = max(0.0, min(1.0, lam.get("nx", 0.5)))
        ny = max(0.0, min(1.0, lam.get("ny", 0.5)))
        lx = int(nx * w)
        ly = int(ny * h)
        color = (255, 0, 255)  # magenta in BGR
        thickness = 2

        # Draw bounding box if available (coords are already mirrored).
        bbox_norm = lam.get("bbox_norm")
        if bbox_norm is not None and len(bbox_norm) == 4:
            bx1 = int(max(0.0, min(1.0, bbox_norm[0])) * w)
            by1 = int(max(0.0, min(1.0, bbox_norm[1])) * h)
            bx2 = int(max(0.0, min(1.0, bbox_norm[2])) * w)
            by2 = int(max(0.0, min(1.0, bbox_norm[3])) * h)
            cv2.rectangle(view, (bx1, by1), (bx2, by2), color, thickness)

        # Center crosshair.
        size = 20
        cv2.line(view, (lx - size, ly), (lx + size, ly), color, thickness + 1)
        cv2.line(view, (lx, ly - size), (lx, ly + size), color, thickness + 1)
        cv2.circle(view, (lx, ly), 8, color, 2)
        label = (
            f"LOOK: {lam.get('description', '?')} "
            f"({lam.get('confidence', '?')})"
        )
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
        )
        ly1 = max(0, ly - size - th - 8)
        cv2.rectangle(
            view, (lx - tw // 2 - 4, ly1),
            (lx + tw // 2 + 4, ly1 + th + 6),
            color, -1,
        )
        cv2.putText(
            view, label, (lx - tw // 2 + 1, ly1 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            cv2.LINE_AA,
        )


# --- Main ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0,
                        help="OpenCV camera index (default: 0). Use "
                             "--list-cameras to see which indices work.")
    parser.add_argument("--list-cameras", action="store_true",
                        help="Probe camera indices 0..5 and exit.")
    parser.add_argument("--reachy-camera", action="store_true",
                        help="Use the Reachy daemon camera instead of a "
                             "direct OpenCV camera.")
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
    parser.add_argument("--no-face-recognition", action="store_true",
                        help="Disable face enrollment / identification "
                             "tools (skips loading FaceNet at startup).")
    parser.add_argument("--no-lip-motion", action="store_true",
                        help="Disable the lip-motion speaker-detection "
                             "tracker (also disables the who_is_speaking "
                             "tool). Implied when face recognition is off.")
    parser.add_argument("--no-vad", action="store_true",
                        help="Disable Silero VAD audio-speech gating of "
                             "lip-motion. Without VAD, chewing/yawning can "
                             "false-fire the speaker detector.")
    parser.add_argument("--no-head-pose", action="store_true",
                        help="Disable MediaPipe head-pose tracker "
                             "(kills the `is_being_addressed` score term).")
    parser.add_argument("--no-gestures", action="store_true",
                        help="Disable MediaPipe Hands gesture recognition "
                             "(no wave / point / thumbs-up events).")
    parser.add_argument("--no-identity-graph", action="store_true",
                        help="Disable the IdentityGraph layer on top of "
                             "FaceDB (voice enrollment + corrections).")
    parser.add_argument("--no-voice-id", action="store_true",
                        help="Disable speaker identification + "
                             "cross-modal voice/face linking "
                             "(SpeechBrain ECAPA-TDNN).")
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

    # --- Shared spine: EventBus + WorldStateHolder ---------------------
    # Every sense writes to `world`; every brain reads it. `bus` is the
    # fire-and-forget pub/sub for transitions (person_detected, etc.).
    # See docs/architecture.md.
    bus = EventBus()
    world = WorldStateHolder()

    # --- Face recognition (optional) -----------------------------------
    face_recognizer: Optional[FaceRecognizer] = None
    face_tools: Optional[list] = None
    face_db: Optional[FaceDB] = None
    if not args.no_face_recognition:
        face_db = FaceDB(FACE_DB_DIR)
        # Stay on CPU: the models are small and face tools run on demand
        # (not every frame), so a GPU is not worth the MPS portability
        # risk.
        face_recognizer = FaceRecognizer(face_db, device="cpu")
        face_tools = build_face_tools()
        log.info(
            "Face recognition enabled. Known people: %s",
            face_db.list_names() or "(none yet)",
        )

    # --- Identity graph (optional) -------------------------------------
    identity_graph: Optional[IdentityGraph] = None
    if face_db is not None and not args.no_identity_graph:
        identity_graph = IdentityGraph(IDENTITY_DIR, face_db)
        log.info("IdentityGraph enabled (%d identities).",
                 len(identity_graph.list_identities()))

    # Lip-motion tracker depends on the face recognizer (shares MTCNN +
    # FaceDB). If face recognition is off, lip motion can't run either.
    lip_tracker: Optional[LipMotionTracker] = None

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

    # Optional humanoid senses -- built later, after the camera is open
    # so they can pull frames from ``latest``.
    audio_vad: Optional[SileroVAD] = None
    head_pose_tracker: Optional[HeadPoseTracker] = None
    gesture_tracker: Optional[GestureTracker] = None
    voice_embedder: Optional[VoiceEmbedder] = None
    voice_identity: Optional[VoiceIdentityWorker] = None
    lip_tracker = None

    if not args.reachy_camera:
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

        # --- Silero VAD (audio speech detector) ----------------------------
        # Built before the lip-motion tracker so we can hand it the
        # ``is_active`` callback. If torch / silero isn't installed the
        # VAD stays disabled and lip-motion falls back to un-gated.
        if not args.no_vad:
            audio_vad = SileroVAD(bus=bus, world=world)
            audio_vad.start()

        # --- Lip-motion speaker tracker ------------------------------------
        if face_recognizer is not None and not args.no_lip_motion:
            audio_active_fn = None
            if audio_vad is not None:
                audio_active_fn = lambda: audio_vad.is_active  # noqa: E731
            lip_tracker = LipMotionTracker(
                recognizer=face_recognizer,
                get_frame=latest.get_frame,
                audio_active_fn=audio_active_fn,
            )
            lip_tracker.start()
            log.info(
                "Lip-motion speaker tracker enabled (audio-VAD gating=%s).",
                "on" if audio_active_fn is not None else "off",
            )

        # --- Focus manager --------------------------------------------------
        focus_manager: Optional[FocusManager] = None
        if face_recognizer is not None:
            # Need the cascade tracker to exist; it was started just above.
            focus_manager = FocusManager(
                tracker=tracker,
                lip_tracker=lip_tracker,
                world=world,
                bus=bus,
                frame_w=frame_w,
                frame_h=frame_h,
            )
            log.info("FocusManager enabled (GazePolicy: normal / person, priors).")

        # --- MediaPipe head-pose tracker -----------------------------------
        # Stamps each HumanView with yaw/pitch/roll + is_facing_camera so
        # the GazePolicy can score `is_being_addressed` (see docs/tech-
        # recommendations.md §1.5).
        if focus_manager is not None and not args.no_head_pose:
            head_pose_tracker = HeadPoseTracker(
                get_frame=latest.get_frame,
                yolo_tracks=tracker.get_all_tracks,
                registry=focus_manager.registry,
            )
            head_pose_tracker.start()

        # --- MediaPipe gesture tracker -------------------------------------
        if focus_manager is not None and not args.no_gestures:
            gesture_tracker = GestureTracker(
                get_frame=latest.get_frame,
                yolo_tracks=tracker.get_all_tracks,
                registry=focus_manager.registry,
                bus=bus,
            )
            gesture_tracker.start()

        # --- Voice identity (speaker ID + cross-modal enrollment) ---------
        # Requires: VAD (utterance source), face_recognizer (who is
        # visibly speaking right now) and an identity graph. Any missing
        # piece disables the whole stack. See docs/identity.md §3 for
        # the co-occurrence enrollment algorithm.
        if (
            audio_vad is not None
            and identity_graph is not None
            and lip_tracker is not None
            and not args.no_voice_id
        ):
            voice_embedder = VoiceEmbedder()

            def _visible_speaker() -> Optional[str]:
                """Return the canonical name of a visible + currently-speaking face.

                Exactly one named face must be speaking; otherwise the
                enrollment attribution is ambiguous and we return None so
                the sample gets skipped.
                """
                try:
                    names_speaking = [
                        s.name for s in lip_tracker.snapshot()
                        if s.is_speaking and s.name
                    ]
                except Exception:
                    return None
                if len(names_speaking) != 1:
                    return None
                return names_speaking[0]

            voice_identity = VoiceIdentityWorker(
                vad=audio_vad,
                embedder=voice_embedder,
                identity_graph=identity_graph,
                world=world,
                bus=bus,
                visible_speaker_supplier=_visible_speaker,
            )
            voice_identity.start()

    # --- Reactive wiring: person_lost -> directed head search ---------
    # The gaze-motion planner (natural_gaze.py) handles the visual
    # "where did they go?" sweep when it gets a notify_person_lost()
    # call. The controller is built below and picks up this hook.
    _last_known_yaw = {"value": 0.0}

    def _on_person_left(evt):
        # Resolve the yaw we were pointing at when they disappeared so
        # the search sweep starts from there. This runs on the publish
        # thread so keep it light.
        _last_known_yaw["value"] = _last_known_yaw.get("value", 0.0)

    bus.subscribe(EVT_PERSON_LEFT, _on_person_left)

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

            # --- Reachy daemon camera (deferred setup) --------------------
            if args.reachy_camera:
                log.info("Waiting for Reachy camera frame ...")
                probe = None
                for _ in range(40):
                    probe = reachy.media.get_frame()
                    if probe is not None:
                        break
                    time.sleep(0.1)
                if probe is None:
                    log.error("Reachy camera: no frame from daemon")
                    sys.exit(1)
                frame_h, frame_w = probe.shape[:2]
                enhance = enhance_brightness if args.enhance else (lambda f: f)
                log.info(
                    "Camera ready: %dx%d (enhance=%s)",
                    frame_w, frame_h, "CLAHE" if args.enhance else "off",
                )
                tracker.warmup(frame_h, frame_w)
                tracker.start()

                if not args.no_vad:
                    audio_vad = SileroVAD(bus=bus, world=world)
                    audio_vad.start()

                if face_recognizer is not None and not args.no_lip_motion:
                    audio_active_fn = None
                    if audio_vad is not None:
                        audio_active_fn = lambda: audio_vad.is_active  # noqa: E731
                    lip_tracker = LipMotionTracker(
                        recognizer=face_recognizer,
                        get_frame=latest.get_frame,
                        audio_active_fn=audio_active_fn,
                    )
                    lip_tracker.start()
                    log.info(
                        "Lip-motion speaker tracker enabled (audio-VAD gating=%s).",
                        "on" if audio_active_fn is not None else "off",
                    )

                if face_recognizer is not None:
                    focus_manager = FocusManager(
                        tracker=tracker,
                        lip_tracker=lip_tracker,
                        world=world,
                        bus=bus,
                        frame_w=frame_w,
                        frame_h=frame_h,
                    )
                    log.info("FocusManager enabled (GazePolicy: normal / person, priors).")

                if focus_manager is not None and not args.no_head_pose:
                    head_pose_tracker = HeadPoseTracker(
                        get_frame=latest.get_frame,
                        yolo_tracks=tracker.get_all_tracks,
                        registry=focus_manager.registry,
                    )
                    head_pose_tracker.start()

                if focus_manager is not None and not args.no_gestures:
                    gesture_tracker = GestureTracker(
                        get_frame=latest.get_frame,
                        yolo_tracks=tracker.get_all_tracks,
                        registry=focus_manager.registry,
                        bus=bus,
                    )
                    gesture_tracker.start()

                if (
                    audio_vad is not None
                    and identity_graph is not None
                    and lip_tracker is not None
                    and not args.no_voice_id
                ):
                    voice_embedder = VoiceEmbedder()

                    def _visible_speaker() -> Optional[str]:
                        """Return the canonical name of a visible + currently-speaking face.

                        Exactly one named face must be speaking; otherwise the
                        enrollment attribution is ambiguous and we return None so
                        the sample gets skipped.
                        """
                        try:
                            names_speaking = [
                                s.name for s in lip_tracker.snapshot()
                                if s.is_speaking and s.name
                            ]
                        except Exception:
                            return None
                        if len(names_speaking) != 1:
                            return None
                        return names_speaking[0]

                    voice_identity = VoiceIdentityWorker(
                        vad=audio_vad,
                        embedder=voice_embedder,
                        identity_graph=identity_graph,
                        world=world,
                        bus=bus,
                        visible_speaker_supplier=_visible_speaker,
                    )
                    voice_identity.start()

            # --- Mic source that also feeds Silero VAD -----------------
            # Gemini Live polls mic_source; we intercept each chunk and
            # push a copy into the VAD so `audio_speech_active` stays
            # live without a second capture path.
            def _mic_source_tee():
                sample = media.get_audio_sample()
                if sample is None:
                    return sample
                if audio_vad is not None and audio_vad.enabled:
                    try:
                        arr = np.asarray(sample, dtype=np.float32)
                        if arr.ndim == 2:
                            if arr.shape[0] < arr.shape[1]:
                                arr = arr.T
                            arr = arr.mean(axis=1)
                        audio_vad.push_samples(arr)
                    except Exception as e:   # pragma: no cover
                        log.debug("VAD push_samples failed: %s", e)
                return sample

            # --- Gemini Live ------------------------------------------------
            reachy.goto_target(head=head_pose(), duration=0.8, body_yaw=0.0)
            time.sleep(0.3)

            # Mutable speaking flag so we can create the controller before
            # gemini exists, then wire it in once the session is live.
            _speaking_fn = {"fn": lambda: False}
            is_talking_fn = lambda: _speaking_fn["fn"]()

            controller = RobotController(
                reachy, tracker, is_talking_fn, no_body=args.no_body,
            )

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
                        mic_source=_mic_source_tee,
                        speaker_sink=media.push_audio_sample,
                    )
                    all_tools = []
                    if face_recognizer is not None and face_tools is not None:
                        all_tools.extend(face_tools)
                    # Always include test tools so Gemini can exercise
                    # function calling even when face recognition is off.
                    all_tools.extend(build_test_tools())
                    if all_tools:
                        kwargs["tools"] = all_tools
                        kwargs["tool_handler"] = make_tool_handler(
                            face_recognizer, latest, lip_tracker,
                            focus_manager, tracker,
                            world=world,
                            identity_graph=identity_graph,
                            gesture_tracker=gesture_tracker,
                            controller=controller,
                            api_key=api_key,
                            frame_w=frame_w,
                            frame_h=frame_h,
                        )
                    model = args.model or os.environ.get("GEMINI_MODEL")
                    if model:
                        kwargs["model"] = model
                    gemini = GeminiLiveSession(**kwargs)
                    gemini.start()
                    _speaking_fn["fn"] = gemini.is_speaking.is_set

            # Hook bus -> controller: on person_left, trigger a
            # directed search sweep from the last-known head yaw.
            def _on_person_left_search(evt):
                try:
                    controller.notify_person_lost(controller.snapshot.sent_yaw)
                except Exception as e:  # pragma: no cover
                    log.debug("person_lost search hook failed: %s", e)
            bus.subscribe(EVT_PERSON_LEFT, _on_person_left_search)

            # Hook bus -> controller: on voice_unseen, start a "who
            # said that?" sweep (currently only fired manually; kept
            # wired so DOA can publish the same event later).
            def _on_voice_unseen(evt):
                try:
                    guess = evt.payload.get("yaw_rad")
                    controller.notify_voice_unseen(guess_yaw_rad=guess)
                except Exception as e:  # pragma: no cover
                    log.debug("voice_unseen hook failed: %s", e)
            bus.subscribe(EVT_VOICE_UNSEEN, _on_voice_unseen)

            # Hook bus -> "who said that?" search when we recognise a
            # voice but the corresponding face isn't in frame. This is
            # what lets the robot turn to look for Husam when it hears
            # him from around a corner.
            def _on_voice_identified(evt):
                name = evt.payload.get("name")
                if not name:
                    return
                try:
                    snap = world.snapshot()
                    if snap.find_by_name(name) is not None:
                        return   # they are already visible, nothing to do
                    controller.notify_voice_unseen()
                except Exception as e:   # pragma: no cover
                    log.debug("voice_identified hook failed: %s", e)
            bus.subscribe(EVT_VOICE_IDENTIFIED, _on_voice_identified)

            # Track the currently focused *name* so we only announce
            # real person-to-person changes to the gaze planner. Raw
            # track_id flips are noisy (ByteTrack re-IDs the same
            # person every occlusion) and must not be treated as new
            # targets.
            prev_focus_name: Optional[str] = None

            last_log = 0.0
            while not stop["flag"]:
                if args.reachy_camera:
                    frame = reachy.media.get_frame()
                    ok = frame is not None
                else:
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

                # Apply the current focus intent (person / speaker /
                # normal) to the cascade tracker's lock. Cheap: only
                # reads snapshots produced by other threads.
                focus_snap: Optional[FocusSnapshot] = None
                face_snap: list = []
                if focus_manager is not None:
                    focus_snap = focus_manager.update(
                        tracker.get_all_tracks(),
                        frame_w=frame_w, frame_h=frame_h,
                    )
                    # Only hint at a saccade when we switch between
                    # *named* people -- not on every ByteTrack re-ID.
                    # The P-controller already handles smooth changes
                    # to an unnamed / moving target naturally.
                    if (
                        focus_snap.focused_name is not None
                        and focus_snap.focused_name != prev_focus_name
                        and prev_focus_name is not None
                        and det is not None
                    ):
                        # Target yaw ≈ current head yaw + err mapped
                        # through the camera's half-FOV. 30° is a
                        # reasonable approximation for a 60° HFOV lens
                        # (AVFoundation default); off by <20% of jump
                        # and natural_gaze only cares about whether
                        # the jump is above its 8° saccade threshold.
                        half_fov_rad = math.radians(30.0)
                        approx_yaw = snap.sent_yaw - det.err_x * half_fov_rad
                        approx_pitch = snap.sent_pitch + det.err_y * half_fov_rad * 0.6
                        controller.notify_target_switched(
                            approx_yaw, approx_pitch,
                        )
                    prev_focus_name = focus_snap.focused_name
                if lip_tracker is not None:
                    face_snap = lip_tracker.snapshot()

                # Humans seen? -> keep idle drift at bay.
                if focus_snap is not None and focus_snap.name_by_track:
                    controller.mark_humans_seen()
                elif tracker.get_all_tracks():
                    controller.mark_humans_seen()

                # Publish the robot's own pose into WorldState so any
                # brain reading the snapshot (Gemini tools / future
                # Hsafa bridge) sees a coherent picture.
                world.set_robot_pose(
                    head_yaw_deg=math.degrees(snap.sent_yaw),
                    head_pitch_deg=math.degrees(snap.sent_pitch),
                    head_roll_deg=0.0,
                    body_yaw_deg=math.degrees(snap.body_yaw),
                    is_speaking=snap.talking,
                )

                # Heartbeat
                now = time.time()
                if now - last_log > 0.75:
                    last_log = now
                    tid = f"#{snap.track_id}" if snap.track_id else "--"
                    mode = "TALK" if snap.talking else "idle"
                    focus_tag = ""
                    if focus_snap is not None and focus_snap.mode != "auto":
                        # In person(name) mode, target_name is the name
                        # we were ASKED to follow; focused_name is who
                        # the policy actually picked this tick (may
                        # differ if target is offscreen or face
                        # recognition flipped).
                        tgt = focus_snap.target_name
                        got = focus_snap.focused_name
                        if tgt and got and tgt != got:
                            label = f"{tgt}->{got}"
                        else:
                            label = tgt or got or "?"
                        focus_tag = f" focus={focus_snap.mode}({label})"
                    log.info(
                        "tier=%-9s %-4s %-4s err=(%+.2f,%+.2f) "
                        "yaw=%+6.1f pitch=%+6.1f body=%+6.1f%s",
                        snap.tier, tid, mode,
                        snap.err_x, snap.err_y,
                        math.degrees(snap.sent_yaw),
                        math.degrees(snap.sent_pitch),
                        math.degrees(snap.body_yaw),
                        focus_tag,
                    )

                # Preview
                if not args.no_preview:
                    view = cv2.flip(frame, 1)
                    draw_overlay(view, snap, det_bbox, face_snap, focus_snap)
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
        if head_pose_tracker is not None:
            head_pose_tracker.stop(timeout=1.0)
        if gesture_tracker is not None:
            gesture_tracker.stop(timeout=1.0)
        if voice_identity is not None:
            voice_identity.stop(timeout=1.0)
        if audio_vad is not None:
            audio_vad.stop(timeout=1.0)
        if lip_tracker is not None and lip_tracker.is_alive():
            lip_tracker.stop()
            lip_tracker.join(timeout=1.0)
        if gemini is not None:
            gemini.stop()
        if cap is not None:
            cap.release()
        # ``tracker.join()`` raises if the thread was never started (e.g. the
        # Reachy daemon rejected us before ``tracker.start()`` was reached).
        if tracker.is_alive():
            tracker.stop()
            tracker.join(timeout=1.0)
        # Flush any pending voice samples accumulated during cross-modal
        # enrollment (no-op until the voice embedder lands).
        if identity_graph is not None:
            committed = identity_graph.commit_pending_voices()
            if committed:
                log.info("IdentityGraph: committed voices for %s", committed)
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
