#!/usr/bin/env python3
"""MediaPipe hand tracking helper.

Used by main.py to turn live hand gestures into five normalized (0..1) values
that map to joint angles.

MediaPipe landmarks used (0 = wrist, then finger joints to tips):
    0 wrist, 5 index MCP, 9 middle MCP, 8 index TIP, 12 middle TIP

Metrics:
    base = wrist.x                        (0..1, left -> right)
    arm1 = wrist.y                        (0..1, top -> bottom)
    arm2 = palm size, calibrated          (0..1, hand near -> far)
    tilt = wrist -> middle MCP vector     (0..1, scale invariant)
    claw = pinch / palm size, calibrated  (0..1, fist -> open hand)

MediaPipe 1.x removed the old `mp.solutions.hands` API, so this uses the
Tasks API (mediapipe.tasks.python.vision.HandLandmarker), which requires the
hand_landmarker.task model file in ./models/.
"""
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_RELPATH = os.path.join("models", "hand_landmarker.task")


def default_model_path():
    """Absolute path to the bundled hand landmarker model."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, MODEL_RELPATH)


def _span(value, lo, hi):
    """Map value from the raw [lo, hi] window onto 0..1, clamped."""
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


class HandTracker:
    """Wraps the MediaPipe HandLandmarker and produces normalized metrics."""

    def __init__(self, model_path=None, palm_min=0.04, palm_max=0.16,
                 pinch_min=0.10, pinch_max=0.90, num_hands=1):
        model_path = model_path or default_model_path()
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                "Hand landmarker model not found at {}. Download it with:\n"
                "  mkdir -p models && curl -L -o models/hand_landmarker.task \\\n"
                "    https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                .format(model_path))

        self._palm_min = float(palm_min)
        self._palm_max = float(palm_max)
        self._pinch_min = float(pinch_min)
        self._pinch_max = float(pinch_max)
        self._ts = 0

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, frame):
        """Return a dict of normalized hand values (0..1), or None if no hand."""
        h, w = frame.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # detect_for_video requires strictly increasing timestamps.
        self._ts += 1
        result = self._landmarker.detect_for_video(image, self._ts)
        if not result.hand_landmarks:
            return None
        lm = result.hand_landmarks[0]

        wrist = lm[0]
        idx_mcp = lm[5]
        mid_mcp = lm[9]
        mid_tip = lm[12]
        idx_tip = lm[8]

        # Palm size in pixels, used to make the other metrics scale invariant.
        palm_cx = (idx_mcp.x + mid_mcp.x) / 2.0
        palm_cy = (idx_mcp.y + mid_mcp.y) / 2.0
        palm_px = float(np.hypot((palm_cx - wrist.x) * w, (palm_cy - wrist.y) * h))
        if palm_px <= 1.0:
            return None

        pinch_px = float(np.hypot((idx_tip.x - mid_tip.x) * w,
                                  (idx_tip.y - mid_tip.y) * h))

        # arm2: apparent palm size -> how far the hand is from the camera.
        arm2 = _span(palm_px / float(w), self._palm_min, self._palm_max)
        # claw: pinch distance relative to the palm, so it survives hand depth.
        claw = _span(pinch_px / palm_px, self._pinch_min, self._pinch_max)
        # tilt: forearm direction (wrist -> palm centre) as a unit vector.
        # Using the direction instead of the raw length keeps it scale
        # invariant: straight down = 1.0, straight up = 0.0, level = 0.5.
        fore_x = (palm_cx - wrist.x) * w
        fore_y = (palm_cy - wrist.y) * h
        fore_len = float(np.hypot(fore_x, fore_y))
        if fore_len <= 1.0:
            return None
        tilt = 0.5 + 0.5 * (fore_y / fore_len)

        return {
            "base": float(np.clip(wrist.x, 0.0, 1.0)),
            "arm1": float(np.clip(wrist.y, 0.0, 1.0)),
            "arm2": arm2,
            "tilt": float(np.clip(tilt, 0.0, 1.0)),
            "claw": claw,
        }

    def close(self):
        """Release the MediaPipe graph. Safe to call more than once."""
        landmarker = getattr(self, "_landmarker", None)
        if landmarker is None:
            return
        self._landmarker = None
        try:
            landmarker.close()
        except Exception:
            pass


def build_hands(cfg=None, **kw):
    """Build a HandTracker using the calibration block from config.json."""
    hand_cfg = (cfg or {}).get("hand", {})
    options = {
        "palm_min": hand_cfg.get("palm_min", 0.04),
        "palm_max": hand_cfg.get("palm_max", 0.16),
        "pinch_min": hand_cfg.get("pinch_min", 0.10),
        "pinch_max": hand_cfg.get("pinch_max", 0.90),
    }
    options.update(kw)
    return HandTracker(**options)


def get_hand_state(frame, hands):
    """Return dict of normalized hand values (0..1), or None if no hand."""
    return hands.process(frame)


def close_hands(hands):
    """Close the tracker if it is still open. Safe to call more than once."""
    if hands is None:
        return
    try:
        hands.close()
    except Exception:
        pass
