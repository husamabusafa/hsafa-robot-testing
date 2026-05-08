"""object_detector.py — lightweight YOLO object detection for hand-held items.

Runs ``yolov8n.pt`` (standard COCO detection) on demand and throttled so
the extra model does not starve the main pose tracker.  Only classes
that people typically hold are returned (phone, bottle, cup, book,
scissors, knife, etc.).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Classes from COCO that a human is likely to hold up / show.
_HOLDABLE_COCO = {
    "cell phone", "bottle", "cup", "wine glass", "book", "scissors",
    "knife", "spoon", "fork", "bowl", "mouse", "remote", "keyboard",
    "handbag", "backpack", "umbrella", "clock", "vase", "teddy bear",
    "hair drier", "toothbrush", "sports ball", "baseball bat",
    "baseball glove", "tennis racket",
}

# Default minimum detection confidence.
_DEFAULT_CONF = 0.35
# Default inference image size (small = fast).
_DEFAULT_IMGSZ = 256
# Max inference rate when running on the gesture thread (5 Hz).
_MAX_HZ = 5.0

Bbox = Tuple[int, int, int, int]


class ObjectDetector:
    """Throttled, filtered YOLO object detector.

    The model is loaded lazily on first call to ``detect`` so creating
    the instance is cheap even when the feature ends up unused.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf: float = _DEFAULT_CONF,
        imgsz: int = _DEFAULT_IMGSZ,
        device: str = "cpu",
        max_hz: float = _MAX_HZ,
        classes: Optional[List[str]] = None,
    ) -> None:
        self._model_path = model_path or str(
            Path(__file__).resolve().parent.parent / "yolov8n.pt"
        )
        self._conf = conf
        self._imgsz = imgsz
        self._device = device
        self._period = 1.0 / max(max_hz, 0.5)
        self._classes = set(classes) if classes is not None else _HOLDABLE_COCO
        self._model = None
        self._last_infer: float = 0.0

    def _load(self) -> bool:
        if self._model is not None:
            return True
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(self._model_path)
            log.info(
                "ObjectDetector: loaded %s (%d classes)",
                self._model_path, len(self._model.names),
            )
            # Warm-up with a dummy frame.
            self._model.predict(
                np.zeros((128, 128, 3), dtype=np.uint8),
                imgsz=self._imgsz, conf=self._conf,
                device=self._device, verbose=False,
            )
            return True
        except Exception as e:
            log.warning("ObjectDetector: could not load YOLO (%s)", e)
            return False

    def detect(
        self, frame: np.ndarray,
    ) -> List[Tuple[str, float, Bbox]]:
        """Return [(class_name, confidence, bbox), ...] for holdable objects.

        Inference is skipped if called faster than ``max_hz`` — in that
        case the *previous* result is returned (empty list if no prior
        result exists).  This keeps the gesture-thread from choking.
        """
        if not self._load():
            return []

        now = time.monotonic()
        if now - self._last_infer < self._period:
            return getattr(self, "_last_result", [])
        self._last_infer = now

        try:
            results = self._model.predict(
                frame, imgsz=self._imgsz, conf=self._conf,
                device=self._device, verbose=False,
            )
        except Exception as e:
            log.warning("ObjectDetector inference failed: %s", e)
            self._last_result: List[Tuple[str, float, Bbox]] = []
            return self._last_result

        out: List[Tuple[str, float, Bbox]] = []
        r0 = results[0] if results else None
        if r0 is None:
            self._last_result = out
            return out

        boxes = getattr(r0, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            self._last_result = out
            return out

        names = self._model.names
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []

        for bb, cf, cid in zip(xyxy, confs, cls_ids):
            label = names.get(cid, str(cid))
            if label not in self._classes:
                continue
            x1, y1, x2, y2 = (int(v) for v in bb.tolist())
            out.append((label, float(cf), (x1, y1, x2, y2)))

        self._last_result = out
        return out
