"""
Model loading and inference logic for YOLOv11 instance segmentation.
All ML-specific code lives here, keeping the Streamlit layer clean.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from app.config import MODEL_CONFIG


# ── Device Selection ───────────────────────────────────────────────────────────

def get_device() -> str:
    """Return the best available compute device."""
    if torch.backends.mps.is_available():
        return "mps"   # Apple Silicon GPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Model Lifecycle ────────────────────────────────────────────────────────────

def load_model(model_path: str | None = None) -> YOLO:
    """
    Load a YOLO model from *model_path* and move it to the best device.

    Args:
        model_path: Path to the ``.pt`` weights file.
                    Falls back to ``MODEL_CONFIG["path"]`` when omitted.

    Returns:
        Loaded ``YOLO`` instance ready for inference.
    """
    path = model_path or MODEL_CONFIG["path"]
    device = get_device()
    model = YOLO(path)
    model.to(device)
    return model


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(
    model: YOLO,
    image: Image.Image,
    conf_threshold: float | None = None,
    iou_threshold: float | None = None,
) -> object:
    """
    Run YOLOv11 detection + segmentation on a PIL image.

    Args:
        model:          Loaded ``YOLO`` instance.
        image:          Input image as PIL ``Image``.
        conf_threshold: Confidence threshold (defaults to ``MODEL_CONFIG`` value).
        iou_threshold:  NMS IoU threshold (defaults to ``MODEL_CONFIG`` value).

    Returns:
        Single Ultralytics ``Results`` object.
    """
    conf = conf_threshold if conf_threshold is not None else MODEL_CONFIG["default_conf"]
    iou  = iou_threshold  if iou_threshold  is not None else MODEL_CONFIG["default_iou"]

    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=MODEL_CONFIG["imgsz"],
        verbose=False,
    )
    return results[0]   # single image → single result


# ── Post-processing ────────────────────────────────────────────────────────────

def draw_results(result) -> np.ndarray:
    """
    Render bounding boxes and segmentation masks onto the image.

    Returns:
        Annotated image as an RGB ``numpy`` array suitable for ``st.image``.
    """
    annotated_bgr = result.plot(
        masks=True,
        boxes=True,
        labels=True,
        conf=True,
        line_width=2,
    )
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


def parse_detections(result) -> list[dict]:
    """
    Convert a raw YOLO ``Results`` object into a list of structured dicts.

    Each dict contains:
        - ``class_id``   – integer class index
        - ``class_name`` – human-readable label
        - ``confidence`` – float in [0, 1]
        - ``bbox_xyxy``  – ``[x1, y1, x2, y2]`` pixel coordinates
        - ``has_mask``   – whether a segmentation mask is present
    """
    detections: list[dict] = []
    names = result.names
    has_masks = result.masks is not None

    for box in result.boxes:
        class_id = int(box.cls[0] if hasattr(box.cls, "__len__") else box.cls)
        confidence = float(box.conf[0] if hasattr(box.conf, "__len__") else box.conf)
        detections.append(
            {
                "class_id":   class_id,
                "class_name": names[class_id],
                "confidence": confidence,
                "bbox_xyxy":  box.xyxy[0].tolist(),
                "has_mask":   has_masks,
            }
        )

    return detections
