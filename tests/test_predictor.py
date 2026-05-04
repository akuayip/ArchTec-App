"""
Basic smoke tests for the predictor module.
Run with: pytest tests/
"""

import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_dummy_image(width: int = 64, height: int = 64) -> Image.Image:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ── parse_detections ───────────────────────────────────────────────────────────

def test_parse_detections_empty():
    """parse_detections should return an empty list when no boxes exist."""
    from core.predictor import parse_detections

    result = MagicMock()
    result.names = {0: "column"}
    result.boxes = []
    result.masks = None

    assert parse_detections(result) == []


def test_parse_detections_single_box():
    """parse_detections should correctly parse a single detection."""
    from core.predictor import parse_detections

    box = MagicMock()
    box.cls = [0]
    box.conf = [0.9]
    box.xyxy = [MagicMock(tolist=lambda: [10, 20, 50, 80])]

    result = MagicMock()
    result.names = {0: "column"}
    result.boxes = [box]
    result.masks = None

    detections = parse_detections(result)

    assert len(detections) == 1
    assert detections[0]["class_name"] == "column"
    assert detections[0]["confidence"] == pytest.approx(0.9)
    assert detections[0]["has_mask"] is False


# ── get_device ─────────────────────────────────────────────────────────────────

def test_get_device_returns_string():
    from core.predictor import get_device
    assert get_device() in {"cpu", "cuda", "mps"}
