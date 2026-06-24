"""
Centralized configuration for the ArchTec Segmentation app.
Edit values here instead of hunting through the codebase.
"""

APP_CONFIG: dict = {
    "page_title": "Traditional Building Detector",
    "page_icon": "🏛️",
    "layout": "wide",
}

MODEL_CONFIG: dict = {
    "name": "YOLOv8 (Ultralytics)",
    "task": "Object Detection",
    "path": "model/Object Detection/YOLOv8-Nano-Detect/best.pt",
    "default_conf": 0.25,
    "default_iou": 0.45,
    "imgsz": 640,
}

# Mapping: task label -> YOLO version -> model and training results path.
TASK_MODEL_OPTIONS: dict = {
    "Object Detection": {
        "YOLOv8": {
            "model": "model/Object Detection/YOLOv8-Nano-Detect/best.pt",
            "results": "model/Object Detection/YOLOv8-Nano-Detect/results.csv",
        },
        "YOLOv9": {
            "model": "model/Object Detection/YOLOv9-Tiny-Detect/best.pt",
            "results": "model/Object Detection/YOLOv9-Tiny-Detect/results.csv",
        },
        "YOLOv10": {
            "model": "model/Object Detection/YOLOv10-Nano-Detect/best.pt",
            "results": "model/Object Detection/YOLOv10-Nano-Detect/results.csv",
        },
        "YOLOv11": {
            "model": "model/Object Detection/YOLOv11-Nano-Detect/best.pt",
            "results": "model/Object Detection/YOLOv11-Nano-Detect/results.csv",
        },
    },
    "Instance Segmentation": {
        "YOLOv8": {
            "model": "model/Segmentation/YOLOv8-Nano-Segment/best.pt",
            "results": "model/Segmentation/YOLOv8-Nano-Segment/results.csv",
        },
        "YOLOv9": {
            "model": "model/Segmentation/YOLOv9-Compact-Segment/best.pt",
            "results": "model/Segmentation/YOLOv9-Compact-Segment/results.csv",
        },
        "YOLOv11": {
            "model": "model/Segmentation/YOLOv11-Nano-Segment/best.pt",
            "results": "model/Segmentation/YOLOv11-Nano-Segment/results.csv",
        },
    },
}

DEFAULT_TASK = "Object Detection"
DEFAULT_MODEL_VERSION = "YOLOv8"

# Backward-compatible flat mapping for places that read only selected_model_key.
MODEL_OPTIONS: dict = {
    f"{task} · {version}": cfg
    for task, versions in TASK_MODEL_OPTIONS.items()
    for version, cfg in versions.items()
}
