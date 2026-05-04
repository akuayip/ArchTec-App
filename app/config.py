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
    "name": "YOLOv11 (Ultralytics)",
    "task": "Instance Segmentation",
    "path": "models/train_v2.pt",
    "default_conf": 0.25,
    "default_iou": 0.45,
    "imgsz": 640,
}

# Mapping: label → (model path, results csv path)
MODEL_OPTIONS: dict = {
    "Model V1  (train_v1.pt)": {
        "model": "models/train_v1.pt",
        "results": "models/results_v1.csv",
    },
    "Model V2  (train_v2.pt)": {
        "model": "models/train_v2.pt",
        "results": "models/results_v2.csv",
    },
}