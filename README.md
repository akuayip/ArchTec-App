# ArchTec Segmentation

ArchTec Segmentation is a Streamlit-based application for analyzing traditional architectural elements using YOLO Object Detection and Instance Segmentation models.

Powered by YOLO Object Detection and Instance Segmentation models (v8-v11), this system automatically identifies traditional architectural elements including roofs, columns, walls, doors, windows, stairs, and ornamental details. Users can select different YOLO versions and inference tasks to analyze building structures and visualize detected components in real time.

## Features

- Select inference task: Object Detection or Instance Segmentation.
- Select YOLO version based on the active task.
- Default inference setup: Object Detection with YOLOv8.
- Upload traditional building images in JPG, JPEG, or PNG format.
- Adjust Confidence Threshold and IoU Threshold from the sidebar.
- Visualize annotated prediction results in real time.
- View detection summaries, confidence charts, and optional detection tables.
- Review training metrics from the `results.csv` file associated with the selected model.
- Read a dynamic "How YOLO Works" explanation focused on Backbone, Neck, and Head for the selected YOLO version and task.

## Available Models

The app expects model files under the `model/` directory:

```text
model/
├── Object Detection/
│   ├── YOLOv8-Nano-Detect/
│   │   ├── best.pt
│   │   └── results.csv
│   ├── YOLOv9-Tiny-Detect/
│   │   ├── best.pt
│   │   └── results.csv
│   ├── YOLOv10-Nano-Detect/
│   │   ├── best.pt
│   │   └── results.csv
│   └── YOLOv11-Nano-Detect/
│       ├── best.pt
│       └── results.csv
│
└── Segmentation/
    ├── YOLOv8-Nano-Segment/
    │   ├── best.pt
    │   └── results.csv
    ├── YOLOv9-Compact-Segment/
    │   ├── best.pt
    │   └── results.csv
    └── YOLOv11-Nano-Segment/
        ├── best.pt
        └── results.csv
```

Model weight files such as `.pt`, `.onnx`, `.engine`, `.torchscript`, `.tflite`, `.pb`, and `.h5` are ignored by Git because they are usually large binary artifacts. Keep `results.csv` files tracked if you want the Training Results tab to work without requiring users to regenerate training logs.

## Getting Started

### Option 1: Conda

If you already have the project environment:

```bash
conda activate arch
streamlit run app/main.py
```

To run commands without activating the shell:

```bash
conda run -n arch python -m pytest tests/
conda run -n arch streamlit run app/main.py
```

### Option 2: uv

Install dependencies:

```bash
uv sync
```

Run the app:

```bash
uv run streamlit run app/main.py
```

The application will be available at:

```text
http://localhost:8501
```

## Configuration

Model paths, defaults, and inference settings are centralized in:

```text
app/config.py
```

Important configuration entries:

| Config | Purpose |
|---|---|
| `TASK_MODEL_OPTIONS` | Maps each task and YOLO version to its model weight and training results CSV |
| `DEFAULT_TASK` | Default selected task, currently `Object Detection` |
| `DEFAULT_MODEL_VERSION` | Default selected YOLO version, currently `YOLOv8` |
| `MODEL_CONFIG["default_conf"]` | Default confidence threshold |
| `MODEL_CONFIG["default_iou"]` | Default IoU threshold |
| `MODEL_CONFIG["imgsz"]` | Inference image size |

## Project Structure

```text
ArchTec-Segmentation/
├── app/
│   ├── main.py                 # Streamlit entry point
│   ├── config.py               # App and model configuration
│   └── __init__.py
│
├── core/
│   ├── predictor.py            # YOLO loading, inference, plotting, and parsing
│   └── __init__.py
│
├── pages/
│   ├── tab_detector.py         # Inference UI
│   ├── tab_training.py         # Training metrics visualization
│   ├── tab_how_it_works.py     # Dynamic YOLO architecture explanation
│   └── __init__.py
│
├── assets/
│   ├── arsitektur.png          # Temporary architecture diagram placeholder
│   └── yolov11_pipeline.png    # YOLOv11 pipeline image
│
├── model/                      # Local model folders and training CSV files
├── tests/                      # Pytest smoke tests
├── scripts/                    # Utility scripts
├── utils/                      # Helper modules
├── pyproject.toml
├── uv.lock
└── README.md
```

## Tests

Run the test suite with the conda environment:

```bash
conda run -n arch python -m pytest tests/
```

Or with uv:

```bash
uv run pytest tests/
```

## Model File Policy

Model weights are intentionally excluded from Git via `.gitignore`.

Ignored examples:

- `model/**/*.pt`
- `model/**/*.onnx`
- `model/**/*.engine`
- `model/**/*.torchscript`
- `model/**/*.tflite`
- `model/**/*.pb`
- `model/**/*.h5`

Recommended workflow:

1. Keep source code, configuration, README, assets, and `results.csv` files in Git.
2. Store large model weights outside Git, or use Git LFS if model files must be versioned.
3. Place the model weights back into the expected `model/.../best.pt` paths before running inference.

## Troubleshooting

### Model does not appear in the selector

Check that the expected `best.pt` file exists under the correct task and version folder in `model/`. The UI label is `Instance Segmentation`, while the local folder is currently named `model/Segmentation/`.

### Training Results tab is empty or fails to load

Check that the selected model folder contains a valid `results.csv` file.

### Port 8501 is already in use

Run Streamlit on another port:

```bash
streamlit run app/main.py --server.port 8502
```

### Environment issues

Run:

```bash
python scripts/check_env.py
```

This checks whether key runtime dependencies such as PyTorch are available.
