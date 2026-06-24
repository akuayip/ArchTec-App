"""
Tab: Detector
Upload image → run inference → show results.
The selected model is stored in session_state so Training Results can follow it.
"""

import streamlit as st
from PIL import Image
import pandas as pd
from collections import Counter
from pathlib import Path

from core.predictor import load_model, run_inference, draw_results, parse_detections
from app.config import (
    DEFAULT_MODEL_VERSION,
    DEFAULT_TASK,
    MODEL_CONFIG,
    TASK_MODEL_OPTIONS,
)


@st.cache_resource
def get_model(model_path: str):
    with st.spinner(f"Loading model from `{model_path}`..."):
        return load_model(model_path)


def _render_sidebar_controls() -> dict:
    with st.sidebar:
        st.markdown("### ⚙️ Inference Settings")
        st.divider()

        # ── Task and model selectors ───────────────────────────────────────────
        task_options = list(TASK_MODEL_OPTIONS.keys())
        default_task_index = (
            task_options.index(DEFAULT_TASK)
            if DEFAULT_TASK in task_options
            else 0
        )
        selected_task = st.selectbox(
            "Select Task",
            options=task_options,
            index=default_task_index,
            help="Choose the inference task to run",
        )

        available_versions = {
            version: cfg
            for version, cfg in TASK_MODEL_OPTIONS[selected_task].items()
            if Path(cfg["model"]).exists()
        }

        if not available_versions:
            st.warning("⚠️ No model found in the `model/` folder.")
            selected_version = next(iter(TASK_MODEL_OPTIONS[selected_task]))
        else:
            version_options = list(available_versions.keys())
            default_version_index = (
                version_options.index(DEFAULT_MODEL_VERSION)
                if DEFAULT_MODEL_VERSION in version_options
                else 0
            )
            selected_version = st.selectbox(
                "Select YOLO Version",
                options=version_options,
                index=default_version_index,
                help="Choose the model version · Training Results will update automatically",
            )

        # Store in session_state so the Training tab can read the active model.
        selected_key = f"{selected_task} · {selected_version}"
        selected_config = TASK_MODEL_OPTIONS[selected_task][selected_version]
        st.session_state["selected_task"] = selected_task
        st.session_state["selected_model_version"] = selected_version
        st.session_state["selected_model_key"] = selected_key
        model_path    = selected_config["model"]
        results_path  = selected_config["results"]

        # Active training CSV info.
        csv_label = Path(results_path).name
        st.markdown(
            f"""
            <div style="font-size:0.72rem;color:#4FC3F7;margin:-8px 0 12px;
                        padding:6px 10px;background:rgba(79,195,247,0.08);
                        border-radius:6px;border:1px solid rgba(79,195,247,0.2);">
                📊 Training data: <b>{csv_label}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1, max_value=1.0,
            value=MODEL_CONFIG["default_conf"], step=0.05,
            help="Minimum confidence required to display detections",
        )
        iou_threshold = st.slider(
            "IoU Threshold (NMS)",
            min_value=0.1, max_value=1.0,
            value=MODEL_CONFIG["default_iou"], step=0.05,
            help="NMS threshold for removing overlapping bounding boxes",
        )
        show_table = st.toggle("Show Detection Table", value=True)

    return {
        "conf_threshold": conf_threshold,
        "iou_threshold":  iou_threshold,
        "show_table":     show_table,
        "model_path":     model_path,
        "results_path":   results_path,
        "selected_key":   selected_key,
        "selected_task":   selected_task,
        "selected_version": selected_version,
    }


def _render_upload_zone() -> object:
    st.markdown(
        """
        <div style="margin: 1.5rem 0 0.5rem;">
            <p style="font-size:0.78rem; font-weight:600; letter-spacing:0.06em;
                      text-transform:uppercase; color:#616161; margin:0 0 8px;">
                Upload Image
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.file_uploader(
        "Choose a traditional building image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="Format: JPG, JPEG, PNG · Max 200MB",
    )


def _render_summary(detections: list[dict], show_table: bool, selected_task: str) -> None:
    if not detections:
        st.warning("No architectural components found. Try lowering the Confidence Threshold.")
        return

    st.divider()
    st.markdown(
        """<p style="font-size:0.78rem; font-weight:600; letter-spacing:0.06em;
                    text-transform:uppercase; color:#616161; margin:0 0 12px;">
            Detection Summary
        </p>""",
        unsafe_allow_html=True,
    )

    class_counts = Counter(d["class_name"] for d in detections)
    cols = st.columns(max(len(class_counts), 1))
    for idx, (cls, cnt) in enumerate(class_counts.items()):
        cols[idx].metric(label=f"🧱 {cls.capitalize()}", value=cnt)

    if show_table:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """<p style="font-size:0.78rem; font-weight:600; letter-spacing:0.06em;
                        text-transform:uppercase; color:#616161; margin:0 0 8px;">
                Detection Details
            </p>""",
            unsafe_allow_html=True,
        )
        rows = []
        for d in detections:
            row = {
                "Component":            d["class_name"].capitalize(),
                "Confidence":          f"{d['confidence'] * 100:.1f}%",
                "BBox [x1,y1,x2,y2]": [round(v) for v in d["bbox_xyxy"]],
            }
            if selected_task == "Instance Segmentation":
                row["Segmentation Mask"] = "Available" if d["has_mask"] else "Not available"
            rows.append(row)
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """<p style="font-size:0.78rem; font-weight:600; letter-spacing:0.06em;
                    text-transform:uppercase; color:#616161; margin:0 0 8px;">
            Confidence per Detection
        </p>""",
        unsafe_allow_html=True,
    )
    chart_data = pd.DataFrame({
        "Component":   [d["class_name"] for d in detections],
        "Confidence": [d["confidence"] for d in detections],
    })
    st.bar_chart(chart_data.set_index("Component"))


def render() -> None:
    settings = _render_sidebar_controls()
    selected_task = settings["selected_task"]
    selected_version = settings["selected_version"]
    task_action = (
        "detect and segment"
        if selected_task == "Instance Segmentation"
        else "detect"
    )
    result_label = (
        "Segmentation Result"
        if selected_task == "Instance Segmentation"
        else "Detection Result"
    )

    # ── Tab content header ─────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="padding: 1.5rem 0 0.5rem;">
            <p style="font-size:0.78rem; font-weight:600; letter-spacing:0.06em;
                      text-transform:uppercase; color:#4FC3F7; margin:0 0 6px;">
                {selected_task}
            </p>
            <h2 style="font-size:1.5rem; font-weight:700; margin:0 0 6px;
                       color:#f0f0f0; letter-spacing:-0.02em;">
                Upload Building Photo
            </h2>
            <p style="color:#616161; margin:0; font-size:0.9rem;">
                Using <b style="color:#9e9e9e;">{selected_version}</b> for
                <b style="color:#9e9e9e;">{selected_task}</b> ·
                The model will automatically {task_action} traditional architectural components.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = _render_upload_zone()

    if not uploaded_file:
        st.markdown(
            """
            <div style="margin-top:2rem; padding:2.5rem;
                        border:1px dashed rgba(255,255,255,0.1);
                        border-radius:12px; text-align:center; color:#424242;">
                <div style="font-size:2.5rem; margin-bottom:12px;">🏛️</div>
                <div style="font-size:0.9rem;">
                    Upload a traditional building photo to start {selected_task.lower()}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Model loading ──────────────────────────────────────────────────────────
    try:
        model = get_model(settings["model_path"])
        st.sidebar.success(f"✅ {settings['selected_key'].strip()} is ready")
    except Exception as e:
        st.sidebar.error("❌ Failed to load model")
        st.error(f"**Model not found** at `{settings['model_path']}`\n\n`{e}`")
        return

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown(
            """<p style="font-size:0.78rem; font-weight:600; letter-spacing:0.06em;
                        text-transform:uppercase; color:#616161; margin:0 0 8px;">
                Original Image
            </p>""",
            unsafe_allow_html=True,
        )
        st.image(image, use_container_width=True)

    with st.spinner(f"🔍 Running {selected_task.lower()} with {selected_version}..."):
        result = run_inference(
            model, image,
            conf_threshold=settings["conf_threshold"],
            iou_threshold=settings["iou_threshold"],
        )
        annotated_image = draw_results(result)
        detections = parse_detections(result)

    with col2:
        st.markdown(
            f"""<p style="font-size:0.78rem; font-weight:600; letter-spacing:0.06em;
                          text-transform:uppercase; color:#616161; margin:0 0 8px;">
                {result_label}
                <span style="color:#4FC3F7; margin-left:8px;">{len(detections)} objects</span>
            </p>""",
            unsafe_allow_html=True,
        )
        st.image(annotated_image, use_container_width=True)

    _render_summary(
        detections,
        show_table=settings["show_table"],
        selected_task=selected_task,
    )
