"""
Tab: How YOLO Works
Explains the selected YOLO model architecture by task and version.
"""

from pathlib import Path

import streamlit as st

from app.config import DEFAULT_MODEL_VERSION, DEFAULT_TASK, TASK_MODEL_OPTIONS


ARCHITECTURE_NOTES: dict = {
    "YOLOv8": {
        "backbone_title": "CSP-style Backbone with C2f Blocks",
        "backbone": (
            "YOLOv8 uses a CSP-inspired feature extractor built around C2f blocks. "
            "Early layers capture edges, roof lines, wall textures, and ornaments; "
            "deeper layers combine those patterns into larger architectural shapes."
        ),
        "backbone_points": [
            "C2f modules improve feature reuse while keeping the model lightweight.",
            "SPPF aggregates wider context before features move into the neck.",
            "Useful for traditional buildings because small details and large structures both matter.",
        ],
        "neck_title": "PAN-FPN Multi-scale Feature Fusion",
        "neck": (
            "The neck combines shallow and deep feature maps so the model can reason "
            "about small details such as windows and ornaments as well as large objects "
            "such as roofs and walls."
        ),
        "neck_points": [
            "Top-down FPN path brings semantic context to high-resolution features.",
            "Bottom-up PAN path strengthens localization signals.",
            "P3, P4, and P5 scales support small, medium, and large components.",
        ],
        "head_title": "Anchor-free Decoupled Head",
        "head": (
            "YOLOv8 separates classification and box regression branches. This helps "
            "the model decide what a component is while independently refining where it is."
        ),
        "head_points": [
            "Predicts bounding boxes, classes, and confidence scores.",
            "Object Detection outputs boxes around architectural components.",
            "Instance Segmentation adds a prototype-mask branch and mask coefficients.",
        ],
    },
    "YOLOv9": {
        "backbone_title": "GELAN Backbone",
        "backbone": (
            "YOLOv9 is built around GELAN, a feature extractor designed to preserve "
            "information flow through efficient layer aggregation. This helps retain "
            "fine architectural cues while still learning high-level building context."
        ),
        "backbone_points": [
            "GELAN improves gradient flow and feature reuse across stages.",
            "Programmable Gradient Information helps training preserve useful features.",
            "Well suited for mixed-scale objects like columns, walls, windows, and ornaments.",
        ],
        "neck_title": "GELAN/PAN Feature Aggregation",
        "neck": (
            "The neck fuses multi-level features so small decorative patterns are not "
            "lost while larger structural elements remain easy to localize."
        ),
        "neck_points": [
            "Aggregates low-level detail with high-level semantic context.",
            "Keeps localization strong for narrow elements such as columns and stairs.",
            "Feeds multiple detection scales into the prediction head.",
        ],
        "head_title": "Multi-scale Detection or Segmentation Head",
        "head": (
            "The head predicts component classes and bounding boxes from several feature "
            "scales. In segmentation mode, it also estimates pixel-level masks for each object."
        ),
        "head_points": [
            "Predicts class probabilities and box coordinates per component.",
            "Object Detection focuses on component location and category.",
            "Instance Segmentation adds masks to separate component shapes from the background.",
        ],
    },
    "YOLOv10": {
        "backbone_title": "Efficient CSP-style Backbone",
        "backbone": (
            "YOLOv10 keeps a compact feature extractor optimized for fast detection. "
            "It learns visual patterns from simple edges and textures up to full "
            "architectural structures."
        ),
        "backbone_points": [
            "Designed for efficient inference with strong feature extraction.",
            "Captures repeated building patterns such as columns, windows, and wall boundaries.",
            "Balances speed and accuracy for real-time object detection.",
        ],
        "neck_title": "PAN-FPN Feature Fusion",
        "neck": (
            "The neck merges features across scales, letting the detector combine "
            "precise spatial detail with broader architectural context."
        ),
        "neck_points": [
            "High-resolution features help locate smaller components.",
            "Low-resolution semantic features help identify large structural parts.",
            "Multi-scale fusion improves robustness across different photo distances.",
        ],
        "head_title": "NMS-free Detection Head",
        "head": (
            "YOLOv10 is known for an end-to-end detection design that reduces reliance "
            "on traditional NMS. The head is optimized to produce cleaner final object predictions."
        ),
        "head_points": [
            "Predicts object boxes, class labels, and confidence scores.",
            "Uses one-to-one style assignment for cleaner final predictions.",
            "In this project, YOLOv10 is available for Object Detection.",
        ],
    },
    "YOLOv11": {
        "backbone_title": "C3k2 Backbone",
        "backbone": (
            "YOLOv11 uses C3k2 blocks to extract richer features with efficient computation. "
            "The backbone gradually transforms pixels into semantic representations of "
            "architectural elements."
        ),
        "backbone_points": [
            "Early layers learn edges, textures, and material transitions.",
            "Middle layers learn repeated forms such as window grids and columns.",
            "Deeper layers capture object-level context such as roof, wall, and stair regions.",
        ],
        "neck_title": "C2PSA-enhanced Multi-scale Neck",
        "neck": (
            "The neck combines features across scales and uses attention-style processing "
            "to emphasize relevant regions. This helps the model focus on architectural "
            "components instead of background clutter."
        ),
        "neck_points": [
            "Fuses P3, P4, and P5 feature maps for small, medium, and large objects.",
            "C2PSA improves attention to visually important regions.",
            "Useful when traditional details appear close to modern or background elements.",
        ],
        "head_title": "Decoupled Detection or Segmentation Head",
        "head": (
            "YOLOv11 separates classification and localization branches. For segmentation, "
            "an additional mask branch estimates component contours at pixel level."
        ),
        "head_points": [
            "Object Detection outputs boxes, labels, and confidence scores.",
            "Instance Segmentation outputs boxes plus per-instance masks.",
            "The selected confidence and IoU thresholds control which predictions are displayed.",
        ],
    },
}

ARCHITECTURE_IMAGES: dict = {
    "YOLOv8": "assets/arsitektur.png",
    "YOLOv9": "assets/arsitektur.png",
    "YOLOv10": "assets/arsitektur.png",
    "YOLOv11": "assets/arsitektur.png",
}


def _active_selection() -> tuple[str, str, str, dict]:
    selected_task = st.session_state.get("selected_task", DEFAULT_TASK)
    selected_version = st.session_state.get("selected_model_version", DEFAULT_MODEL_VERSION)

    if selected_task not in TASK_MODEL_OPTIONS:
        selected_task = DEFAULT_TASK
    if selected_version not in TASK_MODEL_OPTIONS[selected_task]:
        selected_version = next(iter(TASK_MODEL_OPTIONS[selected_task]))

    selected_key = f"{selected_task} · {selected_version}"
    selected_config = TASK_MODEL_OPTIONS[selected_task][selected_version]
    return selected_task, selected_version, selected_key, selected_config


def _render_arch_card(
    *,
    title: str,
    subtitle: str,
    body: str,
    points: list[str],
    accent: str,
) -> None:
    st.markdown(
        f"""
        <div style="
            background:rgba(255,255,255,0.025);
            border:1px solid rgba(255,255,255,0.08);
            border-left:4px solid {accent};
            border-radius:8px;
            padding:20px 22px;
            margin-bottom:14px;
        ">
            <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;
                        text-transform:uppercase;color:{accent};margin-bottom:6px;">
                {subtitle}
            </div>
            <div style="font-size:1.08rem;font-weight:800;color:#f0f0f0;margin-bottom:8px;">
                {title}
            </div>
            <div style="font-size:0.9rem;color:#bdbdbd;line-height:1.55;">
                {body}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for point in points:
        st.markdown(
            f"""
            <div style="display:flex;gap:8px;align-items:flex-start;
                        margin:0 0 8px 10px;font-size:0.86rem;color:#9e9e9e;">
                <span style="color:{accent};flex-shrink:0;">▸</span>
                <span>{point}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render() -> None:
    selected_task, selected_version, selected_key, selected_config = _active_selection()
    notes = ARCHITECTURE_NOTES[selected_version]
    task_output = (
        "bounding boxes and pixel-level instance masks"
        if selected_task == "Instance Segmentation"
        else "bounding boxes, class labels, and confidence scores"
    )

    st.markdown(
        f"""
        <div style="padding: 1.5rem 0 1rem;">
            <p style="font-size:0.78rem;font-weight:600;letter-spacing:0.06em;
                      text-transform:uppercase;color:#4FC3F7;margin:0 0 6px;">
                Selected Architecture
            </p>
            <h2 style="font-size:1.5rem;font-weight:700;margin:0 0 6px;
                       color:#f0f0f0;letter-spacing:-0.02em;">
                How {selected_version} Works for {selected_task}
            </h2>
            <p style="color:#757575;margin:0;font-size:0.92rem;line-height:1.55;">
                This explanation follows the active model selected in Inference Settings:
                <b style="color:#bdbdbd;">{selected_key}</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    image_path = Path("assets/yolov11_pipeline.png")
    if selected_version == "YOLOv11" and image_path.exists():
        st.image(str(image_path), use_container_width=True)

    st.divider()

    architecture_image = Path(ARCHITECTURE_IMAGES.get(selected_version, "assets/arsitektur.png"))
    st.markdown(
        f"""
        <p style="font-size:0.78rem;font-weight:600;letter-spacing:0.06em;
                  text-transform:uppercase;color:#616161;margin:0 0 12px;">
            {selected_version} Architecture Diagram
        </p>
        """,
        unsafe_allow_html=True,
    )
    if architecture_image.exists():
        _, image_col, _ = st.columns([1, 4, 1])
        with image_col:
            st.image(str(architecture_image), use_container_width=True)
    else:
        st.info(
            f"Architecture diagram placeholder not found at `{architecture_image}`.",
            icon="🖼️",
        )

    st.divider()

    st.markdown(
        f"""
        <p style="font-size:0.78rem;font-weight:600;letter-spacing:0.06em;
                  text-transform:uppercase;color:#616161;margin:0 0 16px;">
            Backbone, Neck, and Head
        </p>
        <p style="color:#757575;font-size:0.9rem;line-height:1.55;margin-top:-6px;">
            YOLO processes the image in three major architectural stages. The backbone
            extracts visual features, the neck fuses those features across scales, and
            the head converts them into {task_output}.
        </p>
        """,
        unsafe_allow_html=True,
    )

    _render_arch_card(
        title=notes["backbone_title"],
        subtitle="1. Backbone",
        body=notes["backbone"],
        points=notes["backbone_points"],
        accent="#4FC3F7",
    )
    _render_arch_card(
        title=notes["neck_title"],
        subtitle="2. Neck",
        body=notes["neck"],
        points=notes["neck_points"],
        accent="#66BB6A",
    )
    _render_arch_card(
        title=notes["head_title"],
        subtitle="3. Head",
        body=notes["head"],
        points=notes["head_points"],
        accent="#FFA726",
    )

    st.divider()

    with st.expander("How the selected task changes the output", expanded=True):
        if selected_task == "Object Detection":
            st.markdown(
                """
**Object Detection** uses the YOLO head to identify each architectural component
with a bounding box, class label, and confidence score. This is best when the goal
is to count and locate components such as roofs, columns, walls, doors, windows,
stairs, and ornaments.
"""
            )
        else:
            st.markdown(
                """
**Instance Segmentation** starts from the same detection logic, then adds masks
that trace the visible shape of each component. This is useful when component
boundaries matter, for example separating a column from a wall or isolating
ornamental details from the surrounding structure.
"""
            )

    with st.expander("Version focus"):
        st.markdown(
            f"""
**{selected_version} focus for this project**

- **Backbone:** {notes["backbone_title"]}
- **Neck:** {notes["neck_title"]}
- **Head:** {notes["head_title"]}
- **Active task:** {selected_task}
- **Displayed output:** {task_output}
"""
        )
