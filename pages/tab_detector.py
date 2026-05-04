"""
Tab: Detector
Upload image → run inference → show results.
Pilihan model disimpan ke session_state supaya tab Training Results ikut berubah.
"""

import streamlit as st
from PIL import Image
import pandas as pd
from collections import Counter
from pathlib import Path

from core.predictor import load_model, run_inference, draw_results, parse_detections
from app.config import MODEL_CONFIG, MODEL_OPTIONS


@st.cache_resource
def get_model(model_path: str):
    with st.spinner(f"Loading model dari `{model_path}`..."):
        return load_model(model_path)


def _render_sidebar_controls() -> dict:
    with st.sidebar:
        st.markdown("### ⚙️ Inference Settings")
        st.divider()

        # ── Model selector ─────────────────────────────────────────────────────
        available = {
            name: cfg
            for name, cfg in MODEL_OPTIONS.items()
            if Path(cfg["model"]).exists()
        }

        if not available:
            st.warning("⚠️ Tidak ada model ditemukan di folder `models/`.")
            selected_key = list(MODEL_OPTIONS.keys())[-1]
        else:
            selected_key = st.selectbox(
                "Pilih Model",
                options=list(available.keys()),
                index=len(available) - 1,
                help="Pilih versi model · Training Results akan menyesuaikan otomatis",
            )

        # Simpan ke session_state supaya tab Training bisa baca
        st.session_state["selected_model_key"] = selected_key
        model_path    = MODEL_OPTIONS[selected_key]["model"]
        results_path  = MODEL_OPTIONS[selected_key]["results"]

        # Info csv aktif
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
            help="Minimum confidence untuk menampilkan deteksi",
        )
        iou_threshold = st.slider(
            "IoU Threshold (NMS)",
            min_value=0.1, max_value=1.0,
            value=MODEL_CONFIG["default_iou"], step=0.05,
            help="Threshold NMS untuk menghapus bbox yang tumpang tindih",
        )
        show_table = st.toggle("Tampilkan Tabel Deteksi", value=True)

        st.divider()
        st.markdown(
            f"""
            <div style="font-size:0.8rem; color:#616161; line-height:1.8;">
                <div><span style="color:#9e9e9e;">Model</span> &nbsp; {MODEL_CONFIG['name']}</div>
                <div><span style="color:#9e9e9e;">Task</span> &nbsp;&nbsp; {MODEL_CONFIG['task']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "conf_threshold": conf_threshold,
        "iou_threshold":  iou_threshold,
        "show_table":     show_table,
        "model_path":     model_path,
        "results_path":   results_path,
        "selected_key":   selected_key,
    }


def _render_upload_zone() -> object:
    st.markdown(
        """
        <div style="margin: 1.5rem 0 0.5rem;">
            <p style="font-size:0.78rem; font-weight:600; letter-spacing:0.06em;
                      text-transform:uppercase; color:#616161; margin:0 0 8px;">
                Upload Gambar
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.file_uploader(
        "Pilih gambar bangunan tradisional",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="Format: JPG, JPEG, PNG · Maks 200MB",
    )


def _render_summary(detections: list[dict], show_table: bool) -> None:
    if not detections:
        st.warning("Tidak ada komponen terdeteksi. Coba turunkan Confidence Threshold.")
        return

    st.divider()
    st.markdown(
        """<p style="font-size:0.78rem; font-weight:600; letter-spacing:0.06em;
                    text-transform:uppercase; color:#616161; margin:0 0 12px;">
            Ringkasan Deteksi
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
                Detail Deteksi
            </p>""",
            unsafe_allow_html=True,
        )
        df = pd.DataFrame([
            {
                "Komponen":            d["class_name"].capitalize(),
                "Confidence":          f"{d['confidence'] * 100:.1f}%",
                "Segmentation Mask":   "✅" if d["has_mask"] else "❌",
                "BBox [x1,y1,x2,y2]": [round(v) for v in d["bbox_xyxy"]],
            }
            for d in detections
        ])
        st.dataframe(df, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """<p style="font-size:0.78rem; font-weight:600; letter-spacing:0.06em;
                    text-transform:uppercase; color:#616161; margin:0 0 8px;">
            Confidence per Deteksi
        </p>""",
        unsafe_allow_html=True,
    )
    chart_data = pd.DataFrame({
        "Komponen":   [d["class_name"] for d in detections],
        "Confidence": [d["confidence"] for d in detections],
    })
    st.bar_chart(chart_data.set_index("Komponen"))


def render() -> None:
    settings = _render_sidebar_controls()

    # ── Model loading ──────────────────────────────────────────────────────────
    try:
        model = get_model(settings["model_path"])
        st.sidebar.success(f"✅ {settings['selected_key'].strip()} siap")
    except Exception as e:
        st.sidebar.error("❌ Gagal memuat model")
        st.error(f"**Model tidak ditemukan** di `{settings['model_path']}`\n\n`{e}`")
        return

    # ── Tab content header ─────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="padding: 1.5rem 0 0.5rem;">
            <p style="font-size:0.78rem; font-weight:600; letter-spacing:0.06em;
                      text-transform:uppercase; color:#4FC3F7; margin:0 0 6px;">
                Deteksi Komponen
            </p>
            <h2 style="font-size:1.5rem; font-weight:700; margin:0 0 6px;
                       color:#f0f0f0; letter-spacing:-0.02em;">
                Upload Foto Bangunan
            </h2>
            <p style="color:#616161; margin:0; font-size:0.9rem;">
                Menggunakan <b style="color:#9e9e9e;">{settings['selected_key'].strip()}</b> ·
                Model akan otomatis mendeteksi dan mensegmentasi komponen struktural.
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
                    Upload foto bangunan tradisional untuk memulai deteksi
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown(
            """<p style="font-size:0.78rem; font-weight:600; letter-spacing:0.06em;
                        text-transform:uppercase; color:#616161; margin:0 0 8px;">
                Gambar Asli
            </p>""",
            unsafe_allow_html=True,
        )
        st.image(image, use_container_width=True)

    with st.spinner("🔍 Mendeteksi komponen bangunan..."):
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
                Hasil Deteksi
                <span style="color:#4FC3F7; margin-left:8px;">{len(detections)} objek</span>
            </p>""",
            unsafe_allow_html=True,
        )
        st.image(annotated_image, use_container_width=True)

    _render_summary(detections, show_table=settings["show_table"])