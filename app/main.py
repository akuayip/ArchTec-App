"""
ArchTec Segmentation — Streamlit App
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from app.config import APP_CONFIG, DEFAULT_MODEL_VERSION, DEFAULT_TASK, MODEL_OPTIONS

st.set_page_config(
    page_title=APP_CONFIG["page_title"],
    page_icon=APP_CONFIG["page_icon"],
    layout=APP_CONFIG["layout"],
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .app-hero { padding: 2rem 0 1.5rem; border-bottom: 1px solid rgba(255,255,255,0.07); margin-bottom: 1.5rem; }
    .app-hero h1 { font-size: 2.75rem; font-weight: 800; letter-spacing: -0.03em; margin: 0 0 12px; color: #f5f5f5; }
    .app-hero p  { color: #757575; margin: 0; font-size: 1rem; line-height: 1.5; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: transparent; padding: 0;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px; padding: 0 18px; border-radius: 0;
        font-weight: 600; font-size: 0.88rem; color: #757575;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #4FC3F7 !important;
        border-bottom: 2px solid #4FC3F7 !important;
        background: transparent !important;
    }
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px; padding: 16px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero Header ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="app-hero">
        <h1>{APP_CONFIG['page_title']}</h1>
        <p>Powered by YOLO Object Detection and Instance Segmentation models (v8-v11), this system automatically identifies traditional architectural elements including roofs, columns, walls, doors, windows, stairs, and ornamental details. Users can select different YOLO versions and inference tasks to analyze building structures and visualize detected components in real time.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_detector, tab_training, tab_how = st.tabs([
    "🏛️  Detector",
    "📈  Training Results",
    "🔬  How YOLO Works",
])

with tab_detector:
    from pages.tab_detector import render as render_detector
    render_detector()

with tab_training:
    from pages.tab_training import render as render_training
    # Read the CSV that matches the model selected in the sidebar.
    default_model_key = f"{DEFAULT_TASK} · {DEFAULT_MODEL_VERSION}"
    selected_key = st.session_state.get("selected_model_key", default_model_key)
    if selected_key not in MODEL_OPTIONS:
        selected_key = default_model_key
    csv_path = MODEL_OPTIONS[selected_key]["results"]
    render_training(csv_path=csv_path)

with tab_how:
    from pages.tab_how_it_works import render as render_how
    render_how()
