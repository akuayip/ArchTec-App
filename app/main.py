"""
ArchTec Segmentation — Streamlit App
"""

import streamlit as st
from app.config import APP_CONFIG, MODEL_OPTIONS

st.set_page_config(
    page_title=APP_CONFIG["page_title"],
    page_icon=APP_CONFIG["page_icon"],
    layout=APP_CONFIG["layout"],
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .app-hero { padding: 2rem 0 1.5rem; border-bottom: 1px solid rgba(255,255,255,0.07); margin-bottom: 1.5rem; }
    .app-hero-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(79,195,247,0.12); border: 1px solid rgba(79,195,247,0.3);
        border-radius: 20px; padding: 4px 12px; font-size: 0.72rem;
        font-weight: 600; color: #4FC3F7; letter-spacing: 0.06em;
        text-transform: uppercase; margin-bottom: 12px;
    }
    .app-hero h1 { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em; margin: 0 0 8px; color: #f5f5f5; }
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
        <div class="app-hero-badge">🏛️ Instance Segmentation · YOLOv11</div>
        <h1>{APP_CONFIG['page_title']}</h1>
        <p>Deteksi dan segmentasi otomatis komponen bangunan tradisional —
           kolom, pintu, jendela, dan elemen struktural lainnya.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_detector, tab_training, tab_how = st.tabs([
    "🏛️  Detector",
    "📈  Training Results",
    "🔬  How YOLOv11 Works",
])

with tab_detector:
    from pages.tab_detector import render as render_detector
    render_detector()

with tab_training:
    from pages.tab_training import render as render_training
    # Baca csv sesuai model yang dipilih user di sidebar
    selected_key = st.session_state.get("selected_model_key", list(MODEL_OPTIONS.keys())[-1])
    csv_path = MODEL_OPTIONS[selected_key]["results"]
    render_training(csv_path=csv_path)

with tab_how:
    from pages.tab_how_it_works import render as render_how
    render_how()