"""
Tab: Training Results
Visualizes loss curves, precision/recall/mAP from results.csv
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path


# ── Data Loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def find_results_csv() -> str | None:
    candidates = [
        "results.csv",
        "training/results.csv",
        "runs/segment/train/results.csv",
        "runs/detect/train/results.csv",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


# ── Chart Helpers ──────────────────────────────────────────────────────────────

COLORS = {
    "train": "#4FC3F7",
    "val":   "#FF7043",
    "map50": "#66BB6A",
    "map5095": "#AB47BC",
    "precision": "#FFA726",
    "recall": "#26C6DA",
}


def _line(fig, df, col, name, color, row, col_idx, dash="solid"):
    if col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["epoch"], y=df[col],
                name=name, line=dict(color=color, width=2, dash=dash),
                hovertemplate=f"Epoch %{{x}}<br>{name}: %{{y:.4f}}<extra></extra>",
            ),
            row=row, col=col_idx,
        )


# ── Render Function ────────────────────────────────────────────────────────────

def render(csv_path: str | None = None) -> None:
    st.markdown("## 📈 Training Results")
    st.caption("Semua grafik berasal dari `results.csv` yang dihasilkan Ultralytics YOLOv11.")

    # ── Load data ──────────────────────────────────────────────────────────────
    path = csv_path or find_results_csv()

    if path is None:
        uploaded = st.file_uploader(
            "Upload `results.csv` dari folder training Anda",
            type="csv",
            help="Biasanya ada di runs/segment/train/results.csv",
        )
        if uploaded is None:
            st.info("⬆️ Upload file `results.csv` untuk melihat grafik training.")
            return
        df = pd.read_csv(uploaded)
        df.columns = df.columns.str.strip()
    else:
        df = load_results(path)

    total_epochs = len(df)
    best_map50_row = df.loc[df["metrics/mAP50(B)"].idxmax()]
    best_map50 = best_map50_row["metrics/mAP50(B)"]
    best_epoch = int(best_map50_row["epoch"])

    # ── Summary cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Epoch", total_epochs)
    c2.metric("Best mAP@50", f"{best_map50:.4f}", f"epoch {best_epoch}")
    c3.metric(
        "Best mAP@50-95",
        f"{df['metrics/mAP50-95(B)'].max():.4f}",
        f"epoch {int(df.loc[df['metrics/mAP50-95(B)'].idxmax(), 'epoch'])}",
    )
    c4.metric(
        "Final Precision",
        f"{df['metrics/precision(B)'].iloc[-1]:.4f}",
    )

    st.divider()

    # ── Epoch range slider ─────────────────────────────────────────────────────
    epoch_range = st.slider(
        "Filter Rentang Epoch",
        min_value=1,
        max_value=total_epochs,
        value=(1, total_epochs),
        step=1,
    )
    df_view = df[(df["epoch"] >= epoch_range[0]) & (df["epoch"] <= epoch_range[1])]

    # ── 1. Loss Curves ─────────────────────────────────────────────────────────
    st.subheader("📉 Loss Curves")
    loss_fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Box Loss", "Class Loss", "DFL Loss"),
        shared_xaxes=True,
    )
    for i, loss_key in enumerate(["box_loss", "cls_loss", "dfl_loss"], start=1):
        _line(loss_fig, df_view, f"train/{loss_key}", f"Train", COLORS["train"], 1, i)
        _line(loss_fig, df_view, f"val/{loss_key}",   f"Val",   COLORS["val"],   1, i, dash="dot")

    loss_fig.update_layout(
        height=320,
        showlegend=True,
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        margin=dict(t=40, b=80, l=40, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(gridcolor="#2a2a2a"), yaxis=dict(gridcolor="#2a2a2a"),
        xaxis2=dict(gridcolor="#2a2a2a"), yaxis2=dict(gridcolor="#2a2a2a"),
        xaxis3=dict(gridcolor="#2a2a2a"), yaxis3=dict(gridcolor="#2a2a2a"),
    )
    for ann in loss_fig.layout.annotations:
        ann.font.color = "#9e9e9e"
    st.plotly_chart(loss_fig, use_container_width=True)

    # ── 2. mAP Curves ─────────────────────────────────────────────────────────
    st.subheader("🎯 mAP Curves")
    map_fig = go.Figure()
    map_fig.add_trace(go.Scatter(
        x=df_view["epoch"], y=df_view["metrics/mAP50(B)"],
        name="mAP@50", fill="tozeroy",
        fillcolor="rgba(102,187,106,0.15)",
        line=dict(color=COLORS["map50"], width=2.5),
        hovertemplate="Epoch %{x}<br>mAP@50: %{y:.4f}<extra></extra>",
    ))
    map_fig.add_trace(go.Scatter(
        x=df_view["epoch"], y=df_view["metrics/mAP50-95(B)"],
        name="mAP@50-95", fill="tozeroy",
        fillcolor="rgba(171,71,188,0.10)",
        line=dict(color=COLORS["map5095"], width=2.5),
        hovertemplate="Epoch %{x}<br>mAP@50-95: %{y:.4f}<extra></extra>",
    ))
    # Best epoch marker
    map_fig.add_vline(
        x=best_epoch,
        line_dash="dash", line_color="#FFD54F", line_width=1.5,
        annotation_text=f"Best epoch {best_epoch}",
        annotation_font_color="#FFD54F",
    )
    map_fig.update_layout(
        height=300,
        margin=dict(t=20, b=40, l=40, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        xaxis=dict(gridcolor="#2a2a2a", title="Epoch"),
        yaxis=dict(gridcolor="#2a2a2a", title="mAP"),
    )
    st.plotly_chart(map_fig, use_container_width=True)

    # ── 3. Precision & Recall ──────────────────────────────────────────────────
    st.subheader("⚖️ Precision & Recall")
    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(
        x=df_view["epoch"], y=df_view["metrics/precision(B)"],
        name="Precision", line=dict(color=COLORS["precision"], width=2),
        hovertemplate="Epoch %{x}<br>Precision: %{y:.4f}<extra></extra>",
    ))
    pr_fig.add_trace(go.Scatter(
        x=df_view["epoch"], y=df_view["metrics/recall(B)"],
        name="Recall", line=dict(color=COLORS["recall"], width=2),
        hovertemplate="Epoch %{x}<br>Recall: %{y:.4f}<extra></extra>",
    ))
    pr_fig.update_layout(
        height=280,
        margin=dict(t=20, b=40, l=40, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        xaxis=dict(gridcolor="#2a2a2a", title="Epoch"),
        yaxis=dict(gridcolor="#2a2a2a", title="Score", range=[0, 1]),
    )
    st.plotly_chart(pr_fig, use_container_width=True)

    # ── 4. Learning Rate ───────────────────────────────────────────────────────
    with st.expander("📊 Learning Rate Schedule"):
        lr_fig = go.Figure()
        lr_fig.add_trace(go.Scatter(
            x=df_view["epoch"], y=df_view["lr/pg0"],
            name="LR", line=dict(color="#78909C", width=1.5),
            hovertemplate="Epoch %{x}<br>LR: %{y:.2e}<extra></extra>",
        ))
        lr_fig.update_layout(
            height=220,
            margin=dict(t=10, b=40, l=60, r=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            xaxis=dict(gridcolor="#2a2a2a", title="Epoch"),
            yaxis=dict(gridcolor="#2a2a2a", title="Learning Rate", tickformat=".2e"),
        )
        st.plotly_chart(lr_fig, use_container_width=True)

    # ── 5. Raw Data Table ──────────────────────────────────────────────────────
    with st.expander("🗃️ Raw Data"):
        display_cols = [
            "epoch",
            "train/box_loss", "train/cls_loss", "train/dfl_loss",
            "val/box_loss", "val/cls_loss", "val/dfl_loss",
            "metrics/precision(B)", "metrics/recall(B)",
            "metrics/mAP50(B)", "metrics/mAP50-95(B)",
        ]
        st.dataframe(
            df_view[display_cols].set_index("epoch").style.format("{:.4f}"),
            use_container_width=True,
        )
