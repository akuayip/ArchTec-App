"""
Tab: How YOLOv11 Works
- Bagian atas: gambar pipeline statis (jika ada)
- Bagian bawah: stepper interaktif 7 langkah
"""

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path


STEPS = [
    {
        "num": 1,
        "icon": "🖼️",
        "color": "#2196F3",
        "title": "Input",
        "subtitle": "Foto Bangunan Tradisional",
        "description": (
            "Gambar mentah bangunan tradisional (JPG/PNG) diberikan sebagai input ke sistem. "
            "Foto bisa diambil dari berbagai sudut dan kondisi pencahayaan."
        ),
        "details": [
            "Format: JPG, JPEG, PNG",
            "Ukuran bebas — akan di-resize otomatis",
            "Bisa berisi 1 atau lebih komponen bangunan",
            "Tidak perlu preprocessing manual dari user",
        ],
        "analogy": "Seperti memberikan foto bangunan ke seorang arsitek untuk dianalisis.",
    },
    {
        "num": 2,
        "icon": "⚙️",
        "color": "#4CAF50",
        "title": "Preprocessing",
        "subtitle": "Persiapan Gambar",
        "description": (
            "Gambar diubah ukurannya menjadi **640×640 piksel** menggunakan teknik "
            "*letterboxing* agar aspek rasio asli tetap terjaga. "
            "Nilai piksel dinormalisasi ke rentang [0, 1] untuk stabilitas training."
        ),
        "details": [
            "Resize ke 640×640 dengan letterboxing",
            "Normalisasi piksel: nilai ÷ 255 → [0.0, 1.0]",
            "Konversi warna: BGR → RGB",
            "Ekstraksi fitur awal dari konvolusi pertama",
        ],
        "analogy": "Seperti mencetak foto ke ukuran standar dan mengatur kontrasnya sebelum dianalisis.",
    },
    {
        "num": 3,
        "icon": "🧠",
        "color": "#FF9800",
        "title": "Backbone — C3k2",
        "subtitle": "Ekstraksi Fitur",
        "description": (
            "Backbone **C3k2** (Cross Stage Partial dengan kernel 2) mengekstrak fitur "
            "dari level rendah hingga tinggi secara bertahap. "
            "Setiap layer mendeteksi pola yang semakin kompleks."
        ),
        "details": [
            "Layer awal: deteksi tepi & tekstur",
            "Layer tengah: deteksi bentuk & pola",
            "Layer akhir: deteksi konteks & semantik",
            "C3k2 lebih ringan dari C3 dengan akurasi setara",
        ],
        "analogy": "Seperti mata yang memindai bangunan — pertama melihat garis, lalu mengenali bentuk, lalu memahami konteks.",
    },
    {
        "num": 4,
        "icon": "🔗",
        "color": "#9C27B0",
        "title": "Neck — C2PSA + FPN",
        "subtitle": "Fusi Fitur Multi-skala",
        "description": (
            "Neck menggabungkan feature maps dari berbagai skala menggunakan "
            "**FPN** (Feature Pyramid Network) dan **C2PSA** "
            "(Cross Stage Partial with Positional-Sensitive Attention) "
            "untuk mendeteksi objek besar maupun kecil sekaligus."
        ),
        "details": [
            "P5 (20×20): deteksi objek besar seperti atap & dinding",
            "P4 (40×40): deteksi objek sedang seperti pintu",
            "P3 (80×80): deteksi objek kecil seperti jendela",
            "C2PSA: fokus perhatian ke area yang relevan",
        ],
        "analogy": "Seperti arsitek yang melihat bangunan dari jauh (keseluruhan) dan dekat (detail) secara bersamaan.",
    },
    {
        "num": 5,
        "icon": "🎯",
        "color": "#00BCD4",
        "title": "Detection Head",
        "subtitle": "Prediksi Komponen",
        "description": (
            "Head menghasilkan prediksi menggunakan **decoupled head** — "
            "branch klasifikasi dan regresi dipisah untuk akurasi lebih tinggi. "
            "Setiap sel grid menghasilkan kandidat bounding box dan mask."
        ),
        "details": [
            "Bounding box: koordinat [x1, y1, x2, y2]",
            "Kelas objek: roof, wall, window, column, dll.",
            "Confidence score: seberapa yakin model",
            "32 mask coefficients per deteksi untuk segmentasi",
        ],
        "analogy": "Seperti insinyur yang menunjuk lokasi komponen DAN menggambar konturnya.",
    },
    {
        "num": 6,
        "icon": "✂️",
        "color": "#FF5722",
        "title": "Post-processing — NMS",
        "subtitle": "Non-Maximum Suppression",
        "description": (
            "Model menghasilkan ribuan kandidat deteksi. "
            "**NMS** menyaring kandidat yang tumpang tindih berdasarkan "
            "skor confidence dan **IoU** (Intersection over Union), "
            "menyisakan satu deteksi terbaik per objek."
        ),
        "details": [
            "Filter bbox di bawah confidence threshold (default: 0.25)",
            "Hitung IoU antar bbox yang tumpang tindih",
            "Hapus bbox dengan IoU > threshold (default: 0.45)",
            "Hasilkan deteksi bersih tanpa duplikat",
        ],
        "analogy": "Seperti memilih satu foto terbaik dari banyak foto yang hampir identik.",
    },
    {
        "num": 7,
        "icon": "✅",
        "color": "#009688",
        "title": "Output",
        "subtitle": "Hasil Deteksi & Segmentasi",
        "description": (
            "Hasil akhir berupa **bounding box** berlabel dan **segmentation mask** "
            "berwarna di atas gambar asli. "
            "Setiap komponen ditampilkan dengan nama kelas dan skor confidence."
        ),
        "details": [
            "Bounding box: kotak yang melingkupi komponen",
            "Segmentation mask: kontur piksel yang tepat",
            "Class label: roof, wall, window, column, stairs, floor",
            "Confidence score: persentase keyakinan model",
        ],
        "analogy": "Hasil diagnosis arsitektural lengkap — setiap komponen teridentifikasi dan terpetakan.",
    },
]


def render() -> None:
    # ── Section header ─────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="padding: 1.5rem 0 1rem;">
            <p style="font-size:0.78rem;font-weight:600;letter-spacing:0.06em;
                      text-transform:uppercase;color:#4FC3F7;margin:0 0 6px;">
            </p>
            <h2 style="font-size:1.5rem;font-weight:700;margin:0 0 6px;
                       color:#f0f0f0;letter-spacing:-0.02em;">
                Visualisasi Cara Kerja dari Algoritma YOLOv11
            </h2>
            <p style="color:#616161;margin:0;font-size:0.9rem;">
                Dari input gambar bangunan tradisional hingga output deteksi komponen struktural.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Static pipeline image ──────────────────────────────────────────────────
    image_path = Path("assets/yolov11_pipeline.png")
    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
    else:
        st.info(
            "💡 Taruh file gambar pipeline di **`assets/yolov11_pipeline.png`** "
            "untuk menampilkan diagram di sini.",
            icon="🖼️",
        )

    st.divider()

    # ── Interactive stepper ────────────────────────────────────────────────────
    st.markdown(
        """
        <p style="font-size:0.78rem;font-weight:600;letter-spacing:0.06em;
                  text-transform:uppercase;color:#616161;margin:0 0 16px;">
            Penjelasan Step-by-Step
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Step selector — horizontal pills
    pill_html = '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:20px;">'
    for s in STEPS:
        pill_html += f"""
        <div style="
            background:rgba(255,255,255,0.04);
            border:1px solid rgba(255,255,255,0.1);
            border-radius:20px;
            padding:5px 14px;
            font-size:0.75rem;
            font-weight:600;
            color:#9e9e9e;
            cursor:pointer;
        ">
            <span style="color:{s['color']}">{s['num']}.</span> {s['title']}
        </div>
        """
    pill_html += "</div>"

    if "yolo_step" not in st.session_state:
        st.session_state.yolo_step = 1

    current = st.session_state.yolo_step
    step = STEPS[current - 1]

    # Navigation
    nav_cols = st.columns([1, 10, 1])
    with nav_cols[0]:
        if current > 1:
            if st.button("◀", key="yolo_prev", use_container_width=True):
                st.session_state.yolo_step -= 1
                st.rerun()
    with nav_cols[2]:
        if current < len(STEPS):
            if st.button("▶", key="yolo_next", use_container_width=True):
                st.session_state.yolo_step += 1
                st.rerun()

    # Step dot indicators
    dot_cols = st.columns(len(STEPS))
    for i, s in enumerate(STEPS):
        label = "●" if i + 1 == current else "○"
        if dot_cols[i].button(label, key=f"yolo_dot_{i}", use_container_width=True, help=s["title"]):
            st.session_state.yolo_step = i + 1
            st.rerun()

    # ── Step card ──────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {step['color']}15, {step['color']}05);
            border: 1px solid {step['color']}44;
            border-left: 4px solid {step['color']};
            border-radius: 12px;
            padding: 24px 28px;
            margin: 16px 0 12px;
        ">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
                <div style="
                    width:40px;height:40px;border-radius:50%;
                    background:{step['color']};
                    display:flex;align-items:center;justify-content:center;
                    font-size:1.2rem;flex-shrink:0;
                ">{step['icon']}</div>
                <div>
                    <div style="font-size:0.7rem;color:{step['color']};font-weight:700;
                                letter-spacing:0.08em;text-transform:uppercase;">
                        Step {step['num']} / {len(STEPS)}
                    </div>
                    <div style="font-size:1.25rem;font-weight:800;color:#f0f0f0;letter-spacing:-0.02em;">
                        {step['title']}
                    </div>
                    <div style="font-size:0.82rem;color:#757575;margin-top:1px;">
                        {step['subtitle']}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    desc_col, detail_col = st.columns([3, 2], gap="large")

    with desc_col:
        st.markdown(step["description"])
        st.markdown(
            f"""
            <div style="
                background:rgba(255,255,255,0.03);
                border-radius:8px;
                padding:12px 16px;
                margin-top:14px;
                border-left:3px solid {step['color']}99;
            ">
                <span style="font-size:0.75rem;color:#757575;">💡 </span>
                <em style="font-size:0.85rem;color:#757575;">{step['analogy']}</em>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with detail_col:
        st.markdown(
            f"""
            <p style="font-size:0.75rem;font-weight:700;letter-spacing:0.06em;
                      text-transform:uppercase;color:{step['color']};margin:0 0 10px;">
                Detail Teknis
            </p>
            """,
            unsafe_allow_html=True,
        )
        for item in step["details"]:
            st.markdown(
                f"""
                <div style="
                    display:flex;gap:8px;align-items:flex-start;
                    margin-bottom:8px;font-size:0.85rem;color:#bdbdbd;
                ">
                    <span style="color:{step['color']};margin-top:1px;flex-shrink:0;">▸</span>
                    <span>{item}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.progress(current / len(STEPS))

    st.divider()

    # ── Expandables ────────────────────────────────────────────────────────────
    with st.expander("📖 Penjelasan Metrik Training"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
**Loss Functions:**
- **Box Loss** — kesalahan koordinat bbox (CIoU)
- **Class Loss** — cross-entropy prediksi kelas
- **DFL Loss** — Distribution Focal Loss untuk presisi koordinat
""")
        with c2:
            st.markdown("""
**Metrics:**
- **Precision** — dari semua deteksi, berapa % benar
- **Recall** — dari semua objek, berapa % terdeteksi
- **mAP@50** — mean Average Precision pada IoU 0.50
- **mAP@50-95** — rata-rata mAP dari IoU 0.50–0.95
""")

    with st.expander("📊 YOLOv11 vs Versi Sebelumnya"):
        st.markdown("""
| Fitur | YOLOv8 | YOLOv10 | **YOLOv11** |
|---|---|---|---|
| Backbone | C2f | C2f | **C3k2** |
| Attention | ❌ | ❌ | **C2PSA** ✅ |
| Parameters | ~11M | ~8M | **~9M** |
| mAP COCO | 50.2 | 50.3 | **51.5** |
| Speed (T4) | 80ms | 93ms | **56ms** |
""")
        st.caption("Angka mengacu pada YOLOv11m pada COCO val2017.")