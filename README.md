# ArchTec Segmentation

## 📋 Overview

**ArchTec Segmentation** adalah aplikasi deteksi dan segmentasi komponen bangunan tradisional menggunakan teknologi **YOLOv11**. Proyek ini menggabungkan kekuatan machine learning modern dengan antarmuka web yang user-friendly untuk mengidentifikasi elemen-elemen arsitektur pada gambar bangunan.

### Apa itu Proyek Ini?

Bayangkan Anda memiliki foto sebuah bangunan tradisional, dan ingin mengetahui bagian-bagian mana yang merupakan komponen khusus (seperti kolom, jendela, atap, dll). Aplikasi ini akan:

- 🖼️ **Menganalisis gambar** yang Anda upload
- 🎯 **Mendeteksi** komponen bangunan secara otomatis
- 📍 **Menandai area** masing-masing komponen dengan presisi tinggi
- 📊 **Menampilkan hasil** dalam antarmuka yang mudah dipahami

Aplikasi dibangun dengan stack modern:
- **Backend**: Python + PyTorch (untuk machine learning)
- **Frontend**: Streamlit (antarmuka web interaktif)
- **Model**: YOLOv11 (arsitektur deep learning terdepan untuk deteksi objek)

## 🚀 Cara Menjalankan

### Prasyarat

Sebelum memulai, pastikan Anda memiliki:
- **Python 3.9+** terinstall di sistem
- **Git** untuk cloning repository
- **uv** (package manager Python yang cepat) - [Install di sini](https://github.com/astral-sh/uv)

### Langkah-Langkah Menjalankan

#### 1️⃣ Persiapan Awal
```bash
# Clone repository
git clone <repository-url>
cd ArchTec-Segmentation

# Install semua dependencies
uv sync
```

#### 2️⃣ Verifikasi Lingkungan
```bash
# Pastikan PyTorch dan dependencies lainnya terinstall dengan benar
python scripts/check_env.py
```

Jika semua berjalan lancar, Anda akan melihat output yang mengonfirmasi instalasi PyTorch dan package lainnya.

#### 3️⃣ Jalankan Aplikasi
```bash
# Mulai server Streamlit
streamlit run app/main.py
```

Setelah ini, aplikasi akan otomatis terbuka di browser Anda pada `http://localhost:8501`

### Akses Aplikasi

- **URL Lokal**: http://localhost:8501
- **Fitur Utama**:
  - Pilih model (v1 atau v2)
  - Upload gambar bangunan
  - Atur confidence threshold & NMS IoU
  - Lihat hasil segmentasi real-time

## ⚙️ Konfigurasi

Semua pengaturan aplikasi dapat disesuaikan di file **`app/config.py`**:

| Parameter | Deskripsi |
|-----------|-----------|
| `MODEL_CONFIG["path"]` | Path ke file model weights (.pt) |
| `MODEL_CONFIG["default_conf"]` | Confidence threshold default (0-1) |
| `MODEL_CONFIG["default_iou"]` | NMS IoU threshold untuk mengurangi duplikat deteksi |

**Contoh pengaturan confidence threshold**: Nilai lebih tinggi = deteksi lebih ketat (hanya objek yang sangat yakin), nilai lebih rendah = deteksi lebih sensitif.

## 🧪 Menjalankan Tests

Untuk memastikan semuanya berfungsi dengan baik, jalankan unit tests:

```bash
uv run pytest
```

Ini akan menjalankan semua test di folder `tests/` dan melaporkan hasilnya.

## 📁 Struktur Direktori Detail

```
ArchTec-Segmentation/
│
├── app/                          # Layer UI (Streamlit)
│   ├── main.py                   # Entry point - jalankan dengan streamlit
│   ├── config.py                 # Konfigurasi aplikasi & model (edit di sini)
│   └── __init__.py
│
├── core/                         # Logic ML (loading model, inference, post-processing)
│   ├── predictor.py              # Class untuk melakukan prediksi
│   └── __init__.py
│
├── utils/                        # Helper functions
│   ├── device.py                 # Deteksi device PyTorch (CPU/GPU/MPS)
│   └── __init__.py
│
├── models/                       # Model weight files (.pt) - tidak di-track git
│   ├── train_v1.pt               # Model versi 1
│   ├── train_v2.pt               # Model versi 2
│   ├── results_v1.csv            # Hasil training v1
│   └── results_v2.csv            # Hasil training v2
│
├── pages/                        # Halaman-halaman Streamlit tambahan
│   ├── tab_detector.py           # Tab untuk deteksi/prediksi
│   ├── tab_how_it_works.py       # Tab penjelasan cara kerja
│   ├── tab_training.py           # Tab info training
│   └── __init__.py
│
├── tests/                        # Unit tests dengan pytest
│   ├── test_predictor.py         # Test untuk predictor
│   └── __init__.py
│
├── scripts/                      # Script one-off untuk development/ops
│   └── check_env.py              # Verifikasi PyTorch & dependency
│
├── assets/                       # Aset statis (gambar, dll)
│
├── pyproject.toml                # Dependency configuration (uv)
│
└── README.md                     # File ini
```

## 💡 Tips & Troubleshooting

### GPU tidak terdeteksi?
Jalankan `python scripts/check_env.py` untuk diagnostik. Pastikan CUDA compatible GPU driver terinstall.

### Dependencies tidak ter-install?
```bash
# Clear cache dan reinstall
rm -rf .venv
uv sync --refresh
```

### Port 8501 sudah digunakan?
```bash
streamlit run app/main.py --server.port 8502
```

## 📞 Support

Untuk pertanyaan atau issue, silakan buka GitHub issue atau hubungi tim development.

---

**Terakhir diupdate**: 2026  
**Versi Python**: 3.9+  
**License**: [Sesuaikan sesuai kebutuhan]
