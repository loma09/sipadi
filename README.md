### SIPADI
### Sistem Prediksi Anomali & Deteksi Risiko Pangan Indonesia

[![Azure](https://img.shields.io/badge/Microsoft_Azure-5_Services-0078D4?logo=microsoftazure&logoColor=white)](https://azure.microsoft.com)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![AI Impact Challenge](https://img.shields.io/badge/Microsoft_Elevate-AI_Impact_Challenge_2026-0078D4)](https://www.dicoding.com/challenges/list)

---

> **SiPADI** adalah sistem deteksi dini risiko defisit pangan berbasis anomali iklim,
> citra satelit, dan analisis sentimen berita untuk mendukung keputusan operasional
> **Bulog** dan **Kementan** di level kabupaten Jawa Tengah & Jawa Timur.
> Dibangun di atas **Microsoft Azure** sebagai kontribusi nyata terhadap program
> **Swasembada Pangan 2025–2029** Presiden Prabowo Subianto.

---
Live Demo : https://sipadii.streamlit.app/
##  Problem Statement

Indonesia menghadapi ancaman defisit pangan yang kompleks akibat:
- Anomali iklim El Niño/La Niña yang tidak terprediksi
- Degradasi kualitas lahan pertanian di Jawa
- Volatilitas harga beras yang merugikan petani & konsumen
- Keterlambatan deteksi risiko gagal panen di level kabupaten

**SiPADI hadir sebagai solusi early warning system berbasis AI** yang mampu memprediksi
produktivitas padi, mendeteksi risiko gagal panen, dan memforecast harga beras
hingga 12 bulan ke depan — semua dalam satu dashboard interaktif.

---

##  Arsitektur Sistem
```bash
┌─────────────────────────────────────────────────────────┐
│                    DATA SOURCES                         │
│  BPS · BMKG · NOAA · Sentinel-2 · Kementan · Kemendag  │
└──────────────────────┬──────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────┐
│              AZURE BLOB STORAGE                         │
│      Raw Zone · Processed Zone · Model Zone             │
└──────────────────────┬──────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────┐
│                  AI & ML LAYER                          │
│                                                         │
│  Azure Machine Learning    Azure AI Language            │
│  · Experiment Tracking     · Sentiment Analysis         │
│  · Model Registry          · Berita Pangan NLP          │
│                                                         │
│  Azure Maps                                             │
│  · Peta Risiko Kabupaten                               │
│  · Visualisasi Spasial                                  │
└──────────────────────┬──────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────┐
│                   3 MODEL AI                            │
│                                                         │
│  Model 1: Prediksi Produktivitas  → XGBoost Regressor  │
│  Model 2: Deteksi Risiko          → XGBoost Classifier  │
│  Model 3: Forecast Harga Beras    → SARIMA              │
└──────────────────────┬──────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────┐
│         DASHBOARD — Azure Static Web Apps               │
│  · Peta risiko interaktif per kabupaten                 │
│  · Prediksi real-time dengan input parameter            │
│  · Forecast harga 12 bulan                             │
│  · Analisis sentimen berita pangan                      │
│  · Rekomendasi strategis Bulog & Kementan               │
└─────────────────────────────────────────────────────────┘
```
---

##  Performa Model AI

| Model | Algoritma | Metrik | Nilai |
|-------|-----------|--------|-------|
| Prediksi Produktivitas Padi | XGBoost Regressor | R² | **0.886** |
| Prediksi Produktivitas Padi | XGBoost Regressor | MAPE | **2.25%** |
| Deteksi Risiko Gagal Panen | XGBoost Classifier | F1-Score | **0.833** |
| Deteksi Risiko Gagal Panen | XGBoost Classifier | ROC-AUC | **0.967** |
| Forecast Harga Beras | SARIMA(1,1,1)(1,1,1,12) | MAPE | **1.16%** |

---

##  Microsoft Azure Services

| Service | Fungsi | Tier |
|---------|--------|------|
| **Azure Blob Storage** | Data lake — 7 dataset, 11.616 records | Free (5GB) |
| **Azure Machine Learning** | Experiment tracking, model registry | Free quota |
| **Azure AI Language** | Sentiment analysis berita pangan | Free (F0) |
| **Azure Maps** | Visualisasi peta risiko kabupaten | Free (1000 tx/hari) |
| **Azure Static Web Apps** | Deploy dashboard publik | Free |

---

##  Dataset

| Dataset | Sumber | Records | Periode |
|---------|--------|---------|---------|
| Produksi & Luas Panen Padi | BPS | 1.624 | 2010–2023 |
| Curah Hujan & Suhu | BMKG | 1.680 | 2010–2023 |
| Indeks ENSO | NOAA | 420 | 1990–2023 |
| Harga Beras | Kemendagri | 1.344 | 2010–2023 |
| NDVI Lahan | Sentinel-2 | 5.568 | 2016–2023 |
| Irigasi & Pompanisasi | Kementan | 812 | 2010–2023 |
| Impor Beras | Kemendag | 168 | 2010–2023 |
| **Total** | **7 sumber** | **11.616** | **2010–2023** |

---


##  Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/loma09/sipadi.git
cd sipadi
```

### 2. Setup Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Konfigurasi Azure
```bash
cp .env.example .env
# Isi credentials Azure di file .env
```

### 4. Jalankan Dashboard
```bash
streamlit run src/dashboard/app.py
```

### 5. Jalankan Notebook
```bash
jupyter notebook
# Jalankan notebook secara berurutan: 01 → 07
```

---

##  Key Insights

1. **Program Pompanisasi Efektif** — Korelasi jumlah pompa dengan produktivitas mencapai 0.708, membuktikan program Kementan sejak 2020 berdampak signifikan

2. **NDVI sebagai Early Warning** — Kesehatan lahan (NDVI) adalah prediktor terkuat produktivitas padi dengan korelasi 0.188 di level kabupaten

3. **Musim Tanam 1 Lebih Produktif** — MT1 (Jan–Apr) rata-rata 0.5 ton/ha lebih tinggi dari MT2 karena ketersediaan air lebih baik

4. **Risiko Lonjakan Harga Jul–Sep** — Model SARIMA mendeteksi pola kenaikan harga konsisten setiap musim kemarau, rekomendasi Bulog untuk stok preventif

5. **25% Kabupaten Berisiko** — 1 dari 4 kabupaten di Jateng & Jatim memiliki probabilitas gagal panen di atas threshold, membutuhkan intervensi segera

---

##  Relevansi Kebijakan

SiPADI dirancang langsung untuk mendukung program pemerintah:

| Program Pemerintah | Kontribusi SiPADI |
|-------------------|-------------------|
| Swasembada Pangan 2025–2029 | Prediksi produktivitas & deteksi risiko gagal panen |
| Pompanisasi Kementan | Evaluasi dampak pompanisasi per kabupaten |
| Stabilisasi Harga Bulog | Forecast harga 12 bulan untuk pembelian preventif |
| Lumbung Pangan Nasional | Identifikasi kabupaten surplus untuk pengembangan |

---

##  Tim

**Ahmad** — Data Scientist & AI Solution Architect
Microsoft Elevate Training Center · AI Impact Challenge 2026

---

