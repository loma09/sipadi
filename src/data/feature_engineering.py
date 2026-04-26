"""
feature_engineering.py
=======================
Membuat fitur-fitur untuk model ML dari master_dataset_raw.csv.
Output: data/processed/master_dataset.csv (siap untuk training)

Cara pakai:
    python -m src.data.feature_engineering
    dari src.data.feature_engineering import run_feature_engineering
    run_feature_engineering()
"""

import logging
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Rata-rata historis referensi (dari domain knowledge)
RATA_RATA_CURAH_HUJAN = 600.0  # mm/musim
RATA_RATA_PRODUKTIVITAS = 5.5  # ton/ha


# ─────────────────────────────────────────────
# FEATURE GROUPS
# ─────────────────────────────────────────────

def add_climate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fitur turunan iklim dan ENSO."""
    logger.info("Membuat fitur iklim...")

    if 'curah_hujan_total' in df.columns:
        df['anomali_curah_hujan'] = df['curah_hujan_total'] - RATA_RATA_CURAH_HUJAN
        df['pct_anomali_hujan']   = (df['anomali_curah_hujan'] / RATA_RATA_CURAH_HUJAN) * 100
        df['curah_hujan_sqrt']    = np.sqrt(df['curah_hujan_total'].clip(0))
        df['kekeringan_flag']     = (df['curah_hujan_total'] < 300).astype(int)
        df['banjir_flag']         = (df['curah_hujan_total'] > 1200).astype(int)

    if 'enso_mean' in df.columns:
        df['el_nino_severity'] = df['enso_mean'].clip(0).rename('el_nino_severity')
        df['la_nina_severity'] = (-df['enso_mean']).clip(0).rename('la_nina_severity')
        df['enso_squared']     = df['enso_mean'] ** 2

    if 'suhu_mean' in df.columns:
        df['suhu_anomali'] = df['suhu_mean'] - 27.5  # rata-rata normal
        df['suhu_tinggi_flag'] = (df['suhu_mean'] > 30).astype(int)

    if 'kelembaban_mean' in df.columns:
        df['kelembaban_rendah_flag'] = (df['kelembaban_mean'] < 65).astype(int)

    return df


def add_ndvi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fitur turunan NDVI (kesehatan lahan)."""
    logger.info("Membuat fitur NDVI...")

    if 'ndvi_mean' in df.columns:
        df['ndvi_kategori'] = pd.cut(
            df['ndvi_mean'],
            bins=[-0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)
        df['ndvi_lahan_baik'] = (df['ndvi_mean'] >= 0.5).astype(int)

        if 'ndvi_std' in df.columns:
            df['ndvi_variabilitas_tinggi'] = (df['ndvi_std'] > 0.1).astype(int)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag features produktivitas per kabupaten (time series)."""
    logger.info("Membuat lag features...")

    if 'produktivitas_ton_per_ha' not in df.columns:
        return df

    df = df.sort_values(['kabupaten', 'tahun', 'musim_tanam'] if 'musim_tanam' in df.columns else ['kabupaten', 'tahun'])

    for lag in [1, 2, 3]:
        col_name = f'prod_lag{lag}'
        df[col_name] = df.groupby('kabupaten')['produktivitas_ton_per_ha'].shift(lag)

    df['prod_rolling_mean3'] = df.groupby('kabupaten')['produktivitas_ton_per_ha'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df['prod_rolling_std3'] = df.groupby('kabupaten')['produktivitas_ton_per_ha'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).std()
    )

    df['prod_yoy_change'] = df.groupby('kabupaten')['produktivitas_ton_per_ha'].pct_change() * 100

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fitur interaksi antar variabel."""
    logger.info("Membuat interaction features...")

    if 'enso_mean' in df.columns and 'curah_hujan_total' in df.columns:
        df['enso_x_hujan'] = df['enso_mean'] * df['curah_hujan_total']

    if 'ndvi_mean' in df.columns and 'skor_irigasi' in df.columns:
        df['ndvi_x_irigasi'] = df['ndvi_mean'] * df['skor_irigasi']

    if 'pct_lahan_sangat_baik' in df.columns and 'skor_irigasi' in df.columns:
        df['lahan_irigasi_score'] = df['pct_lahan_sangat_baik'] * df['skor_irigasi']

    if 'jumlah_pompa_unit' in df.columns and 'luas_panen_ha' in df.columns:
        df['kapasitas_pompa_per_ha'] = df['jumlah_pompa_unit'] * 80 / df['luas_panen_ha'].replace(0, np.nan)
        df['kapasitas_pompa_per_ha'] = df['kapasitas_pompa_per_ha'].fillna(0)

    return df


def add_target_and_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buat target variabel dan encoding kategorikal.

    Target:
    - produktivitas_ton_per_ha  → regresi (sudah ada)
    - risiko_gagal_panen        → klasifikasi (1 jika prod < threshold)
    - harga_yoy_change          → diambil dari data harga
    """
    logger.info("Membuat target & encoding...")

    # Target klasifikasi: gagal panen jika produktivitas < 4.5 ton/ha
    if 'produktivitas_ton_per_ha' in df.columns:
        df['risiko_gagal_panen'] = (
            df['produktivitas_ton_per_ha'] < 4.5
        ).astype(int)

    # Encoding musim tanam
    if 'musim_tanam' in df.columns:
        df['musim_tanam_enc'] = df['musim_tanam'].map({'MT1': 1, 'MT2': 0}).fillna(0).astype(int)

    # Encoding provinsi
    if 'provinsi' in df.columns:
        df['provinsi_enc'] = (df['provinsi'] == 'Jawa Tengah').astype(int)

    # Tahun sebagai fitur tren
    if 'tahun' in df.columns:
        df['tahun_norm']     = df['tahun'] - df['tahun'].min()
        df['era_pompanisasi'] = (df['tahun'] >= 2022).astype(int)

    # Harga YoY change
    if 'harga_mean' in df.columns:
        df['harga_yoy_change'] = df.groupby('kabupaten')['harga_mean'].pct_change() * 100
        df['rasio_gabah_beras'] = 0.45  # konstanta konversi gabah→beras

    # Import dummy (jika tidak ada kolom asli)
    if 'impor_volume' not in df.columns:
        df['impor_volume']   = 500000
        df['harga_impor_idr'] = 8500000

    return df


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bersihkan NaN sisa, pastikan semua fitur numerik.
    Simpan daftar feature columns ke disk.
    """
    logger.info("Finalisasi dataset...")

    # Kolom non-fitur
    drop_cols = ['kabupaten', 'provinsi', 'musim_tanam', 'tanggal']
    drop_cols = [c for c in drop_cols if c in df.columns]

    # Isi NaN dengan median
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    logger.info(f"Dataset final: {len(df)} baris, {len(df.columns)} kolom")
    return df


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run_feature_engineering() -> pd.DataFrame:
    """
    Jalankan pipeline feature engineering lengkap.

    Returns:
        DataFrame siap training
    """
    logger.info("=" * 55)
    logger.info("SiPADI — Feature Engineering Pipeline")
    logger.info("=" * 55)

    df = pd.read_csv(PROCESSED_DIR / "master_dataset_raw.csv")
    logger.info(f"Input: {len(df)} baris, {len(df.columns)} kolom")

    df = add_climate_features(df)
    df = add_ndvi_features(df)
    df = add_lag_features(df)
    df = add_interaction_features(df)
    df = add_target_and_encoding(df)
    df = finalize_features(df)

    df.to_csv(PROCESSED_DIR / "master_dataset.csv", index=False)
    logger.info("✅ Feature engineering selesai → data/processed/master_dataset.csv")
    return df


if __name__ == "__main__":
    run_feature_engineering()