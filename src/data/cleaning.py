"""
cleaning.py
===========
Membersihkan data raw dari data/raw/ dan menyimpan ke data/processed/.

Cara pakai:
    python -m src.data.cleaning
    dari src.data.cleaning import run_cleaning
    run_cleaning()
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
RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# INDIVIDUAL CLEANERS
# ─────────────────────────────────────────────

def clean_produksi(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan data produksi padi."""
    logger.info("Cleaning: produksi_padi")
    df = df.copy()

    # Standardisasi kolom
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Drop duplikat
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"  Drop duplikat: {before - len(df)} baris")

    # Isi missing value numerik dengan median per kabupaten
    num_cols = ['produktivitas_ton_per_ha', 'produksi_ton', 'luas_panen_ha']
    for col in num_cols:
        if col in df.columns:
            df[col] = df.groupby('kabupaten')[col].transform(
                lambda x: x.fillna(x.median())
            )

    # Hapus outlier ekstrem (IQR 1.5x)
    for col in ['produktivitas_ton_per_ha']:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            mask = (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
            removed = (~mask).sum()
            df = df[mask]
            logger.info(f"  Outlier {col}: {removed} baris dihapus")

    # Pastikan tipe data
    df['tahun'] = df['tahun'].astype(int)
    df['kabupaten'] = df['kabupaten'].str.strip().str.title()
    df['provinsi']  = df['provinsi'].str.strip().str.title()

    logger.info(f"  Produksi clean: {len(df)} baris")
    return df


def clean_harga(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan data harga beras."""
    logger.info("Cleaning: harga_beras")
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.drop_duplicates()

    # Forward fill harga yang kosong (time series)
    if 'harga_beras_medium_per_kg' in df.columns:
        df = df.sort_values(['tahun', 'bulan'])
        df['harga_beras_medium_per_kg'] = df['harga_beras_medium_per_kg'].ffill().bfill()

    # Clip harga tidak masuk akal (< 5000 atau > 30000)
    df['harga_beras_medium_per_kg'] = df['harga_beras_medium_per_kg'].clip(5000, 30000)

    logger.info(f"  Harga clean: {len(df)} baris")
    return df


def clean_cuaca(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan data cuaca BMKG."""
    logger.info("Cleaning: cuaca_bmkg")
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.drop_duplicates()

    # Interpolasi missing curah hujan dan suhu
    num_cols = [c for c in df.columns if any(k in c for k in ['hujan', 'suhu', 'kelembaban'])]
    for col in num_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            df[col] = df[col].interpolate(method='linear').bfill().ffill()
            logger.info(f"  Interpolasi {col}: {missing} nilai diisi")

    # Clip nilai tidak masuk akal
    if 'suhu_mean' in df.columns:
        df['suhu_mean'] = df['suhu_mean'].clip(18, 38)
    if 'kelembaban_mean' in df.columns:
        df['kelembaban_mean'] = df['kelembaban_mean'].clip(40, 100)
    if 'curah_hujan_total' in df.columns:
        df['curah_hujan_total'] = df['curah_hujan_total'].clip(0, 3000)

    logger.info(f"  Cuaca clean: {len(df)} baris")
    return df


def clean_enso(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan data ENSO index."""
    logger.info("Cleaning: enso_index")
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.drop_duplicates()

    if 'tanggal' in df.columns:
        df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
        df = df.dropna(subset=['tanggal'])
        df['tahun'] = df['tanggal'].dt.year
        df['bulan'] = df['tanggal'].dt.month

    if 'enso_index' in df.columns:
        df['enso_index'] = df['enso_index'].clip(-4.0, 4.0)
        df['enso_index'] = df['enso_index'].interpolate(method='linear').bfill().ffill()

    logger.info(f"  ENSO clean: {len(df)} baris")
    return df


def clean_ndvi(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan data NDVI Sentinel."""
    logger.info("Cleaning: ndvi_sentinel")
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.drop_duplicates()

    if 'ndvi_mean' in df.columns:
        df['ndvi_mean'] = df['ndvi_mean'].clip(-0.1, 1.0)
        df['ndvi_mean'] = df['ndvi_mean'].interpolate(method='linear').bfill().ffill()

    logger.info(f"  NDVI clean: {len(df)} baris")
    return df


# ─────────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────────

def merge_all(
    df_produksi: pd.DataFrame,
    df_harga: pd.DataFrame,
    df_cuaca: pd.DataFrame,
    df_enso: pd.DataFrame,
    df_ndvi: pd.DataFrame,
) -> pd.DataFrame:
    """
    Gabungkan semua dataset menjadi satu master_dataset.
    Join key: kabupaten + tahun + (musim/bulan jika tersedia).
    """
    logger.info("Merging semua dataset...")

    # Base: produksi
    df = df_produksi.copy()

    # Merge harga (agregasi tahunan per kabupaten)
    if 'bulan' in df_harga.columns:
        harga_tahunan = df_harga.groupby(['tahun', 'kabupaten']).agg(
            harga_mean=('harga_beras_medium_per_kg', 'mean'),
            harga_lag1=('harga_beras_medium_per_kg', lambda x: x.shift(1).mean()),
            harga_volatility=('harga_beras_medium_per_kg', 'std'),
        ).reset_index()
    else:
        harga_tahunan = df_harga.groupby(['tahun', 'kabupaten']).agg(
            harga_mean=('harga_beras_medium_per_kg', 'mean'),
        ).reset_index()
        harga_tahunan['harga_lag1'] = harga_tahunan['harga_mean'].shift(1)
        harga_tahunan['harga_volatility'] = harga_tahunan['harga_mean'].std()

    df = df.merge(harga_tahunan, on=['tahun', 'kabupaten'], how='left')

    # Merge cuaca (agregasi tahunan per kabupaten)
    cuaca_cols = [c for c in df_cuaca.columns if c not in ['bulan', 'kabupaten', 'tahun', 'provinsi']]
    if cuaca_cols:
        cuaca_tahunan = df_cuaca.groupby(['tahun', 'kabupaten'])[cuaca_cols].mean().reset_index()
        df = df.merge(cuaca_tahunan, on=['tahun', 'kabupaten'], how='left')

    # Merge ENSO (agregasi tahunan)
    if 'tahun' in df_enso.columns:
        enso_tahunan = df_enso.groupby('tahun').agg(
            enso_mean=('enso_index', 'mean'),
            enso_lag1_mean=('enso_index', lambda x: x.shift(1).mean()),
            enso_lag3_mean=('enso_index', lambda x: x.shift(3).mean()),
        ).reset_index()
        enso_tahunan['bulan_el_nino'] = (enso_tahunan['enso_mean'] > 0.5).astype(int)
        enso_tahunan['bulan_la_nina'] = (enso_tahunan['enso_mean'] < -0.5).astype(int)
        df = df.merge(enso_tahunan, on='tahun', how='left')

    # Merge NDVI
    ndvi_cols = [c for c in df_ndvi.columns if c not in ['bulan', 'kabupaten', 'tahun', 'provinsi']]
    if ndvi_cols:
        ndvi_tahunan = df_ndvi.groupby(['tahun', 'kabupaten'])[ndvi_cols].mean().reset_index()
        df = df.merge(ndvi_tahunan, on=['tahun', 'kabupaten'], how='left')

    # Isi sisa NaN
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    logger.info(f"Master dataset: {len(df)} baris, {len(df.columns)} kolom")
    return df


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run_cleaning() -> pd.DataFrame:
    """
    Jalankan pipeline cleaning lengkap.

    Returns:
        master_dataset DataFrame
    """
    logger.info("=" * 55)
    logger.info("SiPADI — Data Cleaning Pipeline")
    logger.info("=" * 55)

    # Load raw
    df_produksi = pd.read_csv(RAW_DIR / "produksi_padi.csv")
    df_harga    = pd.read_csv(RAW_DIR / "harga_beras.csv")
    df_cuaca    = pd.read_csv(RAW_DIR / "cuaca_bmkg.csv")
    df_enso     = pd.read_csv(RAW_DIR / "enso_index.csv")
    df_ndvi     = pd.read_csv(RAW_DIR / "ndvi_sentinel.csv")

    # Clean individual
    df_produksi = clean_produksi(df_produksi)
    df_harga    = clean_harga(df_harga)
    df_cuaca    = clean_cuaca(df_cuaca)
    df_enso     = clean_enso(df_enso)
    df_ndvi     = clean_ndvi(df_ndvi)

    # Save masing-masing
    df_produksi.to_csv(PROCESSED_DIR / "produksi_clean.csv",  index=False)
    df_harga.to_csv(PROCESSED_DIR / "harga_clean.csv",        index=False)
    df_cuaca.to_csv(PROCESSED_DIR / "cuaca_clean.csv",        index=False)
    df_enso.to_csv(PROCESSED_DIR / "enso_clean.csv",          index=False)
    df_ndvi.to_csv(PROCESSED_DIR / "ndvi_clean.csv",          index=False)

    # Merge jadi master
    df_master = merge_all(df_produksi, df_harga, df_cuaca, df_enso, df_ndvi)
    df_master.to_csv(PROCESSED_DIR / "master_dataset_raw.csv", index=False)

    logger.info("✅ Cleaning selesai. File disimpan di data/processed/")
    return df_master


if __name__ == "__main__":
    run_cleaning()