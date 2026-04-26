"""
predict.py
==========
Inferensi / prediksi menggunakan model SiPADI yang sudah ditraining.
Mendukung: single input dict, DataFrame batch, dan forecast harga.

Cara pakai:
    python -m src.models.predict
    dari src.models.predict import predict_productivity, predict_risk, forecast_price
"""

import logging
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"

# Cache model agar tidak re-load setiap call
_cache = {}


def _load_models():
    """Load semua model dari disk (dengan caching)."""
    if 'loaded' not in _cache:
        logger.info("Loading models dari disk...")
        _cache['prod_model']   = joblib.load(MODELS_DIR / "productivity_model.pkl")
        _cache['risk_model']   = joblib.load(MODELS_DIR / "risk_classifier.pkl")
        _cache['price_model']  = joblib.load(MODELS_DIR / "price_forecaster.pkl")
        _cache['scaler']       = joblib.load(MODELS_DIR / "scaler.pkl")
        _cache['feature_cols'] = joblib.load(MODELS_DIR / "feature_cols.pkl")
        _cache['loaded']       = True
        logger.info("✅ Semua model berhasil dimuat.")
    return (
        _cache['prod_model'],
        _cache['risk_model'],
        _cache['price_model'],
        _cache['scaler'],
        _cache['feature_cols'],
    )


def _prepare_input(
    input_data: Union[dict, pd.DataFrame],
    feature_cols: list,
) -> pd.DataFrame:
    """
    Konversi input (dict atau DataFrame) menjadi DataFrame
    dengan kolom yang sesuai feature_cols. Isi 0 jika kolom tidak ada.
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    # Tambah kolom yang hilang dengan nilai 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df[feature_cols]


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def predict_productivity(input_data: Union[dict, pd.DataFrame]) -> pd.DataFrame:
    """
    Prediksi produktivitas padi (ton/ha).

    Args:
        input_data: dict satu sampel atau DataFrame multi-baris

    Returns:
        DataFrame dengan kolom:
          - prediksi_produktivitas_ton_ha
    """
    prod_model, _, _, _, feature_cols = _load_models()

    X   = _prepare_input(input_data, feature_cols)
    pred = prod_model.predict(X)

    result = pd.DataFrame({'prediksi_produktivitas_ton_ha': pred})
    result['kategori'] = pd.cut(
        pred,
        bins=[0, 4.5, 5.5, 6.5, 100],
        labels=['Rendah (<4.5)', 'Sedang (4.5–5.5)', 'Baik (5.5–6.5)', 'Tinggi (>6.5)']
    )
    return result


def predict_risk(input_data: Union[dict, pd.DataFrame]) -> pd.DataFrame:
    """
    Prediksi risiko gagal panen (klasifikasi biner).

    Args:
        input_data: dict satu sampel atau DataFrame multi-baris

    Returns:
        DataFrame dengan kolom:
          - prediksi_risiko (0=Aman, 1=Berisiko)
          - probabilitas_risiko
          - status
    """
    _, risk_model, _, _, feature_cols = _load_models()

    X    = _prepare_input(input_data, feature_cols)
    pred = risk_model.predict(X)
    prob = risk_model.predict_proba(X)[:, 1]

    result = pd.DataFrame({
        'prediksi_risiko': pred,
        'probabilitas_risiko': prob,
    })
    result['status'] = result['prediksi_risiko'].map({0: '🟢 Aman', 1: '🔴 Berisiko'})
    result['level_risiko'] = pd.cut(
        prob,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Rendah', 'Sedang', 'Tinggi']
    )
    return result


def predict_full(input_data: Union[dict, pd.DataFrame]) -> pd.DataFrame:
    """
    Gabungkan prediksi produktivitas + risiko dalam satu output.

    Returns:
        DataFrame gabungan
    """
    prod_result = predict_productivity(input_data)
    risk_result = predict_risk(input_data)
    return pd.concat([prod_result, risk_result], axis=1)


def forecast_price(n_months: int = 12) -> pd.DataFrame:
    """
    Forecast harga beras N bulan ke depan menggunakan model SARIMA.

    Args:
        n_months: jumlah bulan yang ingin diforecast (default 12)

    Returns:
        DataFrame dengan kolom:
          - tanggal
          - forecast_harga_per_kg
          - batas_bawah_90
          - batas_atas_90
          - status_harga
    """
    _, _, price_model, _, _ = _load_models()

    if price_model is None:
        logger.error("Price model tidak tersedia.")
        return pd.DataFrame()

    try:
        forecast = price_model.get_forecast(steps=n_months)
        pred_mean = forecast.predicted_mean
        ci        = forecast.conf_int(alpha=0.10)  # 90% CI

        df_forecast = pd.DataFrame({
            'tanggal': [str(p) for p in pred_mean.index],
            'forecast_harga_per_kg': pred_mean.values.round(0).astype(int),
            'batas_bawah_90': ci.iloc[:, 0].values.round(0).astype(int),
            'batas_atas_90':  ci.iloc[:, 1].values.round(0).astype(int),
        })

        # Status harga
        THRESHOLD_TINGGI = 14200
        df_forecast['status_harga'] = df_forecast['forecast_harga_per_kg'].apply(
            lambda x: '🔴 Tinggi' if x > THRESHOLD_TINGGI else '🟡 Normal'
        )

        return df_forecast

    except Exception as e:
        logger.error(f"Error forecast harga: {e}")
        return pd.DataFrame()


def batch_predict(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Prediksi batch untuk seluruh baris DataFrame.
    Berguna untuk menghasilkan peta risiko seluruh kabupaten.

    Args:
        df_input: DataFrame dengan fitur input

    Returns:
        df_input + kolom hasil prediksi
    """
    logger.info(f"Batch predict: {len(df_input)} baris")
    hasil = predict_full(df_input)
    return pd.concat([df_input.reset_index(drop=True), hasil], axis=1)


# ─────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Demo prediksi SiPADI")

    # Contoh input satu kabupaten
    sample = {
        'enso_mean': 0.3, 'enso_lag1_mean': 0.27, 'enso_lag3_mean': 0.21,
        'bulan_el_nino': 0, 'bulan_la_nina': 0, 'enso_squared': 0.09,
        'curah_hujan_total': 650, 'curah_hujan_mean': 162, 'curah_hujan_std': 130,
        'curah_hujan_max': 845, 'hari_hujan_total': 43, 'curah_hujan_sqrt': 25.5,
        'anomali_curah_hujan': 50, 'pct_anomali_hujan': 8.3, 'kekeringan_flag': 0,
        'banjir_flag': 0, 'suhu_mean': 27.2, 'suhu_max': 30.1, 'suhu_anomali': -0.3,
        'suhu_tinggi_flag': 0, 'kelembaban_mean': 79, 'kelembaban_rendah_flag': 0,
        'ndvi_mean': 0.58, 'ndvi_max': 0.68, 'ndvi_std': 0.05,
        'ndvi_kategori': 3, 'ndvi_lahan_baik': 1, 'ndvi_variabilitas_tinggi': 0,
        'pct_lahan_sangat_baik': 62, 'pct_lahan_buruk': 7,
        'skor_irigasi': 0.68, 'jumlah_pompa_unit': 35, 'luas_panen_ha': 12000,
        'kapasitas_pompa_per_ha': 0.233, 'ndvi_x_irigasi': 0.394,
        'lahan_irigasi_score': 42.2, 'enso_x_hujan': 195,
        'indeks_pertanaman': 200, 'persen_irigasi_kondisi_baik': 68,
        'era_pompanisasi': 1, 'harga_mean': 13500, 'harga_lag1': 13095,
        'harga_volatility': 148, 'harga_yoy_change': 3.5,
        'rasio_gabah_beras': 0.45, 'impor_volume': 500000, 'harga_impor_idr': 8500000,
        'musim_tanam_enc': 1, 'provinsi_enc': 1,
        'tahun_norm': 13, 'prod_lag1': 5.6, 'prod_lag2': 5.4, 'prod_lag3': 5.5,
        'prod_rolling_mean3': 5.5, 'prod_rolling_std3': 0.1, 'prod_yoy_change': 1.8,
    }

    hasil = predict_full(sample)
    print("\n=== Hasil Prediksi ===")
    print(hasil.to_string())

    print("\n=== Forecast Harga Beras 12 Bulan ===")
    df_fc = forecast_price(12)
    print(df_fc.to_string())