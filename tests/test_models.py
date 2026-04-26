"""
test_models.py
==============
Unit test untuk src/models/ (train, predict, evaluate)
Menggunakan data sintetis — tidak butuh file CSV asli.

Jalankan: pytest tests/test_models.py -v
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.models.predict import (
    predict_productivity,
    predict_risk,
    predict_full,
    forecast_price,
    batch_predict,
    _prepare_input,
)


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

FEATURE_COLS = [
    'enso_mean', 'enso_lag1_mean', 'enso_lag3_mean',
    'bulan_el_nino', 'bulan_la_nina', 'enso_squared',
    'curah_hujan_total', 'curah_hujan_mean', 'curah_hujan_std',
    'curah_hujan_max', 'hari_hujan_total', 'anomali_curah_hujan',
    'pct_anomali_hujan', 'kekeringan_flag', 'banjir_flag',
    'suhu_mean', 'suhu_max', 'suhu_anomali', 'suhu_tinggi_flag',
    'kelembaban_mean', 'kelembaban_rendah_flag',
    'ndvi_mean', 'ndvi_max', 'ndvi_std',
    'ndvi_kategori', 'ndvi_lahan_baik',
    'pct_lahan_sangat_baik', 'pct_lahan_buruk',
    'skor_irigasi', 'jumlah_pompa_unit', 'luas_panen_ha',
    'kapasitas_pompa_per_ha', 'indeks_pertanaman',
    'era_pompanisasi', 'harga_mean', 'harga_lag1',
    'harga_volatility', 'harga_yoy_change', 'rasio_gabah_beras',
    'impor_volume', 'harga_impor_idr', 'musim_tanam_enc',
    'provinsi_enc', 'tahun_norm',
]


def make_sample_dict(n=1) -> dict:
    """Buat sample input dict dengan nilai default yang masuk akal."""
    base = {col: 0.0 for col in FEATURE_COLS}
    base.update({
        'enso_mean': 0.3, 'curah_hujan_total': 650,
        'suhu_mean': 27.5, 'kelembaban_mean': 78,
        'ndvi_mean': 0.55, 'skor_irigasi': 0.65,
        'jumlah_pompa_unit': 30, 'luas_panen_ha': 12000,
        'harga_mean': 13500, 'harga_lag1': 13095,
        'musim_tanam_enc': 1, 'provinsi_enc': 1,
        'tahun_norm': 13, 'era_pompanisasi': 1,
    })
    if n == 1:
        return base
    return {k: [v] * n for k, v in base.items()}


def make_sample_df(n=5) -> pd.DataFrame:
    """Buat sample DataFrame dengan n baris."""
    data = make_sample_dict(n)
    return pd.DataFrame(data)


def make_mock_regressor(predict_value=5.5):
    """Buat mock XGBoost Regressor."""
    mock = MagicMock()
    mock.predict.return_value = np.array([predict_value])
    return mock


def make_mock_classifier(pred_class=0, prob=0.15):
    """Buat mock XGBoost Classifier."""
    mock = MagicMock()
    mock.predict.return_value = np.array([pred_class])
    mock.predict_proba.return_value = np.array([[1 - prob, prob]])
    return mock


def make_mock_sarima(forecast_vals=None):
    """Buat mock SARIMA result."""
    if forecast_vals is None:
        forecast_vals = [13500 + i * 50 for i in range(12)]

    import pandas as pd
    dates = pd.period_range('2024-01', periods=12, freq='M')
    pred_mean = pd.Series(forecast_vals, index=dates)
    ci_df = pd.DataFrame({
        'lower harga_beras_medium_per_kg': [v * 0.9 for v in forecast_vals],
        'upper harga_beras_medium_per_kg': [v * 1.1 for v in forecast_vals],
    }, index=dates)

    mock = MagicMock()
    forecast_obj = MagicMock()
    forecast_obj.predicted_mean = pred_mean
    forecast_obj.conf_int.return_value = ci_df
    mock.get_forecast.return_value = forecast_obj
    return mock


# ─────────────────────────────────────────────
# TEST: _prepare_input
# ─────────────────────────────────────────────

class TestPrepareInput:

    def test_dict_to_dataframe(self):
        sample = make_sample_dict()
        result = _prepare_input(sample, FEATURE_COLS)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_columns_match_feature_cols(self):
        sample = make_sample_dict()
        result = _prepare_input(sample, FEATURE_COLS)
        assert list(result.columns) == FEATURE_COLS

    def test_missing_cols_filled_with_zero(self):
        """Kolom yang tidak ada di input diisi 0."""
        partial = {'enso_mean': 0.5, 'curah_hujan_total': 700}
        result  = _prepare_input(partial, FEATURE_COLS)
        assert result.shape[1] == len(FEATURE_COLS)
        assert result['suhu_mean'].iloc[0] == 0.0

    def test_dataframe_input(self):
        df = make_sample_df(3)
        result = _prepare_input(df, FEATURE_COLS)
        assert len(result) == 3
        assert list(result.columns) == FEATURE_COLS


# ─────────────────────────────────────────────
# TEST: predict_productivity
# ─────────────────────────────────────────────

class TestPredictProductivity:

    def _patch(self, prod_val=5.8):
        return patch(
            'src.models.predict._load_models',
            return_value=(
                make_mock_regressor(prod_val),
                make_mock_classifier(),
                make_mock_sarima(),
                MagicMock(),
                FEATURE_COLS,
            )
        )

    def test_returns_dataframe(self):
        with self._patch():
            result = predict_productivity(make_sample_dict())
        assert isinstance(result, pd.DataFrame)

    def test_contains_prediction_col(self):
        with self._patch():
            result = predict_productivity(make_sample_dict())
        assert 'prediksi_produktivitas_ton_ha' in result.columns

    def test_contains_kategori_col(self):
        with self._patch():
            result = predict_productivity(make_sample_dict())
        assert 'kategori' in result.columns

    def test_prediction_value(self):
        with self._patch(prod_val=6.1):
            result = predict_productivity(make_sample_dict())
        assert result['prediksi_produktivitas_ton_ha'].iloc[0] == pytest.approx(6.1)

    def test_kategori_tinggi(self):
        with self._patch(prod_val=7.0):
            result = predict_productivity(make_sample_dict())
        assert 'Tinggi' in str(result['kategori'].iloc[0])

    def test_kategori_rendah(self):
        with self._patch(prod_val=3.5):
            result = predict_productivity(make_sample_dict())
        assert 'Rendah' in str(result['kategori'].iloc[0])

    def test_batch_multiple_rows(self):
        mock_reg = MagicMock()
        mock_reg.predict.return_value = np.array([5.5, 6.0, 4.8])
        with patch('src.models.predict._load_models',
                   return_value=(mock_reg, make_mock_classifier(),
                                 make_mock_sarima(), MagicMock(), FEATURE_COLS)):
            result = predict_productivity(make_sample_df(3))
        assert len(result) == 3


# ─────────────────────────────────────────────
# TEST: predict_risk
# ─────────────────────────────────────────────

class TestPredictRisk:

    def _patch(self, pred_class=0, prob=0.12):
        return patch(
            'src.models.predict._load_models',
            return_value=(
                make_mock_regressor(),
                make_mock_classifier(pred_class, prob),
                make_mock_sarima(),
                MagicMock(),
                FEATURE_COLS,
            )
        )

    def test_returns_dataframe(self):
        with self._patch():
            result = predict_risk(make_sample_dict())
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        with self._patch():
            result = predict_risk(make_sample_dict())
        for col in ['prediksi_risiko', 'probabilitas_risiko', 'status']:
            assert col in result.columns

    def test_status_aman(self):
        with self._patch(pred_class=0, prob=0.10):
            result = predict_risk(make_sample_dict())
        assert 'Aman' in result['status'].iloc[0]

    def test_status_berisiko(self):
        with self._patch(pred_class=1, prob=0.85):
            result = predict_risk(make_sample_dict())
        assert 'Berisiko' in result['status'].iloc[0]

    def test_probabilitas_range(self):
        with self._patch(prob=0.42):
            result = predict_risk(make_sample_dict())
        prob = result['probabilitas_risiko'].iloc[0]
        assert 0.0 <= prob <= 1.0

    def test_level_risiko_col(self):
        with self._patch(prob=0.75):
            result = predict_risk(make_sample_dict())
        assert 'level_risiko' in result.columns


# ─────────────────────────────────────────────
# TEST: predict_full
# ─────────────────────────────────────────────

class TestPredictFull:

    def _patch(self):
        return patch(
            'src.models.predict._load_models',
            return_value=(
                make_mock_regressor(5.7),
                make_mock_classifier(0, 0.18),
                make_mock_sarima(),
                MagicMock(),
                FEATURE_COLS,
            )
        )

    def test_returns_dataframe(self):
        with self._patch():
            result = predict_full(make_sample_dict())
        assert isinstance(result, pd.DataFrame)

    def test_combined_columns(self):
        with self._patch():
            result = predict_full(make_sample_dict())
        assert 'prediksi_produktivitas_ton_ha' in result.columns
        assert 'prediksi_risiko' in result.columns
        assert 'probabilitas_risiko' in result.columns


# ─────────────────────────────────────────────
# TEST: forecast_price
# ─────────────────────────────────────────────

class TestForecastPrice:

    def _patch(self, n=12):
        vals = [13500 + i * 40 for i in range(n)]
        return patch(
            'src.models.predict._load_models',
            return_value=(
                make_mock_regressor(),
                make_mock_classifier(),
                make_mock_sarima(vals),
                MagicMock(),
                FEATURE_COLS,
            )
        )

    def test_returns_dataframe(self):
        with self._patch():
            result = forecast_price(12)
        assert isinstance(result, pd.DataFrame)

    def test_correct_row_count(self):
        with self._patch(12):
            result = forecast_price(12)
        assert len(result) == 12

    def test_required_columns(self):
        with self._patch():
            result = forecast_price(12)
        for col in ['tanggal', 'forecast_harga_per_kg', 'batas_bawah_90', 'batas_atas_90', 'status_harga']:
            assert col in result.columns, f"Kolom '{col}' tidak ada"

    def test_batas_bawah_less_than_forecast(self):
        with self._patch():
            result = forecast_price(12)
        assert (result['batas_bawah_90'] <= result['forecast_harga_per_kg']).all()

    def test_batas_atas_greater_than_forecast(self):
        with self._patch():
            result = forecast_price(12)
        assert (result['batas_atas_90'] >= result['forecast_harga_per_kg']).all()

    def test_status_harga_values(self):
        with self._patch():
            result = forecast_price(12)
        valid_status = {'🔴 Tinggi', '🟡 Normal'}
        assert set(result['status_harga'].unique()).issubset(valid_status)

    def test_none_model_returns_empty(self):
        with patch('src.models.predict._load_models',
                   return_value=(MagicMock(), MagicMock(), None, MagicMock(), FEATURE_COLS)):
            result = forecast_price(12)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ─────────────────────────────────────────────
# TEST: batch_predict
# ─────────────────────────────────────────────

class TestBatchPredict:

    def _patch(self, n=5):
        mock_reg = MagicMock()
        mock_reg.predict.return_value = np.full(n, 5.5)
        mock_cls = MagicMock()
        mock_cls.predict.return_value = np.zeros(n, dtype=int)
        mock_cls.predict_proba.return_value = np.column_stack(
            [np.full(n, 0.85), np.full(n, 0.15)]
        )
        return patch(
            'src.models.predict._load_models',
            return_value=(mock_reg, mock_cls, make_mock_sarima(), MagicMock(), FEATURE_COLS)
        )

    def test_returns_dataframe(self):
        df = make_sample_df(5)
        with self._patch(5):
            result = batch_predict(df)
        assert isinstance(result, pd.DataFrame)

    def test_row_count_preserved(self):
        df = make_sample_df(5)
        with self._patch(5):
            result = batch_predict(df)
        assert len(result) == 5

    def test_input_cols_preserved(self):
        df = make_sample_df(5)
        with self._patch(5):
            result = batch_predict(df)
        for col in df.columns:
            assert col in result.columns

    def test_prediction_cols_added(self):
        df = make_sample_df(5)
        with self._patch(5):
            result = batch_predict(df)
        assert 'prediksi_produktivitas_ton_ha' in result.columns
        assert 'prediksi_risiko' in result.columns