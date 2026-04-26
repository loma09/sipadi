"""
test_cleaning.py
================
Unit test untuk src/data/cleaning.py
Jalankan: pytest tests/test_cleaning.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.cleaning import (
    clean_produksi,
    clean_harga,
    clean_cuaca,
    clean_enso,
    clean_ndvi,
    merge_all,
)


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def df_produksi_raw():
    return pd.DataFrame({
        'kabupaten':                ['Cilacap', 'Cilacap', 'Banyumas', 'Sragen', 'Klaten'],
        'provinsi':                 ['Jawa Tengah'] * 5,
        'tahun':                    [2020, 2020, 2021, 2021, 2022],
        'musim_tanam':              ['MT1', 'MT1', 'MT2', 'MT1', 'MT2'],
        'produktivitas_ton_per_ha': [5.4, 5.4, None, 8.9, 4.8],   # ada NaN & outlier
        'produksi_ton':             [120000, 120000, 95000, 180000, 75000],  # duplikat
        'luas_panen_ha':            [22000, 22000, 19000, 20000, 15000],
    })


@pytest.fixture
def df_harga_raw():
    return pd.DataFrame({
        'kabupaten':               ['Cilacap', 'Banyumas', 'Sragen', 'Klaten', 'Cilacap'],
        'tahun':                   [2020, 2020, 2021, 2021, 2022],
        'bulan':                   [1, 1, 6, 6, 1],
        'harga_beras_medium_per_kg': [12500, None, 13000, 13200, 99999],  # NaN & outlier
    })


@pytest.fixture
def df_cuaca_raw():
    return pd.DataFrame({
        'kabupaten':        ['Cilacap', 'Banyumas', 'Sragen', 'Klaten', 'Grobogan'],
        'tahun':            [2020, 2020, 2021, 2021, 2022],
        'bulan':            [1, 1, 6, 6, 1],
        'curah_hujan_total': [650, None, 400, 800, 1600],  # NaN & nilai ekstrem
        'suhu_mean':        [27.5, 28.0, 27.8, 55.0, 27.0],  # 55°C = outlier
        'kelembaban_mean':  [78, 80, 75, 82, 20],            # 20% = outlier
    })


@pytest.fixture
def df_enso_raw():
    return pd.DataFrame({
        'tanggal':    ['2020-01-01', '2020-07-01', '2021-01-01', '2021-07-01', None],
        'enso_index': [0.3, -0.8, 1.2, -1.5, 0.0],
    })


@pytest.fixture
def df_ndvi_raw():
    return pd.DataFrame({
        'kabupaten': ['Cilacap', 'Banyumas', 'Sragen', 'Klaten', 'Grobogan'],
        'tahun':     [2020, 2020, 2021, 2021, 2022],
        'bulan':     [1, 1, 6, 6, 1],
        'ndvi_mean': [0.55, None, 1.5, 0.48, 0.62],  # NaN & nilai > 1 (invalid)
        'ndvi_max':  [0.72, 0.68, 0.80, 0.65, 0.75],
        'ndvi_std':  [0.05, 0.04, 0.06, 0.05, 0.03],
    })


# ─────────────────────────────────────────────
# TEST: clean_produksi
# ─────────────────────────────────────────────

class TestCleanProduksi:

    def test_output_is_dataframe(self, df_produksi_raw):
        result = clean_produksi(df_produksi_raw)
        assert isinstance(result, pd.DataFrame)

    def test_drop_duplicates(self, df_produksi_raw):
        result = clean_produksi(df_produksi_raw)
        assert result.duplicated().sum() == 0

    def test_no_missing_numeric(self, df_produksi_raw):
        result = clean_produksi(df_produksi_raw)
        num_cols = ['produktivitas_ton_per_ha', 'produksi_ton', 'luas_panen_ha']
        for col in num_cols:
            if col in result.columns:
                assert result[col].isna().sum() == 0, f"{col} masih ada NaN"

    def test_kabupaten_title_case(self, df_produksi_raw):
        result = clean_produksi(df_produksi_raw)
        assert result['kabupaten'].str.istitle().all()

    def test_tahun_is_int(self, df_produksi_raw):
        result = clean_produksi(df_produksi_raw)
        assert result['tahun'].dtype == int

    def test_produktivitas_range(self, df_produksi_raw):
        """Tidak ada nilai produktivitas ekstrem setelah cleaning."""
        result = clean_produksi(df_produksi_raw)
        assert result['produktivitas_ton_per_ha'].max() < 15.0
        assert result['produktivitas_ton_per_ha'].min() >= 0.0

    def test_returns_copy(self, df_produksi_raw):
        """Pastikan input asli tidak dimodifikasi."""
        original_len = len(df_produksi_raw)
        clean_produksi(df_produksi_raw)
        assert len(df_produksi_raw) == original_len


# ─────────────────────────────────────────────
# TEST: clean_harga
# ─────────────────────────────────────────────

class TestCleanHarga:

    def test_no_missing_harga(self, df_harga_raw):
        result = clean_harga(df_harga_raw)
        assert result['harga_beras_medium_per_kg'].isna().sum() == 0

    def test_harga_clip_max(self, df_harga_raw):
        """Harga tidak boleh melebihi 30.000."""
        result = clean_harga(df_harga_raw)
        assert result['harga_beras_medium_per_kg'].max() <= 30000

    def test_harga_clip_min(self, df_harga_raw):
        """Harga tidak boleh di bawah 5.000."""
        result = clean_harga(df_harga_raw)
        assert result['harga_beras_medium_per_kg'].min() >= 5000

    def test_output_shape(self, df_harga_raw):
        result = clean_harga(df_harga_raw)
        assert len(result) == len(df_harga_raw)


# ─────────────────────────────────────────────
# TEST: clean_cuaca
# ─────────────────────────────────────────────

class TestCleanCuaca:

    def test_no_missing_after_interpolate(self, df_cuaca_raw):
        result = clean_cuaca(df_cuaca_raw)
        num_cols = [c for c in result.columns if result[c].dtype in ['float64', 'int64']]
        for col in num_cols:
            assert result[col].isna().sum() == 0, f"{col} masih ada NaN"

    def test_suhu_clip(self, df_cuaca_raw):
        result = clean_cuaca(df_cuaca_raw)
        if 'suhu_mean' in result.columns:
            assert result['suhu_mean'].max() <= 38
            assert result['suhu_mean'].min() >= 18

    def test_kelembaban_clip(self, df_cuaca_raw):
        result = clean_cuaca(df_cuaca_raw)
        if 'kelembaban_mean' in result.columns:
            assert result['kelembaban_mean'].max() <= 100
            assert result['kelembaban_mean'].min() >= 40

    def test_curah_hujan_clip(self, df_cuaca_raw):
        result = clean_cuaca(df_cuaca_raw)
        if 'curah_hujan_total' in result.columns:
            assert result['curah_hujan_total'].max() <= 3000
            assert result['curah_hujan_total'].min() >= 0


# ─────────────────────────────────────────────
# TEST: clean_enso
# ─────────────────────────────────────────────

class TestCleanEnso:

    def test_tanggal_parsed(self, df_enso_raw):
        result = clean_enso(df_enso_raw)
        assert 'tahun' in result.columns
        assert 'bulan' in result.columns

    def test_invalid_tanggal_dropped(self, df_enso_raw):
        """Baris dengan tanggal None harus dihapus."""
        result = clean_enso(df_enso_raw)
        assert result['tahun'].isna().sum() == 0

    def test_enso_clip(self, df_enso_raw):
        result = clean_enso(df_enso_raw)
        assert result['enso_index'].max() <= 4.0
        assert result['enso_index'].min() >= -4.0

    def test_no_missing_enso(self, df_enso_raw):
        result = clean_enso(df_enso_raw)
        assert result['enso_index'].isna().sum() == 0


# ─────────────────────────────────────────────
# TEST: clean_ndvi
# ─────────────────────────────────────────────

class TestCleanNdvi:

    def test_ndvi_clip_max(self, df_ndvi_raw):
        result = clean_ndvi(df_ndvi_raw)
        assert result['ndvi_mean'].max() <= 1.0

    def test_ndvi_clip_min(self, df_ndvi_raw):
        result = clean_ndvi(df_ndvi_raw)
        assert result['ndvi_mean'].min() >= -0.1

    def test_no_missing_ndvi(self, df_ndvi_raw):
        result = clean_ndvi(df_ndvi_raw)
        assert result['ndvi_mean'].isna().sum() == 0


# ─────────────────────────────────────────────
# TEST: merge_all
# ─────────────────────────────────────────────

class TestMergeAll:

    def test_merge_returns_dataframe(
        self, df_produksi_raw, df_harga_raw,
        df_cuaca_raw, df_enso_raw, df_ndvi_raw
    ):
        dp = clean_produksi(df_produksi_raw)
        dh = clean_harga(df_harga_raw)
        dc = clean_cuaca(df_cuaca_raw)
        de = clean_enso(df_enso_raw)
        dn = clean_ndvi(df_ndvi_raw)

        result = merge_all(dp, dh, dc, de, dn)
        assert isinstance(result, pd.DataFrame)

    def test_merge_no_missing_numeric(
        self, df_produksi_raw, df_harga_raw,
        df_cuaca_raw, df_enso_raw, df_ndvi_raw
    ):
        dp = clean_produksi(df_produksi_raw)
        dh = clean_harga(df_harga_raw)
        dc = clean_cuaca(df_cuaca_raw)
        de = clean_enso(df_enso_raw)
        dn = clean_ndvi(df_ndvi_raw)

        result = merge_all(dp, dh, dc, de, dn)
        num_cols = result.select_dtypes(include='number').columns
        total_na = result[num_cols].isna().sum().sum()
        assert total_na == 0, f"Masih ada {total_na} NaN setelah merge"

    def test_merge_row_count(
        self, df_produksi_raw, df_harga_raw,
        df_cuaca_raw, df_enso_raw, df_ndvi_raw
    ):
        """Jumlah baris master = jumlah baris produksi setelah cleaning."""
        dp = clean_produksi(df_produksi_raw)
        dh = clean_harga(df_harga_raw)
        dc = clean_cuaca(df_cuaca_raw)
        de = clean_enso(df_enso_raw)
        dn = clean_ndvi(df_ndvi_raw)

        result = merge_all(dp, dh, dc, de, dn)
        assert len(result) == len(dp)