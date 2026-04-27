import os
import sys
import json
import requests
import joblib
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pathlib import Path
from dotenv import load_dotenv
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()

try:
    if hasattr(st, 'secrets') and len(st.secrets) > 0:
        for key, val in st.secrets.items():
            os.environ[key] = str(val)
except Exception:
    pass

st.set_page_config(
    page_title="SiPADI — Sistem Prediksi Anomali & Deteksi Risiko Pangan",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR      = Path(__file__).parent.parent.parent
MODELS_DIR    = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR       = BASE_DIR / "data" / "raw"

# ═══════════════════════════════════════════════════════
# LOAD CSS FROM EXTERNAL FILE
# ═══════════════════════════════════════════════════════
_css_path = Path(__file__).parent / "style.css"
with open(_css_path, "r", encoding="utf-8") as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# LOAD DATA & MODELS
# ═══════════════════════════════════════════════════════
@st.cache_data
def load_data():
    from azure.storage.blob import BlobServiceClient
    from io import BytesIO

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        st.error("❌ AZURE_STORAGE_CONNECTION_STRING tidak ditemukan di secrets!")
        st.stop()

    def read_csv_azure(container, blob_name, **kwargs):
        try:
            local_path = BASE_DIR / "data" / "raw" / blob_name.split("/")[-1]
            if local_path.exists():
                return pd.read_csv(local_path, **kwargs)
        except:
            pass
        blob_service = BlobServiceClient.from_connection_string(conn_str)
        blob_client  = blob_service.get_blob_client(container=container, blob=blob_name)
        data         = blob_client.download_blob().readall()
        return pd.read_csv(BytesIO(data), **kwargs)

    def read_csv_processed(blob_name, **kwargs):
        try:
            local_path = BASE_DIR / "data" / "processed" / blob_name
            if local_path.exists():
                return pd.read_csv(local_path, **kwargs)
        except:
            pass
        blob_service = BlobServiceClient.from_connection_string(conn_str)
        blob_client  = blob_service.get_blob_client(
            container="sipadi-data-processed", blob=blob_name)
        data = blob_client.download_blob().readall()
        return pd.read_csv(BytesIO(data), **kwargs)

    df_master   = read_csv_processed("master_dataset.csv")
    df_produksi = read_csv_azure("sipadi-data-raw", "bps/produksi_padi.csv")
    df_harga    = read_csv_azure("sipadi-data-raw", "harga/harga_beras.csv")
    df_cuaca    = read_csv_azure("sipadi-data-raw", "bmkg/cuaca_bmkg.csv")
    df_enso     = read_csv_azure("sipadi-data-raw", "enso/enso_index.csv", parse_dates=['tanggal'])
    df_ndvi     = read_csv_azure("sipadi-data-raw", "satellite/ndvi_sentinel.csv")
    return df_master, df_produksi, df_harga, df_cuaca, df_enso, df_ndvi


@st.cache_resource
def load_models():
    from azure.storage.blob import BlobServiceClient
    from io import BytesIO

    conn_str     = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_service = BlobServiceClient.from_connection_string(conn_str)

    def load_pkl(blob_name):
        local_path = BASE_DIR / "models" / blob_name.split("/")[-1]
        if local_path.exists():
            return joblib.load(local_path)
        blob_client = blob_service.get_blob_client(container="sipadi-models", blob=blob_name)
        data = blob_client.download_blob().readall()
        with open(f"/tmp/{blob_name.split('/')[-1]}", "wb") as f:
            f.write(data)
        return joblib.load(f"/tmp/{blob_name.split('/')[-1]}")

    prod_model   = load_pkl("models/productivity_model.pkl")
    risk_model   = load_pkl("models/risk_classifier.pkl")
    price_model  = load_pkl("models/price_forecaster.pkl")
    scaler       = load_pkl("models/scaler.pkl")
    feature_cols = load_pkl("models/feature_cols.pkl")
    return prod_model, risk_model, price_model, scaler, feature_cols


df_master, df_produksi, df_harga, df_cuaca, df_enso, df_ndvi = load_data()
prod_model, risk_model, price_model, scaler, feature_cols = load_models()


# ═══════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════
C = {
    'green':   '#1da84a',
    'green2':  '#34d464',
    'blue':    '#1a5fd4',
    'red':     '#d92b2b',
    'amber':   '#c97a00',
    'jateng':  '#1da84a',
    'jatim':   '#1a5fd4',
}

def chart_style(fig, h=400, m=None):
    if m is None:
        m = dict(l=0, r=0, t=20, b=0)
    fig.update_layout(
        height=h, margin=m,
        font_family='Inter', font=dict(size=12),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, linecolor='#e8efe9'),
        yaxis=dict(gridcolor='#f0f6f1', zeroline=False),
    )
    return fig


# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-head">
        <div class="sb-logorow">
            <div class="sb-icon">
                <svg viewBox="0 0 24 24">
                    <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                    <path d="M2 17l10 5 10-5"/>
                    <path d="M2 12l10 5 10-5"/>
                </svg>
            </div>
            <div>
                <p class="sb-title">SiPADI</p>
            </div>
        </div>
        <p class="sb-sub">Prediksi Anomali &amp; Deteksi<br>Risiko Pangan Indonesia</p>
    </div>
    <p class="sb-nav-label">Menu</p>
    """, unsafe_allow_html=True)

    menu = st.radio("nav", [
        "Overview & KPI",
        "Peta Risiko Kabupaten",
        "Prediksi Produktivitas",
        "Deteksi Risiko Gagal Panen",
        "Forecast Harga Beras",
        "Analisis Sentimen Berita",
        "Rekomendasi Strategis",
    ], label_visibility="collapsed")

    st.markdown("""
    <hr class="sb-divider">
    <p class="sb-foot-label">Azure Services</p>
    <div class="sb-svc-item"><div class="sb-svc-dot"></div>Blob Storage</div>
    <div class="sb-svc-item"><div class="sb-svc-dot"></div>ML Workspace</div>
    <div class="sb-svc-item"><div class="sb-svc-dot"></div>AI Language</div>
    <div class="sb-svc-item"><div class="sb-svc-dot"></div>Maps</div>
    <div class="sb-svc-item"><div class="sb-svc-dot"></div>Static Web Apps</div>
    <hr class="sb-divider">
    <p class="sb-foot-label">Microsoft Elevate</p>
    <p class="sb-foot-val">AI Impact Challenge 2026</p>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
    <div>
        <div class="hero-live">
            <div class="hero-dot"></div>
            Live Monitoring
        </div>
        <h1>Si<span>PADI</span> Dashboard</h1>
        <p class="hero-desc">
            Sistem Prediksi Anomali &amp; Deteksi Risiko Pangan Indonesia —
            mendukung Program Swasembada Pangan 2025–2029
        </p>
    </div>
    <div class="hero-stats">
        <div class="hs">
            <span class="hs-label">Wilayah</span>
            <span class="hs-val">Jateng &amp; Jatim</span>
        </div>
        <div class="hs">
            <span class="hs-label">Kabupaten</span>
            <span class="hs-val hi">58</span>
        </div>
        <div class="hs">
            <span class="hs-label">Periode Data</span>
            <span class="hs-val">2010–2024</span>
        </div>
        <div class="hs">
            <span class="hs-label">Model AI</span>
            <span class="hs-val">XGBoost + SARIMA</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE: OVERVIEW & KPI
# ═══════════════════════════════════════════════════════
if menu == "Overview & KPI":

    total_kab  = df_produksi['kabupaten'].nunique()
    avg_prod   = df_produksi['produktivitas_ton_per_ha'].mean()
    risiko_pct = df_master['risiko_gagal_panen'].mean() * 100
    total_prod = df_produksi['produksi_ton'].sum() / 1e6
    harga_now  = df_harga['harga_beras_medium_per_kg'].iloc[-1]

    st.markdown('<p class="stitle">Key Performance Indicators</p>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class="kpi kpi-primary">
            <div class="kpi-label">Kabupaten Dipantau</div>
            <div class="kpi-num">{total_kab}</div>
            <div class="kpi-desc">Jawa Tengah &amp; Jawa Timur</div>
            <span class="badge">Aktif 100%</span>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi" style="--kc:#1da84a">
            <div class="kpi-label">Produktivitas Rata-rata</div>
            <div class="kpi-num">{avg_prod:.2f}</div>
            <div class="kpi-desc">ton / hektar</div>
            <span class="badge b-up">+0.04 / tahun</span>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi" style="--kc:#c97a00">
            <div class="kpi-label">Kabupaten Berisiko</div>
            <div class="kpi-num">{risiko_pct:.1f}%</div>
            <div class="kpi-desc">dari total kabupaten</div>
            <span class="badge b-up">−2.3% vs tahun lalu</span>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi" style="--kc:#1a5fd4">
            <div class="kpi-label">Total Produksi</div>
            <div class="kpi-num">{total_prod:.1f}M</div>
            <div class="kpi-desc">ton kumulatif</div>
            <span class="badge b-neu">2010–2024</span>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="kpi" style="--kc:#d92b2b">
            <div class="kpi-label">Harga Beras Kini</div>
            <div class="kpi-num mid">Rp{harga_now:,.0f}</div>
            <div class="kpi-desc">per kilogram</div>
            <span class="badge b-dn">+3.2% YoY</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="stitle">Performa Model AI</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="mc">
            <span class="mc-tag tg">Model 1 — Regresi</span>
            <h3>Prediksi Produktivitas</h3>
            <p class="mc-algo">XGBoost Regressor</p>
            <div class="mrow"><span class="mk">R² Score</span><span class="mv ok">0.886</span></div>
            <div class="mrow"><span class="mk">MAPE</span><span class="mv ok">2.25%</span></div>
            <div class="mrow"><span class="mk">RMSE</span><span class="mv">0.158 ton/ha</span></div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="mc">
            <span class="mc-tag tr">Model 2 — Klasifikasi</span>
            <h3>Deteksi Risiko Gagal Panen</h3>
            <p class="mc-algo">XGBoost Classifier</p>
            <div class="mrow"><span class="mk">F1-Score</span><span class="mv ok">0.833</span></div>
            <div class="mrow"><span class="mk">Recall</span><span class="mv ok">92.6%</span></div>
            <div class="mrow"><span class="mk">ROC-AUC</span><span class="mv ok">0.967</span></div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="mc">
            <span class="mc-tag ta">Model 3 — Time Series</span>
            <h3>Forecast Harga Beras</h3>
            <p class="mc-algo">SARIMA(1,1,1)(1,1,1,12)</p>
            <div class="mrow"><span class="mk">MAPE</span><span class="mv ok">1.16%</span></div>
            <div class="mrow"><span class="mk">Horizon Forecast</span><span class="mv">12 bulan</span></div>
            <div class="mrow"><span class="mk">Range Prediksi</span><span class="mv">Rp13.923–14.394</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="stitle">Tren Produksi Padi (2010–2023)</p>', unsafe_allow_html=True)

    tren = df_produksi.groupby(['tahun', 'provinsi'])['produksi_ton'].sum().reset_index()
    fig = px.line(tren, x='tahun', y='produksi_ton', color='provinsi',
                  markers=True, template='plotly_white',
                  labels={'produksi_ton': 'Total Produksi (ton)', 'tahun': 'Tahun'},
                  color_discrete_map={'Jawa Tengah': C['jateng'], 'Jawa Timur': C['jatim']})
    fig.update_traces(line_width=2.5, marker_size=8)
    chart_style(fig, h=380, m=dict(l=0, r=0, t=10, b=0))
    fig.update_layout(
        legend_title="Provinsi",
        legend=dict(yanchor="bottom", y=0.03, xanchor="right", x=0.99,
                    bgcolor='rgba(255,255,255,0.9)', bordercolor='#eee', borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════
# PAGE: PETA RISIKO
# ═══════════════════════════════════════════════════════
elif menu == "Peta Risiko Kabupaten":
    st.markdown("""
    <div class="page-header">
        <div class="ph-icon"><span class="material-symbols-outlined">map</span></div>
        <div class="ph-text">
            <h1>Peta Risiko per Kabupaten</h1>
            <p>Visualisasi spasial risiko gagal panen berdasarkan model klasifikasi SiPADI</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<span class="az">Azure Maps</span>', unsafe_allow_html=True)

    koordinat = {
        'Cilacap': (-7.73, 109.02), 'Banyumas': (-7.42, 109.23),
        'Kebumen': (-7.67, 109.65), 'Purworejo': (-7.71, 110.02),
        'Klaten': (-7.71, 110.60), 'Sragen': (-7.42, 111.02),
        'Grobogan': (-7.01, 110.92), 'Demak': (-6.90, 110.64),
        'Brebes': (-6.87, 109.04), 'Tegal': (-6.87, 109.14),
        'Surabaya': (-7.25, 112.75), 'Malang': (-7.97, 112.63),
        'Jember': (-8.17, 113.70), 'Banyuwangi': (-8.22, 114.37),
        'Lamongan': (-7.12, 112.42), 'Bojonegoro': (-7.15, 111.88),
        'Ngawi': (-7.40, 111.45), 'Madiun': (-7.63, 111.52),
        'Kediri': (-7.82, 112.01), 'Jombang': (-7.55, 112.23),
    }

    risiko_kab = df_master.groupby('kabupaten').agg(
        risiko_pct=('risiko_gagal_panen', 'mean'),
        avg_prod=('produktivitas_ton_per_ha', 'mean'),
        n_records=('risiko_gagal_panen', 'count')
    ).reset_index()
    risiko_kab['lat'] = risiko_kab['kabupaten'].map(lambda x: koordinat.get(x, (None, None))[0])
    risiko_kab['lon'] = risiko_kab['kabupaten'].map(lambda x: koordinat.get(x, (None, None))[1])
    risiko_kab = risiko_kab.dropna(subset=['lat', 'lon'])
    risiko_kab['risiko_pct_100'] = risiko_kab['risiko_pct'] * 100

    col1, col2 = st.columns([2, 1])
    with col1:
        threshold = st.slider("Threshold Risiko (%)", 0, 100, 25)
    with col2:
        show_all = st.checkbox("Tampilkan semua kabupaten", value=True)

    df_map = risiko_kab if show_all else risiko_kab[risiko_kab['risiko_pct_100'] >= threshold]

    fig_map = px.scatter_mapbox(
        df_map, lat='lat', lon='lon',
        color='risiko_pct_100', size='avg_prod',
        hover_name='kabupaten',
        hover_data={'risiko_pct_100': ':.1f', 'avg_prod': ':.2f', 'lat': False, 'lon': False},
        color_continuous_scale='RdYlGn_r',
        size_max=22, zoom=6,
        center={'lat': -7.5, 'lon': 111.5},
        mapbox_style='open-street-map',
        labels={'risiko_pct_100': 'Risiko Gagal Panen (%)', 'avg_prod': 'Produktivitas (ton/ha)'},
    )
    fig_map.update_layout(
        height=520, margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title="Risiko (%)", thickness=14, len=0.6,
            bgcolor='rgba(255,255,255,0.9)'
        )
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown('<p class="stitle">Kabupaten Risiko Tinggi</p>', unsafe_allow_html=True)
    high_risk = risiko_kab[risiko_kab['risiko_pct_100'] >= threshold].sort_values('risiko_pct_100', ascending=False)
    high_risk['Status'] = high_risk['risiko_pct_100'].apply(lambda x: 'Tinggi' if x >= 30 else 'Sedang')
    st.dataframe(
        high_risk[['kabupaten', 'risiko_pct_100', 'avg_prod', 'Status']].rename(columns={
            'kabupaten': 'Kabupaten', 'risiko_pct_100': 'Risiko (%)', 'avg_prod': 'Produktivitas (ton/ha)'
        }),
        use_container_width=True, hide_index=True
    )


# ═══════════════════════════════════════════════════════
# PAGE: PREDIKSI PRODUKTIVITAS
# ═══════════════════════════════════════════════════════
elif menu == "Prediksi Produktivitas":
    st.markdown("""
    <div class="page-header">
        <div class="ph-icon"><span class="material-symbols-outlined">trending_up</span></div>
        <div class="ph-text">
            <h1>Prediksi Produktivitas Padi</h1>
            <p>Masukkan parameter kondisi lapangan untuk mendapatkan estimasi produktivitas</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<p class="stitle">Kondisi Iklim</p>', unsafe_allow_html=True)
        enso_val    = st.slider("ENSO Index", -2.0, 2.0, 0.0, 0.1)
        curah_hujan = st.slider("Curah Hujan Total (mm)", 100, 1500, 600)
        suhu        = st.slider("Suhu Rata-rata (°C)", 24.0, 32.0, 27.5, 0.1)
        kelembaban  = st.slider("Kelembaban (%)", 60, 95, 78)
    with col2:
        st.markdown('<p class="stitle">Kondisi Lahan</p>', unsafe_allow_html=True)
        ndvi       = st.slider("NDVI (Kesehatan Lahan)", 0.1, 0.9, 0.55, 0.01)
        pct_baik   = st.slider("Lahan Kondisi Baik (%)", 0, 100, 60)
        luas_panen = st.number_input("Luas Panen (ha)", 1000, 50000, 10000)
        musim      = st.selectbox("Musim Tanam", ["MT1 (Jan-Apr)", "MT2 (Jun-Sep)"])
    with col3:
        st.markdown('<p class="stitle">Infrastruktur</p>', unsafe_allow_html=True)
        jumlah_pompa = st.slider("Jumlah Pompa (unit)", 0, 100, 30)
        skor_irigasi = st.slider("Skor Irigasi", 0.0, 1.0, 0.65, 0.01)
        harga_beras  = st.number_input("Harga Beras (Rp/kg)", 8000, 20000, 13500)
        provinsi     = st.selectbox("Provinsi", ["Jawa Tengah", "Jawa Timur"])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Prediksi Sekarang", type="primary", use_container_width=True):
        input_data = {col: 0 for col in feature_cols}
        input_data.update({
            'enso_mean': enso_val, 'enso_lag1_mean': enso_val * 0.9,
            'enso_lag3_mean': enso_val * 0.7,
            'bulan_el_nino': 1 if enso_val > 0.5 else 0,
            'bulan_la_nina': 1 if enso_val < -0.5 else 0,
            'curah_hujan_total': curah_hujan, 'curah_hujan_mean': curah_hujan / 4,
            'curah_hujan_std': curah_hujan * 0.2, 'curah_hujan_max': curah_hujan * 1.3,
            'hari_hujan_total': int(curah_hujan / 15), 'suhu_mean': suhu,
            'suhu_max': suhu + 3, 'kelembaban_mean': kelembaban,
            'anomali_curah_hujan': curah_hujan - 600,
            'pct_anomali_hujan': (curah_hujan - 600) / 600 * 100,
            'ndvi_mean': ndvi, 'ndvi_max': ndvi + 0.1, 'ndvi_std': 0.05,
            'pct_lahan_sangat_baik': pct_baik,
            'pct_lahan_buruk': max(0, 20 - pct_baik * 0.2),
            'skor_irigasi': skor_irigasi,
            'jumlah_pompa_unit': jumlah_pompa,
            'kapasitas_pompa_per_ha': jumlah_pompa * 80 / luas_panen,
            'indeks_pertanaman': 200,
            'persen_irigasi_kondisi_baik': skor_irigasi * 100,
            'era_pompanisasi': 1, 'harga_mean': harga_beras,
            'harga_lag1': harga_beras * 0.97, 'harga_volatility': 150,
            'harga_yoy_change': 3.5, 'rasio_gabah_beras': 0.45,
            'impor_volume': 500000, 'harga_impor_idr': 8500000,
            'luas_panen_ha': luas_panen,
            'musim_tanam_enc': 1 if 'MT1' in musim else 0,
            'provinsi_enc': 1 if provinsi == 'Jawa Tengah' else 0,
        })
        X_input = pd.DataFrame([input_data])[feature_cols]
        pred_prod = prod_model.predict(X_input)[0]
        pred_risk = risk_model.predict(X_input)[0]
        pred_prob = risk_model.predict_proba(X_input)[0][1]

        st.markdown("---")
        st.markdown('<p class="stitle">Hasil Prediksi</p>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Prediksi Produktivitas",
                      f"{pred_prod:.2f} ton/ha",
                      f"{'Di atas' if pred_prod > 5.5 else 'Di bawah'} rata-rata")
        with r2:
            risk_label = "BERISIKO" if pred_risk == 1 else "AMAN"
            st.metric("Status Risiko", risk_label, f"Prob: {pred_prob:.1%}")
        with r3:
            st.metric("Estimasi Total Produksi",
                      f"{pred_prod * luas_panen:,.0f} ton",
                      f"Luas: {luas_panen:,} ha")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_prod,
            delta={'reference': 5.5, 'valueformat': '.2f'},
            title={'text': "Produktivitas (ton/ha)", 'font': {'size': 14, 'family': 'Outfit'}},
            gauge={
                'axis': {'range': [3, 8], 'tickwidth': 1, 'tickcolor': '#ccc'},
                'bar': {'color': C['green'], 'thickness': 0.65},
                'bgcolor': 'white', 'borderwidth': 0,
                'steps': [
                    {'range': [3, 4.5],   'color': "#fee2e2"},
                    {'range': [4.5, 5.5], 'color': "#fef9c3"},
                    {'range': [5.5, 8],   'color': "#dcfce7"},
                ],
                'threshold': {'line': {'color': "#dc2626", 'width': 3},
                              'thickness': 0.75, 'value': 5.14}
            }
        ))
        chart_style(fig_gauge, h=280, m=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if pred_risk == 1:
            st.markdown(f"""
            <div class="al al-err">
                <strong>Peringatan Risiko Gagal Panen</strong>
                Probabilitas gagal panen: <b>{pred_prob:.1%}</b>.
                Segera lakukan intervensi pompanisasi tambahan dan koordinasi dengan Dinas Pertanian setempat.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="al al-ok">
                <strong>Kondisi Aman</strong>
                Probabilitas gagal panen: <b>{pred_prob:.1%}</b>.
                Produktivitas diprediksi {'di atas' if pred_prod > 5.5 else 'mendekati'} rata-rata historis 5.5 ton/ha.
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE: DETEKSI RISIKO
# ═══════════════════════════════════════════════════════
elif menu == "Deteksi Risiko Gagal Panen":
    st.markdown("""
    <div class="page-header">
        <div class="ph-icon"><span class="material-symbols-outlined">warning</span></div>
        <div class="ph-text">
            <h1>Deteksi Risiko Gagal Panen</h1>
            <p>Analisis tren risiko dan performa model klasifikasi XGBoost</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    risiko_trend = df_master.groupby(['tahun', 'provinsi'])['risiko_gagal_panen'].mean().reset_index()
    risiko_trend['risiko_pct'] = risiko_trend['risiko_gagal_panen'] * 100

    fig = px.line(risiko_trend, x='tahun', y='risiko_pct', color='provinsi',
                  markers=True,
                  title='Tren Persentase Kabupaten Berisiko Gagal Panen',
                  labels={'risiko_pct': 'Kabupaten Berisiko (%)', 'tahun': 'Tahun'},
                  template='plotly_white',
                  color_discrete_map={'Jawa Tengah': C['jateng'], 'Jawa Timur': C['red']})
    fig.add_hline(y=25, line_dash="dot", line_color="#c97a00",
                  annotation_text="Threshold 25%", annotation_font_size=11)
    fig.update_traces(line_width=2.5, marker_size=8)
    chart_style(fig, h=380, m=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        risiko_musim = df_master.groupby('musim_tanam')['risiko_gagal_panen'].mean() * 100
        fig2 = px.bar(risiko_musim.reset_index(),
                      x='musim_tanam', y='risiko_gagal_panen',
                      title='Risiko per Musim Tanam',
                      template='plotly_white',
                      color='musim_tanam',
                      color_discrete_map={'MT1': C['jateng'], 'MT2': C['red']},
                      labels={'risiko_gagal_panen': 'Risiko (%)', 'musim_tanam': 'Musim'})
        fig2.update_traces(marker_line_width=0, width=0.4)
        chart_style(fig2, h=340, m=dict(l=0, r=0, t=40, b=0))
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<p class="stitle">Performa Model Klasifikasi</p>', unsafe_allow_html=True)
        metrics_df = pd.DataFrame({
            'Metrik': ['F1-Score', 'Precision', 'Recall', 'ROC-AUC'],
            'Nilai': [0.8333, 0.7576, 0.9259, 0.9665],
            'Keterangan': [
                'Keseimbangan presisi & recall',
                '76% prediksi berisiko terbukti benar',
                '93% kasus berisiko terdeteksi',
                'Kemampuan membedakan kelas sangat baik',
            ]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.markdown("""
        <div class="ic">
            <div class="ic-head"><strong>Mengapa Recall Penting?</strong></div>
            <p>Dalam sistem peringatan dini, lebih baik memberikan <b>peringatan berlebih</b>
            daripada <b>melewatkan kasus berisiko</b>. Recall 92.6% memastikan hampir semua
            kasus gagal panen terdeteksi tepat waktu untuk intervensi.</p>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE: FORECAST HARGA
# ═══════════════════════════════════════════════════════
elif menu == "Forecast Harga Beras":
    st.markdown("""
    <div class="page-header">
        <div class="ph-icon"><span class="material-symbols-outlined">payments</span></div>
        <div class="ph-text">
            <h1>Forecast Harga Beras</h1>
            <p>Prediksi 12 bulan ke depan — SARIMA(1,1,1)(1,1,1,12) · MAPE 1.16%</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ts_hist = df_harga.groupby(['tahun', 'bulan'])['harga_beras_medium_per_kg'].mean().reset_index()
    ts_hist['tanggal'] = pd.to_datetime({'year': ts_hist['tahun'], 'month': ts_hist['bulan'], 'day': 1})
    ts_hist = ts_hist.sort_values('tanggal')

    forecast_dates = pd.date_range('2024-01-01', periods=12, freq='MS')
    forecast_vals  = [13932, 13923, 13938, 13953, 13956, 13971, 14366, 14362, 14394, 13988, 13995, 14020]
    forecast_upper = [v * 1.10 for v in forecast_vals]
    forecast_lower = [v * 0.90 for v in forecast_vals]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_hist['tanggal'], y=ts_hist['harga_beras_medium_per_kg'],
        name='Data Historis', line=dict(color=C['blue'], width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates[::-1]),
        y=forecast_upper + forecast_lower[::-1],
        fill='toself', fillcolor='rgba(217,43,43,0.06)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval ±10%',
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_vals,
        name='Forecast 2024',
        line=dict(color=C['red'], width=2.5, dash='dash'),
        mode='lines+markers', marker=dict(size=8, symbol='circle')
    ))
    fig.add_vline(x=pd.Timestamp('2024-01-01').timestamp() * 1000,
                  line_dash='dot', line_color='#9dada0',
                  annotation_text='Mulai Forecast',
                  annotation_position='top left',
                  annotation_font_size=11)
    chart_style(fig, h=420, m=dict(l=0, r=0, t=20, b=0))
    fig.update_layout(
        xaxis_title='Tanggal', yaxis_title='Harga (Rp/kg)',
        yaxis_tickformat=',.0f',
        legend=dict(yanchor="top", y=0.97, xanchor="left", x=0.01,
                    bgcolor='rgba(255,255,255,0.9)', bordercolor='#eee', borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="stitle">Tabel Prediksi Bulanan</p>', unsafe_allow_html=True)
    df_forecast = pd.DataFrame({
        'Bulan': [d.strftime('%B %Y') for d in forecast_dates],
        'Forecast (Rp/kg)': [f"Rp{v:,.0f}" for v in forecast_vals],
        'Batas Bawah':  [f"Rp{v:,.0f}" for v in forecast_lower],
        'Batas Atas':   [f"Rp{v:,.0f}" for v in forecast_upper],
        'Status': ['Tinggi' if v > 14200 else 'Normal' for v in forecast_vals]
    })
    st.dataframe(df_forecast, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="ic">
        <div class="ic-head"><strong>Insight Forecast Harga</strong></div>
        <p>Harga beras diprediksi naik signifikan pada <b>Juli–September 2024</b> (musim paceklik)
        dengan puncak <b>Rp14.394/kg</b> pada September 2024.<br>
        <b>Rekomendasi Bulog:</b> Lakukan pembelian cadangan beras sebelum Juli 2024
        untuk menghindari tekanan harga di pasar.</p>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE: SENTIMEN BERITA
# ═══════════════════════════════════════════════════════
elif menu == "Analisis Sentimen Berita":
    st.markdown("""
    <div class="page-header">
        <div class="ph-icon"><span class="material-symbols-outlined">article</span></div>
        <div class="ph-text">
            <h1>Analisis Sentimen Berita Pangan</h1>
            <p>Analisis sentimen teks berita menggunakan Azure AI Language</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<span class="az">Azure AI Language</span>', unsafe_allow_html=True)

    sample_texts = [
        "Produksi padi di Jawa Tengah meningkat 15% berkat program pompanisasi pemerintah",
        "Harga beras melonjak tinggi akibat kekeringan panjang di musim kemarau tahun ini",
        "Bulog berhasil menyerap gabah petani di atas HPP untuk stabilkan harga pangan",
        "Banjir melanda lahan pertanian Jawa Timur, ribuan hektar padi terancam gagal panen",
    ]

    selected = st.selectbox("Pilih contoh berita atau ketik sendiri:",
                            ["Ketik sendiri..."] + sample_texts)
    text_input = st.text_area(
        "Teks berita:",
        value="" if selected == "Ketik sendiri..." else selected,
        height=110,
        placeholder="Masukkan teks berita pangan di sini..."
    )

    if st.button("Analisis Sentimen", type="primary") and text_input:
        try:
            endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
            key      = os.getenv("AZURE_LANGUAGE_KEY")
            client   = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))
            response = client.analyze_sentiment([text_input])
            result   = response[0]
            sentiment = result.sentiment
            scores    = result.confidence_scores
            label_map = {'positive': 'Positif', 'negative': 'Negatif', 'neutral': 'Netral'}

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Sentimen", label_map[sentiment])
            with c2: st.metric("Positif",  f"{scores.positive:.1%}")
            with c3: st.metric("Netral",   f"{scores.neutral:.1%}")
            with c4: st.metric("Negatif",  f"{scores.negative:.1%}")

            fig_sent = go.Figure(go.Bar(
                x=['Positif', 'Netral', 'Negatif'],
                y=[scores.positive, scores.neutral, scores.negative],
                marker_color=[C['green'], C['amber'], C['red']],
                text=[f"{v:.1%}" for v in [scores.positive, scores.neutral, scores.negative]],
                textposition='outside', width=0.35
            ))
            chart_style(fig_sent, h=300, m=dict(l=0, r=0, t=20, b=0))
            fig_sent.update_layout(
                yaxis_title='Confidence Score',
                yaxis=dict(gridcolor='#f0f6f1', range=[0, 1.15], zeroline=False),
                xaxis=dict(showgrid=False, zeroline=False),
            )
            st.plotly_chart(fig_sent, use_container_width=True)

            if sentiment == 'negative':
                st.markdown("""
                <div class="al al-err">
                    <strong>Berita Negatif Terdeteksi</strong>
                    Berita negatif tentang pangan dapat mengindikasikan risiko ketidakstabilan
                    pasokan atau harga. Monitor kondisi lapangan segera.
                </div>""", unsafe_allow_html=True)
            elif sentiment == 'positive':
                st.markdown("""
                <div class="al al-ok">
                    <strong>Berita Positif Terdeteksi</strong>
                    Sentimen positif mengindikasikan kondisi pangan yang baik.
                    Pertahankan program yang sedang berjalan.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="al al-warn">
                    <strong>Sentimen Netral</strong>
                    Berita bersifat informatif tanpa indikasi risiko maupun kondisi positif yang signifikan.
                </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error Azure AI Language: {e}")


# ═══════════════════════════════════════════════════════
# PAGE: REKOMENDASI STRATEGIS
# ═══════════════════════════════════════════════════════
elif menu == "Rekomendasi Strategis":
    st.markdown("""
    <div class="page-header">
        <div class="ph-icon"><span class="material-symbols-outlined">target</span></div>
        <div class="ph-text">
            <h1>Rekomendasi Strategis</h1>
            <p>Actionable insights berbasis model SiPADI untuk Bulog &amp; Kementan</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Rekomendasi Bulog", "Rekomendasi Kementan", "Prioritas Kabupaten"])

    with tab1:
        st.markdown('<p class="stitle">Rekomendasi Operasional Bulog</p>', unsafe_allow_html=True)
        recs_bulog = [
            ("Pembelian Cadangan Preventif", "Tinggi", "Juni 2024",
             "Lakukan pembelian gabah di 5 kabupaten dengan prediksi surplus produksi sebelum Oktober 2024 untuk mengantisipasi kenaikan harga Jul–Sep."),
            ("Optimasi Distribusi Stok", "Tinggi", "Juli 2024",
             "Alokasikan cadangan beras ke 12 kabupaten berisiko tinggi di Jawa Timur berdasarkan peta risiko SiPADI sebelum musim paceklik."),
            ("Stabilisasi Harga", "Sedang", "Agt–Sep 2024",
             "Intervensi pasar diperlukan jika harga melampaui Rp14.400/kg (batas atas forecast). Target operasi pasar di Surabaya, Malang, Semarang."),
        ]
        for judul, pri, timeline, detail in recs_bulog:
            bc = "pb-hi" if pri == "Tinggi" else "pb-md"
            st.markdown(f"""
            <div class="ic">
                <div class="ic-head">
                    <strong>{judul}</strong>
                    <span class="pb {bc}">{pri}</span>
                </div>
                <p><b>Timeline:</b> {timeline} — {detail}</p>
            </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown('<p class="stitle">Rekomendasi Kebijakan Kementan</p>', unsafe_allow_html=True)
        recs_kementan = [
            ("Prioritas Pompanisasi", "Tinggi",
             "Alokasi tambahan 500 unit pompa ke 10 kabupaten dengan skor irigasi terendah dan risiko gagal panen tertinggi berdasarkan model SiPADI."),
            ("Program Intensifikasi Lahan", "Sedang",
             "Fokus peningkatan NDVI di 15 kabupaten dengan kondisi lahan buruk melalui subsidi pupuk dan pendampingan teknologi pertanian presisi."),
            ("Integrasi Early Warning System", "Rendah",
             "Integrasikan SiPADI dengan sistem informasi pertanian Kementan untuk monitoring real-time 58 kabupaten di Jawa Tengah & Jawa Timur."),
        ]
        for judul, pri, detail in recs_kementan:
            bc = "pb-hi" if pri == "Tinggi" else ("pb-md" if pri == "Sedang" else "pb-lo")
            st.markdown(f"""
            <div class="ic">
                <div class="ic-head">
                    <strong>{judul}</strong>
                    <span class="pb {bc}">{pri}</span>
                </div>
                <p>{detail}</p>
            </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown('<p class="stitle">Top 20 Kabupaten — Prioritas Intervensi</p>', unsafe_allow_html=True)
        risiko_kab = df_master.groupby(['kabupaten', 'provinsi']).agg(
            risiko_pct=('risiko_gagal_panen', 'mean'),
            avg_prod=('produktivitas_ton_per_ha', 'mean')
        ).reset_index()
        risiko_kab['risiko_pct'] = risiko_kab['risiko_pct'] * 100
        risiko_kab['Prioritas'] = risiko_kab['risiko_pct'].apply(
            lambda x: 'P1 – Segera' if x >= 30 else ('P2 – Monitor' if x >= 20 else 'P3 – Aman'))
        risiko_kab = risiko_kab.sort_values('risiko_pct', ascending=False)

        fig_bar = px.bar(
            risiko_kab.head(20), x='kabupaten', y='risiko_pct', color='provinsi',
            template='plotly_white',
            labels={'risiko_pct': 'Risiko Gagal Panen (%)', 'kabupaten': ''},
            color_discrete_map={'Jawa Tengah': C['jateng'], 'Jawa Timur': C['blue']}
        )
        fig_bar.update_traces(marker_line_width=0)
        chart_style(fig_bar, h=400, m=dict(l=0, r=0, t=10, b=0))
        fig_bar.update_layout(
            xaxis_tickangle=40,
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor='#f0f6f1'),
            legend_title="Provinsi"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(
            risiko_kab[['kabupaten', 'provinsi', 'risiko_pct', 'avg_prod', 'Prioritas']].rename(columns={
                'kabupaten': 'Kabupaten', 'provinsi': 'Provinsi',
                'risiko_pct': 'Risiko (%)', 'avg_prod': 'Produktivitas (ton/ha)'
            }),
            use_container_width=True, hide_index=True
        )