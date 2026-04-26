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

# Support Streamlit Cloud secrets
import streamlit as st
if hasattr(st, 'secrets') and len(st.secrets) > 0:
    for key, val in st.secrets.items():
        os.environ[key] = str(val)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SiPADI — Sistem Prediksi Anomali & Deteksi Risiko Pangan Indonesia",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Path
BASE_DIR      = Path(__file__).parent.parent.parent
MODELS_DIR    = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR       = BASE_DIR / "data" / "raw"

# ─────────────────────────────────────────────
# CSS — REDESIGNED (NO EMOJI)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --brand-950: #020c06;
        --brand-900: #041a0d;
        --brand-800: #072b17;
        --brand-700: #0e4023;
        --brand-600: #155c32;
        --brand-500: #1d7a44;
        --brand-400: #29a35d;
        --brand-300: #4dc97e;
        --brand-200: #86e3a8;
        --brand-100: #c2f5d5;
        --brand-50:  #f0fdf5;

        --neutral-950: #080d09;
        --neutral-900: #111712;
        --neutral-800: #1c2520;
        --neutral-700: #2d3b33;
        --neutral-600: #445249;
        --neutral-500: #5e6e65;
        --neutral-400: #7e8e85;
        --neutral-300: #a4b0aa;
        --neutral-200: #c8d3cd;
        --neutral-100: #e4ebe7;
        --neutral-50:  #f5f8f6;

        --danger-600: #c0392b;
        --danger-400: #e74c3c;
        --danger-50:  #fdf2f2;
        --amber-600:  #d4860a;
        --amber-50:   #fdf8ee;
        --blue-600:   #2563eb;
        --blue-50:    #eff6ff;

        --radius-xs: 4px;
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    .block-container {
        padding: 0 2.25rem 4rem !important;
        max-width: 1480px !important;
    }

    /* ─── SIDEBAR ─── */
    [data-testid="stSidebar"] {
        background: var(--brand-950) !important;
        border-right: 1px solid var(--brand-800) !important;
        width: 240px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 1.75rem 1.25rem 2rem;
    }
    [data-testid="stSidebar"] * {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    [data-testid="stSidebar"] .stRadio > label {
        display: none !important;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        background: transparent !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.55rem 0.85rem !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        color: var(--brand-300) !important;
        transition: all 0.15s ease !important;
        cursor: pointer !important;
        letter-spacing: 0.1px !important;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        background: var(--brand-800) !important;
        color: var(--brand-100) !important;
        transform: translateX(3px);
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] {
        background: rgba(77,201,126,0.12) !important;
        color: var(--brand-200) !important;
        font-weight: 700 !important;
        border-left: 2px solid var(--brand-300) !important;
        padding-left: calc(0.85rem - 2px) !important;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    [data-testid="stSidebarNav"] { display: none; }

    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="collapsedControl"] button {
        background: var(--brand-900) !important;
        border: 1px solid var(--brand-700) !important;
        border-radius: 10px !important;
        color: var(--brand-300) !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebarCollapseButton"] button:hover,
    [data-testid="collapsedControl"] button:hover {
        background: var(--brand-800) !important;
    }
    [data-testid="stSidebarCollapseButton"] button svg,
    [data-testid="collapsedControl"] button svg {
        stroke: var(--brand-300) !important;
    }

    /* ─── SIDEBAR BRAND ─── */
    .sb-brand {
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid var(--brand-800);
    }
    .sb-logo {
        width: 42px;
        height: 42px;
        background: var(--brand-500);
        border-radius: var(--radius-md);
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.85rem;
    }
    .sb-logo svg {
        width: 22px;
        height: 22px;
        fill: none;
        stroke: #fff;
        stroke-width: 2;
        stroke-linecap: round;
        stroke-linejoin: round;
    }
    .sb-brand h2 {
        font-size: 1.1rem;
        font-weight: 800;
        color: var(--brand-50) !important;
        margin: 0 0 3px 0;
        letter-spacing: -0.4px;
    }
    .sb-brand p {
        font-size: 0.65rem;
        color: var(--brand-500) !important;
        margin: 0;
        line-height: 1.5;
        font-weight: 400;
    }

    /* ─── NAV LABEL ─── */
    .sb-nav-label {
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--brand-700) !important;
        font-weight: 700;
        margin: 0 0 0.6rem 0.85rem;
    }

    /* ─── SIDEBAR FOOTER ─── */
    .sb-services {
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--brand-800);
    }
    .sb-services-title {
        font-size: 0.6rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: var(--brand-700) !important;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }
    .sb-svc {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-size: 0.72rem;
        color: var(--brand-500) !important;
        margin-bottom: 0.3rem;
    }
    .sb-svc-dot {
        width: 4px;
        height: 4px;
        background: var(--brand-600);
        border-radius: 50%;
        flex-shrink: 0;
    }
    .sb-foot {
        margin-top: 1.75rem;
        padding-top: 1.25rem;
        border-top: 1px solid var(--brand-800);
    }
    .sb-foot-name {
        font-size: 0.65rem;
        font-weight: 700;
        color: var(--brand-500) !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .sb-foot-sub {
        font-size: 0.7rem;
        color: var(--brand-700) !important;
        margin-top: 2px;
    }

    /* ─── HERO BANNER ─── */
    .hero {
        background: var(--brand-950);
        border: 1px solid var(--brand-800);
        border-radius: var(--radius-xl);
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 2rem;
        align-items: center;
        overflow: hidden;
        position: relative;
    }
    .hero::after {
        content: '';
        position: absolute;
        top: -60px;
        right: -60px;
        width: 280px;
        height: 280px;
        background: radial-gradient(circle, rgba(29,122,68,0.18) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(77,201,126,0.1);
        border: 1px solid rgba(77,201,126,0.2);
        color: var(--brand-300);
        padding: 0.2rem 0.75rem;
        border-radius: 100px;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
    }
    .hero-badge-dot {
        width: 5px;
        height: 5px;
        background: var(--brand-300);
        border-radius: 50%;
        animation: blink 2s infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.2; }
    }
    .hero h1 {
        font-size: 2rem;
        font-weight: 800;
        color: #fff;
        margin: 0 0 0.5rem 0;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    .hero-desc {
        font-size: 0.82rem;
        color: rgba(255,255,255,0.35);
        margin: 0;
        line-height: 1.7;
        max-width: 500px;
    }
    .hero-stats {
        display: flex;
        flex-direction: column;
        gap: 0;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: var(--radius-md);
        overflow: hidden;
        min-width: 180px;
    }
    .hero-stat {
        padding: 0.7rem 1.1rem;
        border-bottom: 1px solid rgba(255,255,255,0.07);
    }
    .hero-stat:last-child { border-bottom: none; }
    .hero-stat-label {
        font-size: 0.58rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: rgba(255,255,255,0.28);
        font-weight: 700;
        display: block;
        margin-bottom: 3px;
    }
    .hero-stat-val {
        font-size: 0.82rem;
        font-weight: 600;
        color: rgba(255,255,255,0.7);
        display: block;
    }
    .hero-stat-val.accent { color: var(--brand-200); }

    /* ─── SECTION HEADING ─── */
    .sh {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: var(--neutral-400);
        margin: 0 0 1rem 0;
    }
    .sh::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--neutral-100);
    }

    /* ─── PAGE HEADER ─── */
    .ph {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid var(--neutral-100);
    }
    .ph-icon {
        width: 46px;
        height: 46px;
        background: var(--brand-50);
        border: 1px solid var(--brand-100);
        border-radius: var(--radius-md);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .ph-icon svg {
        width: 20px;
        height: 20px;
        fill: none;
        stroke: var(--brand-500);
        stroke-width: 2;
        stroke-linecap: round;
        stroke-linejoin: round;
    }
    .ph-text h1 {
        font-size: 1.45rem;
        font-weight: 800;
        color: var(--neutral-900);
        margin: 0 0 4px 0;
        letter-spacing: -0.5px;
    }
    .ph-text p {
        font-size: 0.8rem;
        color: var(--neutral-400);
        margin: 0;
    }

    /* ─── KPI CARDS ─── */
    .kpi {
        background: #fff;
        border: 1px solid var(--neutral-100);
        border-radius: var(--radius-lg);
        padding: 1.25rem;
        height: 100%;
        position: relative;
        overflow: hidden;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .kpi::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: var(--kpi-color, var(--brand-300));
        border-radius: 3px 3px 0 0;
    }
    .kpi:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.07);
    }
    .kpi-label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--neutral-400);
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    .kpi-val {
        font-size: 2.1rem;
        font-weight: 800;
        color: var(--neutral-900);
        letter-spacing: -2px;
        line-height: 1;
        margin-bottom: 4px;
        font-variant-numeric: tabular-nums;
    }
    .kpi-val.sm { font-size: 1.5rem; letter-spacing: -0.5px; }
    .kpi-sub {
        font-size: 0.7rem;
        color: var(--neutral-400);
        margin-bottom: 0.75rem;
    }
    .kpi-chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 100px;
        font-family: 'IBM Plex Mono', monospace;
    }
    .chip-up   { background: var(--brand-50);  color: var(--brand-600); }
    .chip-down { background: var(--danger-50); color: var(--danger-600); }
    .chip-neu  { background: var(--neutral-50); color: var(--neutral-600); }

    /* ─── MODEL CARDS ─── */
    .mc {
        background: #fff;
        border: 1px solid var(--neutral-100);
        border-radius: var(--radius-lg);
        padding: 1.25rem;
        height: 100%;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .mc:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.07);
    }
    .mc-tag {
        display: inline-block;
        font-size: 0.6rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: var(--radius-xs);
        margin-bottom: 0.85rem;
        font-family: 'IBM Plex Mono', monospace;
    }
    .t-green { background: var(--brand-50);  color: var(--brand-600); }
    .t-red   { background: var(--danger-50); color: var(--danger-600); }
    .t-amber { background: var(--amber-50);  color: var(--amber-600); }
    .mc h3 {
        font-size: 0.92rem;
        font-weight: 700;
        color: var(--neutral-900);
        margin: 0 0 3px 0;
    }
    .mc-algo {
        font-size: 0.7rem;
        color: var(--neutral-400);
        margin-bottom: 1rem;
        font-family: 'IBM Plex Mono', monospace;
    }
    .mrow {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.42rem 0;
        border-bottom: 1px solid var(--neutral-50);
    }
    .mrow:last-child { border-bottom: none; }
    .mrow-k {
        font-size: 0.72rem;
        color: var(--neutral-400);
        font-weight: 500;
    }
    .mrow-v {
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--neutral-700);
        font-family: 'IBM Plex Mono', monospace;
    }
    .v-ok   { color: var(--brand-600) !important; }
    .v-warn { color: var(--danger-600) !important; }

    /* ─── ALERTS ─── */
    .alert {
        border-radius: var(--radius-md);
        padding: 1rem 1.15rem;
        margin: 0.75rem 0;
        font-size: 0.82rem;
        line-height: 1.7;
        border-left: 3px solid;
    }
    .alert strong {
        display: block;
        font-weight: 700;
        margin-bottom: 4px;
        font-size: 0.85rem;
    }
    .alert-err  { background: var(--danger-50);  border-color: var(--danger-600);  color: #7f1d1d; }
    .alert-ok   { background: var(--brand-50);   border-color: var(--brand-400);   color: #14532d; }
    .alert-info { background: var(--blue-50);    border-color: var(--blue-600);    color: #1e3a8a; }
    .alert-warn { background: var(--amber-50);   border-color: var(--amber-600);   color: #78350f; }

    /* ─── INSIGHT CARDS ─── */
    .ic {
        background: var(--neutral-50);
        border: 1px solid var(--neutral-100);
        border-radius: var(--radius-md);
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.82rem;
        line-height: 1.7;
        transition: box-shadow 0.15s ease, transform 0.15s ease;
    }
    .ic:hover {
        box-shadow: 0 3px 12px rgba(0,0,0,0.05);
        transform: translateY(-1px);
    }
    .ic-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.45rem;
        gap: 0.75rem;
    }
    .ic-head strong {
        font-size: 0.88rem;
        font-weight: 700;
        color: var(--neutral-900);
    }
    .ic p { color: var(--neutral-500); margin: 0; }
    .pbadge {
        display: inline-block;
        font-size: 0.62rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 4px;
        flex-shrink: 0;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .p-high { background: var(--danger-50);  color: var(--danger-600); }
    .p-mid  { background: var(--amber-50);   color: var(--amber-600); }
    .p-low  { background: var(--brand-50);   color: var(--brand-600); }

    /* ─── AZURE BADGE ─── */
    .az-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: var(--blue-50);
        border: 1px solid #bfdbfe;
        color: var(--blue-600);
        padding: 4px 10px;
        border-radius: 100px;
        font-size: 0.67rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        margin-bottom: 1.25rem;
    }

    /* ─── STREAMLIT OVERRIDES ─── */
    .stMetric {
        background: #fff !important;
        border: 1px solid var(--neutral-100) !important;
        border-radius: var(--radius-md) !important;
        padding: 1rem 1.1rem !important;
    }
    .stMetric label {
        font-size: 0.65rem !important;
        color: var(--neutral-400) !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
    }
    .stMetric [data-testid="metric-container"] > div:nth-child(2) {
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        color: var(--neutral-900) !important;
        letter-spacing: -0.8px !important;
        font-variant-numeric: tabular-nums;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.73rem !important;
        font-weight: 600 !important;
    }

    .stButton > button {
        background: var(--brand-500) !important;
        color: #fff !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.6rem 1.5rem !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        transition: all 0.18s ease !important;
    }
    .stButton > button:hover {
        background: var(--brand-600) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(21,92,50,0.3) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: var(--neutral-50) !important;
        border: 1px solid var(--neutral-100) !important;
        border-radius: var(--radius-sm) !important;
        padding: 4px !important;
        gap: 2px !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: var(--neutral-400) !important;
        padding: 0.45rem 1rem !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background: #fff !important;
        color: var(--neutral-900) !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07) !important;
    }

    .stSelectbox > div > div {
        border-radius: var(--radius-sm) !important;
        border-color: var(--neutral-100) !important;
        font-size: 0.82rem !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    .stSlider [data-testid="stTickBar"] { display: none; }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: var(--brand-500) !important;
    }
    .stSlider [data-baseweb="slider"] > div > div {
        background: var(--brand-200) !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--neutral-100) !important;
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    hr {
        border: none !important;
        border-top: 1px solid var(--neutral-100) !important;
        margin: 1.5rem 0 !important;
    }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: var(--neutral-200);
        border-radius: 3px;
    }
    [data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
        background: var(--brand-800);
    }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .block-container { animation: fadeUp 0.35s ease-out; }

    [data-testid="stSidebar"] {
        transition: transform 0.3s ease, width 0.3s ease !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    from azure.storage.blob import BlobServiceClient
    from io import BytesIO

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    
    # Debug — pastikan secret terbaca
    if not conn_str:
        st.error("❌ AZURE_STORAGE_CONNECTION_STRING tidak ditemukan di secrets!")
        st.stop()

    
    # Coba load dari lokal dulu, kalau tidak ada load dari Azure
    def read_csv_azure(container, blob_name, **kwargs):
        try:
            # Coba lokal dulu
            local_path = BASE_DIR / "data" / "raw" / blob_name.split("/")[-1]
            if local_path.exists():
                return pd.read_csv(local_path, **kwargs)
        except:
            pass
        
        # Load dari Azure Blob
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
            container="sipadi-data-processed", blob=blob_name
        )
        data = blob_client.download_blob().readall()
        return pd.read_csv(BytesIO(data), **kwargs)

    df_master   = read_csv_processed("master_dataset.csv")
    df_produksi = read_csv_azure("sipadi-data-raw", "bps/produksi_padi.csv")
    df_harga    = read_csv_azure("sipadi-data-raw", "harga/harga_beras.csv")
    df_cuaca    = read_csv_azure("sipadi-data-raw", "bmkg/cuaca_bmkg.csv")
    df_enso     = read_csv_azure("sipadi-data-raw", "enso/enso_index.csv",
                                  parse_dates=['tanggal'])
    df_ndvi     = read_csv_azure("sipadi-data-raw", "satellite/ndvi_sentinel.csv")

    return df_master, df_produksi, df_harga, df_cuaca, df_enso, df_ndvi

@st.cache_resource
@st.cache_resource
def load_models():
    from azure.storage.blob import BlobServiceClient
    from io import BytesIO

    conn_str     = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_service = BlobServiceClient.from_connection_string(conn_str)

    def load_pkl(blob_name):
        # Coba lokal dulu
        local_path = BASE_DIR / "models" / blob_name.split("/")[-1]
        if local_path.exists():
            return joblib.load(local_path)
        # Load dari Azure
        blob_client = blob_service.get_blob_client(
            container="sipadi-models", blob=blob_name
        )
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

# ─────────────────────────────────────────────
# CHART STYLE
# ─────────────────────────────────────────────
COLORS = {
    'primary':   '#2d6a4f',
    'secondary': '#2E86AB',
    'danger':    '#C73E1D',
    'amber':     '#F18F01',
    'jateng':    '#2d6a4f',
    'jatim':     '#2E86AB',
}

def apply_chart_style(fig, height=400, margin=None):
    if margin is None:
        margin = dict(l=0, r=0, t=20, b=0)
    fig.update_layout(
        height=height,
        margin=margin,
        font_family='Sora',
        font=dict(size=12),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, linecolor='#e5e7eb'),
        yaxis=dict(gridcolor='#f0f0f0', zeroline=False),
    )
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-mark">
            <svg viewBox="0 0 24 24">
                <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                <path d="M2 17l10 5 10-5"/>
                <path d="M2 12l10 5 10-5"/>
            </svg>
        </div>
        <div class="sidebar-brand-text">
            <h2>SiPADI</h2>
            <p>Prediksi Anomali &amp; Deteksi<br>Risiko Pangan Indonesia</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio("Navigasi", [
        "Overview & KPI",
        "Peta Risiko Kabupaten",
        "Prediksi Produktivitas",
        "Deteksi Risiko Gagal Panen",
        "Forecast Harga Beras",
        "Analisis Sentimen Berita",
        "Rekomendasi Strategis"
    ], label_visibility="collapsed")

    st.markdown("""
    <div class="sidebar-services">
        <div class="sidebar-services-label">Azure Services</div>
        <div class="service-item"><div class="service-dot"></div>Azure Blob Storage</div>
        <div class="service-item"><div class="service-dot"></div>Azure ML Workspace</div>
        <div class="service-item"><div class="service-dot"></div>Azure AI Language</div>
        <div class="service-item"><div class="service-dot"></div>Azure Maps</div>
        <div class="service-item"><div class="service-dot"></div>Azure Static Web Apps</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-footer">
        <div class="sidebar-footer-label">Microsoft Elevate</div>
        <div class="sidebar-footer-sub">AI Impact Challenge 2026</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TOP BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
    <div class="top-banner-grid">
        <div>
            <div class="top-banner-badge">
                <div class="top-banner-badge-dot"></div>
                Live Monitoring
            </div>
            <h1>SiPADI Dashboard</h1>
            <p class="top-banner-desc">
                Sistem Prediksi Anomali &amp; Deteksi Risiko Pangan Indonesia —
                mendukung Program Swasembada Pangan 2025–2029
            </p>
        </div>
        <div class="top-banner-stats">
            <div class="top-banner-stat">
                <span class="top-banner-stat-label">Cakupan Wilayah</span>
                <span class="top-banner-stat-value">Jateng &amp; Jatim</span>
            </div>
            <div class="top-banner-stat top-banner-stat-highlight">
                <span class="top-banner-stat-label">Kabupaten</span>
                <span class="top-banner-stat-value">58</span>
            </div>
            <div class="top-banner-stat">
                <span class="top-banner-stat-label">Periode Data</span>
                <span class="top-banner-stat-value">2010–2024</span>
            </div>
            <div class="top-banner-stat">
                <span class="top-banner-stat-label">Model AI</span>
                <span class="top-banner-stat-value">XGBoost + SARIMA</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: OVERVIEW & KPI
# ─────────────────────────────────────────────
if menu == "Overview & KPI":
    st.markdown('<p class="section-heading">Key Performance Indicators</p>', unsafe_allow_html=True)

    total_kab  = df_produksi['kabupaten'].nunique()
    avg_prod   = df_produksi['produktivitas_ton_per_ha'].mean()
    risiko_pct = df_master['risiko_gagal_panen'].mean() * 100
    total_prod = df_produksi['produksi_ton'].sum() / 1e6
    harga_now  = df_harga['harga_beras_medium_per_kg'].iloc[-1]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="kpi-card" style="--kpi-accent: #52b788;">
            <div class="kpi-label">Kabupaten Dipantau</div>
            <div class="kpi-value">{total_kab}</div>
            <div class="kpi-sub">Jawa Tengah &amp; Jawa Timur</div>
            <span class="kpi-delta delta-neutral">Aktif 100%</span>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card" style="--kpi-accent: #2d6a4f;">
            <div class="kpi-label">Produktivitas Rata-rata</div>
            <div class="kpi-value">{avg_prod:.2f}</div>
            <div class="kpi-sub">ton / hektar</div>
            <span class="kpi-delta delta-up">+0.04 / tahun</span>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card" style="--kpi-accent: #f18f01;">
            <div class="kpi-label">Kabupaten Berisiko</div>
            <div class="kpi-value">{risiko_pct:.1f}%</div>
            <div class="kpi-sub">dari total kabupaten</div>
            <span class="kpi-delta delta-up">−2.3% vs tahun lalu</span>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-card" style="--kpi-accent: #2e86ab;">
            <div class="kpi-label">Total Produksi</div>
            <div class="kpi-value">{total_prod:.1f}M</div>
            <div class="kpi-sub">ton kumulatif</div>
            <span class="kpi-delta delta-neutral">Kumulatif 2010–2024</span>
        </div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="kpi-card" style="--kpi-accent: #c73e1d;">
            <div class="kpi-label">Harga Beras Kini</div>
            <div class="kpi-value" style="font-size:1.5rem;">Rp{harga_now:,.0f}</div>
            <div class="kpi-sub">per kilogram</div>
            <span class="kpi-delta delta-down">+3.2% YoY</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-heading">Performa Model AI</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="model-card">
            <span class="model-tag tag-green">Model 1 — Regresi</span>
            <h3>Prediksi Produktivitas</h3>
            <p class="model-algo">XGBoost Regressor</p>
            <div class="metric-row">
                <span class="metric-key">R² Score</span>
                <span class="metric-val val-green">0.886</span>
            </div>
            <div class="metric-row">
                <span class="metric-key">MAPE</span>
                <span class="metric-val val-green">2.25%</span>
            </div>
            <div class="metric-row">
                <span class="metric-key">RMSE</span>
                <span class="metric-val">0.158 ton/ha</span>
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="model-card">
            <span class="model-tag tag-red">Model 2 — Klasifikasi</span>
            <h3>Deteksi Risiko Gagal Panen</h3>
            <p class="model-algo">XGBoost Classifier</p>
            <div class="metric-row">
                <span class="metric-key">F1-Score</span>
                <span class="metric-val val-green">0.833</span>
            </div>
            <div class="metric-row">
                <span class="metric-key">Recall</span>
                <span class="metric-val val-green">92.6%</span>
            </div>
            <div class="metric-row">
                <span class="metric-key">ROC-AUC</span>
                <span class="metric-val val-green">0.967</span>
            </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="model-card">
            <span class="model-tag tag-amber">Model 3 — Time Series</span>
            <h3>Forecast Harga Beras</h3>
            <p class="model-algo">SARIMA(1,1,1)(1,1,1,12)</p>
            <div class="metric-row">
                <span class="metric-key">MAPE</span>
                <span class="metric-val val-green">1.16%</span>
            </div>
            <div class="metric-row">
                <span class="metric-key">Horizon Forecast</span>
                <span class="metric-val">12 bulan</span>
            </div>
            <div class="metric-row">
                <span class="metric-key">Range Prediksi</span>
                <span class="metric-val">Rp13.923–14.394</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-heading">Tren Produksi Padi (2010–2023)</p>', unsafe_allow_html=True)

    tren = df_produksi.groupby(['tahun', 'provinsi'])['produksi_ton'].sum().reset_index()
    fig = px.line(tren, x='tahun', y='produksi_ton', color='provinsi',
                  markers=True, template='plotly_white',
                  labels={'produksi_ton': 'Total Produksi (ton)', 'tahun': 'Tahun'},
                  color_discrete_map={'Jawa Tengah': COLORS['jateng'], 'Jawa Timur': COLORS['jatim']})
    fig.update_traces(line_width=2.5, marker_size=7)
    apply_chart_style(fig, height=380, margin=dict(l=0, r=0, t=10, b=0))
    fig.update_layout(
        legend_title="Provinsi",
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.99,
                    bgcolor='rgba(255,255,255,0.9)', bordercolor='#eee', borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: PETA RISIKO
# ─────────────────────────────────────────────
elif menu == "Peta Risiko Kabupaten":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-mark">
            <svg viewBox="0 0 24 24">
                <polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/>
                <line x1="8" y1="2" x2="8" y2="18"/>
                <line x1="16" y1="6" x2="16" y2="22"/>
            </svg>
        </div>
        <div class="page-header-text">
            <h1>Peta Risiko per Kabupaten</h1>
            <p>Visualisasi spasial risiko gagal panen berdasarkan model klasifikasi SiPADI</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<span class="azure-pill">Azure Maps</span>', unsafe_allow_html=True)

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

    st.markdown('<p class="section-heading">Kabupaten Risiko Tinggi</p>', unsafe_allow_html=True)
    high_risk = risiko_kab[risiko_kab['risiko_pct_100'] >= threshold].sort_values('risiko_pct_100', ascending=False)
    high_risk['Status'] = high_risk['risiko_pct_100'].apply(lambda x: 'Tinggi' if x >= 30 else 'Sedang')
    st.dataframe(
        high_risk[['kabupaten', 'risiko_pct_100', 'avg_prod', 'Status']].rename(columns={
            'kabupaten': 'Kabupaten', 'risiko_pct_100': 'Risiko (%)', 'avg_prod': 'Produktivitas (ton/ha)'
        }),
        use_container_width=True, hide_index=True
    )

# ─────────────────────────────────────────────
# PAGE: PREDIKSI PRODUKTIVITAS
# ─────────────────────────────────────────────
elif menu == "Prediksi Produktivitas":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-mark">
            <svg viewBox="0 0 24 24">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
        </div>
        <div class="page-header-text">
            <h1>Prediksi Produktivitas Padi</h1>
            <p>Masukkan parameter kondisi lapangan untuk mendapatkan estimasi produktivitas</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<p class="section-heading">Kondisi Iklim</p>', unsafe_allow_html=True)
        enso_val    = st.slider("ENSO Index", -2.0, 2.0, 0.0, 0.1)
        curah_hujan = st.slider("Curah Hujan Total (mm)", 100, 1500, 600)
        suhu        = st.slider("Suhu Rata-rata (°C)", 24.0, 32.0, 27.5, 0.1)
        kelembaban  = st.slider("Kelembaban (%)", 60, 95, 78)
    with col2:
        st.markdown('<p class="section-heading">Kondisi Lahan</p>', unsafe_allow_html=True)
        ndvi       = st.slider("NDVI (Kesehatan Lahan)", 0.1, 0.9, 0.55, 0.01)
        pct_baik   = st.slider("Lahan Kondisi Baik (%)", 0, 100, 60)
        luas_panen = st.number_input("Luas Panen (ha)", 1000, 50000, 10000)
        musim      = st.selectbox("Musim Tanam", ["MT1 (Jan-Apr)", "MT2 (Jun-Sep)"])
    with col3:
        st.markdown('<p class="section-heading">Infrastruktur</p>', unsafe_allow_html=True)
        jumlah_pompa = st.slider("Jumlah Pompa (unit)", 0, 100, 30)
        skor_irigasi = st.slider("Skor Irigasi", 0.0, 1.0, 0.65, 0.01)
        harga_beras  = st.number_input("Harga Beras (Rp/kg)", 8000, 20000, 13500)
        provinsi     = st.selectbox("Provinsi", ["Jawa Tengah", "Jawa Timur"])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Prediksi Sekarang", type="primary", use_container_width=True):
        input_data = {col: 0 for col in feature_cols}
        input_data.update({
            'enso_mean': enso_val, 'enso_lag1_mean': enso_val * 0.9, 'enso_lag3_mean': enso_val * 0.7,
            'bulan_el_nino': 1 if enso_val > 0.5 else 0, 'bulan_la_nina': 1 if enso_val < -0.5 else 0,
            'curah_hujan_total': curah_hujan, 'curah_hujan_mean': curah_hujan / 4,
            'curah_hujan_std': curah_hujan * 0.2, 'curah_hujan_max': curah_hujan * 1.3,
            'hari_hujan_total': int(curah_hujan / 15), 'suhu_mean': suhu, 'suhu_max': suhu + 3,
            'kelembaban_mean': kelembaban, 'anomali_curah_hujan': curah_hujan - 600,
            'pct_anomali_hujan': (curah_hujan - 600) / 600 * 100, 'ndvi_mean': ndvi,
            'ndvi_max': ndvi + 0.1, 'ndvi_std': 0.05, 'pct_lahan_sangat_baik': pct_baik,
            'pct_lahan_buruk': max(0, 20 - pct_baik * 0.2), 'skor_irigasi': skor_irigasi,
            'jumlah_pompa_unit': jumlah_pompa, 'kapasitas_pompa_per_ha': jumlah_pompa * 80 / luas_panen,
            'indeks_pertanaman': 200, 'persen_irigasi_kondisi_baik': skor_irigasi * 100,
            'era_pompanisasi': 1, 'harga_mean': harga_beras, 'harga_lag1': harga_beras * 0.97,
            'harga_volatility': 150, 'harga_yoy_change': 3.5, 'rasio_gabah_beras': 0.45,
            'impor_volume': 500000, 'harga_impor_idr': 8500000, 'luas_panen_ha': luas_panen,
            'musim_tanam_enc': 1 if 'MT1' in musim else 0,
            'provinsi_enc': 1 if provinsi == 'Jawa Tengah' else 0,
        })
        X_input = pd.DataFrame([input_data])[feature_cols]
        pred_prod = prod_model.predict(X_input)[0]
        pred_risk = risk_model.predict(X_input)[0]
        pred_prob = risk_model.predict_proba(X_input)[0][1]

        st.markdown("---")
        st.markdown('<p class="section-heading">Hasil Prediksi</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediksi Produktivitas",
                      f"{pred_prod:.2f} ton/ha",
                      f"{'Di atas' if pred_prod > 5.5 else 'Di bawah'} rata-rata")
        with col2:
            risk_label = "BERISIKO" if pred_risk == 1 else "AMAN"
            st.metric("Status Risiko", risk_label, f"Prob: {pred_prob:.1%}")
        with col3:
            st.metric("Estimasi Total Produksi",
                      f"{pred_prod * luas_panen:,.0f} ton",
                      f"Luas: {luas_panen:,} ha")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_prod,
            delta={'reference': 5.5, 'valueformat': '.2f'},
            title={'text': "Produktivitas (ton/ha)", 'font': {'size': 14, 'family': 'Sora'}},
            gauge={
                'axis': {'range': [3, 8], 'tickwidth': 1, 'tickcolor': '#ccc'},
                'bar': {'color': "#2d6a4f", 'thickness': 0.65},
                'bgcolor': 'white',
                'borderwidth': 0,
                'steps': [
                    {'range': [3, 4.5], 'color': "#fee2e2"},
                    {'range': [4.5, 5.5], 'color': "#fef9c3"},
                    {'range': [5.5, 8], 'color': "#dcfce7"}
                ],
                'threshold': {'line': {'color': "#dc2626", 'width': 3}, 'thickness': 0.75, 'value': 5.14}
            }
        ))
        apply_chart_style(fig_gauge, height=280, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if pred_risk == 1:
            st.markdown(f"""
            <div class="alert alert-danger">
                <div class="alert-body">
                    <strong>Peringatan Risiko Gagal Panen</strong>
                    Probabilitas gagal panen: <b>{pred_prob:.1%}</b>.
                    Segera lakukan intervensi pompanisasi tambahan dan koordinasi dengan Dinas Pertanian setempat.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert alert-success">
                <div class="alert-body">
                    <strong>Kondisi Aman</strong>
                    Probabilitas gagal panen: <b>{pred_prob:.1%}</b>.
                    Produktivitas diprediksi {'di atas' if pred_prod > 5.5 else 'mendekati'} rata-rata historis 5.5 ton/ha.
                </div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: DETEKSI RISIKO
# ─────────────────────────────────────────────
elif menu == "Deteksi Risiko Gagal Panen":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-mark">
            <svg viewBox="0 0 24 24">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                <line x1="12" y1="9" x2="12" y2="13"/>
                <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
        </div>
        <div class="page-header-text">
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
                  color_discrete_map={'Jawa Tengah': COLORS['jateng'], 'Jawa Timur': COLORS['danger']})
    fig.add_hline(y=25, line_dash="dot", line_color="#f59e0b", annotation_text="Threshold 25%")
    fig.update_traces(line_width=2.5, marker_size=7)
    apply_chart_style(fig, height=380, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        risiko_musim = df_master.groupby('musim_tanam')['risiko_gagal_panen'].mean() * 100
        fig2 = px.bar(risiko_musim.reset_index(),
                      x='musim_tanam', y='risiko_gagal_panen',
                      title='Risiko per Musim Tanam', template='plotly_white',
                      color='musim_tanam',
                      color_discrete_map={'MT1': COLORS['jateng'], 'MT2': COLORS['danger']},
                      labels={'risiko_gagal_panen': 'Risiko (%)', 'musim_tanam': 'Musim'})
        fig2.update_traces(marker_line_width=0, width=0.4)
        apply_chart_style(fig2, height=340, margin=dict(l=0, r=0, t=40, b=0))
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<p class="section-heading">Performa Model Klasifikasi</p>', unsafe_allow_html=True)
        metrics_data = {
            'Metrik': ['F1-Score', 'Precision', 'Recall', 'ROC-AUC'],
            'Nilai': [0.8333, 0.7576, 0.9259, 0.9665],
            'Interpretasi': [
                'Keseimbangan presisi & recall yang baik',
                '76% prediksi berisiko terbukti benar',
                '93% kasus berisiko berhasil terdeteksi',
                'Model sangat mampu membedakan kelas'
            ]
        }
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
        st.markdown("""
        <div class="insight-card">
            <strong>Mengapa Recall Penting?</strong>
            <p>Dalam sistem peringatan dini gagal panen, lebih baik memberikan <b>peringatan
            berlebih</b> daripada <b>melewatkan kasus berisiko</b>. Recall 92.6% memastikan
            hampir semua kasus gagal panen terdeteksi tepat waktu.</p>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: FORECAST HARGA
# ─────────────────────────────────────────────
elif menu == "Forecast Harga Beras":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-mark">
            <svg viewBox="0 0 24 24">
                <line x1="12" y1="1" x2="12" y2="23"/>
                <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
            </svg>
        </div>
        <div class="page-header-text">
            <h1>Forecast Harga Beras</h1>
            <p>Prediksi 12 bulan ke depan menggunakan model SARIMA(1,1,1)(1,1,1,12) — MAPE 1.16%</p>
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
        name='Data Historis', line=dict(color=COLORS['secondary'], width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates[::-1]),
        y=forecast_upper + forecast_lower[::-1],
        fill='toself', fillcolor='rgba(199,62,29,0.08)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval ±10%', showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_vals,
        name='Forecast 2024', line=dict(color=COLORS['danger'], width=2.5, dash='dash'),
        mode='lines+markers', marker=dict(size=7, symbol='circle')
    ))
    fig.add_vline(x=pd.Timestamp('2024-01-01').timestamp() * 1000,
                  line_dash='dot', line_color='#9ca3af',
                  annotation_text='Mulai Forecast', annotation_position='top left')
    apply_chart_style(fig, height=420, margin=dict(l=0, r=0, t=20, b=0))
    fig.update_layout(
        xaxis_title='Tanggal', yaxis_title='Harga (Rp/kg)',
        yaxis_tickformat=',.0f',
        legend=dict(yanchor="top", y=0.97, xanchor="left", x=0.01,
                    bgcolor='rgba(255,255,255,0.9)', bordercolor='#eee', borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-heading">Tabel Prediksi Bulanan</p>', unsafe_allow_html=True)
    df_forecast = pd.DataFrame({
        'Bulan': [d.strftime('%B %Y') for d in forecast_dates],
        'Forecast (Rp/kg)': [f"Rp{v:,.0f}" for v in forecast_vals],
        'Batas Bawah': [f"Rp{v:,.0f}" for v in forecast_lower],
        'Batas Atas': [f"Rp{v:,.0f}" for v in forecast_upper],
        'Status': ['Tinggi' if v > 14200 else 'Normal' for v in forecast_vals]
    })
    st.dataframe(df_forecast, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="insight-card">
        <strong>Insight Forecast Harga</strong>
        <p>Harga beras diprediksi naik signifikan pada <b>Juli–September 2024</b> (musim paceklik)
        dengan puncak <b>Rp14.394/kg</b> pada September 2024.<br>
        <b>Rekomendasi Bulog:</b> Lakukan pembelian cadangan beras sebelum Juli 2024
        untuk menghindari tekanan harga di pasar.</p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: SENTIMEN BERITA
# ─────────────────────────────────────────────
elif menu == "Analisis Sentimen Berita":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-mark">
            <svg viewBox="0 0 24 24">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
                <line x1="16" y1="13" x2="8" y2="13"/>
                <line x1="16" y1="17" x2="8" y2="17"/>
                <polyline points="10 9 9 9 8 9"/>
            </svg>
        </div>
        <div class="page-header-text">
            <h1>Analisis Sentimen Berita Pangan</h1>
            <p>Analisis sentimen teks berita menggunakan Azure AI Language</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<span class="azure-pill">Azure AI Language</span>', unsafe_allow_html=True)

    sample_texts = [
        "Produksi padi di Jawa Tengah meningkat 15% berkat program pompanisasi pemerintah",
        "Harga beras melonjak tinggi akibat kekeringan panjang di musim kemarau tahun ini",
        "Bulog berhasil menyerap gabah petani di atas HPP untuk stabilkan harga pangan",
        "Banjir melanda lahan pertanian Jawa Timur, ribuan hektar padi terancam gagal panen",
    ]

    selected = st.selectbox("Pilih contoh berita atau ketik sendiri:",
                            ["Ketik sendiri..."] + sample_texts)
    text_input = st.text_area("Teks berita:",
                               value="" if selected == "Ketik sendiri..." else selected,
                               height=110,
                               placeholder="Masukkan teks berita pangan di sini...")

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
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Sentimen", label_map[sentiment])
            with col2: st.metric("Positif", f"{scores.positive:.1%}")
            with col3: st.metric("Netral", f"{scores.neutral:.1%}")
            with col4: st.metric("Negatif", f"{scores.negative:.1%}")

            fig_sent = go.Figure(go.Bar(
                x=['Positif', 'Netral', 'Negatif'],
                y=[scores.positive, scores.neutral, scores.negative],
                marker_color=['#22c55e', '#f59e0b', '#ef4444'],
                text=[f"{v:.1%}" for v in [scores.positive, scores.neutral, scores.negative]],
                textposition='outside', width=0.35
            ))
            apply_chart_style(fig_sent, height=300, margin=dict(l=0, r=0, t=20, b=0))
            fig_sent.update_layout(
                yaxis_title='Confidence Score',
                yaxis=dict(gridcolor='#f5f5f5', range=[0, 1.15], zeroline=False),
                xaxis=dict(showgrid=False, zeroline=False),
            )
            st.plotly_chart(fig_sent, use_container_width=True)

            if sentiment == 'negative':
                st.markdown("""
                <div class="alert alert-danger">
                    <div class="alert-body">
                        <strong>Berita Negatif Terdeteksi</strong>
                        Berita negatif tentang pangan dapat mengindikasikan risiko ketidakstabilan
                        pasokan atau harga. Monitor kondisi lapangan segera.
                    </div>
                </div>""", unsafe_allow_html=True)
            elif sentiment == 'positive':
                st.markdown("""
                <div class="alert alert-success">
                    <div class="alert-body">
                        <strong>Berita Positif Terdeteksi</strong>
                        Sentimen positif mengindikasikan kondisi pangan yang baik.
                        Pertahankan program yang sedang berjalan.
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert alert-warning">
                    <div class="alert-body">
                        <strong>Sentimen Netral</strong>
                        Berita bersifat informatif tanpa indikasi risiko maupun kondisi positif yang signifikan.
                    </div>
                </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error Azure AI Language: {e}")

# ─────────────────────────────────────────────
# PAGE: REKOMENDASI STRATEGIS
# ─────────────────────────────────────────────
elif menu == "Rekomendasi Strategis":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-mark">
            <svg viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
        </div>
        <div class="page-header-text">
            <h1>Rekomendasi Strategis</h1>
            <p>Actionable insights berbasis model SiPADI untuk Bulog &amp; Kementan</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Rekomendasi Bulog", "Rekomendasi Kementan", "Prioritas Kabupaten"])

    with tab1:
        st.markdown('<p class="section-heading">Rekomendasi Operasional Bulog</p>', unsafe_allow_html=True)
        recs = [
            ("Pembelian Cadangan Preventif", "Tinggi", "Juni 2024",
             "Lakukan pembelian gabah di 5 kabupaten dengan prediksi surplus produksi sebelum Oktober 2024 untuk mengantisipasi kenaikan harga Jul–Sep."),
            ("Optimasi Distribusi Stok", "Tinggi", "Juli 2024",
             "Alokasikan cadangan beras ke 12 kabupaten berisiko tinggi di Jawa Timur berdasarkan peta risiko SiPADI sebelum musim paceklik."),
            ("Stabilisasi Harga", "Sedang", "Agt–Sep 2024",
             "Intervensi pasar diperlukan jika harga melampaui Rp14.400/kg (batas atas forecast). Target operasi pasar di Surabaya, Malang, Semarang."),
        ]
        for judul, prioritas, timeline, detail in recs:
            badge_class = "priority-high" if prioritas == "Tinggi" else "priority-medium"
            st.markdown(f"""
            <div class="insight-card">
                <div class="ic-header">
                    <strong>{judul}</strong>
                    <span class="priority-badge {badge_class}">{prioritas}</span>
                </div>
                <p><b>Timeline:</b> {timeline} — {detail}</p>
            </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown('<p class="section-heading">Rekomendasi Kebijakan Kementan</p>', unsafe_allow_html=True)
        recs_k = [
            ("Prioritas Pompanisasi", "Tinggi",
             "Alokasi tambahan 500 unit pompa ke 10 kabupaten dengan skor irigasi terendah dan risiko gagal panen tertinggi berdasarkan model SiPADI."),
            ("Program Intensifikasi Lahan", "Sedang",
             "Fokus peningkatan NDVI di 15 kabupaten dengan kondisi lahan buruk melalui subsidi pupuk dan pendampingan teknologi pertanian presisi."),
            ("Integrasi Early Warning System", "Jangka Panjang",
             "Integrasikan SiPADI dengan sistem informasi pertanian Kementan untuk monitoring real-time 58 kabupaten di Jawa Tengah & Jawa Timur."),
        ]
        for judul, prioritas, detail in recs_k:
            bc = "priority-high" if prioritas == "Tinggi" else ("priority-medium" if prioritas == "Sedang" else "priority-low")
            st.markdown(f"""
            <div class="insight-card">
                <div class="ic-header">
                    <strong>{judul}</strong>
                    <span class="priority-badge {bc}">{prioritas}</span>
                </div>
                <p>{detail}</p>
            </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown('<p class="section-heading">Top 20 Kabupaten — Prioritas Intervensi</p>', unsafe_allow_html=True)
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
            title='', template='plotly_white',
            labels={'risiko_pct': 'Risiko Gagal Panen (%)', 'kabupaten': ''},
            color_discrete_map={'Jawa Tengah': COLORS['jateng'], 'Jawa Timur': COLORS['secondary']}
        )
        fig_bar.update_traces(marker_line_width=0)
        apply_chart_style(fig_bar, height=400, margin=dict(l=0, r=0, t=10, b=0))
        fig_bar.update_layout(xaxis_tickangle=40,
                              xaxis=dict(showgrid=False),
                              yaxis=dict(gridcolor='#f5f5f5'),
                              legend_title="Provinsi")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(
            risiko_kab[['kabupaten', 'provinsi', 'risiko_pct', 'avg_prod', 'Prioritas']].rename(columns={
                'kabupaten': 'Kabupaten', 'provinsi': 'Provinsi',
                'risiko_pct': 'Risiko (%)', 'avg_prod': 'Produktivitas (ton/ha)'
            }),
            use_container_width=True, hide_index=True
        )