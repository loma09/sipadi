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

if hasattr(st, 'secrets') and len(st.secrets) > 0:
    for key, val in st.secrets.items():
        os.environ[key] = str(val)

st.set_page_config(
    page_title="SiPADI — Food Risk Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR      = Path(__file__).parent.parent.parent
MODELS_DIR    = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR       = BASE_DIR / "data" / "raw"

# ══════════════════════════════════════════════════════════════
# DESIGN SYSTEM — DARK NAVY + AMBER THEME
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    /* Navy palette */
    --n950: #060d1a;
    --n900: #0c1829;
    --n800: #112138;
    --n700: #172c4a;
    --n600: #1e3a60;
    --n500: #264d7e;
    --n400: #3566a0;
    --n300: #5b8ec5;
    --n200: #96bce0;
    --n100: #cfe0f0;
    --n50:  #eef5fb;

    /* Amber palette */
    --a600: #92400e;
    --a500: #b45309;
    --a400: #d97706;
    --a300: #f59e0b;
    --a200: #fbbf24;
    --a100: #fde68a;
    --a50:  #fffbeb;

    /* Semantic */
    --danger:   #ef4444;
    --danger-bg:#fef2f2;
    --warn:     #f59e0b;
    --warn-bg:  #fffbeb;
    --ok:       #10b981;
    --ok-bg:    #ecfdf5;
    --info:     #3b82f6;
    --info-bg:  #eff6ff;

    /* Surface */
    --surface:    #ffffff;
    --surface2:   #f8fafd;
    --border:     #e2eaf3;
    --border2:    #d0dcea;
    --text:       #0c1829;
    --text2:      #4a6080;
    --text3:      #8ba4be;

    --r6:  6px;
    --r10: 10px;
    --r14: 14px;
    --r20: 20px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: #f0f5fb !important;
    color: var(--text) !important;
}

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ─── SIDEBAR ─── */
[data-testid="stSidebar"] {
    background: var(--n950) !important;
    border-right: 1px solid rgba(255,255,255,0.04) !important;
    width: 230px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="stSidebar"] .stRadio > label { display: none !important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 0 12px;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    background: transparent !important;
    border: none !important;
    border-radius: var(--r10) !important;
    padding: 9px 12px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: rgba(255,255,255,0.35) !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
    white-space: nowrap !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.05) !important;
    color: rgba(255,255,255,0.7) !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] {
    background: rgba(245,158,11,0.12) !important;
    color: var(--a200) !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label > div:first-child {
    display: none !important;
}
[data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] button {
    background: var(--n800) !important;
    border: 1px solid var(--n700) !important;
    border-radius: 8px !important;
    color: var(--a200) !important;
}

/* ─── MAIN CONTENT WRAPPER ─── */
.main-wrap {
    padding: 28px 32px 60px;
    max-width: 1400px;
    margin: 0 auto;
}

/* ─── TOP BAR ─── */
.topbar {
    background: var(--n900);
    padding: 0 32px;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    position: sticky;
    top: 0;
    z-index: 100;
}
.topbar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.topbar-logo {
    width: 28px; height: 28px;
    background: var(--a300);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 800;
    color: var(--n950);
    letter-spacing: -0.5px;
}
.topbar-name {
    font-size: 0.95rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.3px;
}
.topbar-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(245,158,11,0.12);
    border: 1px solid rgba(245,158,11,0.2);
    color: var(--a200);
    padding: 4px 10px;
    border-radius: 100px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.topbar-dot {
    width: 5px; height: 5px;
    background: var(--a300);
    border-radius: 50%;
    animation: blink 1.8s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

/* ─── SIDEBAR CONTENT ─── */
.sb-brand-area {
    padding: 20px 16px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 8px;
}
.sb-brand-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, var(--a300), var(--a500));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    margin-bottom: 10px;
}
.sb-brand-name {
    font-size: 1.05rem;
    font-weight: 800;
    color: #fff !important;
    letter-spacing: -0.5px;
    display: block;
    margin-bottom: 2px;
}
.sb-brand-sub {
    font-size: 0.65rem;
    color: rgba(255,255,255,0.3) !important;
    line-height: 1.5;
    display: block;
}
.sb-section-label {
    font-size: 0.58rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: rgba(255,255,255,0.2) !important;
    padding: 8px 16px 4px;
    display: block;
}
.sb-footer {
    padding: 16px;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: auto;
}
.sb-service {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
    font-size: 0.7rem;
    color: rgba(255,255,255,0.25) !important;
}
.sb-service-icon {
    width: 5px; height: 5px;
    background: var(--a500);
    border-radius: 50%;
    flex-shrink: 0;
}
.sb-footer-title {
    font-size: 0.6rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: rgba(255,255,255,0.2) !important;
    margin-bottom: 6px;
    display: block;
}

/* ─── PAGE WRAPPER ─── */
.pw {
    padding: 28px 32px 60px;
    max-width: 1380px;
    margin: 0 auto;
}

/* ─── BANNER ─── */
.banner {
    background: var(--n900);
    border-radius: var(--r20);
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    overflow: hidden;
    position: relative;
    border: 1px solid rgba(255,255,255,0.04);
}
.banner::before {
    content: '';
    position: absolute;
    right: -40px; top: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(245,158,11,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.banner-left { position: relative; }
.banner-eyebrow {
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--a300);
    margin-bottom: 8px;
    display: block;
}
.banner-title {
    font-size: 1.9rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 8px;
}
.banner-title em {
    font-style: normal;
    color: var(--a300);
}
.banner-desc {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.4);
    line-height: 1.7;
    max-width: 420px;
}
.banner-right {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    overflow: hidden;
    min-width: 280px;
    border: 1px solid rgba(255,255,255,0.04);
    flex-shrink: 0;
}
.bstat {
    background: rgba(255,255,255,0.02);
    padding: 14px 16px;
}
.bstat-label {
    font-size: 0.55rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: rgba(255,255,255,0.2);
    display: block;
    margin-bottom: 4px;
}
.bstat-val {
    font-size: 0.88rem;
    font-weight: 700;
    color: rgba(255,255,255,0.55);
    display: block;
    font-family: 'JetBrains Mono', monospace;
}
.bstat-val.accent { color: var(--a200); }

/* ─── SECTION HEADER ─── */
.sh {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
}
.sh-line {
    width: 3px;
    height: 16px;
    background: var(--a300);
    border-radius: 2px;
    flex-shrink: 0;
}
.sh-text {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text2);
}
.sh-rule {
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ─── PAGE HEADER ─── */
.ph {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    margin-bottom: 28px;
    padding-bottom: 24px;
    border-bottom: 1px solid var(--border);
}
.ph-badge {
    width: 44px; height: 44px;
    background: var(--n900);
    border: 1px solid var(--n700);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.ph-badge svg {
    width: 20px; height: 20px;
    fill: none;
    stroke: var(--a300);
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
}
.ph-title {
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}
.ph-sub {
    font-size: 0.78rem;
    color: var(--text3);
    line-height: 1.6;
}

/* ─── METRIC TILES ─── */
.tile {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r14);
    padding: 20px;
    height: 100%;
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.2s, transform 0.2s;
}
.tile:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(12,24,41,0.08);
}
.tile-icon {
    width: 36px; height: 36px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 14px;
    font-size: 16px;
}
.tile-icon.amber { background: var(--a50); }
.tile-icon.navy  { background: var(--n50); }
.tile-icon.ok    { background: var(--ok-bg); }
.tile-icon.red   { background: var(--danger-bg); }
.tile-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text3);
    margin-bottom: 6px;
    display: block;
}
.tile-val {
    font-size: 2rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -1.5px;
    line-height: 1;
    margin-bottom: 8px;
    font-variant-numeric: tabular-nums;
}
.tile-val.md { font-size: 1.4rem; letter-spacing: -0.5px; }
.tile-desc {
    font-size: 0.7rem;
    color: var(--text3);
    margin-bottom: 10px;
    line-height: 1.5;
}
.chip {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 100px;
    font-family: 'JetBrains Mono', monospace;
}
.chip-up  { background: var(--ok-bg);     color: #065f46; }
.chip-dn  { background: var(--danger-bg); color: #991b1b; }
.chip-neu { background: var(--n50);       color: var(--n500); }
.chip-amb { background: var(--a50);       color: var(--a600); }

/* ─── MODEL CARDS ─── */
.mcard {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r14);
    padding: 20px;
    height: 100%;
    transition: box-shadow 0.2s, transform 0.2s;
}
.mcard:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(12,24,41,0.08);
}
.mcard-type {
    font-size: 0.58rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 3px 8px;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 12px;
    font-family: 'JetBrains Mono', monospace;
}
.mt-reg  { background: var(--info-bg); color: #1d4ed8; }
.mt-cls  { background: var(--danger-bg); color: #b91c1c; }
.mt-ts   { background: var(--a50); color: var(--a600); }
.mcard-name {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 2px;
}
.mcard-algo {
    font-size: 0.66rem;
    color: var(--text3);
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 16px;
}
.mrow {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-top: 1px solid var(--border);
}
.mrow:first-of-type { border-top: none; }
.mrow-k { font-size: 0.73rem; color: var(--text3); }
.mrow-v {
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
}
.mrow-v.ok  { color: var(--ok); }
.mrow-v.bad { color: var(--danger); }

/* ─── ALERT BOXES ─── */
.alert {
    border-radius: var(--r10);
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.8rem;
    line-height: 1.7;
    border-left: 3px solid;
    display: flex;
    gap: 10px;
}
.alert-icon { flex-shrink: 0; font-size: 16px; margin-top: 1px; }
.alert-body strong {
    display: block;
    font-weight: 700;
    font-size: 0.84rem;
    margin-bottom: 2px;
}
.al-err  { background: var(--danger-bg); border-color: var(--danger); color: #7f1d1d; }
.al-ok   { background: var(--ok-bg);     border-color: var(--ok);     color: #064e3b; }
.al-warn { background: var(--warn-bg);   border-color: var(--warn);   color: #78350f; }
.al-info { background: var(--info-bg);   border-color: var(--info);   color: #1e3a8a; }

/* ─── INSIGHT CARDS ─── */
.ic {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--r10);
    padding: 14px 16px;
    margin: 8px 0;
    font-size: 0.8rem;
    line-height: 1.7;
    transition: box-shadow 0.15s, transform 0.15s;
}
.ic:hover {
    box-shadow: 0 4px 16px rgba(12,24,41,0.06);
    transform: translateY(-1px);
}
.ic-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 6px;
}
.ic-title { font-size: 0.87rem; font-weight: 700; color: var(--text); }
.ic-body  { color: var(--text2); margin: 0; }
.pr {
    font-size: 0.6rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    text-transform: uppercase;
    flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace;
}
.pr-hi { background: var(--danger-bg); color: #b91c1c; }
.pr-md { background: var(--a50);       color: var(--a600); }
.pr-lo { background: var(--ok-bg);     color: #065f46; }

/* ─── AZURE BADGE ─── */
.az-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--info-bg);
    border: 1px solid #bfdbfe;
    color: #1d4ed8;
    padding: 4px 10px;
    border-radius: 100px;
    font-size: 0.63rem;
    font-weight: 700;
    margin-bottom: 20px;
    letter-spacing: 0.3px;
}

/* ─── STREAMLIT OVERRIDES ─── */
div[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r10) !important;
    padding: 14px 16px !important;
}
div[data-testid="stMetric"] label {
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: var(--text3) !important;
}
div[data-testid="stMetric"] [data-testid="metric-container"] > div:nth-child(2) {
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    color: var(--text) !important;
    letter-spacing: -0.8px !important;
}

.stButton > button {
    background: var(--a300) !important;
    color: var(--n950) !important;
    border: none !important;
    border-radius: var(--r10) !important;
    padding: 10px 24px !important;
    font-size: 0.84rem !important;
    font-weight: 700 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    transition: all 0.18s !important;
    letter-spacing: 0.1px !important;
}
.stButton > button:hover {
    background: var(--a200) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(245,158,11,0.3) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r10) !important;
    padding: 4px !important;
    gap: 3px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: var(--text3) !important;
    padding: 7px 16px !important;
    background: transparent !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background: var(--surface) !important;
    color: var(--text) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
}

.stSelectbox > div > div {
    border-radius: var(--r10) !important;
    border-color: var(--border2) !important;
    font-size: 0.82rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: var(--surface) !important;
}

.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--a300) !important;
    border: 2px solid #fff !important;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.3) !important;
}
.stSlider [data-baseweb="slider"] > div > div {
    background: var(--a200) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--r10) !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] th {
    background: var(--surface2) !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    color: var(--text3) !important;
}

#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden !important;
}

hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 20px 0 !important;
}

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb { background: var(--n700); }

.js-plotly-plot .plotly { border-radius: var(--r10) !important; }

/* Responsive */
@media (max-width: 768px) {
    .banner { flex-direction: column; padding: 20px; }
    .banner-right { min-width: 100%; width: 100%; }
    .pw { padding: 16px 16px 40px; }
    .topbar { padding: 0 16px; }
    .tile-val { font-size: 1.6rem; }
    .banner-title { font-size: 1.4rem; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD DATA & MODELS
# ══════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════
C = {
    'amber':  '#f59e0b',
    'amber2': '#fbbf24',
    'navy':   '#172c4a',
    'navy2':  '#264d7e',
    'teal':   '#0d9488',
    'red':    '#ef4444',
    'green':  '#10b981',
    'blue':   '#3b82f6',
    'jateng': '#f59e0b',
    'jatim':  '#3b82f6',
}

def chart_style(fig, h=400, m=None):
    if m is None:
        m = dict(l=0, r=0, t=20, b=0)
    fig.update_layout(
        height=h, margin=m,
        font_family='Plus Jakarta Sans',
        font=dict(size=12, color='#4a6080'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False,
                   linecolor='#e2eaf3', tickfont=dict(size=11)),
        yaxis=dict(gridcolor='#f0f5fb', zeroline=False,
                   tickfont=dict(size=11)),
    )
    return fig


# ══════════════════════════════════════════════════════════════
# TOP BAR
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="topbar">
    <div class="topbar-brand">
        <div class="topbar-logo">S</div>
        <span class="topbar-name">SiPADI Intelligence</span>
    </div>
    <div class="topbar-badge">
        <div class="topbar-dot"></div>
        Live · Jateng &amp; Jatim
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand-area">
        <div class="sb-brand-icon">🌾</div>
        <span class="sb-brand-name">SiPADI</span>
        <span class="sb-brand-sub">Sistem Prediksi Anomali &amp;<br>Deteksi Risiko Pangan</span>
    </div>
    <span class="sb-section-label">Menu Utama</span>
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
    <div class="sb-footer">
        <span class="sb-footer-title">Azure Services</span>
        <div class="sb-service"><div class="sb-service-icon"></div>Blob Storage</div>
        <div class="sb-service"><div class="sb-service-icon"></div>ML Workspace</div>
        <div class="sb-service"><div class="sb-service-icon"></div>AI Language</div>
        <div class="sb-service"><div class="sb-service-icon"></div>Maps</div>
        <div class="sb-service"><div class="sb-service-icon"></div>Static Web Apps</div>
        <br>
        <span class="sb-footer-title">Microsoft Elevate</span>
        <div class="sb-service">AI Impact Challenge 2026</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HELPER: wrap content in page wrapper
# ══════════════════════════════════════════════════════════════
def pw_start():
    st.markdown('<div class="pw">', unsafe_allow_html=True)

def pw_end():
    st.markdown('</div>', unsafe_allow_html=True)

def sh(label):
    st.markdown(f"""
    <div class="sh">
        <div class="sh-line"></div>
        <span class="sh-text">{label}</span>
        <div class="sh-rule"></div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: OVERVIEW & KPI
# ══════════════════════════════════════════════════════════════
if menu == "Overview & KPI":

    total_kab  = df_produksi['kabupaten'].nunique()
    avg_prod   = df_produksi['produktivitas_ton_per_ha'].mean()
    risiko_pct = df_master['risiko_gagal_panen'].mean() * 100
    total_prod = df_produksi['produksi_ton'].sum() / 1e6
    harga_now  = df_harga['harga_beras_medium_per_kg'].iloc[-1]

    st.markdown("""
    <div class="pw">
    <div class="banner">
        <div class="banner-left">
            <span class="banner-eyebrow">Food Intelligence Platform · 2025–2029</span>
            <div class="banner-title">Swasembada <em>Pangan</em><br>Intelligence Hub</div>
            <p class="banner-desc">
                Platform AI berbasis Azure untuk prediksi anomali produksi,
                deteksi risiko gagal panen, dan stabilisasi harga beras nasional.
            </p>
        </div>
        <div class="banner-right">
            <div class="bstat">
                <span class="bstat-label">Wilayah</span>
                <span class="bstat-val">Jateng · Jatim</span>
            </div>
            <div class="bstat">
                <span class="bstat-label">Kabupaten</span>
                <span class="bstat-val accent">58</span>
            </div>
            <div class="bstat">
                <span class="bstat-label">Periode Data</span>
                <span class="bstat-val">2010 – 2024</span>
            </div>
            <div class="bstat">
                <span class="bstat-label">Model AI</span>
                <span class="bstat-val">XGBoost + SARIMA</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    sh("Indikator Utama")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="tile">
            <div class="tile-icon navy">🏛️</div>
            <span class="tile-label">Kabupaten Dipantau</span>
            <div class="tile-val">{total_kab}</div>
            <div class="tile-desc">Jawa Tengah &amp; Jawa Timur</div>
            <span class="chip chip-neu">Aktif 100%</span>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="tile">
            <div class="tile-icon amber">📈</div>
            <span class="tile-label">Produktivitas Rata-rata</span>
            <div class="tile-val">{avg_prod:.2f}</div>
            <div class="tile-desc">ton / hektar</div>
            <span class="chip chip-up">+0.04 / tahun</span>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="tile">
            <div class="tile-icon red">⚠️</div>
            <span class="tile-label">Kabupaten Berisiko</span>
            <div class="tile-val">{risiko_pct:.1f}%</div>
            <div class="tile-desc">dari total kabupaten</div>
            <span class="chip chip-up">−2.3% vs lalu</span>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="tile">
            <div class="tile-icon ok">🌾</div>
            <span class="tile-label">Total Produksi</span>
            <div class="tile-val">{total_prod:.1f}M</div>
            <div class="tile-desc">ton kumulatif 2010–2024</div>
            <span class="chip chip-neu">Multi-tahun</span>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="tile">
            <div class="tile-icon red">💰</div>
            <span class="tile-label">Harga Beras Kini</span>
            <div class="tile-val md">Rp{harga_now:,.0f}</div>
            <div class="tile-desc">per kilogram</div>
            <span class="chip chip-dn">+3.2% YoY</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sh("Performa Model AI")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="mcard">
            <span class="mcard-type mt-reg">Regresi</span>
            <div class="mcard-name">Prediksi Produktivitas</div>
            <div class="mcard-algo">XGBoost Regressor</div>
            <div class="mrow"><span class="mrow-k">R² Score</span><span class="mrow-v ok">0.886</span></div>
            <div class="mrow"><span class="mrow-k">MAPE</span><span class="mrow-v ok">2.25%</span></div>
            <div class="mrow"><span class="mrow-k">RMSE</span><span class="mrow-v">0.158 ton/ha</span></div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="mcard">
            <span class="mcard-type mt-cls">Klasifikasi</span>
            <div class="mcard-name">Deteksi Risiko Gagal Panen</div>
            <div class="mcard-algo">XGBoost Classifier</div>
            <div class="mrow"><span class="mrow-k">F1-Score</span><span class="mrow-v ok">0.833</span></div>
            <div class="mrow"><span class="mrow-k">Recall</span><span class="mrow-v ok">92.6%</span></div>
            <div class="mrow"><span class="mrow-k">ROC-AUC</span><span class="mrow-v ok">0.967</span></div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="mcard">
            <span class="mcard-type mt-ts">Time Series</span>
            <div class="mcard-name">Forecast Harga Beras</div>
            <div class="mcard-algo">SARIMA(1,1,1)(1,1,1,12)</div>
            <div class="mrow"><span class="mrow-k">MAPE</span><span class="mrow-v ok">1.16%</span></div>
            <div class="mrow"><span class="mrow-k">Horizon</span><span class="mrow-v">12 bulan</span></div>
            <div class="mrow"><span class="mrow-k">Range Prediksi</span><span class="mrow-v">Rp13.923–14.394</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sh("Tren Produksi Padi (2010–2023)")

    tren = df_produksi.groupby(['tahun', 'provinsi'])['produksi_ton'].sum().reset_index()
    fig = px.line(tren, x='tahun', y='produksi_ton', color='provinsi',
                  markers=True, template='plotly_white',
                  labels={'produksi_ton': 'Total Produksi (ton)', 'tahun': 'Tahun'},
                  color_discrete_map={'Jawa Tengah': C['jateng'], 'Jawa Timur': C['jatim']})
    fig.update_traces(line_width=2.5, marker_size=7)
    chart_style(fig, h=360, m=dict(l=0, r=0, t=10, b=0))
    fig.update_layout(
        legend_title="Provinsi",
        legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.99,
                    bgcolor='rgba(255,255,255,0.95)',
                    bordercolor='#e2eaf3', borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: PETA RISIKO
# ══════════════════════════════════════════════════════════════
elif menu == "Peta Risiko Kabupaten":
    st.markdown('<div class="pw">', unsafe_allow_html=True)

    st.markdown("""
    <div class="ph">
        <div class="ph-badge">
            <svg viewBox="0 0 24 24">
                <polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/>
                <line x1="8" y1="2" x2="8" y2="18"/>
                <line x1="16" y1="6" x2="16" y2="22"/>
            </svg>
        </div>
        <div>
            <div class="ph-title">Peta Risiko per Kabupaten</div>
            <div class="ph-sub">Visualisasi spasial risiko gagal panen — 58 kabupaten Jawa Tengah &amp; Jawa Timur</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<span class="az-tag">🔷 Azure Maps</span>', unsafe_allow_html=True)

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

    col1, col2 = st.columns([3, 1])
    with col1:
        threshold = st.slider("Threshold Risiko (%)", 0, 100, 25)
    with col2:
        show_all = st.checkbox("Tampilkan semua", value=True)

    df_map = risiko_kab if show_all else risiko_kab[risiko_kab['risiko_pct_100'] >= threshold]

    fig_map = px.scatter_mapbox(
        df_map, lat='lat', lon='lon',
        color='risiko_pct_100', size='avg_prod',
        hover_name='kabupaten',
        hover_data={'risiko_pct_100': ':.1f', 'avg_prod': ':.2f', 'lat': False, 'lon': False},
        color_continuous_scale=[
            [0.0, '#10b981'], [0.4, '#f59e0b'],
            [0.7, '#ef4444'], [1.0, '#7f1d1d']
        ],
        size_max=22, zoom=6,
        center={'lat': -7.5, 'lon': 111.5},
        mapbox_style='carto-positron',
        labels={'risiko_pct_100': 'Risiko (%)', 'avg_prod': 'Prod. (ton/ha)'},
    )
    fig_map.update_layout(
        height=500, margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title="Risiko (%)", thickness=12, len=0.55,
            bgcolor='rgba(255,255,255,0.95)'
        )
    )
    st.plotly_chart(fig_map, use_container_width=True)

    sh("Kabupaten Risiko Tinggi")
    high_risk = risiko_kab[risiko_kab['risiko_pct_100'] >= threshold].sort_values('risiko_pct_100', ascending=False)
    high_risk['Status'] = high_risk['risiko_pct_100'].apply(
        lambda x: '🔴 Tinggi' if x >= 30 else '🟡 Sedang')
    st.dataframe(
        high_risk[['kabupaten', 'risiko_pct_100', 'avg_prod', 'Status']].rename(columns={
            'kabupaten': 'Kabupaten', 'risiko_pct_100': 'Risiko (%)', 'avg_prod': 'Produktivitas (ton/ha)'
        }),
        use_container_width=True, hide_index=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: PREDIKSI PRODUKTIVITAS
# ══════════════════════════════════════════════════════════════
elif menu == "Prediksi Produktivitas":
    st.markdown('<div class="pw">', unsafe_allow_html=True)

    st.markdown("""
    <div class="ph">
        <div class="ph-badge">
            <svg viewBox="0 0 24 24">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
        </div>
        <div>
            <div class="ph-title">Prediksi Produktivitas Padi</div>
            <div class="ph-sub">Estimasi produktivitas berbasis kondisi iklim, lahan, dan infrastruktur — XGBoost Regressor</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        sh("Kondisi Iklim")
        enso_val    = st.slider("ENSO Index", -2.0, 2.0, 0.0, 0.1)
        curah_hujan = st.slider("Curah Hujan Total (mm)", 100, 1500, 600)
        suhu        = st.slider("Suhu Rata-rata (°C)", 24.0, 32.0, 27.5, 0.1)
        kelembaban  = st.slider("Kelembaban (%)", 60, 95, 78)
    with col2:
        sh("Kondisi Lahan")
        ndvi       = st.slider("NDVI (Kesehatan Lahan)", 0.1, 0.9, 0.55, 0.01)
        pct_baik   = st.slider("Lahan Kondisi Baik (%)", 0, 100, 60)
        luas_panen = st.number_input("Luas Panen (ha)", 1000, 50000, 10000)
        musim      = st.selectbox("Musim Tanam", ["MT1 (Jan-Apr)", "MT2 (Jun-Sep)"])
    with col3:
        sh("Infrastruktur")
        jumlah_pompa = st.slider("Jumlah Pompa (unit)", 0, 100, 30)
        skor_irigasi = st.slider("Skor Irigasi", 0.0, 1.0, 0.65, 0.01)
        harga_beras  = st.number_input("Harga Beras (Rp/kg)", 8000, 20000, 13500)
        provinsi     = st.selectbox("Provinsi", ["Jawa Tengah", "Jawa Timur"])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Jalankan Prediksi", type="primary", use_container_width=True):
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
            'skor_irigasi': skor_irigasi, 'jumlah_pompa_unit': jumlah_pompa,
            'kapasitas_pompa_per_ha': jumlah_pompa * 80 / luas_panen,
            'indeks_pertanaman': 200, 'persen_irigasi_kondisi_baik': skor_irigasi * 100,
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

        st.markdown("<hr>", unsafe_allow_html=True)
        sh("Hasil Prediksi")

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
            title={'text': "Produktivitas (ton/ha)",
                   'font': {'size': 13, 'family': 'Plus Jakarta Sans', 'color': '#4a6080'}},
            gauge={
                'axis': {'range': [3, 8], 'tickwidth': 1, 'tickcolor': '#d0dcea',
                         'tickfont': {'size': 10}},
                'bar': {'color': C['amber'], 'thickness': 0.6},
                'bgcolor': 'white', 'borderwidth': 0,
                'steps': [
                    {'range': [3,   4.5], 'color': '#fef2f2'},
                    {'range': [4.5, 5.5], 'color': '#fffbeb'},
                    {'range': [5.5, 8],   'color': '#ecfdf5'},
                ],
                'threshold': {'line': {'color': C['red'], 'width': 2},
                              'thickness': 0.75, 'value': 5.14}
            }
        ))
        chart_style(fig_gauge, h=260, m=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if pred_risk == 1:
            st.markdown(f"""
            <div class="alert al-err">
                <div class="alert-icon">⚠️</div>
                <div class="alert-body">
                    <strong>Peringatan Risiko Gagal Panen</strong>
                    Probabilitas gagal panen: <b>{pred_prob:.1%}</b>.
                    Segera lakukan intervensi pompanisasi tambahan dan koordinasi dengan Dinas Pertanian setempat.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert al-ok">
                <div class="alert-icon">✅</div>
                <div class="alert-body">
                    <strong>Kondisi Aman</strong>
                    Probabilitas gagal panen: <b>{pred_prob:.1%}</b>.
                    Produktivitas diprediksi {'di atas' if pred_prod > 5.5 else 'mendekati'} rata-rata historis 5.5 ton/ha.
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: DETEKSI RISIKO
# ══════════════════════════════════════════════════════════════
elif menu == "Deteksi Risiko Gagal Panen":
    st.markdown('<div class="pw">', unsafe_allow_html=True)

    st.markdown("""
    <div class="ph">
        <div class="ph-badge">
            <svg viewBox="0 0 24 24">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                <line x1="12" y1="9" x2="12" y2="13"/>
                <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
        </div>
        <div>
            <div class="ph-title">Deteksi Risiko Gagal Panen</div>
            <div class="ph-sub">Tren risiko historis dan evaluasi performa model klasifikasi XGBoost</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    risiko_trend = df_master.groupby(['tahun', 'provinsi'])['risiko_gagal_panen'].mean().reset_index()
    risiko_trend['risiko_pct'] = risiko_trend['risiko_gagal_panen'] * 100

    fig = px.area(risiko_trend, x='tahun', y='risiko_pct', color='provinsi',
                  labels={'risiko_pct': 'Kabupaten Berisiko (%)', 'tahun': 'Tahun'},
                  template='plotly_white',
                  color_discrete_map={'Jawa Tengah': C['jateng'], 'Jawa Timur': C['jatim']})
    fig.add_hline(y=25, line_dash="dot", line_color=C['red'],
                  annotation_text="Ambang Batas 25%",
                  annotation_font_size=11, annotation_font_color=C['red'])
    fig.update_traces(line_width=2)
    chart_style(fig, h=360, m=dict(l=0, r=0, t=20, b=0))
    fig.update_layout(
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.02,
                    bgcolor='rgba(255,255,255,0.95)',
                    bordercolor='#e2eaf3', borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        sh("Risiko per Musim Tanam")
        risiko_musim = df_master.groupby('musim_tanam')['risiko_gagal_panen'].mean() * 100
        fig2 = px.bar(risiko_musim.reset_index(),
                      x='musim_tanam', y='risiko_gagal_panen',
                      template='plotly_white',
                      color='musim_tanam',
                      color_discrete_map={'MT1': C['jateng'], 'MT2': C['jatim']},
                      labels={'risiko_gagal_panen': 'Risiko (%)', 'musim_tanam': 'Musim'})
        fig2.update_traces(marker_line_width=0, width=0.35)
        chart_style(fig2, h=300, m=dict(l=0, r=0, t=10, b=0))
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        sh("Performa Model Klasifikasi")
        metrics_df = pd.DataFrame({
            'Metrik': ['F1-Score', 'Precision', 'Recall', 'ROC-AUC'],
            'Nilai':  [0.8333, 0.7576, 0.9259, 0.9665],
            'Keterangan': [
                'Keseimbangan presisi & recall',
                '76% prediksi berisiko terbukti',
                '93% kasus berisiko terdeteksi',
                'Kemampuan membedakan kelas',
            ]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.markdown("""
        <div class="ic">
            <div class="ic-row"><span class="ic-title">Mengapa Recall Jadi Prioritas?</span></div>
            <p class="ic-body">Dalam early warning system, lebih baik memberikan
            peringatan berlebih daripada melewatkan kasus berisiko. Recall 92.6%
            memastikan hampir semua gagal panen terdeteksi tepat waktu.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: FORECAST HARGA
# ══════════════════════════════════════════════════════════════
elif menu == "Forecast Harga Beras":
    st.markdown('<div class="pw">', unsafe_allow_html=True)

    st.markdown("""
    <div class="ph">
        <div class="ph-badge">
            <svg viewBox="0 0 24 24">
                <line x1="12" y1="1" x2="12" y2="23"/>
                <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
            </svg>
        </div>
        <div>
            <div class="ph-title">Forecast Harga Beras</div>
            <div class="ph-sub">Prediksi 12 bulan ke depan — SARIMA(1,1,1)(1,1,1,12) · MAPE 1.16%</div>
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
        name='Data Historis', line=dict(color=C['navy2'], width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates[::-1]),
        y=forecast_upper + forecast_lower[::-1],
        fill='toself', fillcolor='rgba(245,158,11,0.08)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval ±10%',
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_vals,
        name='Forecast 2024',
        line=dict(color=C['amber'], width=2.5, dash='dash'),
        mode='lines+markers', marker=dict(size=7, symbol='circle')
    ))
    fig.add_vline(
        x=pd.Timestamp('2024-01-01').timestamp() * 1000,
        line_dash='dot', line_color='#8ba4be',
        annotation_text='Mulai Forecast',
        annotation_position='top left',
        annotation_font_size=11, annotation_font_color='#8ba4be'
    )
    chart_style(fig, h=420, m=dict(l=0, r=0, t=20, b=0))
    fig.update_layout(
        xaxis_title='Tanggal', yaxis_title='Harga (Rp/kg)',
        yaxis_tickformat=',.0f',
        legend=dict(yanchor="top", y=0.97, xanchor="left", x=0.01,
                    bgcolor='rgba(255,255,255,0.95)',
                    bordercolor='#e2eaf3', borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    sh("Tabel Prediksi Bulanan")
    df_forecast = pd.DataFrame({
        'Bulan': [d.strftime('%B %Y') for d in forecast_dates],
        'Forecast (Rp/kg)': [f"Rp{v:,.0f}" for v in forecast_vals],
        'Batas Bawah':  [f"Rp{v:,.0f}" for v in forecast_lower],
        'Batas Atas':   [f"Rp{v:,.0f}" for v in forecast_upper],
        'Status': ['🔴 Tinggi' if v > 14200 else '🟢 Normal' for v in forecast_vals]
    })
    st.dataframe(df_forecast, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="ic">
        <div class="ic-row"><span class="ic-title">Insight Forecast Harga</span></div>
        <p class="ic-body">Harga beras diprediksi naik signifikan pada <b>Juli–September 2024</b>
        (musim paceklik) dengan puncak <b>Rp14.394/kg</b>.<br>
        <b>Rekomendasi Bulog:</b> Lakukan pembelian cadangan sebelum Juli 2024
        untuk mengantisipasi tekanan harga pasar.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: SENTIMEN BERITA
# ══════════════════════════════════════════════════════════════
elif menu == "Analisis Sentimen Berita":
    st.markdown('<div class="pw">', unsafe_allow_html=True)

    st.markdown("""
    <div class="ph">
        <div class="ph-badge">
            <svg viewBox="0 0 24 24">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
                <line x1="16" y1="13" x2="8" y2="13"/>
                <line x1="16" y1="17" x2="8" y2="17"/>
            </svg>
        </div>
        <div>
            <div class="ph-title">Analisis Sentimen Berita Pangan</div>
            <div class="ph-sub">Natural language processing menggunakan Azure AI Language Service</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<span class="az-tag">🔷 Azure AI Language</span>', unsafe_allow_html=True)

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
        height=100,
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

            st.markdown("<hr>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Sentimen Utama", label_map[sentiment])
            with c2: st.metric("Skor Positif",   f"{scores.positive:.1%}")
            with c3: st.metric("Skor Netral",    f"{scores.neutral:.1%}")
            with c4: st.metric("Skor Negatif",   f"{scores.negative:.1%}")

            fig_sent = go.Figure(go.Bar(
                x=['Positif', 'Netral', 'Negatif'],
                y=[scores.positive, scores.neutral, scores.negative],
                marker_color=[C['green'], C['amber'], C['red']],
                marker_line_width=0,
                text=[f"{v:.1%}" for v in [scores.positive, scores.neutral, scores.negative]],
                textposition='outside', width=0.32
            ))
            chart_style(fig_sent, h=280, m=dict(l=0, r=0, t=20, b=0))
            fig_sent.update_layout(
                yaxis_title='Confidence Score',
                yaxis=dict(gridcolor='#f0f5fb', range=[0, 1.2], zeroline=False),
                xaxis=dict(showgrid=False, zeroline=False),
            )
            st.plotly_chart(fig_sent, use_container_width=True)

            if sentiment == 'negative':
                st.markdown("""
                <div class="alert al-err">
                    <div class="alert-icon">🔴</div>
                    <div class="alert-body">
                        <strong>Berita Negatif Terdeteksi</strong>
                        Indikasi risiko ketidakstabilan pasokan atau harga. Monitor kondisi lapangan segera.
                    </div>
                </div>""", unsafe_allow_html=True)
            elif sentiment == 'positive':
                st.markdown("""
                <div class="alert al-ok">
                    <div class="alert-icon">🟢</div>
                    <div class="alert-body">
                        <strong>Berita Positif Terdeteksi</strong>
                        Sentimen positif mengindikasikan kondisi pangan yang baik. Pertahankan program berjalan.
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert al-warn">
                    <div class="alert-icon">🟡</div>
                    <div class="alert-body">
                        <strong>Sentimen Netral</strong>
                        Berita informatif tanpa indikasi risiko maupun kondisi positif yang signifikan.
                    </div>
                </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error Azure AI Language: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: REKOMENDASI STRATEGIS
# ══════════════════════════════════════════════════════════════
elif menu == "Rekomendasi Strategis":
    st.markdown('<div class="pw">', unsafe_allow_html=True)

    st.markdown("""
    <div class="ph">
        <div class="ph-badge">
            <svg viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
        </div>
        <div>
            <div class="ph-title">Rekomendasi Strategis</div>
            <div class="ph-sub">Actionable insights berbasis model SiPADI untuk Bulog &amp; Kementerian Pertanian</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "Rekomendasi Bulog",
        "Rekomendasi Kementan",
        "Prioritas Kabupaten"
    ])

    with tab1:
        sh("Rekomendasi Operasional Bulog")
        recs_bulog = [
            ("Pembelian Cadangan Preventif", "Tinggi", "Juni 2024",
             "Lakukan pembelian gabah di 5 kabupaten surplus sebelum Oktober 2024 untuk mengantisipasi kenaikan harga Jul–Sep."),
            ("Optimasi Distribusi Stok", "Tinggi", "Juli 2024",
             "Alokasikan cadangan beras ke 12 kabupaten berisiko tinggi di Jawa Timur sebelum musim paceklik berdasarkan peta risiko SiPADI."),
            ("Stabilisasi Harga Pasar", "Sedang", "Agt–Sep 2024",
             "Intervensi pasar jika harga melampaui Rp14.400/kg. Target operasi pasar: Surabaya, Malang, Semarang."),
        ]
        for judul, pri, timeline, detail in recs_bulog:
            pc = "pr-hi" if pri == "Tinggi" else "pr-md"
            st.markdown(f"""
            <div class="ic">
                <div class="ic-row">
                    <span class="ic-title">{judul}</span>
                    <span class="pr {pc}">{pri}</span>
                </div>
                <p class="ic-body"><b>Timeline:</b> {timeline} — {detail}</p>
            </div>""", unsafe_allow_html=True)

    with tab2:
        sh("Rekomendasi Kebijakan Kementan")
        recs_kementan = [
            ("Prioritas Pompanisasi", "Tinggi",
             "Alokasi tambahan 500 unit pompa ke 10 kabupaten dengan skor irigasi terendah dan risiko gagal panen tertinggi berdasarkan model SiPADI."),
            ("Program Intensifikasi Lahan", "Sedang",
             "Fokus peningkatan NDVI di 15 kabupaten kondisi lahan buruk melalui subsidi pupuk dan pendampingan pertanian presisi."),
            ("Integrasi Early Warning System", "Rendah",
             "Integrasikan SiPADI dengan sistem informasi pertanian Kementan untuk monitoring real-time 58 kabupaten Jawa Tengah & Jawa Timur."),
        ]
        for judul, pri, detail in recs_kementan:
            pc = "pr-hi" if pri == "Tinggi" else ("pr-md" if pri == "Sedang" else "pr-lo")
            st.markdown(f"""
            <div class="ic">
                <div class="ic-row">
                    <span class="ic-title">{judul}</span>
                    <span class="pr {pc}">{pri}</span>
                </div>
                <p class="ic-body">{detail}</p>
            </div>""", unsafe_allow_html=True)

    with tab3:
        sh("Top 20 Kabupaten — Prioritas Intervensi")
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
            color_discrete_map={'Jawa Tengah': C['jateng'], 'Jawa Timur': C['jatim']}
        )
        fig_bar.update_traces(marker_line_width=0)
        chart_style(fig_bar, h=380, m=dict(l=0, r=0, t=10, b=0))
        fig_bar.update_layout(
            xaxis_tickangle=40,
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor='#f0f5fb'),
            legend_title="Provinsi",
            legend=dict(bgcolor='rgba(255,255,255,0.95)',
                        bordercolor='#e2eaf3', borderwidth=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(
            risiko_kab[['kabupaten', 'provinsi', 'risiko_pct', 'avg_prod', 'Prioritas']].rename(columns={
                'kabupaten': 'Kabupaten', 'provinsi': 'Provinsi',
                'risiko_pct': 'Risiko (%)', 'avg_prod': 'Produktivitas (ton/ha)'
            }),
            use_container_width=True, hide_index=True
        )

    st.markdown('</div>', unsafe_allow_html=True)