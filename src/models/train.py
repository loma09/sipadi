"""
train.py
========
Melatih 3 model SiPADI:
  1. XGBoost Regressor  → prediksi produktivitas
  2. XGBoost Classifier → deteksi risiko gagal panen
  3. SARIMA             → forecast harga beras

Cara pakai:
    python -m src.models.train
    dari src.models.train import run_training
    run_training()
"""

import logging
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor, XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR       = BASE_DIR / "data" / "raw"

# Kolom yang diexclude dari fitur
EXCLUDE_COLS = [
    'produktivitas_ton_per_ha', 'risiko_gagal_panen',
    'kabupaten', 'provinsi', 'musim_tanam', 'tanggal',
    'produksi_ton',
]


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list:
    """Ambil kolom fitur (numerik, bukan target/id)."""
    num_cols = df.select_dtypes(include='number').columns.tolist()
    return [c for c in num_cols if c not in EXCLUDE_COLS]


# ─────────────────────────────────────────────
# MODEL 1: XGBOOST REGRESSOR
# ─────────────────────────────────────────────

def train_productivity_model(df: pd.DataFrame, feature_cols: list) -> XGBRegressor:
    """
    Latih XGBoost Regressor untuk prediksi produktivitas ton/ha.

    Returns:
        Trained model
    """
    logger.info("=" * 45)
    logger.info("Training Model 1: Productivity Regressor")

    X = df[feature_cols]
    y = df['produktivitas_ton_per_ha']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric='rmse',
        early_stopping_rounds=20,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    mape   = mean_absolute_percentage_error(y_test, y_pred) * 100
    rmse   = np.sqrt(((y_test - y_pred) ** 2).mean())
    r2     = 1 - ((y_test - y_pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()

    logger.info(f"  R²   : {r2:.4f}")
    logger.info(f"  MAPE : {mape:.2f}%")
    logger.info(f"  RMSE : {rmse:.4f} ton/ha")

    # Feature importance top 10
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top10 = importances.nlargest(10)
    logger.info(f"  Top 5 features: {list(top10.head(5).index)}")

    return model


# ─────────────────────────────────────────────
# MODEL 2: XGBOOST CLASSIFIER
# ─────────────────────────────────────────────

def train_risk_classifier(df: pd.DataFrame, feature_cols: list) -> XGBClassifier:
    """
    Latih XGBoost Classifier untuk deteksi risiko gagal panen.

    Returns:
        Trained model
    """
    logger.info("=" * 45)
    logger.info("Training Model 2: Risk Classifier")

    X = df[feature_cols]
    y = df['risiko_gagal_panen']

    # Class imbalance handling
    pos_weight = (y == 0).sum() / (y == 1).sum()
    logger.info(f"  Class distribution: {y.value_counts().to_dict()}")
    logger.info(f"  scale_pos_weight   : {pos_weight:.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=pos_weight,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss',
        early_stopping_rounds=20,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]
    f1          = f1_score(y_test, y_pred)
    precision   = precision_score(y_test, y_pred)
    recall      = recall_score(y_test, y_pred)
    roc_auc     = roc_auc_score(y_test, y_prob)

    logger.info(f"  F1-Score  : {f1:.4f}")
    logger.info(f"  Precision : {precision:.4f}")
    logger.info(f"  Recall    : {recall:.4f}")
    logger.info(f"  ROC-AUC   : {roc_auc:.4f}")

    return model


# ─────────────────────────────────────────────
# MODEL 3: SARIMA HARGA BERAS
# ─────────────────────────────────────────────

def train_price_forecaster(df_harga: pd.DataFrame):
    """
    Latih SARIMA(1,1,1)(1,1,1,12) untuk forecast harga beras.

    Returns:
        Fitted SARIMA model result
    """
    logger.info("=" * 45)
    logger.info("Training Model 3: SARIMA Price Forecaster")

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        logger.error("statsmodels tidak terinstall. Install: pip install statsmodels")
        return None

    # Buat time series bulanan
    df_harga = df_harga.copy()
    df_harga.columns = df_harga.columns.str.lower()

    if 'tanggal' in df_harga.columns:
        df_harga['tanggal'] = pd.to_datetime(df_harga['tanggal'])
    else:
        df_harga['tanggal'] = pd.to_datetime({
            'year': df_harga['tahun'],
            'month': df_harga['bulan'],
            'day': 1
        })

    ts = df_harga.groupby('tanggal')['harga_beras_medium_per_kg'].mean().sort_index()
    ts.index = pd.DatetimeIndex(ts.index).to_period('M')

    model = SARIMAX(
        ts,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)

    logger.info(f"  AIC  : {result.aic:.2f}")
    logger.info(f"  BIC  : {result.bic:.2f}")

    # Evaluasi in-sample MAPE
    fitted   = result.fittedvalues
    actual   = ts[-len(fitted):]
    mask     = actual != 0
    mape     = np.mean(np.abs((actual[mask] - fitted[mask]) / actual[mask])) * 100
    logger.info(f"  MAPE (in-sample) : {mape:.2f}%")

    return result


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run_training():
    """
    Latih semua model dan simpan ke models/.
    """
    logger.info("=" * 55)
    logger.info("SiPADI — Model Training Pipeline")
    logger.info("=" * 55)

    # Load data
    df       = pd.read_csv(PROCESSED_DIR / "master_dataset.csv")
    df_harga = pd.read_csv(RAW_DIR / "harga_beras.csv")

    # Drop baris tanpa target
    df = df.dropna(subset=['produktivitas_ton_per_ha', 'risiko_gagal_panen'])

    feature_cols = get_feature_cols(df)
    logger.info(f"Jumlah fitur: {len(feature_cols)}")

    # Scaler
    scaler = StandardScaler()
    scaler.fit(df[feature_cols])

    # Train
    prod_model  = train_productivity_model(df, feature_cols)
    risk_model  = train_risk_classifier(df, feature_cols)
    price_model = train_price_forecaster(df_harga)

    # Simpan model & artefak
    joblib.dump(prod_model,   MODELS_DIR / "productivity_model.pkl")
    joblib.dump(risk_model,   MODELS_DIR / "risk_classifier.pkl")
    joblib.dump(scaler,       MODELS_DIR / "scaler.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")

    if price_model is not None:
        joblib.dump(price_model, MODELS_DIR / "price_forecaster.pkl")

    logger.info("=" * 55)
    logger.info("✅ Semua model tersimpan di models/")
    logger.info("=" * 55)

    return prod_model, risk_model, price_model, scaler, feature_cols


if __name__ == "__main__":
    run_training()