"""
evaluate.py
===========
Evaluasi komprehensif semua model SiPADI.
Output: laporan metrik + confusion matrix + feature importance + residual plot.

Cara pakai:
    python -m src.models.evaluate
    dari src.models.evaluate import run_evaluation
    run_evaluation()
"""

import logging
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    mean_absolute_percentage_error, mean_squared_error, r2_score,
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"
REPORTS_DIR   = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_COLS = [
    'produktivitas_ton_per_ha', 'risiko_gagal_panen',
    'kabupaten', 'provinsi', 'musim_tanam', 'tanggal', 'produksi_ton',
]


def load_artifacts():
    prod_model   = joblib.load(MODELS_DIR / "productivity_model.pkl")
    risk_model   = joblib.load(MODELS_DIR / "risk_classifier.pkl")
    scaler       = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
    return prod_model, risk_model, scaler, feature_cols


# ─────────────────────────────────────────────
# EVALUASI REGRESSOR
# ─────────────────────────────────────────────

def evaluate_regressor(model, X_test, y_test) -> dict:
    """Evaluasi lengkap XGBoost Regressor."""
    logger.info("Evaluasi Model 1: Productivity Regressor")

    y_pred = model.predict(X_test)
    mape   = mean_absolute_percentage_error(y_test, y_pred) * 100
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)
    mae    = np.abs(y_test - y_pred).mean()

    logger.info(f"  R²   : {r2:.4f}")
    logger.info(f"  MAPE : {mape:.2f}%")
    logger.info(f"  RMSE : {rmse:.4f}")
    logger.info(f"  MAE  : {mae:.4f}")

    return {
        'R2': round(r2, 4),
        'MAPE (%)': round(mape, 2),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
    }


def plot_residuals(model, X_test, y_test, out_path: Path):
    """Plot residuals aktual vs prediksi."""
    y_pred    = model.predict(X_test)
    residuals = y_test.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Aktual vs Prediksi
    axes[0].scatter(y_test, y_pred, alpha=0.5, color='#2d6a4f', s=20)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=1.5)
    axes[0].set_xlabel("Aktual (ton/ha)")
    axes[0].set_ylabel("Prediksi (ton/ha)")
    axes[0].set_title("Aktual vs Prediksi Produktivitas")
    axes[0].grid(alpha=0.3)

    # Residual distribution
    axes[1].hist(residuals, bins=40, color='#2d6a4f', alpha=0.7, edgecolor='white')
    axes[1].axvline(0, color='red', linestyle='--', lw=1.5)
    axes[1].set_xlabel("Residual (ton/ha)")
    axes[1].set_title("Distribusi Residual")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Residual plot disimpan: {out_path}")


def plot_feature_importance(model, feature_cols, out_path: Path, top_n=20):
    """Plot feature importance."""
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8, top_n * 0.35 + 1))
    top.plot(kind='barh', ax=ax, color='#2d6a4f', edgecolor='white')
    ax.set_title(f"Top {top_n} Feature Importance")
    ax.set_xlabel("Importance Score")
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Feature importance plot: {out_path}")


# ─────────────────────────────────────────────
# EVALUASI CLASSIFIER
# ─────────────────────────────────────────────

def evaluate_classifier(model, X_test, y_test) -> dict:
    """Evaluasi lengkap XGBoost Classifier."""
    logger.info("Evaluasi Model 2: Risk Classifier")

    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]
    f1      = f1_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred)
    rec     = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    logger.info(f"  F1-Score  : {f1:.4f}")
    logger.info(f"  Precision : {prec:.4f}")
    logger.info(f"  Recall    : {rec:.4f}")
    logger.info(f"  ROC-AUC   : {roc_auc:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=['Aman', 'Berisiko']))

    return {
        'F1-Score': round(f1, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'ROC-AUC': round(roc_auc, 4),
    }


def plot_confusion_matrix(model, X_test, y_test, out_path: Path):
    """Plot confusion matrix."""
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Greens')
    plt.colorbar(im, ax=ax)

    labels = ['Aman (0)', 'Berisiko (1)']
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    ax.set_title("Confusion Matrix — Risk Classifier")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Confusion matrix disimpan: {out_path}")


# ─────────────────────────────────────────────
# CROSS-VALIDATION
# ─────────────────────────────────────────────

def cross_validate_models(df, feature_cols):
    """5-Fold cross-validation untuk kedua model."""
    from xgboost import XGBRegressor, XGBClassifier

    logger.info("Cross-Validation (5-Fold)...")
    X = df[feature_cols]

    # Regressor CV
    y_reg  = df['produktivitas_ton_per_ha']
    reg_cv = cross_validate(
        XGBRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
        X, y_reg, cv=5, scoring=['r2', 'neg_mean_absolute_percentage_error'],
        return_train_score=False
    )
    logger.info(f"  Regressor CV R²   : {reg_cv['test_r2'].mean():.4f} ± {reg_cv['test_r2'].std():.4f}")

    # Classifier CV
    y_cls  = df['risiko_gagal_panen']
    cls_cv = cross_validate(
        XGBClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1,
                       use_label_encoder=False, eval_metric='logloss'),
        X, y_cls, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring=['f1', 'roc_auc'],
        return_train_score=False
    )
    logger.info(f"  Classifier CV F1  : {cls_cv['test_f1'].mean():.4f} ± {cls_cv['test_f1'].std():.4f}")
    logger.info(f"  Classifier CV AUC : {cls_cv['test_roc_auc'].mean():.4f} ± {cls_cv['test_roc_auc'].std():.4f}")

    return reg_cv, cls_cv


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run_evaluation():
    """
    Evaluasi semua model SiPADI. Simpan laporan ke reports/.
    """
    logger.info("=" * 55)
    logger.info("SiPADI — Model Evaluation")
    logger.info("=" * 55)

    df = pd.read_csv(PROCESSED_DIR / "master_dataset.csv")
    df = df.dropna(subset=['produktivitas_ton_per_ha', 'risiko_gagal_panen'])

    prod_model, risk_model, scaler, feature_cols = load_artifacts()

    X = df[feature_cols]
    y_reg = df['produktivitas_ton_per_ha']
    y_cls = df['risiko_gagal_panen']

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

    # Evaluasi
    reg_metrics = evaluate_regressor(prod_model, X_test_r, y_test_r)
    cls_metrics = evaluate_classifier(risk_model, X_test_c, y_test_c)

    # Plot
    plot_residuals(prod_model, X_test_r, y_test_r, REPORTS_DIR / "residuals.png")
    plot_feature_importance(prod_model, feature_cols, REPORTS_DIR / "feature_importance_reg.png")
    plot_confusion_matrix(risk_model, X_test_c, y_test_c, REPORTS_DIR / "confusion_matrix.png")

    # Cross-validation
    cross_validate_models(df, feature_cols)

    # Simpan ringkasan
    summary = pd.DataFrame({
        'Model': ['XGBoost Regressor', 'XGBoost Classifier'],
        'Tugas': ['Prediksi Produktivitas', 'Deteksi Risiko Gagal Panen'],
        'Metrik Utama': [f"R²={reg_metrics['R2']}, MAPE={reg_metrics['MAPE (%)']}%",
                         f"F1={cls_metrics['F1-Score']}, AUC={cls_metrics['ROC-AUC']}"],
    })
    summary.to_csv(REPORTS_DIR / "evaluation_summary.csv", index=False)

    logger.info("✅ Evaluasi selesai. Laporan tersimpan di reports/")
    return reg_metrics, cls_metrics


if __name__ == "__main__":
    run_evaluation()