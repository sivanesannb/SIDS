"""Metrics, plots, feature importance and the water-saving simulation."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
                             balanced_accuracy_score, confusion_matrix, classification_report)
from sklearn.inspection import permutation_importance

from .config import CLASSES, DOSE_MM, BASELINE_DOSE_MM, OUT_DIR, RANDOM_STATE


def metrics_row(name, y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    rec_high = precision_recall_fscore_support(y_true, y_pred, labels=[2], zero_division=0)[1][0]
    return {"model": name, "accuracy": accuracy_score(y_true, y_pred),
            "balanced_acc": balanced_accuracy_score(y_true, y_pred),
            "precision_macro": p, "recall_macro": r, "f1_macro": f,
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "recall_High": rec_high}


def plot_confusion(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Greens", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix - {name}")
    plt.tight_layout(); plt.savefig(OUT_DIR / f"confusion_{name.replace(' ', '_')}.png", dpi=150); plt.close()


def plot_importance(model, X_te, y_te, name, n=15, sample=5000):
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_te), min(sample, len(X_te)), replace=False)
    r = permutation_importance(model, X_te.iloc[idx], y_te[idx], scoring="f1_macro",
                               n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
    imp = pd.Series(r.importances_mean, index=X_te.columns).sort_values()
    imp.tail(n).plot.barh(color="seagreen", figsize=(7, 5))
    plt.xlabel("Drop in macro-F1 when shuffled"); plt.title(f"Top features - {name}")
    plt.tight_layout(); plt.savefig(OUT_DIR / "feature_importance.png", dpi=150); plt.close()
    imp = imp.sort_values(ascending=False)
    imp.to_csv(OUT_DIR / "feature_importance.csv")
    return imp


def water_simulation(y_true, y_pred):
    """Compare water use of a fixed schedule vs model-guided irrigation.
    Doses are ASSUMPTIONS set in config.py - state them in the report.
    Also reports the agronomic risk: fields that needed MORE water than they got."""
    dose = np.array([DOSE_MM[c] for c in CLASSES])
    n = len(y_true)
    baseline_total = BASELINE_DOSE_MM * n
    model_total = int(dose[y_pred].sum())
    ideal_total = int(dose[y_true].sum())
    return {"fixed_schedule_mm_total": baseline_total, "model_guided_mm_total": model_total,
            "oracle_mm_total": ideal_total,
            "water_saved_vs_fixed_pct": 100 * (baseline_total - model_total) / baseline_total,
            "oracle_saving_pct": 100 * (baseline_total - ideal_total) / baseline_total,
            "under_irrigated_pct": 100 * float(np.mean(y_pred < y_true)),
            "over_irrigated_pct": 100 * float(np.mean(y_pred > y_true))}


def save_report(y_true, y_pred, name):
    rep = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    (OUT_DIR / f"classification_report_{name.replace(' ', '_')}.txt").write_text(rep)
    return rep
