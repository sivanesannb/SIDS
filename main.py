"""Run the whole pipeline: load -> features -> split -> train -> evaluate -> save.

Usage (from the project root, the folder that contains src/ and data/):
    python main.py                      # full data (slow: Random Forest on 630k rows)
    python main.py --sample 100000      # faster run for development
    python main.py --sample 100000 --cv # also run 5-fold CV on the sample
    python main.py --synthetic          # smoke-test with fake data (never report these results)
"""
import argparse
import json

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from src.config import TARGET, CLASSES, CLASS_TO_INT, RANDOM_STATE, OUT_DIR
from src.data import load_data, make_synthetic
from src.features import add_features
from src.train import MoistureRuleBaseline, get_models, cross_validate_models
from src.evaluate import (metrics_row, plot_confusion, plot_importance,
                          water_simulation, save_report)


def main(sample=None, run_cv=False, synthetic=False):
    # 1. Data
    df = make_synthetic() if synthetic else load_data(sample=sample)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(df[TARGET].value_counts(normalize=True).round(3).to_string())

    # 2. Features + integer target (Low=0, Medium=1, High=2)
    df = add_features(df)
    y = df[TARGET].map(CLASS_TO_INT).values
    X = df.drop(columns=[TARGET])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # 3. Models (+ simple moisture-rule baseline for comparison)
    models = {"Moisture Rule Baseline": MoistureRuleBaseline(), **get_models(X_tr)}

    if run_cv:
        print("\nCross-validation (macro-F1):")
        cv_models = {k: v for k, v in models.items() if k != "Moisture Rule Baseline"}
        cv_n = min(len(X_tr), 100_000)
        cv_df = cross_validate_models(cv_models, X_tr.iloc[:cv_n], y_tr[:cv_n])
        cv_df.to_csv(OUT_DIR / "cv_results.csv", index=False)

    # 4. Train + evaluate on the unseen test split
    rows, fitted, preds = [], {}, {}
    for name, m in models.items():
        print(f"\nTraining {name} ...")
        m = clone(m).fit(X_tr, y_tr)
        pred = m.predict(X_te)
        fitted[name], preds[name] = m, pred
        rows.append(metrics_row(name, y_te, pred))
        plot_confusion(y_te, pred, name)
        print(save_report(y_te, pred, name))

    results = pd.DataFrame(rows).set_index("model").round(4)
    results.to_csv(OUT_DIR / "model_comparison.csv")
    print("\nModel comparison:\n", results.to_string())

    # 5. Best model = highest balanced accuracy (baseline excluded)
    cand = results.drop(index="Moisture Rule Baseline")
    best = cand["balanced_acc"].idxmax()
    print(f"\nBest model: {best}")

    # 6. Feature importance + water-saving simulation for the best model
    imp = plot_importance(fitted[best], X_te, y_te, best)
    print("\nTop features:\n", imp.head(10).round(4).to_string())
    water = water_simulation(y_te, preds[best])
    (OUT_DIR / "water_simulation.json").write_text(json.dumps(water, indent=2))
    print("\nWater simulation:\n", json.dumps(water, indent=2))

    # 7. Save everything the app needs
    joblib.dump({"model": fitted[best], "model_name": best,
                 "input_columns": list(X.columns), "classes": CLASSES},
                OUT_DIR / "model.pkl")
    print(f"\nSaved model + outputs to {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="use only N rows")
    ap.add_argument("--cv", action="store_true", help="run 5-fold CV on <=100k rows")
    ap.add_argument("--synthetic", action="store_true", help="fake data, smoke test only")
    a = ap.parse_args()
    main(a.sample, a.cv, a.synthetic)