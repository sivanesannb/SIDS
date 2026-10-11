"""Model training: baseline + Logistic Regression + Random Forest + Gradient Boosting, with CV."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE
from .features import build_preprocessor


class MoistureRuleBaseline(BaseEstimator, ClassifierMixin):
    """Simple farmer-style rule on soil moisture only (what a cheap sensor gives)."""
    def __init__(self, low_thr=25, high_thr=40):
        self.low_thr, self.high_thr = low_thr, high_thr
    def fit(self, X, y):
        self.classes_ = np.array(sorted(set(y)))
        return self
    def predict(self, X):
        m = X["soil_moisture"].values
        return np.where(m < self.low_thr, 2, np.where(m < self.high_thr, 1, 0))


def get_models(X_sample):
    pre, _, _ = build_preprocessor(X_sample)
    return {
        "Logistic Regression": Pipeline([("pre", pre), ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))]),
        "Random Forest": Pipeline([("pre", pre),
            ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=-1,
                    class_weight="balanced_subsample", random_state=RANDOM_STATE))]),
        "Gradient Boosting": Pipeline([("pre", pre),
            ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                    class_weight="balanced", random_state=RANDOM_STATE))]),
    }


def cross_validate_models(models, X, y, folds=5):
    skf = StratifiedKFold(folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, m in models.items():
        s = cross_val_score(m, X, y, cv=skf, scoring="f1_macro", n_jobs=1)
        rows.append({"model": name, "cv_f1_macro_mean": s.mean(), "cv_f1_macro_std": s.std()})
        print(f"  CV {name:20s} F1-macro = {s.mean():.4f} +/- {s.std():.4f}")
    return pd.DataFrame(rows)
