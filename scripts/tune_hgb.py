#!/usr/bin/env python3
"""HistGradientBoosting hiperparametre arama (Optuna).

Hedef: validation seti PR-AUC'yi maksimize etmek.
Validation seti, configs/split.yaml'daki time-based ayrımdan gelir — random kfold YOK,
çünkü temporal drift büyük.

Output: artifacts/best_hgb_params.json
"""
from __future__ import annotations
import json
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline

import sys
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.loader import load_validated  # noqa: E402
from src.features.instant import add_derived  # noqa: E402
from src.features.historical import add_label_free_aggregates, add_label_dependent_aggregates  # noqa: E402
from src.models.split import time_based_split  # noqa: E402
from src.models.train import _feature_columns  # noqa: E402

warnings.filterwarnings("ignore")
N_TRIALS = int(__import__("os").environ.get("N_TRIALS", "30"))


def build_data():
    df, _ = load_validated()
    df = add_derived(df, daytype_nan="unknown")
    df = add_label_free_aggregates(df)
    df = add_label_dependent_aggregates(df, label_lag_days=7, prior_strength=50)
    return df


def make_pipeline(num, cat, params):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat),
    ])
    cat_idx = list(range(len(num), len(num) + len(cat)))
    clf = HistGradientBoostingClassifier(
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        max_leaf_nodes=params["max_leaf_nodes"],
        min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"],
        max_depth=params.get("max_depth"),
        max_features=params.get("max_features", 1.0),
        class_weight="balanced",
        categorical_features=cat_idx,
        random_state=42,
        early_stopping=False,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def main():
    print(f"[tune] Loading + FE …")
    df = build_data()
    split = time_based_split(df)
    print(f"[tune] Train {len(split.train):,} | Val {len(split.val):,} | Test {len(split.test):,}")

    num, cat = _feature_columns(df, drop_demographic=True)
    feat = num + cat
    X_train = split.train[feat]
    y_train = split.train["IsFraudTransaction"].to_numpy()
    X_val = split.val[feat]
    y_val = split.val["IsFraudTransaction"].to_numpy()
    print(f"[tune] feature count: {len(feat)}, demographic_free variant")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_iter": trial.suggest_int("max_iter", 200, 800, step=100),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 500, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 5.0),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        }
        pipe = make_pipeline(num, cat, params)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_val)[:, 1]
        return float(average_precision_score(y_val, proba))

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    elapsed = time.time() - t0
    print(f"\n[tune] {N_TRIALS} trials in {elapsed:.1f}s")
    print(f"[tune] Best val PR-AUC: {study.best_value:.4f}")
    print(f"[tune] Best params:")
    for k, v in study.best_params.items():
        print(f"  {k} = {v}")

    out = REPO / "artifacts" / "best_hgb_params.json"
    out.write_text(json.dumps({
        "best_val_pr_auc": study.best_value,
        "n_trials": N_TRIALS,
        "params": study.best_params,
    }, indent=2))
    print(f"[tune] wrote {out}")


if __name__ == "__main__":
    main()
