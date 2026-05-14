#!/usr/bin/env python3
"""Tüm 4 ML modeli için time-aware Optuna HPO.

Test set'e DOKUNULMAZ. Sadece configs/split.yaml'daki train + val kullanılır.
Objective: val PR-AUC. Sampler: TPE (seed sabit).

Çıktılar:
  artifacts/hpo/{model_name}_best_params.json
  artifacts/hpo/{model_name}_trials.csv
  reports/hpo_summary.md (toplu)
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from src.data.loader import load_validated  # noqa: E402
from src.features.instant import add_derived  # noqa: E402
from src.features.historical import add_label_free_aggregates, add_label_dependent_aggregates  # noqa: E402
from src.models.split import time_based_split  # noqa: E402
from src.models.train import _feature_columns  # noqa: E402
from src.models.transformers import CatBoostPrep  # noqa: E402

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_frame() -> pd.DataFrame:
    df, _ = load_validated()
    df = add_derived(df, daytype_nan="unknown")
    df = add_label_free_aggregates(df)
    df = add_label_dependent_aggregates(df, label_lag_days=7, prior_strength=50)
    return df


# ---------------- model builders -------------------------------------------------
def _make_pre(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler(with_mean=False)),
        ]), num),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)),
        ]), cat),
    ])


def _make_pre_ordinal(num, cat):
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat),
    ])


def build_lr(params, num, cat):
    return Pipeline([
        ("pre", _make_pre(num, cat)),
        ("clf", LogisticRegression(
            penalty=params.get("penalty", "l2"),
            C=params["C"],
            solver=params.get("solver", "liblinear"),
            max_iter=int(params.get("max_iter", 300)),
            class_weight="balanced",
            random_state=42,
        )),
    ])


def build_rf(params, num, cat):
    return Pipeline([
        ("pre", _make_pre(num, cat)),
        ("clf", RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params.get("max_features", "sqrt"),
            class_weight="balanced_subsample",
            n_jobs=-1, random_state=42,
        )),
    ])


def build_hgb(params, num, cat):
    pre = _make_pre_ordinal(num, cat)
    cat_idx = list(range(len(num), len(num) + len(cat)))
    return Pipeline([
        ("pre", pre),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"],
            max_iter=int(params["max_iter"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            l2_regularization=params["l2_regularization"],
            max_features=params.get("max_features", 1.0),
            class_weight="balanced",
            categorical_features=cat_idx,
            random_state=42,
        )),
    ])


def build_catboost(params, num, cat):
    from catboost import CatBoostClassifier
    cat_idx = list(range(len(num), len(num) + len(cat)))
    return Pipeline([
        ("pre", CatBoostPrep(num, cat)),
        ("clf", CatBoostClassifier(
            iterations=int(params["iterations"]),
            learning_rate=params["learning_rate"],
            depth=int(params["depth"]),
            l2_leaf_reg=params["l2_leaf_reg"],
            bagging_temperature=params.get("bagging_temperature", 1.0),
            auto_class_weights="Balanced",
            cat_features=cat_idx, random_seed=42, verbose=False,
            allow_writing_files=False,
        )),
    ])


# ---------------- objectives -----------------------------------------------------
def make_objective(model_name, X_train, y_train, X_val, y_val, num, cat):
    def objective(trial: optuna.Trial) -> float:
        if model_name == "lr":
            penalty = trial.suggest_categorical("penalty", ["l2", "l1"])
            solver = "liblinear"  # only liblinear supports l1+l2 with balanced weight
            params = {
                "penalty": penalty,
                "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                "solver": solver,
                "max_iter": trial.suggest_int("max_iter", 200, 600, step=100),
            }
            pipe = build_lr(params, num, cat)
        elif model_name == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_depth": trial.suggest_int("max_depth", 6, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, log=True),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.75]),
            }
            pipe = build_rf(params, num, cat)
        elif model_name == "hgb":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_iter": trial.suggest_int("max_iter", 200, 800, step=100),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 500, log=True),
                "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 5.0),
                "max_features": trial.suggest_float("max_features", 0.5, 1.0),
            }
            pipe = build_hgb(params, num, cat)
        elif model_name == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 300, 1000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            }
            pipe = build_catboost(params, num, cat)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_val)[:, 1]
        return float(average_precision_score(y_val, proba))

    return objective


# ---------------- driver ---------------------------------------------------------
def tune_one_model(model_name: str, n_trials: int, X_train, y_train, X_val, y_val, num, cat, out_dir: Path):
    print(f"\n[hpo] === {model_name.upper()} === ({n_trials} trials)")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    obj = make_objective(model_name, X_train, y_train, X_val, y_val, num, cat)
    t0 = time.time()
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - t0

    best = study.best_params
    best_val = study.best_value

    # Trials CSV
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state", "duration"))
    trials_df.to_csv(out_dir / f"{model_name}_trials.csv", index=False)

    # Best params JSON
    payload = {
        "model": model_name,
        "n_trials": n_trials,
        "best_val_pr_auc": float(best_val),
        "best_params": best,
        "elapsed_s": round(elapsed, 1),
    }
    (out_dir / f"{model_name}_best_params.json").write_text(json.dumps(payload, indent=2))

    print(f"[hpo] {model_name}: best_val_pr_auc = {best_val:.4f}  ({elapsed:.1f}s)")
    print(f"[hpo] {model_name}: best_params = {best}")
    return payload


def evaluate_baseline(model_name, X_train, y_train, X_val, y_val, num, cat) -> dict:
    """Sabit (default) parametrelerle baseline val PR-AUC."""
    if model_name == "lr":
        pipe = build_lr({"C": 1.0, "penalty": "l2", "solver": "liblinear", "max_iter": 200}, num, cat)
    elif model_name == "rf":
        pipe = build_rf({"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 50}, num, cat)
    elif model_name == "hgb":
        pipe = build_hgb({"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 63,
                          "min_samples_leaf": 100, "l2_regularization": 1.0}, num, cat)
    elif model_name == "catboost":
        pipe = build_catboost({"iterations": 600, "learning_rate": 0.05, "depth": 8,
                                "l2_leaf_reg": 3.0, "bagging_temperature": 1.0}, num, cat)
    else:
        return {"baseline_val_pr_auc": None}
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_val)[:, 1]
    return {"baseline_val_pr_auc": float(average_precision_score(y_val, proba))}


def write_summary(all_results: dict, out_path: Path):
    md = []
    md.append("# HPO Özeti — Time-Aware (val PR-AUC objective)\n")
    md.append("> Auto-generated. Test set'e DOKUNULMAMIŞTIR. Tüm tuning train + val üzerinde.\n")
    md.append("## Modeller × baseline vs tuned val PR-AUC\n")
    md.append("| Model | n_trials | Baseline val PR-AUC | Best val PR-AUC | Δ | Elapsed |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for name, r in all_results.items():
        b = r.get("baseline_val_pr_auc")
        t = r.get("best_val_pr_auc")
        if b is None or t is None:
            continue
        md.append(f"| {name} | {r.get('n_trials', '?')} | {b:.4f} | {t:.4f} | {t-b:+.4f} | {r.get('elapsed_s', 0)}s |")
    md.append("")
    md.append("## Tune edilen parametre aralıkları\n")
    md.append("- **lr**: penalty {l2, l1}, C log-uniform [1e-3, 10], solver=liblinear, max_iter ∈ {200..600}")
    md.append("- **rf**: n_estimators ∈ {100..400}, max_depth ∈ {6..20}, min_samples_leaf log-uniform [10, 200], max_features ∈ {sqrt, log2, 0.5, 0.75}")
    md.append("- **hgb**: learning_rate log [0.01, 0.2], max_iter ∈ {200..800}, max_leaf_nodes ∈ {15..127}, min_samples_leaf log [20, 500], l2_reg ∈ [0, 5], max_features ∈ [0.5, 1.0]")
    md.append("- **catboost**: iterations ∈ {300..1000}, learning_rate log [0.02, 0.15], depth ∈ {4..10}, l2_leaf_reg ∈ [1, 10], bagging_temperature ∈ [0, 5]\n")
    md.append("## Her model için en iyi parametreler\n")
    for name, r in all_results.items():
        if r.get("best_params"):
            md.append(f"### {name}")
            md.append("```json")
            md.append(json.dumps(r["best_params"], indent=2))
            md.append("```\n")
    md.append("## Kurallar (uygulanmıştır)\n")
    md.append("- ✅ Test set'e DOKUNULMADI. Tuning yalnız train + val.")
    md.append("- ✅ Sampler: TPESampler(seed=42), reproducible.")
    md.append("- ✅ Objective: val PR-AUC (`average_precision_score`).")
    md.append("- ✅ Random shuffle CV YOK (time-based train/val ayrımı).")
    out_path.write_text("\n".join(md), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["lr", "rf", "hgb", "catboost"])
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--feature-set", default="full_safe")
    parser.add_argument("--output-dir", default="artifacts/hpo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = REPO / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    print("[hpo] building frame …")
    t0 = time.time()
    df = build_frame()
    sp = time_based_split(df)
    print(f"[hpo] frame: {df.shape}, train: {len(sp.train):,}, val: {len(sp.val):,}, test: {len(sp.test):,} (test KULLANILMAYACAK)")

    num, cat = _feature_columns(df, drop_demographic=True)
    feat = num + cat
    X_train, y_train = sp.train[feat], sp.train["IsFraudTransaction"].to_numpy()
    X_val, y_val = sp.val[feat], sp.val["IsFraudTransaction"].to_numpy()

    all_results = {}
    for m in args.models:
        # baseline
        base = evaluate_baseline(m, X_train, y_train, X_val, y_val, num, cat)
        # tune
        tuned = tune_one_model(m, args.n_trials, X_train, y_train, X_val, y_val, num, cat, out_dir)
        tuned.update(base)
        all_results[m] = tuned

    # Combined report
    summary_md = REPO / "reports" / "hpo_summary.md"
    write_summary(all_results, summary_md)
    print(f"\n[hpo] wrote summary → {summary_md}")
    print(f"[hpo] total elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
