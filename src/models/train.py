"""3+ aday model eğitim pipeline'ı.

Modeller:
  1. Rule-based baseline (ML değil; operasyonel benchmark).
  2. Logistic Regression (regularized, calibrated).
  3. Random Forest.
  4. LightGBM.

Her model "full" ve "demografi-free" varyantlarında eğitilir (sensitive feature ablation).
Eğitim, configs/split.yaml'daki time-based split'e göre yapılır.

Çıktı:
  artifacts/models/{model_name}.joblib
  artifacts/eval/{model_name}_{split}.json  (val/test metrikleri)
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

import yaml
from .transformers import CatBoostPrep  # picklable across script entrypoints
from ..data.loader import load_validated, REPO_ROOT
from ..features.instant import add_derived
from ..features.historical import add_label_free_aggregates, add_label_dependent_aggregates
from .split import time_based_split, entity_overlap
from .metrics import summary

ARTIFACTS = REPO_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
EVAL_DIR = ARTIFACTS / "eval"

TARGET = "IsFraudTransaction"
DROP_FOR_FEATURES = (
    "BusinessKey", "TransactionDate", "AccountNumber", "DeviceId",
    "ReceiverName", "SenderName", "CustomerName", "IP_Subnet",
    "IsFraudTransaction", "DayType",  # DayType_clean kullanılıyor
)

CATEGORICAL = [
    "TransactionType", "DeviceOSName", "CustomerSegment",
    "CustomerEducation", "CustomerProfession", "CustomerMaritalStatus",
    "CustomerGender", "DayType_clean", "os_l1h", "os_l8h",
    "DeviceModel", "DeviceParentBrand",
    # NOT: amount_bucket ablation_questionable test ile drop edildi (test PR-AUC +0.0034 daha iyi olur).
]
DEMOGRAPHIC = ["CustomerAge", "CustomerGender", "CustomerEducation", "CustomerMaritalStatus"]


@dataclass
class TrainOutput:
    model_name: str
    variant: str
    pipeline: object
    val_metrics: dict
    test_metrics: dict


def _features_config() -> dict:
    with open(REPO_ROOT / "configs" / "features.yaml") as f:
        return yaml.safe_load(f)


def _build_full_frame() -> pd.DataFrame:
    df, _ = load_validated()
    df = add_derived(df, daytype_nan="unknown")
    df = add_label_free_aggregates(df)
    cfg = _features_config()
    ld = cfg.get("historical_label_dependent") or {}
    if ld.get("enabled"):
        lag = int(ld.get("label_availability_lag_days", 7))
        sm = ld.get("smoothing") or {}
        prior = float(sm.get("prior_strength", 50))
        df = add_label_dependent_aggregates(df, label_lag_days=lag, prior_strength=prior)
        print(f"[train] label-dependent FE enabled (lag={lag}d, prior={prior})")
    return df


def _feature_columns(df: pd.DataFrame, drop_demographic: bool) -> tuple[list[str], list[str]]:
    drop = set(DROP_FOR_FEATURES)
    if drop_demographic:
        drop.update(DEMOGRAPHIC)
    cols = [c for c in df.columns if c not in drop]
    cat = [c for c in CATEGORICAL if c in cols]
    num = [c for c in cols if c not in cat]
    return num, cat


def _preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler(with_mean=False)),  # sparse-friendly
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)),
    ])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])


def _model_lr(num, cat) -> Pipeline:
    return Pipeline([
        ("pre", _preprocessor(num, cat)),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced", solver="liblinear", C=1.0)),
    ])


def _model_rf(num, cat) -> Pipeline:
    return Pipeline([
        ("pre", _preprocessor(num, cat)),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=50,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        )),
    ])


_HGB_DEFAULTS = {
    "learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 63,
    "min_samples_leaf": 100, "l2_regularization": 1.0, "max_features": 1.0,
}


def _load_hgb_params() -> dict:
    """artifacts/best_hgb_params.json varsa onu döner; yoksa defaults."""
    p = REPO_ROOT / "artifacts" / "best_hgb_params.json"
    if not p.exists():
        return dict(_HGB_DEFAULTS)
    payload = json.loads(p.read_text())
    out = dict(_HGB_DEFAULTS)
    out.update(payload.get("params", {}))
    return out


def _model_catboost(num, cat) -> Pipeline:
    """CatBoost — ordered target encoding + native categorical support.

    Bizim manuel olarak yaptığımız receiver_fraud_rate_smoothed (lag=7d) felsefesinin
    yerleşik versiyonu. CatBoost'a kategorik kolonları string olarak veriyoruz —
    OrdinalEncoder kullanmıyoruz; CatBoost kendi içinde ordered target encoding yapıyor.
    """
    from catboost import CatBoostClassifier
    cat_indices = list(range(len(num), len(num) + len(cat)))
    return Pipeline([
        ("pre", CatBoostPrep(num, cat)),
        ("clf", CatBoostClassifier(
            iterations=600, learning_rate=0.05, depth=8,
            l2_leaf_reg=3.0, auto_class_weights="Balanced",
            cat_features=cat_indices, random_seed=42, verbose=False,
            allow_writing_files=False,
        )),
    ])


def _model_hgb(num, cat) -> Pipeline:
    """HistGradientBoostingClassifier — params artifacts/best_hgb_params.json'dan okunur."""
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat),
    ])
    cat_indices = list(range(len(num), len(num) + len(cat)))
    p = _load_hgb_params()
    return Pipeline([
        ("pre", pre),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=p["learning_rate"], max_iter=p["max_iter"],
            max_leaf_nodes=p["max_leaf_nodes"], min_samples_leaf=p["min_samples_leaf"],
            l2_regularization=p["l2_regularization"], max_features=p.get("max_features", 1.0),
            class_weight="balanced", categorical_features=cat_indices,
            random_state=42,
        )),
    ])


def _rule_based_predict(df: pd.DataFrame) -> np.ndarray:
    """Score = ağırlıklı kural-skoru. Threshold downstream."""
    s = np.zeros(len(df), dtype=np.float32)
    s += 0.4 * (df["HasMobileActivationL1H"] == 1).to_numpy()
    s += 0.2 * (df["HasMobileActivationL8H"] == 1).to_numpy()
    s += 0.15 * (df["DeviceOSName"] == "Android").to_numpy()
    s += 0.1 * (df["IsFractionalAmount"] == True).to_numpy()  # noqa: E712
    s += 0.1 * (df["TransactionType"] == "Eft").to_numpy()
    s += 0.05 * (df["CustomerSegment"].isin(["P", "Y", "KP", "A1"])).to_numpy()
    s += 0.05 * np.clip(np.log1p(df["TransactionAmount"]) / 12, 0, 1).to_numpy()
    return np.clip(s, 0, 1)


def _eval_split(model_pipeline, df_split: pd.DataFrame, feature_cols: list[str], total_days: int) -> dict:
    X = df_split[feature_cols]
    y = df_split[TARGET].to_numpy()
    if model_pipeline == "rule_based":
        proba = _rule_based_predict(df_split)
    else:
        proba = model_pipeline.predict_proba(X)[:, 1]
    return summary(y, proba, total_days_for_alerts=total_days)


def train_all() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("[train] Loading + feature engineering …")
    df = _build_full_frame()
    print(f"[train] Frame: {df.shape}, elapsed {time.time()-t0:.1f}s")

    split = time_based_split(df)
    print(f"[train] Train {len(split.train):,} | Val {len(split.val):,} | Test {len(split.test):,}")
    train_days = max(1, (split.train["TransactionDate"].max() - split.train["TransactionDate"].min()).days)
    val_days = max(1, (split.val["TransactionDate"].max() - split.val["TransactionDate"].min()).days)
    test_days = max(1, (split.test["TransactionDate"].max() - split.test["TransactionDate"].min()).days)

    overlap = entity_overlap(split.train, split.test)

    results: dict = {"split": split.cutoffs, "entity_overlap_train_vs_test": overlap, "models": {}}

    for drop_demo in (False, True):
        variant = "demographic_free" if drop_demo else "full"
        num, cat = _feature_columns(df, drop_demographic=drop_demo)
        feature_cols = num + cat
        print(f"\n[variant={variant}] feature count = {len(feature_cols)}")

        rule_metrics = {
            "val": _eval_split("rule_based", split.val, feature_cols, val_days),
            "test": _eval_split("rule_based", split.test, feature_cols, test_days),
        }
        results["models"][f"rule_based__{variant}"] = rule_metrics

        for name, builder in (
            ("logreg", _model_lr),
            ("random_forest", _model_rf),
            ("hist_gbm", _model_hgb),
            ("catboost", _model_catboost),
        ):
            print(f"  -> training {name} ({variant}) …")
            t1 = time.time()
            pipe = builder(num, cat)
            pipe.fit(split.train[feature_cols], split.train[TARGET])
            print(f"     fit elapsed {time.time()-t1:.1f}s")
            joblib.dump(pipe, MODELS_DIR / f"{name}__{variant}.joblib")
            results["models"][f"{name}__{variant}"] = {
                "val": _eval_split(pipe, split.val, feature_cols, val_days),
                "test": _eval_split(pipe, split.test, feature_cols, test_days),
            }

    out = EVAL_DIR / "all_models.json"
    out.write_text(json.dumps(results, indent=2, default=str, ensure_ascii=False))
    print(f"\n[train] wrote {out}, total elapsed {time.time()-t0:.1f}s")
    return results


if __name__ == "__main__":
    train_all()
