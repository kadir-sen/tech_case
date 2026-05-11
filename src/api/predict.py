"""Predict fonksiyonu — eğitilmiş model artefaktını yükler ve skor üretir.

Reason codes: SHAP TreeExplainer ile, sadece request'te `explain=true` ile.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import os
import yaml
import joblib
import numpy as np
import pandas as pd

from ..data.loader import REPO_ROOT
from ..features.instant import add_derived
from .schemas import TransactionInput, ScoreResponse, ReasonCode

MODEL_NAME = os.environ.get("MODEL_NAME", "hist_gbm__demographic_free")
MODEL_PATH = REPO_ROOT / "artifacts" / "models" / f"{MODEL_NAME}.joblib"

_state: dict = {"pipeline": None, "thresholds": None, "feature_cols": None,
                "explainer": None, "feature_names_after_pre": None}


def _load_thresholds() -> dict:
    with open(REPO_ROOT / "configs" / "thresholds.yaml") as f:
        return yaml.safe_load(f)


def _load_pipeline():
    if _state["pipeline"] is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artefaktı bulunamadı: {MODEL_PATH}. "
                f"Önce `python -m src.models.train` çalıştırın."
            )
        _state["pipeline"] = joblib.load(MODEL_PATH)
        _state["thresholds"] = _load_thresholds()
    return _state["pipeline"]


def _build_features_row(tx: TransactionInput) -> pd.DataFrame:
    row = tx.model_dump()
    hist = row.pop("historical_features", None) or {}
    df = pd.DataFrame([row])
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    df = add_derived(df)
    for k, v in hist.items():
        df[k] = v
    expected_hist = [
        "device_tx_count_all", "device_tx_count_1d", "device_tx_count_7d", "device_tx_count_30d",
        "device_distinct_accounts_all", "device_distinct_receivers_all", "device_first_seen_days_ago",
        "receiver_tx_count_all", "receiver_tx_count_7d", "receiver_tx_count_30d",
        "receiver_distinct_senders_all", "receiver_distinct_devices_all", "receiver_first_seen_days_ago",
        "subnet_tx_count_all", "subnet_tx_count_7d", "subnet_distinct_devices_all",
        "account_tx_count_all", "account_first_seen_days_ago",
        "device_is_first_seen", "receiver_is_first_seen", "account_is_first_seen",
    ]
    for k in expected_hist:
        if k not in df.columns:
            df[k] = 0
    return df


def predict(tx: TransactionInput, explain: bool = False) -> ScoreResponse:
    pipe = _load_pipeline()
    thresholds = _state["thresholds"]
    df = _build_features_row(tx)

    proba = float(pipe.predict_proba(df)[0, 1])

    high_t = thresholds["policy"]["bands"]["HIGH"]["percentile"]
    medium_t = thresholds["policy"]["bands"]["MEDIUM"]["percentile"]
    cutoffs = _load_score_cutoffs()
    high_cut = cutoffs.get("HIGH", 0.95)
    medium_cut = cutoffs.get("MEDIUM", 0.5)
    if proba >= high_cut:
        band = "HIGH"; threshold_used = high_cut
    elif proba >= medium_cut:
        band = "MEDIUM"; threshold_used = medium_cut
    else:
        band = "LOW"; threshold_used = medium_cut

    reason_codes = _reason_codes(pipe, df) if explain else None

    return ScoreResponse(
        fraud_score=proba,
        risk_band=band,
        is_fraud=band in ("HIGH", "MEDIUM"),
        threshold_used=threshold_used,
        model_version=MODEL_NAME,
        score_calculated_at=datetime.now(timezone.utc),
        reason_codes=reason_codes,
    )


def _load_score_cutoffs() -> dict:
    cut_path = REPO_ROOT / "artifacts" / "score_cutoffs.json"
    if not cut_path.exists():
        return {}
    import json
    return json.loads(cut_path.read_text())


def _reason_codes(pipe, df: pd.DataFrame, top_n: int = 5) -> list[ReasonCode]:
    try:
        import shap
        pre = pipe.named_steps["pre"]
        clf = pipe.named_steps["clf"]
        X = pre.transform(df)
        if hasattr(X, "toarray"):
            X = X.toarray()
        try:
            feature_names = pre.get_feature_names_out()
        except Exception:
            feature_names = np.array([f"f{i}" for i in range(X.shape[1])])
        if _state["explainer"] is None:
            try:
                _state["explainer"] = shap.TreeExplainer(clf)
            except Exception:
                _state["explainer"] = shap.LinearExplainer(clf, np.zeros((1, X.shape[1])))
        sv = _state["explainer"](X)
        vals = sv.values[0] if hasattr(sv, "values") else sv[0]
        if vals.ndim == 2:
            vals = vals[:, 1]
        order = np.argsort(-np.abs(vals))[:top_n]
        return [
            ReasonCode(
                feature=str(feature_names[i]),
                contribution=float(vals[i]),
                direction="increases_risk" if vals[i] > 0 else "decreases_risk",
            )
            for i in order
        ]
    except Exception as e:
        return [ReasonCode(feature="_unavailable", contribution=0.0,
                           direction="decreases_risk")]
