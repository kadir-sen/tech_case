#!/usr/bin/env python3
"""Leakage audit — kapsamlı kontrol.

Tests:
  1) Permutation: target'i train+val içinde shuffle et, retrain. Test PR-AUC ≈ baseline.
  2) Lag sensitivity: label_lag_days ∈ {7, 30, 60, 90}, retrain, PR-AUC karşılaştır.
  3) Drop fraud_rate features: kazancın ne kadarı bu spesifik feature'lardan.
  4) New-vs-seen receiver: test set'i receiver overlap'a göre ayır, performans nasıl.

Output: artifacts/leakage_audit.json + reports/leakage_audit.md
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.loader import load_validated  # noqa: E402
from src.features.instant import add_derived  # noqa: E402
from src.features.historical import add_label_free_aggregates, add_label_dependent_aggregates  # noqa: E402
from src.models.split import time_based_split  # noqa: E402
from src.models.train import _feature_columns, _model_hgb  # noqa: E402
from src.models.metrics import summary  # noqa: E402

TARGET = "IsFraudTransaction"
RAW = None  # cached raw frame


def get_raw() -> pd.DataFrame:
    global RAW
    if RAW is None:
        df, _ = load_validated()
        RAW = df
    return RAW.copy()


def build_frame(use_label_dep: bool = True, lag_days: int = 7, prior: float = 50.0) -> pd.DataFrame:
    df = get_raw()
    df = add_derived(df, daytype_nan="unknown")
    df = add_label_free_aggregates(df)
    if use_label_dep:
        df = add_label_dependent_aggregates(df, label_lag_days=lag_days, prior_strength=prior)
    return df


def eval_pipeline(df: pd.DataFrame, drop_demo: bool = True, drop_cols: list[str] | None = None) -> dict:
    split = time_based_split(df)
    num, cat = _feature_columns(df, drop_demographic=drop_demo)
    if drop_cols:
        num = [c for c in num if c not in drop_cols]
        cat = [c for c in cat if c not in drop_cols]
    feat = num + cat
    pipe = _model_hgb(num, cat)
    pipe.fit(split.train[feat], split.train[TARGET])
    proba = pipe.predict_proba(split.test[feat])[:, 1]
    y = split.test[TARGET].to_numpy()
    test_days = max(1, (split.test["TransactionDate"].max() - split.test["TransactionDate"].min()).days)
    s = summary(y, proba, total_days_for_alerts=test_days)
    r1 = next((x for x in s["top_k_percentile"] if abs(x["k_frac"] - 0.01) < 1e-9), None)
    return {
        "n_features": len(feat),
        "pr_auc": s["core"]["pr_auc"],
        "roc_auc": s["core"]["roc_auc"],
        "recall_at_1pct": r1["recall"] if r1 else None,
        "precision_at_1pct": r1["precision"] if r1 else None,
    }, proba, split


# -------------------------------------------------------------------------------
# Test 1: Permutation — train+val target shuffle
# -------------------------------------------------------------------------------
def test_permutation(seed: int = 0) -> dict:
    print("\n[1] PERMUTATION TEST — train+val target shuffle …")
    t0 = time.time()
    df = build_frame(use_label_dep=True, lag_days=7)
    split = time_based_split(df)
    # train+val içindeki target'leri shuffle et
    rng = np.random.default_rng(seed)
    train_val_idx = df["TransactionDate"] <= pd.Timestamp("2024-05-31 23:59:59")
    perm_target = df[TARGET].copy()
    sub = perm_target[train_val_idx].to_numpy().copy()
    rng.shuffle(sub)
    perm_target.loc[train_val_idx] = sub
    df_perm = df.copy()
    df_perm[TARGET] = perm_target
    # Re-derive label-dependent features with shuffled labels (kritik!)
    df_perm = df_perm.drop(columns=[c for c in df_perm.columns if c.startswith("device_fraud_rate") or c.startswith("receiver_fraud_rate") or c.endswith("_label_n")], errors="ignore")
    df_perm = add_label_dependent_aggregates(df_perm, label_lag_days=7, prior_strength=50)
    # Now train on shuffled train; evaluate on REAL test target
    real_target = df[TARGET].to_numpy()
    df_perm[TARGET] = perm_target  # permuted on train+val
    # ama test bölgesinde real target ile değerlendirmek istiyoruz
    # Yani modeli shuffled train ile eğit, test'in gerçek target'ı ile değerlendir
    train = df_perm[df_perm["TransactionDate"] <= pd.Timestamp("2024-03-31 23:59:59")]
    test = df_perm[df_perm["TransactionDate"] >= pd.Timestamp("2024-06-01")]
    num, cat = _feature_columns(df_perm, drop_demographic=True)
    feat = num + cat
    pipe = _model_hgb(num, cat)
    pipe.fit(train[feat], train[TARGET])
    proba = pipe.predict_proba(test[feat])[:, 1]
    # Test target gerçek (orijinal df'te)
    real_test_target = df.loc[df["TransactionDate"] >= pd.Timestamp("2024-06-01"), TARGET].to_numpy()
    s = summary(real_test_target, proba, total_days_for_alerts=120)
    r1 = next((x for x in s["top_k_percentile"] if abs(x["k_frac"] - 0.01) < 1e-9), None)
    out = {
        "pr_auc": s["core"]["pr_auc"],
        "roc_auc": s["core"]["roc_auc"],
        "recall_at_1pct": r1["recall"] if r1 else None,
        "expected": "PR-AUC ≈ baseline (~0.002 = test fraud rate) if no leakage",
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"  → PR-AUC={out['pr_auc']:.4f}, Recall@1%={out['recall_at_1pct']:.4f}")
    return out


# -------------------------------------------------------------------------------
# Test 2: Lag sensitivity
# -------------------------------------------------------------------------------
def test_lag_sensitivity(lags: tuple[int, ...] = (7, 30, 60, 90)) -> list[dict]:
    print("\n[2] LAG SENSITIVITY TEST …")
    out = []
    for lag in lags:
        t0 = time.time()
        df = build_frame(use_label_dep=True, lag_days=lag)
        res, _, _ = eval_pipeline(df)
        res["lag_days"] = lag
        res["elapsed_s"] = round(time.time() - t0, 1)
        out.append(res)
        print(f"  lag={lag:>3d}d → PR-AUC={res['pr_auc']:.4f}, Recall@1%={res['recall_at_1pct']:.4f}  ({res['elapsed_s']}s)")
    return out


# -------------------------------------------------------------------------------
# Test 3: Drop fraud_rate features
# -------------------------------------------------------------------------------
def test_drop_fraud_rate() -> dict:
    print("\n[3] DROP fraud_rate FEATURES …")
    df = build_frame(use_label_dep=True, lag_days=7)
    drop_cols = [c for c in df.columns if "fraud_rate" in c or c.endswith("_label_n")]
    print(f"  dropping: {drop_cols}")
    t0 = time.time()
    full_res, _, _ = eval_pipeline(df)
    drop_res, _, _ = eval_pipeline(df, drop_cols=drop_cols)
    out = {
        "with_fraud_rate": full_res,
        "without_fraud_rate": drop_res,
        "pr_auc_delta": full_res["pr_auc"] - drop_res["pr_auc"],
        "recall_at_1pct_delta": (full_res["recall_at_1pct"] or 0) - (drop_res["recall_at_1pct"] or 0),
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"  with fraud_rate    → PR-AUC={full_res['pr_auc']:.4f}, R@1%={full_res['recall_at_1pct']:.4f}")
    print(f"  without fraud_rate → PR-AUC={drop_res['pr_auc']:.4f}, R@1%={drop_res['recall_at_1pct']:.4f}")
    print(f"  Δ PR-AUC = {out['pr_auc_delta']:+.4f}")
    return out


# -------------------------------------------------------------------------------
# Test 4: New-vs-seen receiver subset
# -------------------------------------------------------------------------------
def test_new_vs_seen_receivers() -> dict:
    print("\n[4] NEW-vs-SEEN RECEIVER SUBSET TEST …")
    df = build_frame(use_label_dep=True, lag_days=7)
    split = time_based_split(df)
    num, cat = _feature_columns(df, drop_demographic=True)
    feat = num + cat
    pipe = _model_hgb(num, cat)
    pipe.fit(split.train[feat], split.train[TARGET])
    proba = pipe.predict_proba(split.test[feat])[:, 1]
    test = split.test.copy()
    train_receivers = set(split.train["ReceiverName"].unique())
    test["receiver_seen"] = test["ReceiverName"].isin(train_receivers)
    test["proba"] = proba

    from sklearn.metrics import average_precision_score, roc_auc_score
    out = {}
    for label, sub in (("seen_receivers", test[test["receiver_seen"]]),
                        ("new_receivers", test[~test["receiver_seen"]])):
        if len(sub) == 0:
            continue
        y = sub[TARGET].to_numpy()
        p = sub["proba"].to_numpy()
        n_pos = int(y.sum())
        out[label] = {
            "n_rows": len(sub),
            "n_positives": n_pos,
            "fraud_rate": float(y.mean()),
            "pr_auc": float(average_precision_score(y, p)) if n_pos >= 1 else None,
            "roc_auc": float(roc_auc_score(y, p)) if n_pos >= 1 and n_pos < len(sub) else None,
        }
    print(f"  Seen receivers: n={out.get('seen_receivers',{}).get('n_rows',0):,}, fraud={out.get('seen_receivers',{}).get('n_positives',0):,}, PR-AUC={out.get('seen_receivers',{}).get('pr_auc')}")
    print(f"  NEW receivers:  n={out.get('new_receivers',{}).get('n_rows',0):,}, fraud={out.get('new_receivers',{}).get('n_positives',0):,}, PR-AUC={out.get('new_receivers',{}).get('pr_auc')}")
    return out


def main():
    results = {}
    results["permutation"] = test_permutation()
    results["lag_sensitivity"] = test_lag_sensitivity()
    results["drop_fraud_rate"] = test_drop_fraud_rate()
    results["new_vs_seen_receivers"] = test_new_vs_seen_receivers()

    out_path = REPO / "artifacts" / "leakage_audit.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[audit] wrote {out_path}")
    return results


if __name__ == "__main__":
    main()
