"""Fraud detection metrikleri.

Operasyonel metrikler (precision@k, recall@k, fraud capture @ top X%) burada üretilir.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    fbeta_score, precision_score, recall_score, confusion_matrix,
)


def core_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "base_rate": float(y_true.mean()),
        "n": int(len(y_true)),
        "n_positives": int(y_true.sum()),
    }


def at_top_k_percentile(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> dict:
    """Top k_frac (örn 0.01 = top %1) skor için precision, recall, n_alerts, threshold."""
    n = len(y_score)
    k = max(1, int(round(n * k_frac)))
    order = np.argsort(-y_score, kind="stable")
    top_idx = order[:k]
    y_pred = np.zeros(n, dtype=np.int8)
    y_pred[top_idx] = 1
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {
        "k_frac": k_frac,
        "n_alerts": k,
        "threshold": float(y_score[order[k - 1]]),
        "precision": tp / k,
        "recall": tp / max(int(y_true.sum()), 1),
        "tp": tp, "fp": fp, "fn": fn,
        "fraud_capture_pct": tp / max(int(y_true.sum()), 1) * 100,
    }


def at_alerts_per_day(y_true: np.ndarray, y_score: np.ndarray,
                      alerts_per_day: int, total_days: int) -> dict:
    """Günlük X alert kapasitesi varsayımıyla; total_days üzerinden total alert sayısı."""
    n_total_alerts = max(1, alerts_per_day * total_days)
    n_total_alerts = min(n_total_alerts, len(y_score))
    k_frac = n_total_alerts / len(y_score)
    res = at_top_k_percentile(y_true, y_score, k_frac)
    res["alerts_per_day_target"] = alerts_per_day
    return res


def at_fixed_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(fbeta_score(y_true, y_pred, beta=1, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
    }


def summary(y_true: np.ndarray, y_score: np.ndarray,
            k_percentiles=(0.001, 0.005, 0.01, 0.05),
            alerts_per_day=(50, 100, 350, 1000),
            total_days_for_alerts: int | None = None) -> dict:
    out = {"core": core_metrics(y_true, y_score), "top_k_percentile": [], "by_alerts_per_day": []}
    for kf in k_percentiles:
        out["top_k_percentile"].append(at_top_k_percentile(y_true, y_score, kf))
    if total_days_for_alerts:
        for a in alerts_per_day:
            out["by_alerts_per_day"].append(
                at_alerts_per_day(y_true, y_score, a, total_days_for_alerts)
            )
    return out
