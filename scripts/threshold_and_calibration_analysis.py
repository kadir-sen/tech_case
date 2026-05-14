#!/usr/bin/env python3
"""Threshold + calibration analizi — final aday model üzerinde.

Üç tip threshold perspektifi:
  1) Global top-k (top %0.1 / %0.5 / %1 / %5)
  2) Daily top-k (her gün için top %k; günler arası varyans)
  3) Alerts/day business scenario (50, 100, 350, 1000)

Calibration:
  - Brier score
  - Reliability bins (10 bin)
  - Platt vs Isotonic vs uncalibrated karşılaştırması (val fit, test eval)

Çıktılar:
  artifacts/thresholds/threshold_table.csv
  artifacts/thresholds/daily_topk.csv
  artifacts/thresholds/risk_bands.json
  artifacts/thresholds/calibration.csv
  reports/threshold_analysis.md
  reports/calibration_report.md
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss, average_precision_score, precision_score, recall_score,
    f1_score, fbeta_score,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.loader import load_validated  # noqa: E402
from src.features.instant import add_derived  # noqa: E402
from src.features.historical import add_label_free_aggregates, add_label_dependent_aggregates  # noqa: E402
from src.models.split import time_based_split  # noqa: E402
from src.models.train import _feature_columns  # noqa: E402

warnings.filterwarnings("ignore")


def build_frame() -> pd.DataFrame:
    df, _ = load_validated()
    df = add_derived(df, daytype_nan="unknown")
    df = add_label_free_aggregates(df)
    df = add_label_dependent_aggregates(df, label_lag_days=7, prior_strength=50)
    return df


def find_model_path(model_name: str) -> Path:
    """Final candidates artifactlerinden öncelikli olarak yükle, yoksa eski models/'tan."""
    p = REPO / "artifacts" / "models" / f"final_{model_name}.joblib"
    if p.exists():
        return p
    p2 = REPO / "artifacts" / "models" / f"{model_name}.joblib"
    return p2


def global_topk_table(y, proba, ks=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.10)) -> pd.DataFrame:
    rows = []
    order = np.argsort(-proba, kind="stable")
    n = len(proba)
    for k in ks:
        kk = max(1, int(round(n * k)))
        thr = float(proba[order[kk - 1]])
        y_pred = np.zeros(n, dtype=int); y_pred[order[:kk]] = 1
        tp = int(((y == 1) & (y_pred == 1)).sum())
        fp = int(((y == 0) & (y_pred == 1)).sum())
        fn = int(((y == 1) & (y_pred == 0)).sum())
        n_pos = int(y.sum())
        rows.append({
            "k_frac": k,
            "n_alerts": kk,
            "threshold": thr,
            "precision": tp / kk,
            "recall": tp / max(n_pos, 1),
            "capture_pct": tp / max(n_pos, 1) * 100,
            "f1": f1_score(y, y_pred, zero_division=0),
            "f2": fbeta_score(y, y_pred, beta=2, zero_division=0),
            "tp": tp, "fp": fp, "fn": fn,
        })
    return pd.DataFrame(rows)


def daily_topk_table(dates, y, proba, ks=(0.005, 0.01, 0.05)) -> pd.DataFrame:
    """Her gün için top-k alert seç, precision/recall hesapla."""
    df = pd.DataFrame({"date": pd.to_datetime(dates).date, "y": y, "p": proba})
    rows = []
    for k in ks:
        per_day = []
        for d, g in df.groupby("date"):
            n = len(g); kk = max(1, int(round(n * k)))
            ord_idx = np.argsort(-g["p"].to_numpy(), kind="stable")
            y_pred = np.zeros(n, dtype=int); y_pred[ord_idx[:kk]] = 1
            tp = int(((g["y"].to_numpy() == 1) & (y_pred == 1)).sum())
            n_pos = int(g["y"].sum())
            per_day.append({
                "date": d, "k_frac": k, "n": n, "n_alerts": kk,
                "precision": tp / kk if kk > 0 else 0.0,
                "recall": tp / max(n_pos, 1) if n_pos > 0 else 0.0,
                "n_positives": n_pos,
            })
        rows.extend(per_day)
    return pd.DataFrame(rows)


def alerts_per_day_table(y, proba, total_days, alerts_per_day=(50, 100, 350, 1000)) -> pd.DataFrame:
    rows = []
    n = len(proba)
    order = np.argsort(-proba, kind="stable")
    n_pos = int(y.sum())
    for apd in alerts_per_day:
        total = min(n, max(1, apd * total_days))
        thr = float(proba[order[total - 1]])
        y_pred = np.zeros(n, dtype=int); y_pred[order[:total]] = 1
        tp = int(((y == 1) & (y_pred == 1)).sum())
        rows.append({
            "alerts_per_day": apd,
            "total_alerts": total,
            "threshold": thr,
            "precision": tp / total,
            "recall": tp / max(n_pos, 1),
        })
    return pd.DataFrame(rows)


def reliability_bins(y, proba, n_bins=10) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (proba >= lo) & (proba < hi if i < n_bins - 1 else proba <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "bin_lo": float(lo), "bin_hi": float(hi), "n": n,
            "avg_proba": float(proba[mask].mean()),
            "fraction_positive": float(y[mask].mean()),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="catboost__demographic_free")
    parser.add_argument("--output-dir", default="artifacts/thresholds")
    args = parser.parse_args()
    out_dir = REPO / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    print("[thr] building frame …")
    df = build_frame()
    sp = time_based_split(df)
    num, cat = _feature_columns(df, drop_demographic=True)
    feat = num + cat

    model_path = find_model_path(args.model)
    print(f"[thr] loading {model_path}")
    pipe = joblib.load(model_path)
    val_proba = pipe.predict_proba(sp.val[feat])[:, 1]
    test_proba = pipe.predict_proba(sp.test[feat])[:, 1]
    y_val = sp.val["IsFraudTransaction"].to_numpy()
    y_test = sp.test["IsFraudTransaction"].to_numpy()
    val_days = max(1, (sp.val["TransactionDate"].max() - sp.val["TransactionDate"].min()).days)
    test_days = max(1, (sp.test["TransactionDate"].max() - sp.test["TransactionDate"].min()).days)

    # 1) Global top-k
    print("[thr] global top-k …")
    gtk_val = global_topk_table(y_val, val_proba); gtk_val["split"] = "val"
    gtk_test = global_topk_table(y_test, test_proba); gtk_test["split"] = "test"
    gtk = pd.concat([gtk_val, gtk_test], ignore_index=True)
    gtk.to_csv(out_dir / "threshold_table.csv", index=False)

    # 2) Daily top-k
    print("[thr] daily top-k (test) …")
    dtk = daily_topk_table(sp.test["TransactionDate"].to_numpy(), y_test, test_proba)
    dtk.to_csv(out_dir / "daily_topk.csv", index=False)
    dtk_summary = dtk.groupby("k_frac").agg(
        n_days=("date", "nunique"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        n_alerts_mean=("n_alerts", "mean"),
    ).reset_index()
    dtk_summary.to_csv(out_dir / "daily_topk_summary.csv", index=False)

    # 3) Alerts/day
    print("[thr] alerts/day scenarios (test) …")
    apd = alerts_per_day_table(y_test, test_proba, test_days)
    apd.to_csv(out_dir / "alerts_per_day.csv", index=False)

    # 4) Risk band (val seti percentile, test'te raporla)
    print("[thr] risk bands …")
    order_val = np.argsort(-val_proba, kind="stable")
    high_cut = float(val_proba[order_val[max(1, int(0.001 * len(val_proba))) - 1]])  # top 0.1%
    med_cut = float(val_proba[order_val[max(1, int(0.01 * len(val_proba))) - 1]])    # top 1%
    bands = {
        "best_model": args.model,
        "HIGH": high_cut, "MEDIUM": med_cut,
        "policy": {"HIGH": "BLOCK_OR_STEP_UP", "MEDIUM": "MANUAL_REVIEW", "LOW": "ALLOW"},
        "derived_from": "val_set_percentiles",
    }
    (out_dir / "risk_bands.json").write_text(json.dumps(bands, indent=2))

    # 5) Calibration
    print("[thr] calibration (Platt + Isotonic, val fit, test eval) …")
    # Uncalibrated baseline
    brier_unc = float(brier_score_loss(y_test, test_proba))
    rel_unc = reliability_bins(y_test, test_proba)
    rel_unc["calibration"] = "uncalibrated"

    # Platt (sigmoid) — fit on val (raw probs)
    # CalibratedClassifierCV requires base estimator; alternative: fit a LR on val_proba→y
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
    platt.fit(val_proba.reshape(-1, 1), y_val)
    test_proba_platt = platt.predict_proba(test_proba.reshape(-1, 1))[:, 1]
    brier_platt = float(brier_score_loss(y_test, test_proba_platt))
    pr_auc_platt = float(average_precision_score(y_test, test_proba_platt))
    rel_platt = reliability_bins(y_test, test_proba_platt); rel_platt["calibration"] = "platt"

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_proba, y_val)
    test_proba_iso = iso.predict(test_proba)
    brier_iso = float(brier_score_loss(y_test, test_proba_iso))
    pr_auc_iso = float(average_precision_score(y_test, test_proba_iso))
    rel_iso = reliability_bins(y_test, test_proba_iso); rel_iso["calibration"] = "isotonic"

    rel_all = pd.concat([rel_unc, rel_platt, rel_iso], ignore_index=True)
    rel_all.to_csv(out_dir / "calibration.csv", index=False)

    cal_summary = {
        "uncalibrated": {"brier": brier_unc, "pr_auc": float(average_precision_score(y_test, test_proba))},
        "platt": {"brier": brier_platt, "pr_auc": pr_auc_platt},
        "isotonic": {"brier": brier_iso, "pr_auc": pr_auc_iso},
    }
    (out_dir / "calibration_summary.json").write_text(json.dumps(cal_summary, indent=2))

    # --- Markdown reports ---
    md_thr = ["# Threshold Analizi — Final aday model\n",
              f"> Model: `{args.model}`. Auto-generated.\n",
              "## 1. Global top-k (test set)\n",
              "| k | n_alerts | threshold | Precision | Recall | F1 | F2 | capture% |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in gtk_test.iterrows():
        md_thr.append(f"| {r['k_frac']:.4f} | {r['n_alerts']} | {r['threshold']:.4f} | "
                      f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['f2']:.4f} | {r['capture_pct']:.1f}% |")
    md_thr.append("")
    md_thr.append("## 2. Daily top-k (test set) — günler arası varyans\n")
    md_thr.append("| k_frac | n_days | precision_mean | precision_std | recall_mean | recall_std | n_alerts_mean/day |")
    md_thr.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in dtk_summary.iterrows():
        md_thr.append(f"| {r['k_frac']:.3f} | {int(r['n_days'])} | {r['precision_mean']:.4f} | {r['precision_std']:.4f} | "
                      f"{r['recall_mean']:.4f} | {r['recall_std']:.4f} | {r['n_alerts_mean']:.1f} |")
    md_thr.append("")
    md_thr.append("## 3. Business scenario — alerts/day (test set)\n")
    md_thr.append("| alerts/day | total_alerts | threshold | Precision | Recall |")
    md_thr.append("|---|---:|---:|---:|---:|")
    for _, r in apd.iterrows():
        md_thr.append(f"| {r['alerts_per_day']} | {r['total_alerts']} | {r['threshold']:.4f} | "
                      f"{r['precision']:.4f} | {r['recall']:.4f} |")
    md_thr.append("")
    md_thr.append("## 4. Risk band politikası (val'den)\n")
    md_thr.append(f"```json\n{json.dumps(bands, indent=2)}\n```\n")
    md_thr.append("## Yorum\n")
    md_thr.append("- Model bir **olasılık skoru** üretir; karar threshold ile verilir.")
    md_thr.append("- Default 0.5 fraud detection için yanıltıcı (imbalance %0.15).")
    md_thr.append("- **Global top-k** = modelin saf ranking gücü.")
    md_thr.append("- **Daily top-k** = günlük operasyonel kapasiteye yakın; günler arası varyansı görmek için.")
    md_thr.append("- **Business scenario** = iş tarafının kapasitesi belliyse direkt karar verilebilir.")
    md_thr.append("- Final threshold seçimi val seti üzerinden yapıldı; test sadece raporlama.\n")
    (REPO / "reports" / "threshold_analysis.md").write_text("\n".join(md_thr), encoding="utf-8")

    md_cal = ["# Calibration Raporu\n",
              f"> Model: `{args.model}`. Val seti üzerinde calibrator fit, test üzerinde eval.\n",
              "## Brier + PR-AUC karşılaştırması\n",
              "| Calibration | Brier (↓ iyi) | Test PR-AUC |",
              "|---|---:|---:|",
              f"| Uncalibrated | {cal_summary['uncalibrated']['brier']:.5f} | {cal_summary['uncalibrated']['pr_auc']:.4f} |",
              f"| Platt (sigmoid) | {cal_summary['platt']['brier']:.5f} | {cal_summary['platt']['pr_auc']:.4f} |",
              f"| Isotonic | {cal_summary['isotonic']['brier']:.5f} | {cal_summary['isotonic']['pr_auc']:.4f} |",
              ""]
    md_cal.append("## Reliability — bin başına gözlenen pozitif oranı (test, uncalibrated)\n")
    md_cal.append("| bin_lo | bin_hi | n | avg_proba | observed_positive_rate |")
    md_cal.append("|---|---|---:|---:|---:|")
    for _, r in rel_unc.iterrows():
        md_cal.append(f"| {r['bin_lo']:.2f} | {r['bin_hi']:.2f} | {r['n']} | {r['avg_proba']:.4f} | {r['fraction_positive']:.4f} |")
    md_cal.append("")
    md_cal.append("## Yorum\n")
    md_cal.append("- Brier düşükse model olasılığı iyi kalibre. Imbalance'da düşük base rate Brier'ı küçük gösterebilir, dikkat.")
    md_cal.append("- PR-AUC kalibrasyondan etkilenmez (ranking metriği). Brier ve reliability bin'leri etkilenir.")
    md_cal.append("- En düşük Brier'a sahip yöntem production önerilir. Daha iyi kalibre skor → daha güvenilir threshold davranışı.")
    md_cal.append("- Bu rapor val fit + test eval ile leakage-free.")
    (REPO / "reports" / "calibration_report.md").write_text("\n".join(md_cal), encoding="utf-8")

    print(f"[thr] wrote threshold_analysis.md + calibration_report.md")
    print(f"[thr] best calibration (Brier): "
          f"unc={cal_summary['uncalibrated']['brier']:.5f}, "
          f"platt={cal_summary['platt']['brier']:.5f}, "
          f"iso={cal_summary['isotonic']['brier']:.5f}")


if __name__ == "__main__":
    main()
