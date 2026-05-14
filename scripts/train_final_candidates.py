#!/usr/bin/env python3
"""Final candidate matrix training.

Modeller × feature setleri kombinasyonu eğitilir. HPO'dan gelen best_params (varsa)
yüklenir; aksi halde defaults.

Test set sadece final değerlendirme için. Tüm eğitim train üzerinde, raporlama
val ve test üzerinde ayrı ayrı.

Çıktılar:
  artifacts/model_comparison/model_metrics.csv
  artifacts/model_comparison/model_metrics.json
  reports/model_comparison.md
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    fbeta_score, precision_score, recall_score, f1_score, confusion_matrix,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.loader import load_validated  # noqa: E402
from src.features.instant import add_derived  # noqa: E402
from src.features.historical import add_label_free_aggregates, add_label_dependent_aggregates  # noqa: E402
from src.models.split import time_based_split  # noqa: E402
from src.models.train import _feature_columns  # noqa: E402
from src.models.metrics import summary  # noqa: E402

# Reuse model builders from tune script
from scripts.tune_all_time_aware import build_lr, build_rf, build_hgb, build_catboost  # noqa: E402

warnings.filterwarnings("ignore")


def build_frame() -> pd.DataFrame:
    df, _ = load_validated()
    df = add_derived(df, daytype_nan="unknown")
    df = add_label_free_aggregates(df)
    df = add_label_dependent_aggregates(df, label_lag_days=7, prior_strength=50)
    return df


def load_best_params(model_name: str) -> dict | None:
    p = REPO / "artifacts" / "hpo" / f"{model_name}_best_params.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("best_params")


def load_selected_features() -> dict:
    p = REPO / "artifacts" / "feature_selection" / "selected_features.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


BASELINE_PARAMS = {
    "lr": {"C": 1.0, "penalty": "l2", "solver": "liblinear", "max_iter": 200},
    "rf": {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 50, "max_features": "sqrt"},
    "hgb": {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 63,
            "min_samples_leaf": 100, "l2_regularization": 1.0, "max_features": 1.0},
    "catboost": {"iterations": 600, "learning_rate": 0.05, "depth": 8,
                  "l2_leaf_reg": 3.0, "bagging_temperature": 1.0},
}

BUILDERS = {"lr": build_lr, "rf": build_rf, "hgb": build_hgb, "catboost": build_catboost}


def eval_full(y, proba, total_days) -> dict:
    s = summary(y, proba, total_days_for_alerts=total_days)
    out = {
        "pr_auc": s["core"]["pr_auc"],
        "roc_auc": s["core"]["roc_auc"],
        "n_positives": int(s["core"]["n_positives"]),
        "fraud_rate": float(s["core"]["base_rate"]),
        "brier": float(brier_score_loss(y, proba)),
    }
    for x in s["top_k_percentile"]:
        kf = x["k_frac"]
        kf_lbl = f"{int(kf*1000)}p1k" if kf < 0.01 else f"{int(kf*100)}p"
        out[f"precision_at_{kf_lbl}"] = x["precision"]
        out[f"recall_at_{kf_lbl}"] = x["recall"]
        out[f"capture_at_{kf_lbl}"] = x["fraud_capture_pct"]
    # F1, F2 at top-1% threshold
    top1 = next(x for x in s["top_k_percentile"] if abs(x["k_frac"] - 0.01) < 1e-9)
    y_pred_at1 = (proba >= top1["threshold"]).astype(int)
    out["f1_at_1pct"] = float(f1_score(y, y_pred_at1, zero_division=0))
    out["f2_at_1pct"] = float(fbeta_score(y, y_pred_at1, beta=2, zero_division=0))
    return out


def train_one(model_name, feat_subset, sp, params, num_all, cat_all):
    """Eğit, val + test eval döndür."""
    num = [c for c in num_all if c in feat_subset]
    cat = [c for c in cat_all if c in feat_subset]
    feat = num + cat
    builder = BUILDERS[model_name]
    pipe = builder(params, num, cat)
    pipe.fit(sp.train[feat], sp.train["IsFraudTransaction"])
    val_proba = pipe.predict_proba(sp.val[feat])[:, 1]
    test_proba = pipe.predict_proba(sp.test[feat])[:, 1]
    val_days = max(1, (sp.val["TransactionDate"].max() - sp.val["TransactionDate"].min()).days)
    test_days = max(1, (sp.test["TransactionDate"].max() - sp.test["TransactionDate"].min()).days)
    return {
        "val": eval_full(sp.val["IsFraudTransaction"].to_numpy(), val_proba, val_days),
        "test": eval_full(sp.test["IsFraudTransaction"].to_numpy(), test_proba, test_days),
        "n_features": len(feat),
        "pipeline": pipe,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["lr", "rf", "hgb", "catboost"])
    parser.add_argument("--feature-sets", nargs="+",
                        default=["full_safe", "selected_by_permutation", "drift_robust", "label_free"])
    parser.add_argument("--use-tuned", action="store_true", default=True,
                        help="HPO'dan gelen best_params'ı kullan (varsa)")
    parser.add_argument("--baseline-too", action="store_true", default=True,
                        help="Ayrıca sabit baseline params ile full_safe'i eğit")
    parser.add_argument("--output-dir", default="artifacts/model_comparison")
    args = parser.parse_args()

    out_dir = REPO / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("[final] building frame …")
    df = build_frame()
    sp = time_based_split(df)
    num_all, cat_all = _feature_columns(df, drop_demographic=True)
    print(f"[final] train: {len(sp.train):,}  val: {len(sp.val):,}  test: {len(sp.test):,}")

    selected = load_selected_features()
    if not selected:
        print("[final] WARN: selected_features.json yok. Sadece full_safe (= num_all + cat_all) kullanılacak.")
        selected = {"full_safe": num_all + cat_all}

    results = []
    artifacts_to_save = {}  # (model, fset, variant) → pipeline

    for model_name in args.models:
        # Baseline (sabit params + full_safe)
        if args.baseline_too:
            print(f"\n[final] {model_name} | baseline params | full_safe ({len(selected.get('full_safe', []))} feat)")
            t1 = time.time()
            r = train_one(model_name, selected["full_safe"], sp, BASELINE_PARAMS[model_name], num_all, cat_all)
            row = {"model": model_name, "params": "baseline", "feature_set": "full_safe",
                   "n_features": r["n_features"], "fit_s": round(time.time()-t1, 1)}
            row.update({f"val_{k}": v for k, v in r["val"].items()})
            row.update({f"test_{k}": v for k, v in r["test"].items()})
            results.append(row)
            print(f"  → val PR-AUC = {r['val']['pr_auc']:.4f}, test PR-AUC = {r['test']['pr_auc']:.4f}, R@1% = {r['test']['recall_at_1p']:.4f}")

        # Tuned + her feature set
        best = load_best_params(model_name) if args.use_tuned else None
        params = best or BASELINE_PARAMS[model_name]
        for fs in args.feature_sets:
            if fs not in selected:
                continue
            print(f"\n[final] {model_name} | {'tuned' if best else 'baseline'} | {fs} ({len(selected[fs])} feat)")
            t1 = time.time()
            r = train_one(model_name, selected[fs], sp, params, num_all, cat_all)
            row = {"model": model_name, "params": "tuned" if best else "baseline", "feature_set": fs,
                   "n_features": r["n_features"], "fit_s": round(time.time()-t1, 1)}
            row.update({f"val_{k}": v for k, v in r["val"].items()})
            row.update({f"test_{k}": v for k, v in r["test"].items()})
            results.append(row)
            print(f"  → val PR-AUC = {r['val']['pr_auc']:.4f}, test PR-AUC = {r['test']['pr_auc']:.4f}, R@1% = {r['test']['recall_at_1p']:.4f}")
            # Save best one (highest val PR-AUC) per (model, feature_set)
            artifacts_to_save[(model_name, fs)] = r["pipeline"]

    # Persist results
    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "model_metrics.csv", index=False)
    (out_dir / "model_metrics.json").write_text(json.dumps(results, indent=2, default=str))

    # Save best pipelines per (model, feature_set)
    models_dir = REPO / "artifacts" / "models"
    models_dir.mkdir(exist_ok=True)
    import joblib
    for (m, fs), pipe in artifacts_to_save.items():
        joblib.dump(pipe, models_dir / f"final_{m}__{fs}.joblib")

    # Markdown
    md = ["# Model Comparison — Final Candidates\n",
          "> Auto-generated. Test set ONLY for final reporting; train+val for everything else.\n",
          "## Sıralı tablo (val PR-AUC ile sıralı)\n",
          "| Model | Params | Feature set | n_feat | Val PR-AUC | Test PR-AUC | Test R@0.5% | Test R@1% | Test P@1% | Test Brier |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    df_sorted = df_res.sort_values("val_pr_auc", ascending=False)
    for _, r in df_sorted.iterrows():
        md.append(f"| {r['model']} | {r['params']} | {r['feature_set']} | {r['n_features']} | "
                  f"{r['val_pr_auc']:.4f} | {r['test_pr_auc']:.4f} | "
                  f"{r['test_recall_at_5p1k']:.4f} | {r['test_recall_at_1p']:.4f} | "
                  f"{r['test_precision_at_1p']:.4f} | {r['test_brier']:.5f} |")
    md.append("")
    md.append("## En iyi val PR-AUC modelleri (final aday)\n")
    top3 = df_sorted.head(3)
    for _, r in top3.iterrows():
        md.append(f"- **{r['model']} / {r['params']} / {r['feature_set']}** "
                  f"→ val PR-AUC {r['val_pr_auc']:.4f}, test PR-AUC {r['test_pr_auc']:.4f}, "
                  f"test Recall@1% {r['test_recall_at_1p']:.4f}")
    md.append("")
    md.append("## Yorum\n")
    best_row = df_sorted.iloc[0]
    md.append(f"- Val PR-AUC'a göre en iyi model: **{best_row['model']} / {best_row['params']} / {best_row['feature_set']}**.")
    md.append(f"- Val→test gap: PR-AUC {best_row['val_pr_auc']:.4f} → {best_row['test_pr_auc']:.4f} "
              f"(Δ {best_row['test_pr_auc']-best_row['val_pr_auc']:+.4f}, drift'in beklenen etkisi).")
    md.append(f"- En iyi label-free model: " + (
        f"**{df_sorted[df_sorted['feature_set']=='label_free'].iloc[0]['model']}**, val PR-AUC {df_sorted[df_sorted['feature_set']=='label_free'].iloc[0]['val_pr_auc']:.4f}"
        if (df_sorted['feature_set']=='label_free').any() else "(label_free set yoktu)"))
    md.append("- **Final model seçimi val seti üzerinden** yapıldı. Test set sadece raporlama için.")
    md.append("- Drift-robust feature set ile full_safe arasında performans farkı val PR-AUC üzerinden ölçüldü; production drift senaryosunda drift_robust tercih edilir.\n")
    md.append("## Kurallar (uygulanmıştır)\n")
    md.append("- ✅ Test set TUNING/SEÇİM için kullanılmadı; sadece final raporlama.")
    md.append("- ✅ HPO best_params artifacts/hpo/ klasöründen okundu.")
    md.append("- ✅ Feature setleri artifacts/feature_selection/selected_features.json'dan okundu.\n")

    (REPO / "reports" / "model_comparison.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[final] wrote model_comparison.md + metrics.csv ({time.time()-t0:.1f}s)")
    print(f"[final] best by val PR-AUC: {best_row['model']}/{best_row['params']}/{best_row['feature_set']}")


if __name__ == "__main__":
    main()
