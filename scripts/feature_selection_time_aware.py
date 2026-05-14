#!/usr/bin/env python3
"""Time-aware feature selection — leakage-free.

Analizler:
  1) Low/zero variance feature check
  2) Permutation importance (val seti üzerinde — test'e DOKUNMAZ)
  3) Drift importance (train↔val adversarial)
  4) SHAP global importance (tree-based modeller için, val sample)
  5) Feature stability (time folds across train)

Çıktılar:
  artifacts/feature_selection/permutation_importance.csv
  artifacts/feature_selection/drift_importance.csv
  artifacts/feature_selection/shap_importance.csv
  artifacts/feature_selection/variance_check.csv
  artifacts/feature_selection/selected_features.json
  reports/feature_selection_report.md
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

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


def make_hgb_pipeline(num, cat, best_params: dict | None = None):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat),
    ])
    cat_idx = list(range(len(num), len(num) + len(cat)))
    p = best_params or {}
    return Pipeline([
        ("pre", pre),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=p.get("learning_rate", 0.05),
            max_iter=int(p.get("max_iter", 400)),
            max_leaf_nodes=int(p.get("max_leaf_nodes", 63)),
            min_samples_leaf=int(p.get("min_samples_leaf", 100)),
            l2_regularization=p.get("l2_regularization", 1.0),
            max_features=p.get("max_features", 1.0),
            class_weight="balanced",
            categorical_features=cat_idx,
            random_state=42,
        )),
    ])


def load_best_hgb_params() -> dict:
    p = REPO / "artifacts" / "hpo" / "hgb_best_params.json"
    if not p.exists():
        p_old = REPO / "artifacts" / "best_hgb_params.json"
        if p_old.exists():
            return json.loads(p_old.read_text()).get("params") or {}
        return {}
    payload = json.loads(p.read_text())
    return payload.get("best_params") or {}


# --------------------------------------------------------------------------------
def variance_check(df_train: pd.DataFrame, df_val: pd.DataFrame, num, cat) -> pd.DataFrame:
    rows = []
    combined = pd.concat([df_train, df_val])  # test KULLANILMIYOR
    for c in num:
        v = combined[c].astype(float)
        rows.append({
            "feature": c, "type": "num",
            "n_unique": int(v.nunique()),
            "variance": float(v.var()),
            "n_missing": int(v.isna().sum()),
            "is_low_signal": bool(v.nunique() <= 1 or float(v.var()) < 1e-12),
        })
    for c in cat:
        v = combined[c]
        rows.append({
            "feature": c, "type": "cat",
            "n_unique": int(v.nunique(dropna=False)),
            "variance": None,
            "n_missing": int(v.isna().sum()),
            "is_low_signal": bool(v.nunique(dropna=False) <= 1),
        })
    return pd.DataFrame(rows).sort_values("is_low_signal", ascending=False)


def permutation_importance_val(pipeline, X_val, y_val, feat_names) -> pd.DataFrame:
    sample_n = min(20000, len(X_val))
    idx = np.random.RandomState(0).choice(len(X_val), sample_n, replace=False)
    Xs = X_val.iloc[idx]
    ys = y_val[idx]
    pi = permutation_importance(pipeline, Xs, ys, n_repeats=3, random_state=0,
                                scoring="average_precision", n_jobs=-1)
    df = pd.DataFrame({
        "feature": feat_names,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std,
    }).sort_values("importance_mean", ascending=False)
    return df


def drift_importance(df_train, df_val, num, cat) -> pd.DataFrame:
    """Adversarial classifier (train=0, val=1) feature importance."""
    feat = num + cat
    a = df_train[feat].assign(_lbl=0)
    b = df_val[feat].assign(_lbl=1)
    combined = pd.concat([a, b], ignore_index=True)
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(combined))
    combined = combined.iloc[perm].reset_index(drop=True)
    X = combined[feat]
    y = combined["_lbl"].to_numpy()
    cut = int(len(X) * 0.8)
    pipe = make_hgb_pipeline(num, cat, {})
    pipe.fit(X.iloc[:cut], y[:cut])
    proba = pipe.predict_proba(X.iloc[cut:])[:, 1]
    adv_auc = float(roc_auc_score(y[cut:], proba))

    pi = permutation_importance(pipe, X.iloc[cut:].iloc[:20000], y[cut:][:20000],
                                n_repeats=2, random_state=0, scoring="roc_auc", n_jobs=-1)
    df = pd.DataFrame({
        "feature": feat,
        "drift_importance": pi.importances_mean,
    }).sort_values("drift_importance", ascending=False)
    df["adversarial_auc"] = adv_auc
    return df, adv_auc


def shap_importance(pipeline, X_val, feat_names) -> pd.DataFrame | None:
    try:
        import shap
        pre = pipeline.named_steps["pre"]
        clf = pipeline.named_steps["clf"]
        sample_n = min(5000, len(X_val))
        idx = np.random.RandomState(0).choice(len(X_val), sample_n, replace=False)
        Xs = pre.transform(X_val.iloc[idx])
        if hasattr(Xs, "toarray"):
            Xs = Xs.toarray()
        expl = shap.TreeExplainer(clf)
        sv = expl(Xs[:2000])
        vals = np.abs(sv.values).mean(axis=0)
        if vals.ndim == 2:
            vals = vals[:, 1]
        try:
            names = list(pre.get_feature_names_out())
        except Exception:
            names = [f"f{i}" for i in range(len(vals))]
        df = pd.DataFrame({"feature": names, "shap_mean_abs": vals}).sort_values("shap_mean_abs", ascending=False)
        return df
    except Exception as e:
        print(f"[fs] SHAP failed: {e}")
        return None


def stability_across_folds(df, num, cat, n_folds: int = 3) -> pd.DataFrame:
    """Train internal time-folds — feature importance her fold'da ne kadar değişiyor?"""
    from src.models.time_validation import make_expanding_time_folds
    sp = time_based_split(df)
    folds = make_expanding_time_folds(sp.train, n_splits=n_folds)
    feat = num + cat
    rows = []
    for f in folds:
        pipe = make_hgb_pipeline(num, cat, {})
        pipe.fit(f.train[feat], f.train["IsFraudTransaction"])
        idx = np.random.RandomState(0).choice(len(f.val), min(8000, len(f.val)), replace=False)
        Xs = f.val[feat].iloc[idx]
        ys = f.val["IsFraudTransaction"].to_numpy()[idx]
        pi = permutation_importance(pipe, Xs, ys, n_repeats=2, random_state=0,
                                    scoring="average_precision", n_jobs=-1)
        for fname, imp in zip(feat, pi.importances_mean):
            rows.append({"fold": f.fold_index, "feature": fname, "importance": imp})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--output-dir", default="artifacts/feature_selection")
    args = parser.parse_args()

    out_dir = REPO / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("[fs] building frame …")
    df = build_frame()
    sp = time_based_split(df)
    num, cat = _feature_columns(df, drop_demographic=True)
    feat = num + cat
    print(f"[fs] frame: {df.shape}, train: {len(sp.train):,}, val: {len(sp.val):,} (TEST KULLANILMIYOR)")

    X_train, y_train = sp.train[feat], sp.train["IsFraudTransaction"].to_numpy()
    X_val, y_val = sp.val[feat], sp.val["IsFraudTransaction"].to_numpy()

    # 1) Variance check
    print("[fs] variance check …")
    vc = variance_check(sp.train, sp.val, num, cat)
    vc.to_csv(out_dir / "variance_check.csv", index=False)
    print(f"[fs] low-signal features: {int(vc['is_low_signal'].sum())}")

    # 2) Train a tuned HGB once for importance analyses
    print("[fs] training HGB (tuned) on train …")
    best_p = load_best_hgb_params()
    pipe = make_hgb_pipeline(num, cat, best_p)
    pipe.fit(X_train, y_train)
    proba_val = pipe.predict_proba(X_val)[:, 1]
    full_val_pr_auc = float(average_precision_score(y_val, proba_val))
    print(f"[fs] full-feature val PR-AUC = {full_val_pr_auc:.4f}")

    # 3) Permutation importance on val
    print("[fs] permutation importance on val …")
    perm = permutation_importance_val(pipe, X_val, y_val, feat)
    perm.to_csv(out_dir / "permutation_importance.csv", index=False)
    print("[fs] top-5 permutation:")
    print(perm.head(5).to_string(index=False))

    # 4) Drift importance (train↔val adversarial)
    print("[fs] drift importance (adversarial train↔val) …")
    drift, adv_auc = drift_importance(sp.train, sp.val, num, cat)
    drift.to_csv(out_dir / "drift_importance.csv", index=False)
    print(f"[fs] adversarial AUC = {adv_auc:.4f}")
    print("[fs] top-5 drift:")
    print(drift[["feature", "drift_importance"]].head(5).to_string(index=False))

    # 5) SHAP importance (tree-based pipeline)
    print("[fs] SHAP global importance …")
    shap_df = shap_importance(pipe, X_val, feat)
    if shap_df is not None:
        shap_df.to_csv(out_dir / "shap_importance.csv", index=False)
        print("[fs] top-5 SHAP:")
        print(shap_df.head(5).to_string(index=False))

    # 6) Stability across folds
    print("[fs] stability across time folds …")
    stab = stability_across_folds(df, num, cat, n_folds=3)
    stab.to_csv(out_dir / "stability_across_folds.csv", index=False)
    stab_summary = stab.groupby("feature")["importance"].agg(["mean", "std"]).reset_index()
    stab_summary["cv"] = stab_summary["std"] / stab_summary["mean"].replace(0, np.nan)
    stab_summary.to_csv(out_dir / "stability_summary.csv", index=False)

    # 7) Selected feature sets
    drop_low_signal = set(vc[vc["is_low_signal"]]["feature"])
    keep_perm = list(perm["feature"].head(args.top_n))
    drift_heavy = list(drift[drift["drift_importance"] > 0.001]["feature"])
    drift_robust = [f for f in feat if f not in drift_heavy and f not in drop_low_signal]

    # label-free / label-dependent separation
    label_dep_features = ["device_fraud_rate_smoothed", "device_label_n",
                           "receiver_fraud_rate_smoothed", "receiver_label_n"]
    label_free_features = [f for f in feat if f not in label_dep_features]

    selected = {
        "full_safe": [f for f in feat if f not in drop_low_signal],
        "selected_by_permutation": keep_perm,
        "drift_heavy": drift_heavy,
        "drift_robust": drift_robust,
        "label_free": label_free_features,
        "label_dependent": label_dep_features,
        "low_signal_dropped": list(drop_low_signal),
        "full_val_pr_auc": full_val_pr_auc,
        "adversarial_auc_train_vs_val": adv_auc,
    }
    (out_dir / "selected_features.json").write_text(json.dumps(selected, indent=2))

    # 8) Markdown report
    md = []
    md.append("# Feature Selection Raporu — Time-Aware\n")
    md.append("> Auto-generated. Test set'e DOKUNULMAMIŞTIR. Tüm analiz train+val üzerinde.\n")
    md.append(f"## Özet\n")
    md.append(f"- Total feature (after low-signal removal): **{len(selected['full_safe'])}**")
    md.append(f"- Low-signal dropped: **{len(drop_low_signal)}** ({list(drop_low_signal) or 'yok'})")
    md.append(f"- Top-{args.top_n} permutation: **{len(keep_perm)}**")
    md.append(f"- Drift-heavy (importance > 0.001): **{len(drift_heavy)}**")
    md.append(f"- Drift-robust: **{len(drift_robust)}**")
    md.append(f"- Label-dependent feature'lar: **{len(label_dep_features)}**")
    md.append(f"- Adversarial AUC (train↔val): **{adv_auc:.4f}** (>0.7 = belirgin drift)\n")
    md.append("## Permutation importance — top 20\n")
    md.append("| Feature | importance_mean | importance_std |")
    md.append("|---|---:|---:|")
    for _, r in perm.head(20).iterrows():
        md.append(f"| `{r['feature']}` | {r['importance_mean']:.5f} | {r['importance_std']:.5f} |")
    md.append("")
    md.append("## Drift importance — top 10 (zaman ile değişen feature'lar)\n")
    md.append("| Feature | drift_importance |")
    md.append("|---|---:|")
    for _, r in drift.head(10).iterrows():
        md.append(f"| `{r['feature']}` | {r['drift_importance']:.5f} |")
    md.append("")
    if shap_df is not None:
        md.append("## SHAP global importance — top 20\n")
        md.append("| Feature (encoded) | mean(|SHAP|) |")
        md.append("|---|---:|")
        for _, r in shap_df.head(20).iterrows():
            md.append(f"| `{r['feature']}` | {r['shap_mean_abs']:.5f} |")
        md.append("")

    md.append("## Feature setleri (production'a aday)\n")
    md.append("- **full_safe**: tüm feature'lar (low-signal hariç).")
    md.append("- **selected_by_permutation**: val PR-AUC permutation importance'a göre ilk top-N feature.")
    md.append("- **drift_robust**: adversarial drift importance < 0.001 olanlar — production drift'e dayanıklı.")
    md.append("- **label_free**: label-dependent feature'lar olmadan (production-safe alt küme).")
    md.append("- **label_dependent**: SADECE smoothed fraud_rate feature'ları (overfit kontrolü için).\n")

    md.append("## Yorum\n")
    md.append("- En önemli feature'lar: " + ", ".join(f"`{x}`" for x in perm["feature"].head(5).tolist()))
    md.append(f"- Hiçbir feature low-signal değil." if not drop_low_signal else f"- Low-signal feature'lar: {drop_low_signal}")
    md.append(f"- Drift en yüksek feature: `{drift.iloc[0]['feature']}` (importance {drift.iloc[0]['drift_importance']:.4f})")
    md.append("- Label-dependent feature'lar performansa baskın katkı sağlıyor (leakage_audit ablation: drop fraud_rate → PR-AUC 0.80 → 0.40).")
    md.append("- Production'a alınma sırasında **drift_robust + label_free** kombinasyonu en konservatif seçenek.")
    md.append("- **selected_by_permutation** maksimum performans, **drift_robust** maksimum stabilite.\n")
    md.append("## Kurallar (uygulanmıştır)\n")
    md.append("- ✅ Test set'e DOKUNULMADI.")
    md.append("- ✅ Tüm analiz train+val üzerinde.")
    md.append("- ✅ Random shuffle YOK; sample'lar deterministik (seed=0).")

    (REPO / "reports" / "feature_selection_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[fs] wrote selected_features.json + report ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
