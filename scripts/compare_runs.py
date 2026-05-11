#!/usr/bin/env python3
"""İki run sonucunu kıyaslar (baseline vs improved).

Usage: python scripts/compare_runs.py <baseline.json> <improved.json> [--out reports/improvement_diff.md]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


def load(p: str) -> dict:
    return json.loads(Path(p).read_text())


def get_at_k(test, k_frac):
    for x in test["top_k_percentile"]:
        if abs(x["k_frac"] - k_frac) < 1e-9:
            return x
    return None


def render(base: dict, imp: dict) -> str:
    out = ["# Run Comparison — Baseline vs Improved\n"]
    out.append("| Model | Variant | PR-AUC (base) | PR-AUC (imp) | Δ PR-AUC | Recall@1% (base) | Recall@1% (imp) | Δ Recall@1% |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|")
    keys = sorted(set(base["models"]) | set(imp["models"]))
    for k in keys:
        bm = base["models"].get(k); im = imp["models"].get(k)
        if not (bm and im): continue
        parts = k.split("__")
        name, var = parts[0], parts[1] if len(parts) > 1 else ""
        b_pr = bm["test"]["core"]["pr_auc"]
        i_pr = im["test"]["core"]["pr_auc"]
        b_r1 = (get_at_k(bm["test"], 0.01) or {}).get("recall", 0)
        i_r1 = (get_at_k(im["test"], 0.01) or {}).get("recall", 0)
        out.append(f"| {name} | {var} | {b_pr:.4f} | {i_pr:.4f} | {i_pr-b_pr:+.4f} | "
                   f"{b_r1:.4f} | {i_r1:.4f} | {i_r1-b_r1:+.4f} |")
    return "\n".join(out) + "\n"


def main():
    baseline_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/eval/all_models__baseline.json"
    improved_path = sys.argv[2] if len(sys.argv) > 2 else "artifacts/eval/all_models.json"
    out_path = "reports/improvement_diff.md"
    for i, a in enumerate(sys.argv):
        if a == "--out" and i + 1 < len(sys.argv):
            out_path = sys.argv[i + 1]

    base = load(baseline_path)
    imp = load(improved_path)
    md = render(base, imp)
    Path(out_path).write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[compare] wrote {out_path}")


if __name__ == "__main__":
    main()
