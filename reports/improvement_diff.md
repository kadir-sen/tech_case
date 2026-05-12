# Run Comparison — Baseline vs Improved

| Model | Variant | PR-AUC (base) | PR-AUC (imp) | Δ PR-AUC | Recall@1% (base) | Recall@1% (imp) | Δ Recall@1% |
|---|---|---:|---:|---:|---:|---:|---:|
| hist_gbm | demographic_free | 0.8091 | 0.8091 | +0.0000 | 0.8992 | 0.8992 | +0.0000 |
| hist_gbm | full | 0.8119 | 0.8119 | +0.0000 | 0.8992 | 0.8992 | +0.0000 |
| logreg | demographic_free | 0.3523 | 0.3523 | +0.0000 | 0.8908 | 0.8908 | +0.0000 |
| logreg | full | 0.3553 | 0.3553 | +0.0000 | 0.8739 | 0.8739 | +0.0000 |
| random_forest | demographic_free | 0.7088 | 0.7088 | +0.0000 | 0.8908 | 0.8908 | +0.0000 |
| random_forest | full | 0.7038 | 0.7038 | +0.0000 | 0.8992 | 0.8992 | +0.0000 |
| rule_based | demographic_free | 0.0519 | 0.0519 | +0.0000 | 0.2479 | 0.2479 | +0.0000 |
| rule_based | full | 0.0519 | 0.0519 | +0.0000 | 0.2479 | 0.2479 | +0.0000 |
