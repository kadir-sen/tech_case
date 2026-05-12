# Run Comparison — Baseline vs Improved

| Model | Variant | PR-AUC (base) | PR-AUC (imp) | Δ PR-AUC | Recall@1% (base) | Recall@1% (imp) | Δ Recall@1% |
|---|---|---:|---:|---:|---:|---:|---:|
| hist_gbm | demographic_free | 0.8046 | 0.8091 | +0.0045 | 0.8992 | 0.8992 | +0.0000 |
| hist_gbm | full | 0.8052 | 0.8119 | +0.0067 | 0.9034 | 0.8992 | -0.0042 |
| logreg | demographic_free | 0.3486 | 0.3523 | +0.0037 | 0.8782 | 0.8908 | +0.0126 |
| logreg | full | 0.3503 | 0.3553 | +0.0050 | 0.8824 | 0.8739 | -0.0084 |
| random_forest | demographic_free | 0.6920 | 0.7088 | +0.0168 | 0.8908 | 0.8908 | +0.0000 |
| random_forest | full | 0.7261 | 0.7038 | -0.0223 | 0.8950 | 0.8992 | +0.0042 |
| rule_based | demographic_free | 0.0519 | 0.0519 | +0.0000 | 0.2479 | 0.2479 | +0.0000 |
| rule_based | full | 0.0519 | 0.0519 | +0.0000 | 0.2479 | 0.2479 | +0.0000 |
