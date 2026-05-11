# Run Comparison — Baseline vs Improved

| Model | Variant | PR-AUC (base) | PR-AUC (imp) | Δ PR-AUC | Recall@1% (base) | Recall@1% (imp) | Δ Recall@1% |
|---|---|---:|---:|---:|---:|---:|---:|
| hist_gbm | demographic_free | 0.7673 | 0.8046 | +0.0373 | 0.9076 | 0.8992 | -0.0084 |
| hist_gbm | full | 0.7718 | 0.8052 | +0.0334 | 0.9160 | 0.9034 | -0.0126 |
| logreg | demographic_free | 0.3486 | 0.3486 | +0.0000 | 0.8782 | 0.8782 | +0.0000 |
| logreg | full | 0.3503 | 0.3503 | +0.0000 | 0.8824 | 0.8824 | +0.0000 |
| random_forest | demographic_free | 0.6920 | 0.6920 | +0.0000 | 0.8908 | 0.8908 | +0.0000 |
| random_forest | full | 0.7261 | 0.7261 | +0.0000 | 0.8950 | 0.8950 | +0.0000 |
| rule_based | demographic_free | 0.0519 | 0.0519 | +0.0000 | 0.2479 | 0.2479 | +0.0000 |
| rule_based | full | 0.0519 | 0.0519 | +0.0000 | 0.2479 | 0.2479 | +0.0000 |
