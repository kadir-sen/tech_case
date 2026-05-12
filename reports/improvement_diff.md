# Run Comparison — Baseline vs Improved

| Model | Variant | PR-AUC (base) | PR-AUC (imp) | Δ PR-AUC | Recall@1% (base) | Recall@1% (imp) | Δ Recall@1% |
|---|---|---:|---:|---:|---:|---:|---:|
| hist_gbm | demographic_free | 0.8106 | 0.8091 | -0.0015 | 0.9034 | 0.8992 | -0.0042 |
| hist_gbm | full | 0.8092 | 0.8119 | +0.0027 | 0.9034 | 0.8992 | -0.0042 |
| logreg | demographic_free | 0.3386 | 0.3523 | +0.0137 | 0.8824 | 0.8908 | +0.0084 |
| logreg | full | 0.3434 | 0.3553 | +0.0119 | 0.8697 | 0.8739 | +0.0042 |
| random_forest | demographic_free | 0.6862 | 0.7088 | +0.0226 | 0.8908 | 0.8908 | +0.0000 |
| random_forest | full | 0.7086 | 0.7038 | -0.0048 | 0.9034 | 0.8992 | -0.0042 |
| rule_based | demographic_free | 0.0519 | 0.0519 | +0.0000 | 0.2479 | 0.2479 | +0.0000 |
| rule_based | full | 0.0519 | 0.0519 | +0.0000 | 0.2479 | 0.2479 | +0.0000 |
