# Run Comparison — Baseline vs Improved

| Model | Variant | PR-AUC (base) | PR-AUC (imp) | Δ PR-AUC | Recall@1% (base) | Recall@1% (imp) | Δ Recall@1% |
|---|---|---:|---:|---:|---:|---:|---:|
| hist_gbm | demographic_free | 0.8046 | 0.8106 | +0.0059 | 0.8992 | 0.9034 | +0.0042 |
| hist_gbm | full | 0.8052 | 0.8092 | +0.0039 | 0.9034 | 0.9034 | +0.0000 |
| logreg | demographic_free | 0.3486 | 0.3386 | -0.0101 | 0.8782 | 0.8824 | +0.0042 |
| logreg | full | 0.3503 | 0.3434 | -0.0068 | 0.8824 | 0.8697 | -0.0126 |
| random_forest | demographic_free | 0.6920 | 0.6862 | -0.0058 | 0.8908 | 0.8908 | +0.0000 |
| random_forest | full | 0.7261 | 0.7086 | -0.0175 | 0.8950 | 0.9034 | +0.0084 |
| rule_based | demographic_free | 0.0519 | 0.0519 | +0.0000 | 0.2479 | 0.2479 | +0.0000 |
| rule_based | full | 0.0519 | 0.0519 | +0.0000 | 0.2479 | 0.2479 | +0.0000 |
