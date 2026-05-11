# Run Comparison — Baseline vs Improved

| Model | Variant | PR-AUC (base) | PR-AUC (imp) | Δ PR-AUC | Recall@1% (base) | Recall@1% (imp) | Δ Recall@1% |
|---|---|---:|---:|---:|---:|---:|---:|
| hist_gbm | demographic_free | 0.3546 | 0.7673 | +0.4128 | 0.7269 | 0.9076 | +0.1807 |
| hist_gbm | full | 0.3575 | 0.7718 | +0.4144 | 0.7143 | 0.9160 | +0.2017 |
| logreg | demographic_free | 0.1626 | 0.3486 | +0.1860 | 0.6723 | 0.8782 | +0.2059 |
| logreg | full | 0.1780 | 0.3503 | +0.1723 | 0.6933 | 0.8824 | +0.1891 |
| random_forest | demographic_free | 0.2368 | 0.6920 | +0.4552 | 0.5798 | 0.8908 | +0.3109 |
| random_forest | full | 0.2468 | 0.7261 | +0.4793 | 0.6008 | 0.8950 | +0.2941 |
| rule_based | demographic_free | 0.0519 | 0.0519 | +0.0000 | 0.2479 | 0.2479 | +0.0000 |
| rule_based | full | 0.0519 | 0.0519 | +0.0000 | 0.2479 | 0.2479 | +0.0000 |
