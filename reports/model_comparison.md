# Model Comparison

> Auto-generated from `artifacts/eval/all_models.json`.

## Split (time-based)
```
{
  "train_end": "2024-03-31 23:59:59",
  "val": [
    "2024-04-01 00:00:00",
    "2024-05-31 23:59:59"
  ],
  "test": [
    "2024-06-01 00:00:00",
    "2024-09-30 23:59:59"
  ]
}
```

## Train ↔ Test entity overlap

| Entity | unique (train) | unique (test) | overlap | % of test seen in train |
|---|---:|---:|---:|---:|
| AccountNumber | 568,914 | 151,516 | 7,882 | 5.2% |
| DeviceId | 12,667 | 11,345 | 10,557 | 93.05% |
| ReceiverName | 109,645 | 74,938 | 54,879 | 73.23% |
| IP_Subnet | 26,610 | 22,347 | 20,486 | 91.67% |

## Test seti — tüm modeller × tüm metrikler

| Model | Variant | PR-AUC | ROC-AUC | Recall@0.1% | Recall@0.5% | Recall@1% | Recall@5% | Precision@1% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hist_gbm | full | 0.3575 | 0.9700 | 0.3277 | 0.5924 | 0.7143 | 0.8782 | 0.1073 |
| hist_gbm | demographic_free | 0.3546 | 0.9666 | 0.3151 | 0.5966 | 0.7269 | 0.8571 | 0.1091 |
| random_forest | full | 0.2468 | 0.9596 | 0.2563 | 0.4916 | 0.6008 | 0.7941 | 0.0902 |
| random_forest | demographic_free | 0.2368 | 0.9570 | 0.2353 | 0.4790 | 0.5798 | 0.7731 | 0.0871 |
| logreg | full | 0.1780 | 0.9666 | 0.1975 | 0.5756 | 0.6933 | 0.8403 | 0.1041 |
| logreg | demographic_free | 0.1626 | 0.9660 | 0.1765 | 0.5420 | 0.6723 | 0.8445 | 0.1009 |
| rule_based | full | 0.0519 | 0.8194 | 0.0714 | 0.1723 | 0.2479 | 0.3950 | 0.0372 |
| rule_based | demographic_free | 0.0519 | 0.8194 | 0.0714 | 0.1723 | 0.2479 | 0.3950 | 0.0372 |

## Validation seti

| Model | Variant | PR-AUC | ROC-AUC | Recall@1% | Precision@1% |
|---|---|---:|---:|---:|---:|
| hist_gbm | demographic_free | 0.5750 | 0.9798 | 0.7362 | 0.2800 |
| hist_gbm | full | 0.5729 | 0.9800 | 0.7239 | 0.2754 |
| random_forest | full | 0.4845 | 0.9742 | 0.6687 | 0.2544 |
| random_forest | demographic_free | 0.4801 | 0.9732 | 0.6626 | 0.2520 |
| logreg | full | 0.3847 | 0.9826 | 0.7055 | 0.2684 |
| logreg | demographic_free | 0.3565 | 0.9817 | 0.6810 | 0.2590 |
| rule_based | full | 0.0502 | 0.8240 | 0.1779 | 0.0677 |
| rule_based | demographic_free | 0.0502 | 0.8240 | 0.1779 | 0.0677 |

## Demografi-Free vs Full ablation (test PR-AUC)

| Model | Full PR-AUC | Demografi-Free PR-AUC | Δ |
|---|---:|---:|---:|
| rule_based | 0.0519 | 0.0519 | +0.0000 |
| logreg | 0.1780 | 0.1626 | -0.0154 |
| random_forest | 0.2468 | 0.2368 | -0.0100 |
| hist_gbm | 0.3575 | 0.3546 | -0.0029 |

## Önerilen model

**`hist_gbm__full`** — Test PR-AUC = **0.3575**, Recall@1% = **0.7143** (Precision@1% = 0.1073).

Demografi-free varyant ile fark < 0.02 PR-AUC: **demografi-free model production'a önerilir** (etik/regülatör avantajı).

