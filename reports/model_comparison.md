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
| catboost | demographic_free | 0.8554 | 0.9940 | 0.6513 | 0.8992 | 0.9244 | 0.9622 | 0.1388 |
| catboost | full | 0.8553 | 0.9930 | 0.6471 | 0.9034 | 0.9244 | 0.9622 | 0.1388 |
| hist_gbm | full | 0.8119 | 0.9938 | 0.6134 | 0.8739 | 0.8992 | 0.9664 | 0.1350 |
| hist_gbm | demographic_free | 0.8091 | 0.9918 | 0.6176 | 0.8697 | 0.8992 | 0.9622 | 0.1350 |
| random_forest | demographic_free | 0.7088 | 0.9927 | 0.5546 | 0.8361 | 0.8908 | 0.9538 | 0.1338 |
| random_forest | full | 0.7038 | 0.9917 | 0.5588 | 0.8319 | 0.8992 | 0.9454 | 0.1350 |
| logreg | full | 0.3553 | 0.9863 | 0.2983 | 0.7941 | 0.8739 | 0.9286 | 0.1312 |
| logreg | demographic_free | 0.3523 | 0.9859 | 0.2983 | 0.7983 | 0.8908 | 0.9370 | 0.1338 |
| rule_based | full | 0.0519 | 0.8194 | 0.0714 | 0.1723 | 0.2479 | 0.3950 | 0.0372 |
| rule_based | demographic_free | 0.0519 | 0.8194 | 0.0714 | 0.1723 | 0.2479 | 0.3950 | 0.0372 |

## Validation seti

| Model | Variant | PR-AUC | ROC-AUC | Recall@1% | Precision@1% |
|---|---|---:|---:|---:|---:|
| catboost | full | 0.9641 | 0.9996 | 0.9847 | 0.3746 |
| catboost | demographic_free | 0.9628 | 0.9995 | 0.9816 | 0.3734 |
| hist_gbm | full | 0.9516 | 0.9995 | 0.9755 | 0.3711 |
| hist_gbm | demographic_free | 0.9500 | 0.9993 | 0.9724 | 0.3699 |
| random_forest | demographic_free | 0.8976 | 0.9989 | 0.9663 | 0.3676 |
| random_forest | full | 0.8955 | 0.9989 | 0.9632 | 0.3664 |
| logreg | full | 0.7710 | 0.9975 | 0.9540 | 0.3629 |
| logreg | demographic_free | 0.7679 | 0.9975 | 0.9509 | 0.3617 |
| rule_based | full | 0.0502 | 0.8240 | 0.1779 | 0.0677 |
| rule_based | demographic_free | 0.0502 | 0.8240 | 0.1779 | 0.0677 |

## Demografi-Free vs Full ablation (test PR-AUC)

| Model | Full PR-AUC | Demografi-Free PR-AUC | Δ |
|---|---:|---:|---:|
| rule_based | 0.0519 | 0.0519 | +0.0000 |
| logreg | 0.3553 | 0.3523 | -0.0030 |
| random_forest | 0.7038 | 0.7088 | +0.0050 |
| hist_gbm | 0.8119 | 0.8091 | -0.0028 |
| catboost | 0.8553 | 0.8554 | +0.0001 |

## Önerilen model

**`catboost__demographic_free`** — Test PR-AUC = **0.8554**, Recall@1% = **0.9244** (Precision@1% = 0.1388).

Demografi-free varyant ile fark < 0.02 PR-AUC: **demografi-free model production'a önerilir** (etik/regülatör avantajı).

