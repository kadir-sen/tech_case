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
| hist_gbm | full | 0.7718 | 0.9813 | 0.5924 | 0.8655 | 0.9160 | 0.9454 | 0.1375 |
| hist_gbm | demographic_free | 0.7673 | 0.9789 | 0.5840 | 0.8613 | 0.9076 | 0.9496 | 0.1363 |
| random_forest | full | 0.7261 | 0.9919 | 0.5714 | 0.8445 | 0.8950 | 0.9538 | 0.1344 |
| random_forest | demographic_free | 0.6920 | 0.9921 | 0.5378 | 0.8151 | 0.8908 | 0.9538 | 0.1338 |
| logreg | full | 0.3503 | 0.9878 | 0.2899 | 0.7941 | 0.8824 | 0.9454 | 0.1325 |
| logreg | demographic_free | 0.3486 | 0.9877 | 0.2899 | 0.7983 | 0.8782 | 0.9370 | 0.1319 |
| rule_based | full | 0.0519 | 0.8194 | 0.0714 | 0.1723 | 0.2479 | 0.3950 | 0.0372 |
| rule_based | demographic_free | 0.0519 | 0.8194 | 0.0714 | 0.1723 | 0.2479 | 0.3950 | 0.0372 |

## Validation seti

| Model | Variant | PR-AUC | ROC-AUC | Recall@1% | Precision@1% |
|---|---|---:|---:|---:|---:|
| hist_gbm | full | 0.9330 | 0.9967 | 0.9724 | 0.3699 |
| hist_gbm | demographic_free | 0.9275 | 0.9954 | 0.9693 | 0.3687 |
| random_forest | full | 0.9148 | 0.9987 | 0.9755 | 0.3711 |
| random_forest | demographic_free | 0.8925 | 0.9985 | 0.9724 | 0.3699 |
| logreg | full | 0.7692 | 0.9977 | 0.9540 | 0.3629 |
| logreg | demographic_free | 0.7666 | 0.9976 | 0.9540 | 0.3629 |
| rule_based | full | 0.0502 | 0.8240 | 0.1779 | 0.0677 |
| rule_based | demographic_free | 0.0502 | 0.8240 | 0.1779 | 0.0677 |

## Demografi-Free vs Full ablation (test PR-AUC)

| Model | Full PR-AUC | Demografi-Free PR-AUC | Δ |
|---|---:|---:|---:|
| rule_based | 0.0519 | 0.0519 | +0.0000 |
| logreg | 0.3503 | 0.3486 | -0.0016 |
| random_forest | 0.7261 | 0.6920 | -0.0341 |
| hist_gbm | 0.7718 | 0.7673 | -0.0045 |

## Önerilen model

**`hist_gbm__full`** — Test PR-AUC = **0.7718**, Recall@1% = **0.9160** (Precision@1% = 0.1375).

Demografi-free varyant ile fark < 0.02 PR-AUC: **demografi-free model production'a önerilir** (etik/regülatör avantajı).

