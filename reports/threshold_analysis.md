# Threshold Analysis

> Auto-generated. İki perspektif: (a) dataset-relative percentile, (b) business-scenario alerts/day.

## Perspektif 1: Dataset-relative percentile (test set)

| Model | k | n_alerts | threshold | precision | recall | fraud_capture |
|---|---|---:|---:|---:|---:|---:|
| rule_based__full | 0.001 | 158 | 0.5869 | 0.1076 | 0.0714 | 7.14% |
| rule_based__full | 0.005 | 792 | 0.3906 | 0.0518 | 0.1723 | 17.23% |
| rule_based__full | 0.01 | 1,585 | 0.3377 | 0.0372 | 0.2479 | 24.79% |
| rule_based__full | 0.05 | 7,924 | 0.2850 | 0.0119 | 0.3950 | 39.50% |
| logreg__full | 0.001 | 158 | 1.0000 | 0.4367 | 0.2899 | 28.99% |
| logreg__full | 0.005 | 792 | 0.9085 | 0.2386 | 0.7941 | 79.41% |
| logreg__full | 0.01 | 1,585 | 0.3741 | 0.1325 | 0.8824 | 88.24% |
| logreg__full | 0.05 | 7,924 | 0.0224 | 0.0284 | 0.9454 | 94.54% |
| random_forest__full | 0.001 | 158 | 0.7786 | 0.8608 | 0.5714 | 57.14% |
| random_forest__full | 0.005 | 792 | 0.4770 | 0.2538 | 0.8445 | 84.45% |
| random_forest__full | 0.01 | 1,585 | 0.3826 | 0.1344 | 0.8950 | 89.50% |
| random_forest__full | 0.05 | 7,924 | 0.1632 | 0.0286 | 0.9538 | 95.38% |
| hist_gbm__full | 0.001 | 158 | 0.9940 | 0.8924 | 0.5924 | 59.24% |
| hist_gbm__full | 0.005 | 792 | 0.1949 | 0.2601 | 0.8655 | 86.55% |
| hist_gbm__full | 0.01 | 1,585 | 0.0427 | 0.1375 | 0.9160 | 91.60% |
| hist_gbm__full | 0.05 | 7,924 | 0.0043 | 0.0284 | 0.9454 | 94.54% |
| rule_based__demographic_free | 0.001 | 158 | 0.5869 | 0.1076 | 0.0714 | 7.14% |
| rule_based__demographic_free | 0.005 | 792 | 0.3906 | 0.0518 | 0.1723 | 17.23% |
| rule_based__demographic_free | 0.01 | 1,585 | 0.3377 | 0.0372 | 0.2479 | 24.79% |
| rule_based__demographic_free | 0.05 | 7,924 | 0.2850 | 0.0119 | 0.3950 | 39.50% |
| logreg__demographic_free | 0.001 | 158 | 1.0000 | 0.4367 | 0.2899 | 28.99% |
| logreg__demographic_free | 0.005 | 792 | 0.9090 | 0.2399 | 0.7983 | 79.83% |
| logreg__demographic_free | 0.01 | 1,585 | 0.3739 | 0.1319 | 0.8782 | 87.82% |
| logreg__demographic_free | 0.05 | 7,924 | 0.0239 | 0.0281 | 0.9370 | 93.70% |
| random_forest__demographic_free | 0.001 | 158 | 0.7969 | 0.8101 | 0.5378 | 53.78% |
| random_forest__demographic_free | 0.005 | 792 | 0.4891 | 0.2449 | 0.8151 | 81.51% |
| random_forest__demographic_free | 0.01 | 1,585 | 0.3876 | 0.1338 | 0.8908 | 89.08% |
| random_forest__demographic_free | 0.05 | 7,924 | 0.1820 | 0.0286 | 0.9538 | 95.38% |
| hist_gbm__demographic_free | 0.001 | 158 | 0.9948 | 0.8797 | 0.5840 | 58.40% |
| hist_gbm__demographic_free | 0.005 | 792 | 0.2262 | 0.2588 | 0.8613 | 86.13% |
| hist_gbm__demographic_free | 0.01 | 1,585 | 0.0513 | 0.1363 | 0.9076 | 90.76% |
| hist_gbm__demographic_free | 0.05 | 7,924 | 0.0051 | 0.0285 | 0.9496 | 94.96% |

## Perspektif 2: Business scenario (alerts/day) — test seti hacmine göre

| Model | alerts/day target | n_alerts (test) | threshold | precision | recall |
|---|---|---:|---:|---:|---:|
| rule_based__full | 50/day | 5,200 | 0.2897 | 0.0165 | 0.3613 |
| rule_based__full | 100/day | 10,400 | 0.2796 | 0.0093 | 0.4076 |
| rule_based__full | 350/day | 36,400 | 0.1905 | 0.0048 | 0.7353 |
| rule_based__full | 1000/day | 104,000 | 0.0764 | 0.0021 | 0.9370 |
| logreg__full | 50/day | 5,200 | 0.0475 | 0.0429 | 0.9370 |
| logreg__full | 100/day | 10,400 | 0.0132 | 0.0217 | 0.9496 |
| logreg__full | 350/day | 36,400 | 0.0007 | 0.0065 | 0.9874 |
| logreg__full | 1000/day | 104,000 | 0.0000 | 0.0023 | 1.0000 |
| random_forest__full | 50/day | 5,200 | 0.2140 | 0.0427 | 0.9328 |
| random_forest__full | 100/day | 10,400 | 0.1352 | 0.0219 | 0.9580 |
| random_forest__full | 350/day | 36,400 | 0.0492 | 0.0065 | 0.9916 |
| random_forest__full | 1000/day | 104,000 | 0.0125 | 0.0023 | 1.0000 |
| hist_gbm__full | 50/day | 5,200 | 0.0060 | 0.0431 | 0.9412 |
| hist_gbm__full | 100/day | 10,400 | 0.0037 | 0.0218 | 0.9538 |
| hist_gbm__full | 350/day | 36,400 | 0.0031 | 0.0063 | 0.9706 |
| hist_gbm__full | 1000/day | 104,000 | 0.0031 | 0.0023 | 0.9874 |
| rule_based__demographic_free | 50/day | 5,200 | 0.2897 | 0.0165 | 0.3613 |
| rule_based__demographic_free | 100/day | 10,400 | 0.2796 | 0.0093 | 0.4076 |
| rule_based__demographic_free | 350/day | 36,400 | 0.1905 | 0.0048 | 0.7353 |
| rule_based__demographic_free | 1000/day | 104,000 | 0.0764 | 0.0021 | 0.9370 |
| logreg__demographic_free | 50/day | 5,200 | 0.0500 | 0.0425 | 0.9286 |
| logreg__demographic_free | 100/day | 10,400 | 0.0143 | 0.0216 | 0.9454 |
| logreg__demographic_free | 350/day | 36,400 | 0.0008 | 0.0065 | 0.9916 |
| logreg__demographic_free | 1000/day | 104,000 | 0.0000 | 0.0023 | 1.0000 |
| random_forest__demographic_free | 50/day | 5,200 | 0.2304 | 0.0427 | 0.9328 |
| random_forest__demographic_free | 100/day | 10,400 | 0.1550 | 0.0222 | 0.9706 |
| random_forest__demographic_free | 350/day | 36,400 | 0.0615 | 0.0065 | 0.9958 |
| random_forest__demographic_free | 1000/day | 104,000 | 0.0160 | 0.0023 | 1.0000 |
| hist_gbm__demographic_free | 50/day | 5,200 | 0.0074 | 0.0429 | 0.9370 |
| hist_gbm__demographic_free | 100/day | 10,400 | 0.0044 | 0.0218 | 0.9538 |
| hist_gbm__demographic_free | 350/day | 36,400 | 0.0038 | 0.0064 | 0.9748 |
| hist_gbm__demographic_free | 1000/day | 104,000 | 0.0037 | 0.0022 | 0.9832 |

## 3-Bant politika önerisi
- **HIGH**: top %0.1 (otomatik blok / step-up auth).
- **MEDIUM**: top %0.1 - %1 arası (manuel inceleme kuyruğu).
- **LOW**: geri kalan (sadece logla).

Production trafik hacmi netleşince iş tarafıyla birlikte percentile vs alerts/day perspektifinden son threshold seçilir.

