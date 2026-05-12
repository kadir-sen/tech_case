# Threshold Analysis

> Auto-generated. İki perspektif: (a) dataset-relative percentile, (b) business-scenario alerts/day.

## Perspektif 1: Dataset-relative percentile (test set)

| Model | k | n_alerts | threshold | precision | recall | fraud_capture |
|---|---|---:|---:|---:|---:|---:|
| rule_based__full | 0.001 | 158 | 0.5869 | 0.1076 | 0.0714 | 7.14% |
| rule_based__full | 0.005 | 792 | 0.3906 | 0.0518 | 0.1723 | 17.23% |
| rule_based__full | 0.01 | 1,585 | 0.3377 | 0.0372 | 0.2479 | 24.79% |
| rule_based__full | 0.05 | 7,924 | 0.2850 | 0.0119 | 0.3950 | 39.50% |
| logreg__full | 0.001 | 158 | 1.0000 | 0.4494 | 0.2983 | 29.83% |
| logreg__full | 0.005 | 792 | 0.9028 | 0.2386 | 0.7941 | 79.41% |
| logreg__full | 0.01 | 1,585 | 0.4312 | 0.1312 | 0.8739 | 87.39% |
| logreg__full | 0.05 | 7,924 | 0.0291 | 0.0279 | 0.9286 | 92.86% |
| random_forest__full | 0.001 | 158 | 0.7593 | 0.8418 | 0.5588 | 55.88% |
| random_forest__full | 0.005 | 792 | 0.4515 | 0.2500 | 0.8319 | 83.19% |
| random_forest__full | 0.01 | 1,585 | 0.3592 | 0.1350 | 0.8992 | 89.92% |
| random_forest__full | 0.05 | 7,924 | 0.1625 | 0.0284 | 0.9454 | 94.54% |
| hist_gbm__full | 0.001 | 158 | 0.9869 | 0.9241 | 0.6134 | 61.34% |
| hist_gbm__full | 0.005 | 792 | 0.3768 | 0.2626 | 0.8739 | 87.39% |
| hist_gbm__full | 0.01 | 1,585 | 0.1049 | 0.1350 | 0.8992 | 89.92% |
| hist_gbm__full | 0.05 | 7,924 | 0.0056 | 0.0290 | 0.9664 | 96.64% |
| catboost__full | 0.001 | 158 | 0.9879 | 0.9747 | 0.6471 | 64.71% |
| catboost__full | 0.005 | 792 | 0.1206 | 0.2715 | 0.9034 | 90.34% |
| catboost__full | 0.01 | 1,585 | 0.0309 | 0.1388 | 0.9244 | 92.44% |
| catboost__full | 0.05 | 7,924 | 0.0018 | 0.0289 | 0.9622 | 96.22% |
| rule_based__demographic_free | 0.001 | 158 | 0.5869 | 0.1076 | 0.0714 | 7.14% |
| rule_based__demographic_free | 0.005 | 792 | 0.3906 | 0.0518 | 0.1723 | 17.23% |
| rule_based__demographic_free | 0.01 | 1,585 | 0.3377 | 0.0372 | 0.2479 | 24.79% |
| rule_based__demographic_free | 0.05 | 7,924 | 0.2850 | 0.0119 | 0.3950 | 39.50% |
| logreg__demographic_free | 0.001 | 158 | 1.0000 | 0.4494 | 0.2983 | 29.83% |
| logreg__demographic_free | 0.005 | 792 | 0.9023 | 0.2399 | 0.7983 | 79.83% |
| logreg__demographic_free | 0.01 | 1,585 | 0.4242 | 0.1338 | 0.8908 | 89.08% |
| logreg__demographic_free | 0.05 | 7,924 | 0.0297 | 0.0281 | 0.9370 | 93.70% |
| random_forest__demographic_free | 0.001 | 158 | 0.7716 | 0.8354 | 0.5546 | 55.46% |
| random_forest__demographic_free | 0.005 | 792 | 0.4604 | 0.2513 | 0.8361 | 83.61% |
| random_forest__demographic_free | 0.01 | 1,585 | 0.3688 | 0.1338 | 0.8908 | 89.08% |
| random_forest__demographic_free | 0.05 | 7,924 | 0.1557 | 0.0286 | 0.9538 | 95.38% |
| hist_gbm__demographic_free | 0.001 | 158 | 0.9890 | 0.9304 | 0.6176 | 61.76% |
| hist_gbm__demographic_free | 0.005 | 792 | 0.3929 | 0.2614 | 0.8697 | 86.97% |
| hist_gbm__demographic_free | 0.01 | 1,585 | 0.1257 | 0.1350 | 0.8992 | 89.92% |
| hist_gbm__demographic_free | 0.05 | 7,924 | 0.0057 | 0.0289 | 0.9622 | 96.22% |
| catboost__demographic_free | 0.001 | 158 | 0.9871 | 0.9810 | 0.6513 | 65.13% |
| catboost__demographic_free | 0.005 | 792 | 0.1050 | 0.2702 | 0.8992 | 89.92% |
| catboost__demographic_free | 0.01 | 1,585 | 0.0267 | 0.1388 | 0.9244 | 92.44% |
| catboost__demographic_free | 0.05 | 7,924 | 0.0017 | 0.0289 | 0.9622 | 96.22% |

## Perspektif 2: Business scenario (alerts/day) — test seti hacmine göre

| Model | alerts/day target | n_alerts (test) | threshold | precision | recall |
|---|---|---:|---:|---:|---:|
| rule_based__full | 50/day | 5,200 | 0.2897 | 0.0165 | 0.3613 |
| rule_based__full | 100/day | 10,400 | 0.2796 | 0.0093 | 0.4076 |
| rule_based__full | 350/day | 36,400 | 0.1905 | 0.0048 | 0.7353 |
| rule_based__full | 1000/day | 104,000 | 0.0764 | 0.0021 | 0.9370 |
| logreg__full | 50/day | 5,200 | 0.0623 | 0.0423 | 0.9244 |
| logreg__full | 100/day | 10,400 | 0.0170 | 0.0218 | 0.9538 |
| logreg__full | 350/day | 36,400 | 0.0008 | 0.0064 | 0.9790 |
| logreg__full | 1000/day | 104,000 | 0.0000 | 0.0023 | 1.0000 |
| random_forest__full | 50/day | 5,200 | 0.2090 | 0.0425 | 0.9286 |
| random_forest__full | 100/day | 10,400 | 0.1374 | 0.0221 | 0.9664 |
| random_forest__full | 350/day | 36,400 | 0.0508 | 0.0065 | 0.9958 |
| random_forest__full | 1000/day | 104,000 | 0.0126 | 0.0023 | 1.0000 |
| hist_gbm__full | 50/day | 5,200 | 0.0093 | 0.0437 | 0.9538 |
| hist_gbm__full | 100/day | 10,400 | 0.0043 | 0.0224 | 0.9790 |
| hist_gbm__full | 350/day | 36,400 | 0.0018 | 0.0065 | 1.0000 |
| hist_gbm__full | 1000/day | 104,000 | 0.0012 | 0.0023 | 1.0000 |
| catboost__full | 50/day | 5,200 | 0.0036 | 0.0433 | 0.9454 |
| catboost__full | 100/day | 10,400 | 0.0011 | 0.0222 | 0.9706 |
| catboost__full | 350/day | 36,400 | 0.0001 | 0.0065 | 0.9916 |
| catboost__full | 1000/day | 104,000 | 0.0000 | 0.0023 | 1.0000 |
| rule_based__demographic_free | 50/day | 5,200 | 0.2897 | 0.0165 | 0.3613 |
| rule_based__demographic_free | 100/day | 10,400 | 0.2796 | 0.0093 | 0.4076 |
| rule_based__demographic_free | 350/day | 36,400 | 0.1905 | 0.0048 | 0.7353 |
| rule_based__demographic_free | 1000/day | 104,000 | 0.0764 | 0.0021 | 0.9370 |
| logreg__demographic_free | 50/day | 5,200 | 0.0627 | 0.0421 | 0.9202 |
| logreg__demographic_free | 100/day | 10,400 | 0.0180 | 0.0218 | 0.9538 |
| logreg__demographic_free | 350/day | 36,400 | 0.0009 | 0.0064 | 0.9832 |
| logreg__demographic_free | 1000/day | 104,000 | 0.0000 | 0.0023 | 1.0000 |
| random_forest__demographic_free | 50/day | 5,200 | 0.2066 | 0.0425 | 0.9286 |
| random_forest__demographic_free | 100/day | 10,400 | 0.1297 | 0.0223 | 0.9748 |
| random_forest__demographic_free | 350/day | 36,400 | 0.0463 | 0.0065 | 0.9958 |
| random_forest__demographic_free | 1000/day | 104,000 | 0.0118 | 0.0023 | 1.0000 |
| hist_gbm__demographic_free | 50/day | 5,200 | 0.0098 | 0.0435 | 0.9496 |
| hist_gbm__demographic_free | 100/day | 10,400 | 0.0044 | 0.0223 | 0.9748 |
| hist_gbm__demographic_free | 350/day | 36,400 | 0.0017 | 0.0065 | 0.9916 |
| hist_gbm__demographic_free | 1000/day | 104,000 | 0.0012 | 0.0023 | 1.0000 |
| catboost__demographic_free | 50/day | 5,200 | 0.0034 | 0.0437 | 0.9538 |
| catboost__demographic_free | 100/day | 10,400 | 0.0011 | 0.0223 | 0.9748 |
| catboost__demographic_free | 350/day | 36,400 | 0.0001 | 0.0065 | 0.9958 |
| catboost__demographic_free | 1000/day | 104,000 | 0.0000 | 0.0023 | 1.0000 |

## 3-Bant politika önerisi
- **HIGH**: top %0.1 (otomatik blok / step-up auth).
- **MEDIUM**: top %0.1 - %1 arası (manuel inceleme kuyruğu).
- **LOW**: geri kalan (sadece logla).

Production trafik hacmi netleşince iş tarafıyla birlikte percentile vs alerts/day perspektifinden son threshold seçilir.

