# Threshold Analysis

> Auto-generated. İki perspektif: (a) dataset-relative percentile, (b) business-scenario alerts/day.

## Perspektif 1: Dataset-relative percentile (test set)

| Model | k | n_alerts | threshold | precision | recall | fraud_capture |
|---|---|---:|---:|---:|---:|---:|
| rule_based__full | 0.001 | 158 | 0.5869 | 0.1076 | 0.0714 | 7.14% |
| rule_based__full | 0.005 | 792 | 0.3906 | 0.0518 | 0.1723 | 17.23% |
| rule_based__full | 0.01 | 1,585 | 0.3377 | 0.0372 | 0.2479 | 24.79% |
| rule_based__full | 0.05 | 7,924 | 0.2850 | 0.0119 | 0.3950 | 39.50% |
| logreg__full | 0.001 | 158 | 0.9988 | 0.2975 | 0.1975 | 19.75% |
| logreg__full | 0.005 | 792 | 0.9391 | 0.1730 | 0.5756 | 57.56% |
| logreg__full | 0.01 | 1,585 | 0.7753 | 0.1041 | 0.6933 | 69.33% |
| logreg__full | 0.05 | 7,924 | 0.1121 | 0.0252 | 0.8403 | 84.03% |
| random_forest__full | 0.001 | 158 | 0.8733 | 0.3861 | 0.2563 | 25.63% |
| random_forest__full | 0.005 | 792 | 0.6630 | 0.1477 | 0.4916 | 49.16% |
| random_forest__full | 0.01 | 1,585 | 0.5419 | 0.0902 | 0.6008 | 60.08% |
| random_forest__full | 0.05 | 7,924 | 0.3168 | 0.0239 | 0.7941 | 79.41% |
| hist_gbm__full | 0.001 | 158 | 0.9716 | 0.4937 | 0.3277 | 32.77% |
| hist_gbm__full | 0.005 | 792 | 0.7533 | 0.1780 | 0.5924 | 59.24% |
| hist_gbm__full | 0.01 | 1,585 | 0.4958 | 0.1073 | 0.7143 | 71.43% |
| hist_gbm__full | 0.05 | 7,924 | 0.0886 | 0.0264 | 0.8782 | 87.82% |
| rule_based__demographic_free | 0.001 | 158 | 0.5869 | 0.1076 | 0.0714 | 7.14% |
| rule_based__demographic_free | 0.005 | 792 | 0.3906 | 0.0518 | 0.1723 | 17.23% |
| rule_based__demographic_free | 0.01 | 1,585 | 0.3377 | 0.0372 | 0.2479 | 24.79% |
| rule_based__demographic_free | 0.05 | 7,924 | 0.2850 | 0.0119 | 0.3950 | 39.50% |
| logreg__demographic_free | 0.001 | 158 | 0.9990 | 0.2658 | 0.1765 | 17.65% |
| logreg__demographic_free | 0.005 | 792 | 0.9457 | 0.1629 | 0.5420 | 54.20% |
| logreg__demographic_free | 0.01 | 1,585 | 0.7982 | 0.1009 | 0.6723 | 67.23% |
| logreg__demographic_free | 0.05 | 7,924 | 0.1226 | 0.0254 | 0.8445 | 84.45% |
| random_forest__demographic_free | 0.001 | 158 | 0.8688 | 0.3544 | 0.2353 | 23.53% |
| random_forest__demographic_free | 0.005 | 792 | 0.6734 | 0.1439 | 0.4790 | 47.90% |
| random_forest__demographic_free | 0.01 | 1,585 | 0.5597 | 0.0871 | 0.5798 | 57.98% |
| random_forest__demographic_free | 0.05 | 7,924 | 0.3333 | 0.0232 | 0.7731 | 77.31% |
| hist_gbm__demographic_free | 0.001 | 158 | 0.9701 | 0.4747 | 0.3151 | 31.51% |
| hist_gbm__demographic_free | 0.005 | 792 | 0.7530 | 0.1793 | 0.5966 | 59.66% |
| hist_gbm__demographic_free | 0.01 | 1,585 | 0.5078 | 0.1091 | 0.7269 | 72.69% |
| hist_gbm__demographic_free | 0.05 | 7,924 | 0.0912 | 0.0257 | 0.8571 | 85.71% |

## Perspektif 2: Business scenario (alerts/day) — test seti hacmine göre

| Model | alerts/day target | n_alerts (test) | threshold | precision | recall |
|---|---|---:|---:|---:|---:|
| rule_based__full | 50/day | 5,200 | 0.2897 | 0.0165 | 0.3613 |
| rule_based__full | 100/day | 10,400 | 0.2796 | 0.0093 | 0.4076 |
| rule_based__full | 350/day | 36,400 | 0.1905 | 0.0048 | 0.7353 |
| rule_based__full | 1000/day | 104,000 | 0.0764 | 0.0021 | 0.9370 |
| logreg__full | 50/day | 5,200 | 0.2312 | 0.0371 | 0.8109 |
| logreg__full | 100/day | 10,400 | 0.0673 | 0.0200 | 0.8739 |
| logreg__full | 350/day | 36,400 | 0.0030 | 0.0062 | 0.9538 |
| logreg__full | 1000/day | 104,000 | 0.0000 | 0.0023 | 1.0000 |
| random_forest__full | 50/day | 5,200 | 0.3766 | 0.0342 | 0.7479 |
| random_forest__full | 100/day | 10,400 | 0.2775 | 0.0192 | 0.8403 |
| random_forest__full | 350/day | 36,400 | 0.1202 | 0.0063 | 0.9622 |
| random_forest__full | 1000/day | 104,000 | 0.0356 | 0.0023 | 0.9958 |
| hist_gbm__full | 50/day | 5,200 | 0.1504 | 0.0383 | 0.8361 |
| hist_gbm__full | 100/day | 10,400 | 0.0614 | 0.0203 | 0.8866 |
| hist_gbm__full | 350/day | 36,400 | 0.0106 | 0.0063 | 0.9664 |
| hist_gbm__full | 1000/day | 104,000 | 0.0031 | 0.0023 | 0.9958 |
| rule_based__demographic_free | 50/day | 5,200 | 0.2897 | 0.0165 | 0.3613 |
| rule_based__demographic_free | 100/day | 10,400 | 0.2796 | 0.0093 | 0.4076 |
| rule_based__demographic_free | 350/day | 36,400 | 0.1905 | 0.0048 | 0.7353 |
| rule_based__demographic_free | 1000/day | 104,000 | 0.0764 | 0.0021 | 0.9370 |
| logreg__demographic_free | 50/day | 5,200 | 0.2533 | 0.0365 | 0.7983 |
| logreg__demographic_free | 100/day | 10,400 | 0.0744 | 0.0195 | 0.8529 |
| logreg__demographic_free | 350/day | 36,400 | 0.0034 | 0.0063 | 0.9580 |
| logreg__demographic_free | 1000/day | 104,000 | 0.0000 | 0.0023 | 0.9958 |
| random_forest__demographic_free | 50/day | 5,200 | 0.3953 | 0.0344 | 0.7521 |
| random_forest__demographic_free | 100/day | 10,400 | 0.2929 | 0.0184 | 0.8025 |
| random_forest__demographic_free | 350/day | 36,400 | 0.1306 | 0.0062 | 0.9538 |
| random_forest__demographic_free | 1000/day | 104,000 | 0.0390 | 0.0023 | 0.9958 |
| hist_gbm__demographic_free | 50/day | 5,200 | 0.1554 | 0.0383 | 0.8361 |
| hist_gbm__demographic_free | 100/day | 10,400 | 0.0635 | 0.0201 | 0.8782 |
| hist_gbm__demographic_free | 350/day | 36,400 | 0.0108 | 0.0062 | 0.9538 |
| hist_gbm__demographic_free | 1000/day | 104,000 | 0.0031 | 0.0023 | 0.9916 |

## 3-Bant politika önerisi
- **HIGH**: top %0.1 (otomatik blok / step-up auth).
- **MEDIUM**: top %0.1 - %1 arası (manuel inceleme kuyruğu).
- **LOW**: geri kalan (sadece logla).

Production trafik hacmi netleşince iş tarafıyla birlikte percentile vs alerts/day perspektifinden son threshold seçilir.

