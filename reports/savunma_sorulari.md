# Sunum Savunma Soruları — Hızlı Referans

> Her soru altında: kısa cevap + sayısal/teknik gerekçe. Uzatma; ezberle.

---

## A. Veri & Split

**S1. Neden time-based split? Random ile kıyasladın mı?**
Fraud verisinde zaman drift'i var (çeyreklik fraud oranı 0.04% → 1.26% → 0.13%). Random split, ileriden geçmişe sızıntı yaratır ve model performansını şişirir. Time-based gerçek üretim koşulunu simüle ediyor.

**S2. Train fraud rate %0.79, test %0.15 — bu sapma sorun değil mi?**
Sorun değil — bilinçli. Veri setinde fraud oranı zamanla düşmüş (concept drift). Bunu test setinde de görmek üretim koşuluna sadık kalmamızı sağlıyor. Bu yüzden tek metrik olarak accuracy değil PR-AUC + Recall@k kullanıyoruz.

**S3. Test PR-AUC 0.84, Val 0.96 — büyük gap. Overfitting mi?**
Hayır, drift'in beklenen etkisi. Modeli train üzerinde fit, val üzerinde seçim, test sadece raporlama için kullanıldı. Adversarial AUC train↔val = 0.989 — yani train ve val dağılımı bile birbirinden ayırt edilebilir durumda. Gap drift kaynaklı, overfit değil. ROC-AUC 0.999 → 0.992 sadece 0.007 düştü.

**S4. Validasyon stratejisi neden tek-pencere? Time-series CV (expanding/walk-forward) kullanmadın mı?**
Kod altyapısı mevcut (`src/models/time_validation.py` — `make_expanding_time_folds`, `make_walk_forward_folds`). Tek pencere final raporlama için; HPO ve feature selection bu pencere üzerinde yapıldı. CV ekstra robustness verirdi ama tek günlük case'te tek-pencere yeterli sinyal verdi.

---

## B. Feature Engineering

**S5. Neden bu kadar fazla historical aggregate? Overengineering değil mi?**
Tek-tx feature'ları (`TransactionAmount`, `IsFractionalAmount` vb.) zayıf sinyal veriyor (rule-based baseline PR-AUC ~0.30). Historical aggregate'ler fraud'un asıl sinyal kaynağı: "bu cihaz daha önce kaç farklı hesabı kullandı", "bu alıcının geçmiş fraud oranı ne". Permutation importance top-2 (device/receiver fraud_rate_smoothed) toplam katkının ~%50'sini taşıyor.

**S6. `device_fraud_rate_smoothed` — target leakage değil mi?**
Hayır, **lag uygulanmış**: 7 günlük label availability lag + Bayesian smoothing (prior=50). Yani t anındaki transaction için, `t-7gün`'den ÖNCEKİ fraud etiketlerinden hesaplanıyor. Üretimde de fraud labelları gecikmeli gelir; bu lag o gerçekliği modelliyor. `_smoothed_fraud_rate_with_lag` fonksiyonu `searchsorted(t - lag, "left")` ile bunu garanti ediyor.

**S7. PII isimleri (ReceiverName, SenderName) neden aggregate üretmek için kullanılıyor? KVKK ihlali değil mi?**
İsimler **modele feature olarak girmiyor**, sadece grouping key. Çıktı bir sayı (`receiver_tx_count_30d`). PII üretim sisteminde hash'lenebilir; pipeline değişmiyor. Modelin kendisi isim görmüyor.

**S8. Demografi (yaş, cinsiyet, eğitim, medeni hal) neden çıkardın?**
Üç gerekçe: (1) Yasal/etik risk — protected attributes üzerinde karar. (2) Permutation importance düşük (`CustomerAge` ve `Gender` top-15'te değil). (3) Demografi-free vs full ablation'da PR-AUC farkı negligible. Risk artıyor, kazanç yok — drop.

**S9. `hour`, `dow`, `is_weekend` neden modelde yok? EDA'da bunlara baktın mı?**
Baktım, EDA'da fraud oranı saat/gün-haftası kırılımında %0.55-%0.85 arasında düz. Ablation testinde Δ PR-AUC ≈ 0. Sinyal taşımıyor, bu yüzden drop. `is_holiday` ise tutuldu çünkü Resmi Tatil fraud %0.86 vs normal gün %0.63 — küçük ama tutarlı sinyal.

**S10. `TransactionChannel` neden modelde yok?**
Tek değer ("Mobile") — varyans yok, model için bilgi taşımıyor. `configs/data.yaml` içinde `drop_columns` ile çıkarılıyor.

**S11. AccountNumber neden kullanılmıyor? Account-bazlı history üretmedin mi?**
AccountNumber 849,564 satırda 789,861 unique — neredeyse her satır farklı hesap. Account-bazlı historical aggregate üretmek için yeterli tekrar yok (medyan tx/account = 1). Bu yüzden DeviceId (med=21) ve ReceiverName (med=2) üzerinden aggregate üretildi.

**S12. DeviceModel 1,004 unique — direkt one-hot çok büyük olmaz mı?**
Evet. Top-100 model + `_OTHER_` bucket'a indirgendi. Ayrıca DeviceParentBrand (iPhone/Samsung/Xiaomi/...) ile granular bir seviye daha eklendi. HGB native categorical 255 sınırı için de gerekli.

---

## C. Model Seçimi

**S13. Neden 4 farklı model eğittin? Tek model yetmez miydi?**
Karşılaştırma + risk yönetimi. Rule-based = operasyonel baseline. LR = lineer benchmark. RF/HGB/CatBoost = ağaç-bazlı aileler. Her birinin farklı bias-variance trade-off'u var. Sonuçta val PR-AUC: LR 0.79, RF 0.94, HGB 0.95, CatBoost 0.96 — kazanan CatBoost.

**S14. Neden CatBoost? XGBoost veya LightGBM denedin mi?**
HGB (sklearn'ün histogram-based GBM) kullandık — LightGBM'le aynı paradigma. CatBoost'un avantajı: native categorical handling + ordered target encoding. Bizim manuel olarak ürettiğimiz `receiver_fraud_rate_smoothed` mantığının yerleşik versiyonu. Test sonucu doğruladı: val PR-AUC 0.96 CatBoost vs 0.95 HGB.

**S15. Neden Neural Net denemedin?**
Tabular + 850K satır + ağır kategorik veri için gradient boosting standardı yener (Kaggle / IEEE-CIS deneyimi). NN ekstra complexity getirir, marginal kazanç olur. Bir günlük case için ROI yok.

**S16. CatBoost neden tuned değil? HGB ve LR tuned ama CatBoost baseline.**
CatBoost'un baseline params'ı zaten val PR-AUC 0.96 verdi. Optuna HPO ekstra zaman maliyetiyle gelirdi ve baseline zaten en yüksekti. Riski-getirisi olumsuzdu, vakit kısıtlıydı.

**S17. `class_weight="balanced"` yerine SMOTE / undersampling denedin mi?**
Hayır. SMOTE tabular'da synthetic gürültü üretir; time-series'te zaman ekseni bozulur. Class weight matematiksel olarak loss'u rescale eder, veri yapısını bozmaz. CatBoost'un `auto_class_weights="Balanced"`'ı aynı şeyi yapar.

---

## D. Feature Selection & 30 Feature

**S18. Neden 30 feature? 50 feature'lı `full_safe` daha iyi test PR-AUC (0.854) vermedi mi?**
Test PR-AUC 0.854 vs 0.842 — çok küçük fark. Final seçim val PR-AUC üzerinden yapıldı; orada 30 feature 0.962 vs 50 feature 0.961 → 30 daha iyi. Test sadece raporlama için, seçim kriteri değil. Ayrıca 30 feature daha az drift surface, daha az kompleksite.

**S19. Permutation importance vs SHAP — neden ikisini birden kullandın?**
Permutation = "bu feature'ı shuffle edersem ne kadar kaybederim" — direkt model-agnostic ölçüm. SHAP = lokal-toplam katkı dekompozisyonu. İkisi farklı sinyal: permutation feature selection için, SHAP global + lokal açıklanabilirlik için (API reason codes).

**S20. Adversarial AUC train↔val = 0.989 — bu çok yüksek değil mi?**
Çok yüksek, drift'in açık göstergesi. Bu yüzden `drift_robust` feature set'i de hazırlandı (drift importance < 0.001 olan 41 feature). Üretimde drift artarsa drift_robust sete geçilebilir. Şu an seçilen `selected_by_permutation` maksimum performans için.

---

## E. Threshold & Calibration

**S21. Eşik neden 0.5 değil?**
Imbalance %0.6'da 0.5 eşiği fraud'u kaçırır. Bunun yerine **percentile-based** eşik kullanılıyor: top 0.1% = HIGH (block), top 1% = MEDIUM (manual review). HIGH cut-off = 0.99999 (val'den), MEDIUM cut-off = 0.0449. İş kapasitesine göre `alerts/day` perspektifi de raporlanıyor.

**S22. HIGH 0.99999 — çok agresif değil mi?**
Bilinçli. HIGH = "block/step-up authentication" → otomatik blok. Yanlış pozitif maliyet yüksek. Bu yüzden çok yüksek confidence istendi. Test'te top-0.1%'te Precision = 0.97 (158 alert'in 154'ü fraud).

**S23. Brier score 0.00096 — bu iyi mi kötü mü?**
Bağlama göre iyi. Base rate %0.15 olduğu için "hep 0 söyle" stratejisi bile düşük Brier verir. Önemli olan: Platt calibration sonrası Brier 0.00056'ya düşüyor, PR-AUC değişmiyor. Reliability tablosu 0.9-1.0 bin'inde 87.9% gerçek pozitif → model olasılığı güvenilir.

**S24. Isotonic calibration daha iyi Brier veriyor (0.00045). Neden Platt?**
Isotonic daha düşük Brier ama test PR-AUC'u -0.02 düşürüyor (0.842 → 0.825). Ranking gücü kaybı kabul edilemez. Platt hem Brier'ı düşürüyor hem ranking'i koruyor.

---

## F. API & Production

**S25. `historical_features` neden client'tan geliyor? API içinde hesaplamak gerekmez mi?**
Üretimde feature store / online feature service'ten anlık sorgulanır. Bu case'te o altyapı yok, bu yüzden client gönderiyor; eksikse 0-fill ile fallback. Pipeline'da feature_list.json üzerinden hangi kolonların gerektiği zaten dokümante.

**S26. Cold-start: yeni device / yeni receiver geldiğinde model ne yapar?**
`device_is_first_seen=1`, `receiver_is_first_seen=1` flag'leri ile yakalanır. Fraud rate ise smoothing sayesinde `p_global ≈ 0.006`'a düşer (prior=50). Aggregate'ler 0; model bunu da bir sinyal olarak öğrenir (yeni cihaz fraud-yatkın).

**S27. SHAP reason codes neden `f21`, `f22` gibi encoded isim dönüyor?**
CatBoost pipeline'da preprocessor `get_feature_names_out()` desteği sınırlı. Fallback olarak generic index isim dönüyor. Üretimde feature mapping table (`f{i}` → orijinal isim) eklenmeli; üretim TODO. Şu an SHAP değerleri doğru, sadece label cosmetic.

**S28. API latency? 30 feature CatBoost predict + opsiyonel SHAP.**
Predict: ~5-15ms. SHAP `?explain=true` ile +30-100ms (TreeExplainer). Production'da SHAP async / batched yapılabilir.

**S29. Healthcheck gerçek predict yapmıyor, sadece model adı dönüyor — bu yeterli mi?**
Yeterli değil, gerçek warm-up predict ile değiştirilmeli. Mevcut implementasyon "process ayakta + model yüklü mü" kontrolü. Üretim için warm-up endpoint TODO listesinde.

---

## G. Metrik & Karar

**S30. Neden PR-AUC, ROC-AUC değil?**
Imbalance'da ROC-AUC yanıltıcı. Negative sınıf çok büyük; FPR çok küçük kalır, model performansı şişer. PR-AUC pozitif sınıfa odaklı; fraud detection'ın doğal metriği. Yine de ROC-AUC raporlandı (0.999 val / 0.992 test).

**S31. Operasyonel metrik nedir? PR-AUC'u risk ekibine nasıl satarsın?**
PR-AUC değil, **Recall@top-1%** ve **alerts/day** ile satarım. Örnek: "test setinde top-1% alert'te fraud'ların %92.4'ünü yakalıyoruz, günde ~75 alert/100 = 1158 tx-day'de". Risk ekibi günlük inceleme kapasitesini bilir, bu cümle direkt aksiyona dönüşür.

**S32. F1 @ 1% sadece 0.24 — bu zayıf değil mi?**
F1 imbalance'da yanıltıcı. Pozitif sınıf çok az olduğu için precision otomatik düşer. Önemli olan: Recall 0.92 + alert budget kabul edilebilir. F2 (recall ağırlıklı) 0.43 — fraud detection için F1'den daha anlamlı.

---

## H. Hızlı Yanıt Cebi

| Soru | Cevap özü |
|---|---|
| Neden CatBoost? | Native categorical + en yüksek val PR-AUC (0.962) |
| Neden 30 feature? | Permutation top-30, val'de full_safe ile aynı performans, daha az drift |
| Demografi neden yok? | Yasal/etik risk + permutation top-15'te yok + ablation kazanç vermedi |
| `fraud_rate_smoothed` leakage mi? | 7-day lag + Bayesian prior=50, leakage yok |
| Test PR-AUC 0.84 yeterli mi? | Recall@1% = 0.92, alerts/day operasyonel olarak ölçeklenebilir |
| Class weight vs SMOTE? | Class weight, time-series'i bozmaz, mathematically equivalent |
| Threshold 0.5 değil neden? | Imbalance %0.6 → percentile-based; HIGH=0.99999, MEDIUM=0.045 |
| Neural net neden yok? | Tabular + 850K satır → GBM standardı yener, 1 günlük case ROI yok |
| Random split neden olmaz? | Drift var, ileri-geri sızıntısı yaratır, performansı şişirir |
| Adversarial AUC 0.989 ne demek? | Train↔val ayırt edilebilir → drift var → drift_robust set hazır |

---

## I. Kolon Sözlüğü — Aklında Tut

### Ham 26 Kolon

| Grup | Kolonlar |
|---|---|
| **ID / Zaman** | `BusinessKey` (tx ID), `TransactionDate`, `AccountNumber` |
| **İşlem** | `TransactionType` (Fast/Havale/Eft), `TransactionChannel` (Mobile — drop), `TransactionAmount`, `IsFractionalAmount` (kuruşlu mu) |
| **Taraflar (PII)** | `ReceiverName`, `SenderName`, `CustomerName` |
| **Müşteri** | `CustomerSegment` (A1/A2/B/C/D/P/KP/T/Y), `CustomerAge`, `CustomerTenure`, `CustomerEducation`, `CustomerProfession`, `CustomerMaritalStatus`, `CustomerGender` |
| **Cihaz / IP** | `DeviceModel` (1004 unique), `DeviceOSName` (Android/IOS), `DeviceId` (13580 unique), `IP_Subnet`, `UniqueIPCount` |
| **Risk sinyali** | `HasMobileActivationL1H`, `HasMobileActivationL8H` (son 1/8 saat içinde mobil aktivasyon) |
| **Zaman** | `DayType` (NaN=Normal / Resmi Tatil / Yarım Gün) — %96 NaN |
| **Hedef** | `IsFraudTransaction` (0/1) |

### Instant Türevler (FE)

| Kolon | Üretim | Anlam |
|---|---|---|
| `amount_log` | `log1p(TransactionAmount)` | Sağa çarpık tutarı normalize eder |
| `DayType_clean` | NaN → "Unknown" | Normal günler ayrı kategori |
| `is_holiday` | `~DayType.isna()` (int8) | Tatil günü flag |
| `os_l1h` | `DeviceOSName + "_" + HasMobileActivationL1H` | Örn: `"Android_1"` (Android cihaz + son 1h aktivasyon) |
| `os_l8h` | aynı, 8h | Örn: `"IOS_0"` |
| `DeviceParentBrand` | DeviceModel string'inden çıkarım | iPhone / Samsung / Xiaomi / Huawei / ... |
| `DeviceModel` | top-100 + `_OTHER_` | Yüksek kardinaliteyi düşürür |

### Historical Aggregate'ler — DEVICE (cihaz başına geçmişe bakar)

| Kolon | Örnek değer | Anlam |
|---|---|---|
| `device_tx_count_all` | 47 | Bu cihaz şimdiye kadar 47 tx yapmış |
| `device_tx_count_1d` | 3 | Son 24 saatte 3 tx |
| `device_tx_count_7d` | 12 | Son 7 günde 12 tx |
| `device_tx_count_30d` | 47 | Son 30 günde 47 tx |
| `device_distinct_accounts_all` | 18 | 18 farklı hesapla işlem yapmış (= paylaşımlı/şüpheli) |
| `device_distinct_receivers_all` | 31 | 31 farklı alıcı |
| `device_first_seen_days_ago` | 2.4 | Bu cihaz 2.4 gün önce ilk görüldü (= yeni) |
| `device_is_first_seen` | 1 | Bu cihazın ilk tx'i mi (binary) |
| `device_amount_mean_prev` | 3120.5 | Bu cihazın geçmiş ortalama tutarı (TL) |
| `device_amount_std_prev` | 1840.2 | Geçmiş tutar std |
| `device_amount_max_prev` | 8200.0 | Geçmişteki max tutar |
| `device_amount_ratio_to_mean` | 2.55 | Şu anki tutar / geçmiş ortalama (anomali sinyali) |

### Historical Aggregate'ler — RECEIVER (alıcı başına)

| Kolon | Örnek | Anlam |
|---|---|---|
| `receiver_tx_count_all` | 220 | Bu alıcı 220 tx almış |
| `receiver_tx_count_7d` | 8 | Son 7 günde 8 |
| `receiver_tx_count_30d` | 34 | Son 30 günde 34 |
| `receiver_distinct_senders_all` | 12 | 12 farklı gönderici (= popüler alıcı veya mule) |
| `receiver_distinct_devices_all` | 6 | 6 farklı cihazdan gelen para |
| `receiver_first_seen_days_ago` | 5.0 | 5 gün önce ilk göründü |
| `receiver_is_first_seen` | 0 | İlk değil |
| `receiver_amount_mean_prev` | 4800.0 | Geçmiş ortalama |
| `receiver_amount_std_prev` | 2100.0 | Std |
| `receiver_amount_max_prev` | 9300.0 | Max |
| `receiver_amount_ratio_to_mean` | 1.65 | Anomali sinyali |

### Historical Aggregate'ler — SUBNET (IP_Subnet) & ACCOUNT

| Kolon | Örnek | Anlam |
|---|---|---|
| `subnet_tx_count_all` | 156 | Bu IP subnet'ten 156 tx |
| `subnet_tx_count_7d` | 18 | Son 7 günde |
| `subnet_distinct_devices_all` | 23 | 23 farklı cihaz aynı subnet'ten (= proxy/VPN sinyali) |
| `account_tx_count_all` | 0 | Bu hesabın ilk tx'i (med=1) |
| `account_first_seen_days_ago` | 0.0 | Hesap bugün ilk göründü |
| `account_is_first_seen` | 1 | Yeni hesap flag |

### Label-DEPENDENT (lag=7 gün + smoothing, prior=50)

| Kolon | Formül | Anlam |
|---|---|---|
| `device_fraud_rate_smoothed` | `(k + 50 * p_global) / (n + 50)` | Cihazın t-7gün öncesine kadar fraud oranı, smoothed |
| `device_label_n` | n | Lag'ten önceki etiket sayısı (rate güvenilirlik göstergesi) |
| `receiver_fraud_rate_smoothed` | aynı | Alıcı fraud oranı |
| `receiver_label_n` | n | Etiket sayısı |

`p_global ≈ 0.006`. Yeni cihaz/alıcıda n=0 → rate ≈ p_global. Mantık: "geçmişte fraud oranı yüksek alıcı/cihaz tekrar yatkın" + "etiket gecikmesi modellendi".

---

## J. Final Modelin 30 Kolonu — Sıralı (önem)

Permutation importance sırasıyla:

```
1.  device_fraud_rate_smoothed          ★ en güçlü sinyal (importance 0.298)
2.  receiver_fraud_rate_smoothed        ★ ikinci (0.193)
3.  DeviceModel                         categorical, top-100 bucket
4.  device_first_seen_days_ago          yeni cihaz mı
5.  receiver_first_seen_days_ago        yeni alıcı mı
6.  CustomerProfession                  meslek (112 kategori)
7.  UniqueIPCount                       hesabın gördüğü IP sayısı
8.  os_l8h                              cross feature (OS × 8h aktivasyon)
9.  CustomerTenure                      hesabın yaşı
10. IsFractionalAmount                  kuruşlu tutar (fraud %2.15 vs %0.43)
11. receiver_distinct_devices_all       alıcıya kaç cihazdan para gelmiş
12. DeviceParentBrand                   iPhone/Samsung/...
13. CustomerSegment                     A1/B/P/...
14. receiver_amount_mean_prev           alıcı ortalama tutar
15. receiver_amount_max_prev            alıcı max tutar
16. receiver_distinct_senders_all       alıcıya kaç gönderici göndermiş
17. account_tx_count_all                hesap geçmişi
18. HasMobileActivationL8H              son 8h aktivasyon
19. device_is_first_seen                cihaz ilk mi
20. receiver_label_n                    alıcı için lag'li etiket sayısı
21. receiver_tx_count_all               alıcı toplam tx
22. device_tx_count_7d                  cihaz son 7 gün
23. receiver_is_first_seen              alıcı ilk mi
24. device_tx_count_1d                  cihaz son 24 saat
25. device_amount_ratio_to_mean         cihaz anomali
26. amount_log                          tutar (log)
27. DayType_clean                       Normal/Resmi Tatil/...
28. device_amount_mean_prev             cihaz ortalama tutar
29. HasMobileActivationL1H              son 1h aktivasyon
30. account_is_first_seen               yeni hesap mı
```

**Sunumda "en önemli 5 feature" sorulursa**: 1, 2, 4, 5, 11. (iki fraud rate + iki first_seen + receiver_distinct_devices)

---

## K. Senaryo Örnekleri — "Bu işleme model nasıl bakar?"

### Örnek 1: Klasik Fraud Pattern (HIGH score beklenir)

```
TransactionAmount     = 7951.97 TL
IsFractionalAmount    = True              ← .97 kuruş, fraud %2.15
HasMobileActivationL1H = 1                ← son 1h aktivasyon → fraud %10.99
HasMobileActivationL8H = 1
DeviceOSName          = "Android"          ← fraud %0.92 vs IOS %0.20
os_l1h                = "Android_1"        ← cross sinyal güçlü
device_first_seen_days_ago    = 2.1        ← YENİ cihaz
receiver_first_seen_days_ago  = 5.0        ← YENİ alıcı
device_fraud_rate_smoothed    = 0.082      ← cihaz geçmişte fraud yapmış
receiver_fraud_rate_smoothed  = 0.041      ← alıcı geçmişte fraud almış
account_is_first_seen         = 1          ← yeni hesap
device_amount_ratio_to_mean   = 2.55       ← olağandışı tutar

Model çıktısı: fraud_score = 0.9998, band = MEDIUM (>0.045)
```

### Örnek 2: Klasik Legit Pattern (LOW score beklenir)

```
TransactionAmount     = 3159.65
IsFractionalAmount    = False              ← yuvarlak
HasMobileActivationL1H = 0
HasMobileActivationL8H = 0
DeviceOSName          = "IOS"              ← IOS fraud %0.20
device_first_seen_days_ago    = 421.0      ← köklü cihaz
receiver_first_seen_days_ago  = 380.0      ← köklü alıcı
device_fraud_rate_smoothed    = 0.0008     ← cihaz temiz geçmiş
receiver_fraud_rate_smoothed  = 0.0009     ← alıcı temiz
receiver_distinct_devices_all = 1          ← stabil tek cihaz
account_tx_count_all          = 38         ← aktif hesap
device_amount_ratio_to_mean   = 1.04       ← normal tutar

Model çıktısı: fraud_score = 3.7e-08, band = LOW
```

### Örnek 3: Belirsiz / Borderline

```
TransactionAmount     = 12500
IsFractionalAmount    = False
HasMobileActivationL1H = 0
device_first_seen_days_ago    = 45.0       ← orta yaş cihaz
device_fraud_rate_smoothed    = 0.012      ← hafif yüksek
receiver_fraud_rate_smoothed  = 0.006      ← global ortalama
device_amount_ratio_to_mean   = 4.2        ← anomali var
```

Bu tip işlemler **MEDIUM** band'a düşer → manual review. Model belirsizliği insana devreder.

---

## L. Üretim Akışı Hatırlatma

API'ye gelen JSON'da `historical_features` boş gelirse → 0-fill (cold start). Üretimde bu kolonlar **feature store**'dan PIT-correct sorgulanır. Şu an client gönderiyor (case kapsamı). 30 kolonun hangileri client'tan gelmeli:

- **API otomatik üretir** (request'in kendi alanlarından): `amount_log`, `DayType_clean`, `is_holiday`, `os_l1h`, `os_l8h`, `DeviceParentBrand`, `DeviceModel` (bucket)
- **Client/feature store gönderir** (historical): tüm `device_*`, `receiver_*`, `account_*`, `subnet_*` aggregate'ler + label-dependent rate'ler

---

## M. Model Eğitiminde Uyguladığım Yöntemler ve Neden Kritik

> "Hangi teknikleri kullandın, neden önemli?" sorusuna direkt cevap. Her birini 2-3 cümleyle anlat.

### 1. Time-Based Split (Random shuffle yok)

Veri zamana göre artan sıralandı; train 2024-03'e kadar, val Nisan-Mayıs, test Haziran-Eylül. **Asla random shuffle yapmadım.**

**Neden kritik:** Fraud zaman içinde değişen bir olgu — fraudcuların pattern'i, banka kontrolleri, kullanıcı davranışı sürekli kayıyor. Random split, geleceği görüp geçmişi tahmin etmek demek; model üretimde çökerken kâğıt üstünde harika görünür. Time-based split gerçek üretim koşulunu simüle eder.

### 2. Point-in-Time (PIT) Correct Feature Engineering

Bir satırın tüm historical feature'ları (`device_tx_count_30d`, `receiver_amount_mean_prev` vb.), o satırın `TransactionDate`'inden **strictly öncekinden** hesaplandı. **Current row daima hariç.**

**Neden kritik:** Eğer bir cihazın "geçmiş ortalama tutarı"nı hesaplarken current transaction'ı da dahil edersem, model fraud sinyalini tahmin yerine "ezberler". Üretimde bu mümkün değil — feature store'da sadece geçmiş veri olur. PIT correctness, eğitim ile üretim arasındaki sapmanın (train/serving skew) sıfırlanmasını sağlar.

### 3. Label-Lag + Bayesian Smoothing (target encoding'in güvenli versiyonu)

`device_fraud_rate_smoothed` ve `receiver_fraud_rate_smoothed` üretirken **7 günlük label availability lag** uyguladım: t anındaki transaction için `t-7gün`'den öncesine kadar olan etiketleri kullandım. Üstüne Bayesian smoothing `(k + 50*p_global) / (n + 50)` ekledim.

**Neden kritik:** Gerçekte fraud etiketi anında belli olmaz — müşteri şikayeti gelir, manuel review yapılır, 1-7 gün sonra label oluşur. Bunu modellemezsem model "fraud etiketini anında biliyormuşum gibi" eğitilir ve üretimde aynı bilgiye sahip olmadığı için çöker. Smoothing ise yeni cihaz/alıcı için global ortalamaya yaklaştırır (cold-start güvenliği).

**Sonuç:** Bu iki feature modelin **en güçlü iki sinyali** oldu — permutation importance toplam katkısının ~%50'si.

### 4. Çok Pencereli Velocity Aggregations

Cihaz için 1d/7d/30d, alıcı için 7d/30d, subnet için 7d pencerelerinde tx sayısı hesapladım (`device_tx_count_1d` … `subnet_tx_count_7d`).

**Neden kritik:** Fraud "burst" davranışı gösterir — bir cihaz aniden 24 saatte 10 işlem yaparsa şüpheli. Tek pencere yetmez: bazı fraud tipi anlık burst, bazıları haftalık yayılır. Çok pencere modele "kısa-orta vadeli hareket" sinyalini hep birden verir.

### 5. Entity Linking — Distinct Count Aggregations

Bir cihazın gördüğü farklı hesap/alıcı sayısı (`device_distinct_accounts_all`, `device_distinct_receivers_all`), bir alıcıya gelen farklı gönderici/cihaz sayısı (`receiver_distinct_senders_all`, `receiver_distinct_devices_all`).

**Neden kritik:** Mule (para aracı) hesaplar burada belli olur — tek bir alıcıya **çok farklı** göndericiden, **çok farklı** cihazdan para gelir. Tek başına bir transaction normal görünür ama entity graf'ında işaretlenir. Bu sinyal sade tx-level feature'larla yakalanamaz.

### 6. Amount Anomaly per Entity (mean/std/max/ratio)

Her cihaz ve alıcı için geçmiş tutarın `mean / std / max` ve "şu anki tutar / geçmiş ortalama" oranı (`device_amount_ratio_to_mean`).

**Neden kritik:** Tek başına 7,000 TL transferi normaldir ama bir cihaz hep 500 TL transfer yapıyorsa **o cihaz için** anomalidir. Ratio feature'ı bu bağıl anomaliyi modele direkt verir. Mutlak tutar yerine **kontekstli tutar**.

### 7. First-Seen / Cold-Start Sinyalleri

Her entity için "kaç gün önce ilk görüldü" (`*_first_seen_days_ago`) + binary "ilk mi" flag (`*_is_first_seen`).

**Neden kritik:** Fraudcular sıklıkla **yeni cihaz, yeni alıcı, yeni hesap** kullanır. Bu sinyalin modelde olması, geçmişi olmayan entity'lerin doğal olarak yüksek risk almasını sağlar. Aynı zamanda cold-start probleminin de cevabı — bilinmeyen alıcı için smoothed_rate = global_avg, days_ago = 0 sinyaller birlikte çalışır.

### 8. Class Imbalance — Class Weight (SMOTE / undersampling değil)

`class_weight="balanced"` (LR/RF/HGB) ve `auto_class_weights="Balanced"` (CatBoost) kullandım. Loss function'da pozitif sınıf 158x daha fazla ağırlık alıyor.

**Neden kritik:** İmbalance 1:159. SMOTE synthetic fraud üretir → time-series'te zaman ekseni bozulur, PIT aggregate'ler anlamsızlaşır. Undersampling negative örnekleri atar → bilgi kaybı. Class weight matematiksel olarak loss'u yeniden ölçekler, veriyi bozmadan modeli pozitif sınıfa odaklar.

### 9. Native Categorical Handling (CatBoost)

CatBoost'a kategorik kolonları string olarak verdim (`CatBoostPrep` transformer), OneHotEncoder kullanmadım. `cat_features=cat_indices` parametresi ile CatBoost kendi içinde **ordered target encoding** yapıyor.

**Neden kritik:** `DeviceModel` (1004 unique), `CustomerProfession` (112), `IP_Subnet` (29K) gibi yüksek kardinaliteli kolonlarda one-hot devasa sparse matris üretir. CatBoost'un ordered target encoding'i hem boyutu küçük tutuyor hem leakage-safe hem benim manuel ürettiğim `fraud_rate_smoothed` mantığıyla uyumlu.

### 10. Yüksek Kardinalite Bucketing

`DeviceModel` 1,004 unique → **top-100 + `_OTHER_`** bucket'a indirgendi.

**Neden kritik:** Long-tail kategoriler tek tek model için gürültüdür (her biri 2-3 örnek görür). Üst 100 model ana sinyali taşır, gerisi "_OTHER_" altında toplanır. Aynı zamanda `DeviceParentBrand` (iPhone/Samsung/Xiaomi) ile daha kaba bir granular seviye eklendi — model iki seviyeden de okuyabiliyor.

### 11. Cross Features (Interaction)

`os_l1h = DeviceOSName + "_" + HasMobileActivationL1H` (örn. `"Android_1"`), `os_l8h` aynı pattern.

**Neden kritik:** EDA'da gördüm: "Android + son 1h aktivasyon var" kombinasyonu **tek başına Android veya tek başına L1H'den çok daha yüksek fraud oranı** veriyor. Ağaç-bazlı modeller iki feature'ı kendiliğinden split'lerde birleştirebilir ama explicit cross feature daha az derinlikte yakalanır → daha hızlı convergence + daha az overfit.

### 12. Demografi Çıkarma (Sensitive Attribute Ablation)

`CustomerAge`, `CustomerGender`, `CustomerEducation`, `CustomerMaritalStatus` modelden tamamen çıkarıldı (`demographic_excluded=True`).

**Neden kritik:** Yasal/etik risk — protected attributes üzerinden karar bias yaratır. Ablation testinde "demografi-free" model "full" modelle aynı val PR-AUC'u verdi → kazanç yok. Risk var ama kazanç yok → drop. Ek bonus: model audit edilebilirliği artıyor.

### 13. Çoklu Feature Set (Risk Yönetimi)

4 feature seti üretildi: `full_safe` (50), `selected_by_permutation` (30), `drift_robust` (41), `label_free` (46). Her biri için ayrı model eğitildi.

**Neden kritik:** Üretim koşulları değişebilir — drift artarsa `drift_robust`'a, label gecikmesi sorun olursa `label_free`'ye, performans kritikse `selected_by_permutation`'a geçilebilir. Tek modele bağımlı kalmamak için **hazır alternatif**ler.

### 14. Permutation Importance ile Feature Selection

Sklearn `permutation_importance` ile val seti üzerinde her feature'ı shuffle edip PR-AUC kaybını ölçtüm. Top-30 seçildi.

**Neden kritik:** Model-agnostic ölçüm — feature'ın gerçekten predictive katkısını ölçüyor, ağaç importance'ı gibi yüzeysel split count değil. Test setine **dokunmadan** seçim yaptım (data leakage'ı önlemek için).

### 15. Adversarial Validation ile Drift Tespiti

Train (label=0) + val (label=1) birleştirilip ikinci bir model bunları ayırt etmeye çalıştı. AUC = **0.989** çıktı.

**Neden kritik:** AUC 0.5'e ne kadar yakınsa train↔val benzerliği o kadar yüksek. 0.989 demek "iki dağılım kolay ayırt ediliyor" → ciddi drift var. Bunu erken görmek `drift_robust` feature setini hazırlamamı, üretimde PSI monitoring önermemi sağladı.

### 16. Time-Aware Hiperparametre Optimizasyonu (Optuna)

LR ve HGB için Optuna TPE sampler ile 15-30 trial HPO yapıldı. **Objective = val PR-AUC**. Test setine asla dokunulmadı.

**Neden kritik:** Random search yerine TPE (Bayesian) → daha az trial'da daha iyi optimum. Time-aware = HPO içinde val ayrılırken zaman sırası bozulmuyor. Test seti tamamen el değmemiş kalıyor → final raporlama bias-free.

### 17. Threshold Politikası — Percentile-Based

Default 0.5 yerine **val seti percentile'ına göre** eşik: HIGH = top 0.1% (0.99999), MEDIUM = top 1% (0.0449). Ayrıca `alerts/day` perspektifi raporlandı.

**Neden kritik:** İmbalance %0.6'da 0.5 eşiği fraud'u tamamen kaçırır. Percentile bazlı eşik **iş kapasitesiyle eşleşir** — risk ekibinin günlük 100 alert kapasitesi varsa "0.0010 threshold" diye söyler, model bu kapasiteye göre çalışır.

### 18. Calibration — Platt (Sigmoid)

Val'de fit, test'te eval. Platt calibration Brier'ı 0.00096 → 0.00056'ya düşürdü, PR-AUC değişmedi (ranking metriği).

**Neden kritik:** Ham model olasılığı yanıltıcı olabilir (örn. 0.8 dönüyor ama gerçekte %30 fraud). Calibrated olasılık → güvenilir threshold, güvenilir risk ekibi kararı. Isotonic daha düşük Brier verdi ama PR-AUC'u düşürdü → Platt tercih edildi.

### 19. SHAP Reason Codes (API'de Per-Request)

API'de `?explain=true` ile her tahmine top-5 SHAP katkısı dönüyor (TreeExplainer).

**Neden kritik:** Fraud kararı **açıklanabilir olmalı** — operasyon ekibi neden bloklandığını görmeli, müşteri itirazında banka cevap verebilmeli, regülasyon audit gerektirebilir. Per-request SHAP, lokal karar açıklaması veriyor.

### 20. Multi-Model Karşılaştırma + Rule-Based Baseline

5 model eğitildi: rule-based, LR, RF, HGB, CatBoost. Her biri 4 feature seti ile → 16 ML aday + 1 baseline.

**Neden kritik:** Tek modele güvenmek tehlikeli. Rule-based **operasyonel benchmark** — "ML olmadan ne yapardık?" sorusunun cevabı. LR **lineer benchmark** — non-lineerlik gerçekten gerekli mi? RF/HGB/CatBoost farklı ağaç aileleri — final seçimde val PR-AUC ile karar verildi (CatBoost 0.962).

---

## N. IEEE-CIS Fraud Detection Yarışmasından Alınan Yöntemler (Yedek Referans)

IEEE-CIS, Kaggle'ın en bilinen fraud yarışması (2019). Birinci ekibin (Chris Deotte, Konstantin Yakovlev) ünlü olduğu temel teknikler ve bu projede hangi şekilde uygulandı:

### 1. UID-Aggregation Pattern (en kritik teknik)

**IEEE-CIS'te ne yapılmıştı:** `card1` (kart kimliği) bazında `TransactionAmt`'in `mean/std/min/max` aggregate'leri üretildi (`card1_TransactionAmt_mean` vb.). Sonra "current amount / card mean" oranı feature olarak eklendi → fraud genelde geçmiş davranıştan sapan tutarlardır.

**Burada nasıl uygulandı:**

```python
# src/features/historical.py — _amount_aggregates_prev
# Per-group (DeviceId, ReceiverName), current row HARİÇ, expanding mean/std/max
for col, prefix in (("DeviceId","device"), ("ReceiverName","receiver")):
    agg = _amount_aggregates_prev(out[col].to_numpy(), amt)
    out[f"{prefix}_amount_mean_prev"] = agg["mean"]
    out[f"{prefix}_amount_std_prev"]  = agg["std"]
    out[f"{prefix}_amount_max_prev"]  = agg["max"]
    out[f"{prefix}_amount_ratio_to_mean"] = np.where(agg["mean"] > 0, amt/agg["mean"], 1.0)
```

→ Sonuç: `device_amount_mean_prev`, `device_amount_std_prev`, `device_amount_max_prev`, `device_amount_ratio_to_mean` + aynı set receiver için. **Final modelde 4 tanesi var** (top-30'da).

**Sunum cümlesi:** "IEEE-CIS 1.'sinin `card1_TransactionAmt_mean` deseni; burada `card1` yerine `DeviceId` ve `ReceiverName` üzerinde, **point-in-time correct** (current row hariç) şekilde uyguladım."

### 2. UID Linking — "Bir kimlik kaç farklı şeye bağlı?"

**IEEE-CIS'te:** Bir kartın kaç farklı email/adres/cihazla ilişkili olduğu sayıldı. Çok sayıda farklı entity ile ilişkili kart = mule / paylaşımlı / ele geçirilmiş.

**Burada nasıl uygulandı:**

```python
# distinct count: bir cihazın kaç farklı hesap/alıcı gördüğü
out["device_distinct_accounts_all"]  = _distinct_count_prev(dev, accounts)
out["device_distinct_receivers_all"] = _distinct_count_prev(dev, receivers)
out["receiver_distinct_senders_all"] = _distinct_count_prev(rec, senders)
out["receiver_distinct_devices_all"] = _distinct_count_prev(rec, devices)
out["subnet_distinct_devices_all"]   = _distinct_count_prev(sub, devices)
```

→ **Final modelde 3 tanesi var**: `receiver_distinct_devices_all`, `receiver_distinct_senders_all` (top-15'te).

**Sunum cümlesi:** "IEEE-CIS UID linking — bir cihaz çok farklı hesabı kullanıyorsa, bir alıcıya çok farklı göndericiden para geliyorsa, mule sinyali. Aynı pattern."

### 3. Time-Window Velocity Aggregations (C-features pattern)

**IEEE-CIS'te:** `C1..C14` kolonları farklı zaman pencerelerinde tx sayıları taşıyordu (kartla ilişkili sayımlar).

**Burada nasıl uygulandı:**

```python
# Rolling count: son N günde aynı entity'nin tx sayısı, current row hariç
for w in (1, 7, 30):
    out[f"device_tx_count_{w}d"] = _rolling_count_prev(times, dev, w)
for w in (7, 30):
    out[f"receiver_tx_count_{w}d"] = _rolling_count_prev(times, rec, w)
out["subnet_tx_count_7d"] = _rolling_count_prev(times, sub, 7)
```

→ **Final modelde `device_tx_count_1d` ve `device_tx_count_7d` var.**

**Sunum cümlesi:** "C-feature analoğu — IEEE-CIS'teki çok pencereli velocity sayımları. Burada 1d / 7d / 30d pencereleri."

### 4. First-Seen / Days-Since (D-features pattern)

**IEEE-CIS'te:** `D1..D15` kolonları "kartın ilk görülmesinden bu yana geçen gün" gibi metrikler. En güçlü sinyallerden biri.

**Burada nasıl uygulandı:**

```python
def _first_seen_days_ago(times, by_vals):
    # Per-group, current row için ilk görülme tarihinden geçen gün
    ...
out["device_first_seen_days_ago"]    = _first_seen_days_ago(times, dev)
out["receiver_first_seen_days_ago"]  = _first_seen_days_ago(times, rec)
out["account_first_seen_days_ago"]   = _first_seen_days_ago(times, acc)
# + first_seen binary flag
out["device_is_first_seen"]   = (out["device_tx_count_all"] == 0).astype("int8")
out["receiver_is_first_seen"] = (out["receiver_tx_count_all"] == 0).astype("int8")
out["account_is_first_seen"]  = (out["account_tx_count_all"] == 0).astype("int8")
```

→ **Final modelde 4-5 tanesi top-10'da**: `device_first_seen_days_ago` (#4), `receiver_first_seen_days_ago` (#5), `device_is_first_seen` (#19), `receiver_is_first_seen` (#23), `account_is_first_seen` (#30).

**Sunum cümlesi:** "D-feature analoğu — entity'nin ilk görülmesinden bu yana geçen gün. Yeni cihaz / yeni alıcı = en güçlü 5 sinyalden ikisi."

### 5. Target Encoding (out-of-fold smoothed)

**IEEE-CIS'te:** Yüksek kardinaliteli kategoriklerde out-of-fold target encoding kullanıldı.

**Burada nasıl uygulandı (önemli fark):**

```python
# src/features/historical.py — _smoothed_fraud_rate_with_lag
# Time-lagged Bayesian smoothing — OOF değil, TIME-LAGGED
rate[idx] = (k + prior_strength * p_global) / (n + prior_strength)
# k, n = t - 7 gün öncesine kadar olan etiket toplam ve sayısı
```

→ `device_fraud_rate_smoothed` ve `receiver_fraud_rate_smoothed` — **modelin en güçlü iki feature'ı**.

**Sunum cümlesi:** "Target encoding'in time-series güvenli versiyonu. IEEE-CIS OOF (out-of-fold) yapıyordu — bu time-aware veride bozulur. Onun yerine 7-day lag + Bayesian smoothing (prior=50). Aynı sinyali leakage'sız taşır."

### 6. CatBoost — Native Categorical + Ordered Target Encoding

**IEEE-CIS'te:** XGBoost + LightGBM + CatBoost ensemble kullanıldı.

**Burada nasıl uygulandı:**

```python
CatBoostClassifier(
    iterations=600, learning_rate=0.05, depth=8,
    l2_leaf_reg=3.0, auto_class_weights="Balanced",
    cat_features=cat_indices,  # native categorical
    ...
)
```

→ CatBoost'un **ordered target encoding**'i, bizim manuel ürettiğimiz `fraud_rate_smoothed` mantığının built-in versiyonu. İkisi birden çalışınca model hem zaman-aware smoothed rate'i hem CatBoost'un kendi target encoding'ini görüyor.

**Sunum cümlesi:** "CatBoost'u tercih ettim çünkü native ordered target encoding zaten benim smoothed_fraud_rate mantığımın yerleşik versiyonu. Manuel pattern + built-in pattern birbirini destekliyor."

### 7. Diğer IEEE-CIS Yöntemleri ve Burada Uygulanmayanlar

| IEEE-CIS Tekniği | Burada? | Gerekçe |
|---|---|---|
| UID-aggregation (amount mean/std/max) | ✅ Uygulandı | Top-15 feature içinde |
| UID linking (distinct counts) | ✅ Uygulandı | Top-15 feature içinde |
| Time-window velocity (C-features) | ✅ Uygulandı | 1d/7d/30d |
| Days-since-first-seen (D-features) | ✅ Uygulandı | Top-5 feature içinde |
| Target encoding (smoothed) | ✅ Uygulandı (time-lagged) | OOF yerine 7-day lag |
| Negative downsampling | ❌ Yapılmadı | Class weight tercih edildi (time-series'i bozmamak için) |
| Magic UID combinations (card1+addr1+...) | ❌ Yapılmadı | Veride email/addr yok; DeviceId + ReceiverName yeterli proxy |
| Frequency encoding | ❌ Direkt yok | DeviceModel top-100 bucket benzer effekt verir |
| Stacking / ensemble | ❌ Yapılmadı | Tek model (CatBoost) yeterli sinyal verdi, 1 günlük case |
| Adversarial validation | ✅ Yapıldı | Train↔val AUC = 0.989 → drift tespiti |

---

## O. "Hangi pattern senin sinyalini taşıdı?" (Eğer derinleşirlerse)

Top-30 importance üzerinden IEEE-CIS pattern eşleştirmesi:

```
IEEE-CIS Pattern          → Bu projede karşılığı                  → Top-30'da kaç tanesi
─────────────────────────────────────────────────────────────────────────────────────
Target encoding (smoothed)  → *_fraud_rate_smoothed                 → 2 (#1, #2 — en güçlü)
D-features (first seen)     → *_first_seen_days_ago, _is_first_seen → 5
UID-aggregation (amount)    → *_amount_{mean,max,ratio}_prev        → 4
UID linking (distinct)      → *_distinct_*_all                      → 3
C-features (velocity)       → *_tx_count_{1d,7d,30d,all}            → 4
```

Toplam: **18 / 30 feature** IEEE-CIS pattern'lerinden türetildi. Geri kalan 12 → ham veri kolonları (DeviceModel, CustomerProfession, UniqueIPCount, IsFractionalAmount, vb.).

---

## P. Eğer "Ne Eksik?" diye sorulursa

1. **CV yok**: tek-pencere val. Expanding-window CV ile robustness artırılabilirdi.
2. **SHAP reason code feature isimleri encoded** (`f21`): üretim öncesi mapping table eklenmeli.
3. **API feature store entegrasyonu yok**: historical_features şu an client'tan geliyor.
4. **Drift monitoring yok**: PSI ölçümü + alert pipeline eklenmeli.
5. **Cold-start politikası yumuşak**: yeni receiver için smoothed rate = global avg; rule-based fallback eklenebilir.
6. **CatBoost HPO yapılmadı**: baseline yeterli geldi, Optuna run eklenmedi.
7. **Two-stage / cost-sensitive learning denenmedi**: notebook 09'da deney var ama final pipeline'a girmedi.
8. **API auth + rate limit yok**: production-öncesi şart.
