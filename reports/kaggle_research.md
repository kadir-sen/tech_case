# Kaggle Benzer Yarışma Araştırması — Bizim Case'e Entegre Edilebilir Teknikler

> Bu rapor, banking/transaction fraud detection alanında benzer Kaggle yarışmalarındaki üst-sıra
> çözümleri inceler ve **bizim case'imize entegre edilebilecek teknikleri** kanıt-temelli olarak
> önceliklendirir.

## En yakın benchmark: IEEE-CIS Fraud Detection (2019)

**Neden bu?** Bizim veriyle güçlü örtüşme:
- ~590K transaction (bizim 850K)
- Tx-level fraud detection (bizim case ile aynı)
- Device fingerprint, IP, customer info, amount alanları var (bizim DeviceId/IP_Subnet/CustomerSegment/TransactionAmount karşılığı)
- 6.351 takım, Vesta Corporation sponsorlu
- Birinci yer: Chris Deotte (NVIDIA) + Konstantin Yakovlev — Private LB AUC **0.9459**

**Bizim case'le benzemeyen tarafları:**
- IEEE-CIS'te 399 anonim "V" feature var; bizde 26 isimli kolon
- IEEE-CIS'te `card1, card2, card3, addr1, addr2, P_emaildomain, R_emaildomain, D1-D15` var; bizde bunlar yok
- IEEE-CIS metric'i ROC-AUC; biz PR-AUC kullandık (daha doğru imbalance için)

## Diğer ilgili kaynaklar (özet)

- **PaySim / BankSim** (Kaggle): %0.13 fraud, mobile money sim. Bizim sentetik veriye yapı olarak benziyor ama 11 feature ile çok dar.
- **Credit Card Fraud (PCA features)**: Tamamen anonim — feature engineering için yararsız.
- **Production fraud detection makaleleri**: Threshold/latency/calibration insights.

---

## 1. IEEE-CIS 1st place — kritik teknikler

### 1.1 "Magic UID" — sentetik kullanıcı kimliği (en büyük tek kazanç)

**Ne yaptılar?**
- Anonim verideki gerçek kullanıcıyı rekonstrükte ettiler: `UID = card1 + addr1 + D1` (kart no + adres + ilk kullanım gününden delta).
- Bu UID üzerinden `TransactionAmt_card1_mean`, `TransactionAmt_card1_std`, `D9_card1_addr1_mean` gibi **45+ aggregate feature** üretti.
- Local CV: AUC 0.9363 → 0.9472 (sadece bu adımdan **+0.011 AUC kazanç**).

**Felsefe:** "Bu işlem, bu kullanıcı için **olağandışı** mı?" sorusunu modelin sorabilmesi.

**Bizim case'e entegrasyon:** **EVET ama farklı** — bizde `AccountNumber` ham olarak var ama %97'sinin tek tx'i olduğu için kullanıcı-bazlı aggregate üretemiyoruz. Bunun yerine bizim "UID benzeri" entity'mız **DeviceId**'dir (medyan 21 tx/cihaz, %66'sı ≥10 hesap kullanıyor). **`TransactionAmount_DeviceId_mean / std / max` PIT-correct şekilde eklenebilir.** Bu en kuvvetli aday.

### 1.2 Frequency encoding

**Ne?** Yüksek-cardinality kategorik için, kategorinin değeri yerine **train'de görülme frekansı** verilir. Örn: `card1_FE = freq(card1)`.

**Mantık:** Tree-based modeller categorical ham değeri ezberlemek yerine, kategorinin "ne kadar yaygın" olduğunu öğrenir; yeni kategorilerde generalize eder.

**Bizim case'e:** **EVET, doğrudan uygulanabilir.** Uygun hedefler:
- `DeviceModel` (1004 unique → top-100 + Other bucket'lanmış zaten; ek olarak `DeviceModel_FE` eklenebilir)
- `CustomerProfession` (112 unique)
- `ReceiverName` ham metin değil ama `ReceiverName_FE` (train'de kaç kez görüldüğü) bilgi taşır

### 1.3 Column splitting (Dollars + Cents)

**Ne?** `TransactionAmt` → `Dollars` (int) + `Cents` (decimal part). Tree-based model bu iki sinyali ayrı ayrı öğrenir.

**Bizim case'e:** **KISMEN var.** Biz `IsFractionalAmount` (binary) tutuyoruz. Daha güçlü versiyon: `TransactionAmount_cents = (TransactionAmount * 100) % 100` (0-99 arası). Easy win.

### 1.4 GroupKFold (time-based, ay bazlı)

**Ne?** Random k-fold yerine, ayları grup olarak alıp time-respecting CV.

**Bizim case'e:** **KISMEN var.** Biz time-based train/val/test ayrımı kullanıyoruz, ama within-train için ek olarak `TimeSeriesSplit` (örn. çeyrek-bazlı 4 fold) ekleyebiliriz. Notebook 09'da cascade için zaten bu yapı var; ana training pipeline'a taşınabilir.

### 1.5 Ensemble (LGBM + XGBoost + CatBoost rank-averaging)

**Ne?** Üç farklı gradient boosting'in olasılıklarını rank-average ile birleştir.

**Sonuç:** Single model AUC 0.945 → ensemble ~0.95+ (small ama net kazanç).

**Bizim case'e:** **EVET, kolay.** Zaten 4 modelimiz (LR/RF/HGB + rule_based) eğitilmiş. HGB + RF + LR rank-average yapılabilir. Tek satır kod, +0.01-0.03 PR-AUC tipik kazanç.

### 1.6 Hyperparameter pattern (XGBoost 5000 iter, LR 0.002, depth 12)

**Karakteristik:** Çok yavaş öğrenen, çok derin tree'ler, regularization az.

**Bizim case'e:** **KISMEN var.** Optuna ile 30 trial tuning yaptık (max_iter 200-800, LR 0.01-0.2). Daha agresif arama (max_iter 1000-3000, LR 0.001-0.05) marjinal kazanç sağlayabilir.

---

## 2. CatBoost — özel olarak değerli

### Ordered target encoding (yapısal anti-leakage)
- CatBoost, kategorik feature'lara **time-aware target encoding** uygular: her satır için target stats yalnız o satırdan önceki satırlardan hesaplanır.
- Bizim manuel olarak yaptığımız `receiver_fraud_rate_smoothed` (lag=7d) felsefesinin **yerleşik versiyonu**.

**Bizim case'e:** **DEĞERLİ.** Olası faydalar:
- Modelin kendi içinde leak-safe target encoding yapar → manuel FE'ye gerek kalmaz
- ReceiverName, DeviceModel gibi yüksek-card kategorikleri doğrudan verebiliriz (hash veya ordinal değil)
- macOS'te `libomp` sorunu olmayabilir (CatBoost own runtime)

**Risk:** Ordered encoding "satır içi" zaman varsayar. Bizde explicit zaman var; CatBoost'a `has_time=True` ile söylemek gerekir.

---

## 3. Adversarial Validation — drift teşhisi için

### Algoritma (kanonik)
1. Train'i `is_test=0`, test'i `is_test=1` ile etiketle.
2. Birleştir, shuffle.
3. Binary classifier (örn. XGBoost) eğit, ROC-AUC'a bak.
4. **AUC ≈ 0.5** → train ve test aynı dağılım (iyi).
5. **AUC >> 0.5** → drift var. Feature importance ile **drift'i hangi feature'lar yaratıyor** bulunur.
6. Drift'li feature'ları drop et veya yeniden transform et.

### Bizim case'in tam ihtiyacı
- Bizde val→test fraud rate düşüşü dramatik (val %0.34 → test %0.21).
- HGB val PR-AUC 0.93, test PR-AUC 0.80 — bu fark bir kısmı drift olabilir.
- **Adversarial validation ile drift'i hangi feature'lar yaratıyor net görebiliriz.**
- Drift'i azaltırsak test performansı val'e yaklaşabilir.

**Bizim case'e:** **YÜKSEK DEĞER**, doğrudan uygulanabilir. ~50 satır kod.

### Bonus: Test-like validation set kurma
- Train'i bu adversarial classifier ile skorla.
- Test'e en çok benzeyen train satırlarını ayrı bir "validation seti" yap.
- Modeli bu set üzerinde tune et — gerçek test'te daha güvenilir performans tahmini.

---

## 4. Production threshold strategy (Kaggle dışı, makale-derlenmiş)

### Business-constrained threshold
Statik PR-AUC değil, iş-kısıtlı threshold:
- **Min precision constraint**: "Precision ≥ %30 olmalı" → PR eğrisinden τ bulunur, karşı Recall raporlanır.
- **Max FPR constraint**: "Günlük ≤350 alert" → bizim mevcut "alerts/day" perspektifimiz bunu zaten yapıyor.
- **FN cost / FP cost matrisi**: ortalama fraud kaybı (örn. ₺5K) vs review maliyeti (örn. ₺10) → break-even threshold.

### Latency budget
- Production: **<50 ms p99 hedef**.
- Bizim API HGB inference ~5-10ms (test ettim) — bütçenin çok altında. SHAP `?explain=true` 30-100ms — istisnai isteklere uygun.

### Shadow mode + A/B
- Modeli production'a koymadan önce "shadow mode" (skorla ama aksiyon alma, sadece logla) ile gerçek trafikte ölçüm.
- Sonra %10 → %50 → %100 ramp.

---

## 5. Bizim case'e ENTEGRE EDİLEBİLİR teknikler — önceliklendirilmiş

### 🟢 Yüksek öncelik (kolay implement, beklenen kazanç ≥0.02 PR-AUC)

| # | Teknik | Beklenen kazanç | Effort | Risk |
|---|---|---|---|---|
| 1 | **Device-level Amount aggregation** (`amount_mean/std/max @ DeviceId`, PIT-correct) | +0.02-0.04 PR-AUC | Düşük | Düşük |
| 2 | **Frequency encoding** for DeviceModel, CustomerProfession, ReceiverName | +0.01-0.02 PR-AUC | Düşük | Çok düşük |
| 3 | **Amount cents feature** (`(amount*100)%100`) | +0.005-0.01 PR-AUC | Çok düşük | Çok düşük |
| 4 | **Rank-averaging ensemble** (HGB + RF + LR) | +0.01-0.03 PR-AUC | Düşük | Düşük |
| 5 | **Adversarial validation diagnostic** (drift teşhisi) | Drift'in nedenini anlama | Düşük | Yok (read-only) |

### 🟡 Orta öncelik (deneme değer, kazanç belirsiz)

| # | Teknik | Beklenen kazanç | Effort | Risk |
|---|---|---|---|---|
| 6 | **CatBoost** ana model olarak (ordered target encoding) | +0.01-0.03 PR-AUC | Orta | macOS libomp riski (test edilmeli) |
| 7 | **Time-since-last-tx per Device/IP_Subnet** (velocity) | +0.01-0.02 PR-AUC | Düşük | Düşük |
| 8 | **TimeSeriesSplit-based within-train CV** (HPO + ensemble için) | Daha güvenilir tuning | Orta | Düşük |
| 9 | **DeviceModel "parent brand" extraction** (samsung / iPhone / Xiaomi) | +0.005 PR-AUC | Düşük | Yok |
| 10 | **More aggressive HPO** (max_iter 2000+, LR 0.005) | +0.005-0.01 PR-AUC | Düşük (3-5 saat compute) | Düşük |

### 🔴 Düşük öncelik / önerilmez

| # | Teknik | Neden değil |
|---|---|---|
| 11 | **UID = card+addr+D1** | Bizde bu kolonlar yok; AccountNumber zaten unique-per-row |
| 12 | **V-column reduction (PCA)** | Bizde 399 anonim feature yok; 26 isimli kolon var |
| 13 | **Email domain features** | Bizde email yok |
| 14 | **Pseudo-labeling** | Fraud'da çok riskli — label noise modeli bozar |
| 15 | **Account-level historical aggregates** | %97 hesap tek-tx, üretemiyoruz |

---

## 6. Önerilen sonraki sprint (`improvements/v2`)

Şu sırayla yapılması önerilir (her biri ayrı commit, ölçülebilir kazanç):

1. **`feat(fe): device-level amount aggregations (PIT)`** — Test 1
2. **`feat(fe): frequency encoding for high-card categoricals`** — Test 2
3. **`feat(fe): amount cents + device parent-brand`** — küçük ama ücretsiz
4. **`feat(models): rank-average ensemble of HGB/RF/LR`** — `src/models/ensemble.py`
5. **`diag(drift): adversarial validation script`** — `scripts/adversarial_validation.py`. Çıktı: hangi feature'lar drift yaratıyor.
6. **(opsiyonel) `feat(models): catboost variant`** — libomp testi geçerse

**Beklenen kümülatif kazanç:** PR-AUC 0.805 → **0.84-0.86** bandı. Recall@1% 0.90 → 0.92-0.94.

**Test gereksinimi:** Her commit sonrası `scripts/compare_runs.py` ile önceki state'le karşılaştırma + `leakage_audit.py` permutation testi tekrarla (yeni feature'lar leakage yaratmadığını doğrula).

---

## 7. NE YAPMAMAK gerek

Kaggle benchmark'larında performans için kullanılan ama **production'a giderken yanlış olan** bazı şeyler:

1. **Public LB'ye göre threshold seçmek** — bizim out-of-time test bunu zaten engelliyor.
2. **Test set'in features distribution'ını train'e geri-feed etmek** (pseudo-labeling agresif kullanım) — fraud için riskli.
3. **Ensemble büyütmek için tüm aile** (20 LGBM + 5 CatBoost + ...) — production latency'i öldürür.
4. **Tüm Kaggle "magic feature"larını körü körüne adapt etmek** — UID-like aggregate'ler bizde DeviceId üzerinden anlamlı; AccountNumber üzerinden anlamsız.

---

## Sources

- **IEEE-CIS Fraud Detection competition**: [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection)
- **1st place writeup (Konstantin Yakovlev & Chris Deotte)**: [Kaggle writeup](https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2)
- **NVIDIA technical blog (Chris Deotte) — magic feature detayları**: [NVIDIA Blog](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/)
- **XGB Fraud with Magic notebook**: [Kaggle code](https://www.kaggle.com/code/cdeotte/xgb-fraud-with-magic-0-9600)
- **Top 5% solution (TDS)**: [Medium / Towards Data Science](https://medium.com/data-science/ieee-cis-fraud-detection-top-5-solution-5488fc66e95f)
- **Adversarial validation — Ilias Antonopoulos**: [Blog](https://ilias-ant.github.io/blog/adversarial-validation/)
- **Production fraud metrics — César Soto Valero**: [Blog](https://www.cesarsotovalero.net/blog/evaluation-metrics-for-real-time-financial-fraud-detection-ml-models.html)
- **PaySim Synthetic Dataset**: [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **BankSim Synthetic Dataset**: [Kaggle](https://www.kaggle.com/datasets/ealaxi/banksim1)
- **CatBoost ordered target encoding**: [NeurIPS paper](https://papers.neurips.cc/paper/7898-catboost-unbiased-boosting-with-categorical-features.pdf)
