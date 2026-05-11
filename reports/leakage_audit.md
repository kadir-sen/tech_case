# Leakage Audit — `hist_gbm + label-dependent FE`

> Auto-generated tarafından `scripts/leakage_audit.py`. JSON: `artifacts/leakage_audit.json`.
>
> Soru: PR-AUC 0.358 → 0.805 sıçraması (label-dependent FE eklenmesinden) gerçek bir
> performans mı yoksa data leakage mı?
>
> **Cevap: Gerçek performans. Dört bağımsız test bu sonucu destekliyor.**

---

## Önce: PIT formülünün doğruluğu (Step 1 — manuel)

`src/features/historical._smoothed_fraud_rate_with_lag` fonksiyonu, her satır için yalnızca
`t_current - lag_days` günden **strictly ÖNCEKİ** etiketleri kullanır.

8 ardışık fraud satırı üzerinde manuel hesap koddaki çıktıyla birebir eşleşti
(receiver = "Abiye Yaman Durdu", `recv_fraud_rate=100%`):

| row | t | label | manual n / k / rate | code n / rate |
|---|---|---|---|---|
| 0 | 2023-09-08 | 1 | 0 / 0 / 0.006283 (=p_global) | 0 / 0.006283 ✓ |
| 1 | 2023-09-24 (16d sonra) | 1 | 1 / 1 / 0.025768 | 1 / 0.025768 ✓ |
| 2 | 2023-09-26 (2d sonra) | 1 | 1 / 1 / 0.025768 (önceki satır <7d olduğu için DAHIL DEĞIL) | 1 / 0.025768 ✓ |
| 5 | 2023-10-20 (1d sonra row 4'ten) | 1 | 4 / 4 (row 4 lag içinde, dahil değil) | 4 ✓ |
| 7 | 2023-10-27 (12d sonra row 4'ten) | 1 | 6 / 6 (row 4 ve row 5 artık dahil) | 6 ✓ |

→ **Kendi satır labelı dahil değil. Future labels dahil değil. Lag içindeki labellar dahil değil.**

---

## Test 1 — Permutation

**Yöntem:** Train+val içinde `IsFraudTransaction` shuffle edildi, label-dependent feature'lar
shuffled etiketlerle yeniden hesaplandı, model retrain edildi, test seti (REAL etiketleri)
üzerinde değerlendirildi.

**Beklenti:** Eğer leakage varsa, model shuffle'a rağmen bir şey öğrenir, PR-AUC yüksek kalır.
Leakage yoksa, PR-AUC ≈ test fraud rate (= 0.0021).

**Sonuç:**

| Metrik | Değer |
|---|---:|
| Test PR-AUC | **0.0017** |
| Test fraud rate (taban) | 0.0021 |
| Test Recall@1% | 0.0126 |

→ Şans düzeyi. **Leakage YOK** ✓

---

## Test 2 — Lag sensitivity

**Yöntem:** `label_lag_days` parametresini değiştirip retrain.

| `lag_days` | Test PR-AUC | Test Recall@1% |
|---:|---:|---:|
| 7 | 0.8046 | 0.8992 |
| 30 | 0.8010 | 0.9076 |
| 60 | 0.7428 | 0.8739 |
| 90 | 0.6861 | 0.8529 |

- 7 ↔ 30 günde fark **−0.004 PR-AUC** — pratikte ihmal edilebilir.
- 60+ günde düşüş başlıyor (receiver davranışı zamanla drift ediyor).

→ Sinyal "son haftadaki labellar" değil, **kalıcı receiver davranışı**. Gerçek bankada
30 günlük lag bile bu performansı destekler.

→ **7 günlük lag varsayımı sentetik veride sağlam**; ama production'da bankanın gerçek
chargeback/itiraz süresi (genelde 30+ gün) ölçülüp kullanılmalı. Audit, lag=30'da
kazancın **kaybedilmediğini** gösteriyor — bu önemli bir konfor.

---

## Test 3 — fraud_rate ablation

**Yöntem:** `device_fraud_rate_smoothed`, `receiver_fraud_rate_smoothed`,
`{device,receiver}_label_n` feature'larını çıkar, retrain.

| Feature seti | Test PR-AUC | Test Recall@1% |
|---|---:|---:|
| Tüm label-dependent feature'lar (default) | **0.8046** | 0.8992 |
| fraud_rate kaldırıldı (count + first_seen kaldı) | 0.4002 | 0.7437 |
| Baseline (label-dependent KAPALI) | 0.3575 | 0.7143 |

- **fraud_rate alone tek başına +0.40 PR-AUC** taşıyor.
- fraud_rate olmadan bile (label_n + tx_count + distinct_*) baseline'dan **+0.04** daha iyi.
- Yani kazanç tamamen tek bir feature'a bağlı değil, ama **dominant katkı `receiver_fraud_rate_smoothed`'ten**.

→ fraud_rate kontrollü ve PIT correct. Ama production'da bu feature'a aşırı bağımlılık
**risk**: receiver populasyonu drift ederse model bozulur. Drift monitoring şart.

---

## Test 4 — New vs Seen receiver subset

**Yöntem:** Test seti, train'de görülmüş (seen) ve görülmemiş (new) receiver'lara ayrıldı.

| Subset | n_rows | n_fraud | fraud_rate | PR-AUC |
|---|---:|---:|---:|---:|
| Seen receivers | 132.928 | 145 | 0.11% | **0.826** |
| **NEW receivers** | 25.546 | 93 | 0.36% | **0.781** |

- Eğer model receiver-ismi ezberleseydi, new receivers'da çökerdi. **Düşüş sadece -0.045** PR-AUC.
- Yeni receiver'larda fraud rate (0.36%) bile seen'lerden (0.11%) yüksek — yani new subset
zaten daha zor değil; aksine fraud yoğunluğu nedeniyle teorik olarak daha kolay.
- Model PIT-correct historical aggregate'leri kullanıyor; **NEW receiver'ın geçmişi sıfır olsa
bile, device/IP_Subnet/account behavior'undan sinyal alıyor**.

→ Memorization YOK. Model gerçek davranışsal sinyali kullanıyor.

---

## Sonuç ve uyarılar

### Leakage YOK — performans gerçek

Dört bağımsız test (permutation, lag invariance, ablation, new-receiver subset) bir arada
bakıldığında PR-AUC 0.80'in **leakage değil, kalıcı receiver/device davranış sinyalinden**
geldiği doğrulanıyor.

### Hâlâ var olan riskler (production'a alınmadan önce)

1. **Sentetik veri artefaktı**: Veri jeneratörü "bir receiver fraud yapıyorsa sonra da yapar"
   tarzında güçlü bir autocorrelation kurmuş olabilir. Gerçek dünyada mule account'lar zamanla
   dönüşür — gerçek veride bu sinyalin yarısı kadar olabilir.

2. **Label availability lag**: 7 gün varsayım. Bankada gerçek lag 30+ gün olabilir.
   Test 2 lag=30'da kazancın kaybolmadığını gösteriyor → bu konforlu; ama yine de doğrulanmalı.

3. **Drift**: Test PR-AUC 0.80 — val PR-AUC 0.97'den daha düşük (drift gerçek). Production'da
   drift devam ederse daha da düşebilir. Drift monitoring (PSI, AUC trend, fraud rate trend) şart.

4. **fraud_rate bağımlılığı**: Performansın çoğu bu tek feature ailesinden. Receiver populasyonu
   değişirse model çöker. Periyodik retraining + fallback (label-free feature set) plan B olmalı.

### Skor güvenle raporlanabilir

Mevcut model — hist_gbm, demographic_free, label-dependent FE (lag=7d), tuned hyperparams —
test setinde PR-AUC **0.805**, Recall@1% **0.90**, ROC-AUC **0.98**. Bu rakamlar **leakage-free**
ve final teknik rapora bu güvenle yazılabilir.
