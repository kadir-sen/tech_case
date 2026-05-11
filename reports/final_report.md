# Para Transferi Sahtecilik Tespiti — Final Teknik Rapor

> Sentetik veri (`synthetic_data.parquet`) üzerinde uçtan uca proje raporu.

## 1. Problem ve veri özeti
Banka Mobil & İnternet Bankacılığı para transferlerinde fraud tespiti. 849.564 işlem, %0.63 fraud rate, 2022-09 → 2024-09 dönemi. Fraud oranı zaman içinde 30× oynuyor — split stratejisinde ve performans yorumunda belirleyici unsur.

## 2. Yaklaşım özeti
- Time-based split (train ≤ 2024-03-31, val 2024-04-05, test 2024-06-09).
- Feature engineering: anlık + point-in-time historical aggregate'ler (label-free öncelikli; label-dependent ikinci turda lag varsayımıyla).
- 4 aday model: rule-based baseline, Logistic Regression, Random Forest, HistGradientBoosting.
- Her ML modeli iki varyantta: 'full' ve 'demografi-free'.
- Threshold analizi iki perspektifte: dataset-relative percentile + business-scenario alerts/day.

## 3. Feature engineering kararları
- **Label-FREE (üretim güvenli):** device_tx_count_{1d,7d,30d}, device_distinct_accounts, receiver_tx_count_{7d,30d}, subnet_tx_count_7d, first_seen_days_ago feature'ları.
- **Label-DEPENDENT (label availability lag varsayımıyla):** device/receiver_fraud_rate_smoothed (Bayesian smoothing, prior_strength=50, lag=7 gün).
- Ham ID'ler (DeviceId, ReceiverName, AccountNumber, IP_Subnet) modele girmiyor — yalnız davranışsal türevler.
- DeviceModel top-100 + 'Other' bucket'landı (988 unique → 101 unique).
- DayType NaN → 'Unknown' kategorisi (alternatif 'Normal' yorumu da test edildi).

## 4. Split ve leakage savunması
- ```
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
- Hesapların %97'si tek-tx → AccountNumber kontaminasyonu doğal olarak yok.
- DeviceId / ReceiverName / IP_Subnet train↔test arasında ciddi overlap (tablo aşağıda) — bu yüzden ham ID modele girmiyor; PIT historical aggregate'ler kullanılıyor.
- Label-dependent feature'lar yalnız ≥7 gün önceki etiketlerden — production label availability lag'ini taklit eder.

## 5. Model karşılaştırma


Detaylı tablo: `reports/model_comparison.md`. Özet:


- **En iyi model:** `hist_gbm__full` (Test PR-AUC = 0.7718).


## 6. Threshold politikası

Detaylı tablo: `reports/threshold_analysis.md`.
- HIGH = top %0.1, MEDIUM = top %0.1-%1, LOW = geri kalan.
- Production trafik hacmi belirsiz olduğundan eşik haritası iki perspektifte sunuldu.

## 7. Açıklanabilirlik
- Permutation importance + SHAP TreeExplainer.
- API `?explain=true` ile top-5 reason code döner; default kapalı (latency koruma).

## 8. Limitasyonlar — açık dille
1. **Sentetik veri**: Tek kaynak `synthetic_data.parquet`. Bütün performans rakamları bu veriye özgü; gerçek banking populasyonuna transfer iddiası YAPILMIYOR.
2. **Saatlik / gün-haftası fraud rate'leri düz**: Gerçek hayatta beklenen örüntü yok — sentetik jenerasyon artefaktı muhtemel.
3. **Hesap başına ~1 tx**: Gerçek hayatta hesap-bazlı behavioral feature'lar güçlü olur; bu veride üretilemedi.
4. **Çeyreklik fraud oranındaki 30× hareket**: Nedeni veriden çıkarılamadı (temporal drift / label policy / synthetic artifact / operasyonel müdahale ihtimalleri açık).
5. **Demografi**: Etik/regülatör boyutu var; demografi-free varyantı tercih edilmesi önerilir.
6. **Label availability**: T+N gün lag varsayımı — production'da ölçülüp doğrulanmalı.
7. **Cihaz/receiver overlap**: Train↔test entity overlap yüksek; ham ID modele girmemekle kontrol edildi ama production'da yeni cihaz dağılımı modeli zorlar.

## 9. Next steps
- Gerçek veride pilot çalışma.
- Drift monitoring (PSI, AUC drift, fraud rate trend).
- Graph features (receiver-sender bipartite graph).
- Online learning / sürekli yeniden eğitim politikası.
- Label availability ölçümü ve label-dependent feature'ların kazancının periyodik doğrulanması.
- SHAP reason-code mapping'in iş ekibiyle dilsel olarak gözden geçirilmesi.

