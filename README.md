# Para Transferi Sahtecilik Tespiti (Fraud Detection)

Banka Mobil & İnternet Bankacılığı para transferi işlemleri için gerçek zamana yakın fraud
tespit modeli ve REST API.

> **Veri uyarısı:** Tek veri kaynağı `synthetic_data.parquet`. Tüm sonuçlar bu sentetik veri
> üzerindedir; gerçek üretim performansı bu çalışmadan kestirilemez. Limitasyonlar için
> `reports/final_report.md`'a bakın.

## Hızlı kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Çalıştırma akışı

1. **Veri profili:**

   ```bash
   python scripts/00_inspect_data.py
   ```
   Çıktı: `reports/initial_data_profile.md` + `artifacts/data_profile.json`.

2. **EDA & FE notebookları (sırayla):**

   ```
   notebooks/01_data_overview.ipynb
   notebooks/02_target_and_imbalance_analysis.ipynb
   notebooks/03_temporal_analysis.ipynb
   notebooks/04_entity_behavior_analysis.ipynb
   notebooks/05_feature_engineering_design.ipynb
   notebooks/06_split_strategy_and_leakage_tests.ipynb
   ```

3. **Modelleme ve değerlendirme:**

   ```
   notebooks/07_baseline_modeling.ipynb
   notebooks/08_metrics_and_threshold_analysis.ipynb
   notebooks/09_two_stage_model_experiment.ipynb
   notebooks/10_explainability.ipynb
   ```

   Veya headless:
   ```bash
   python -m src.models.train
   python -m src.models.evaluate
   ```

4. **API:**

   ```bash
   uvicorn src.api.app:app --reload
   ```
   `POST /score` ile bir işlem nesnesi gönderilir, fraud skoru + risk band dönülür.
   SHAP açıklamaları için `?explain=true`.

## Repo yapısı

```
configs/        # data, features, split, thresholds — YAML config
scripts/        # headless runnable script'ler
src/data/       # parquet loader + şema doğrulama
src/features/   # anlık + historical (PIT) feature pipeline
src/models/     # train, evaluate, registry
src/api/        # FastAPI servisi
artifacts/      # eğitimli model + encoder + profile JSON
notebooks/      # analiz ve karşılaştırma notebook'ları
reports/        # markdown raporlar (data profile, model comparison, final)
```

## Proje planı

Detaylı proje planı: `/Users/kadirsen/.claude/plans/sen-senior-fraud-detection-hashed-pizza.md`.

## Lisans / Veri

Sentetik veri üzerinde araştırma amaçlı. İsim alanları gerçek müşterileri temsil etmez.
KVKK uyumluluğu üretime alınmadan önce yeniden değerlendirilmelidir.
