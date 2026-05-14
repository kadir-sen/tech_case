# Para Transferi Sahtecilik Tespiti (Fraud Detection)

Banka Mobil & İnternet Bankacılığı para transferi işlemleri için fraud tespit modeli ve REST API.

Teknik rapor: `reports/tech_case_teknik_rapor.md`.

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Veri dosyasını (`synthetic_data.parquet`) repo köküne yerleştirin.

## Çalıştırma Akışı

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
   notebooks/07_baseline_modeling.ipynb
   notebooks/08_metrics_and_threshold_analysis.ipynb
   notebooks/09_two_stage_model_experiment.ipynb
   notebooks/10_explainability.ipynb
   notebooks/11_api_design.ipynb
   ```

3. **Pipeline (headless):**

   ```bash
   # HPO (Optuna, train+val üzerinde)
   python scripts/tune_all_time_aware.py

   # Feature selection (permutation + drift + SHAP)
   python scripts/feature_selection_time_aware.py

   # 4 model × 4 feature set final eğitim
   python scripts/train_final_candidates.py

   # Threshold + calibration analizi
   python scripts/threshold_and_calibration_analysis.py

   # Final model explainability
   python scripts/explain_final_model.py
   ```

   Tek model baseline eğitimi için:

   ```bash
   python -m src.models.train
   ```

4. **API:**

   ```bash
   uvicorn src.api.app:app --reload
   ```

   `POST /score` ile işlem skorlanır; `?explain=true` ile SHAP reason code'ları döner.

## Repo Yapısı

```
configs/        # data, features, split, thresholds — YAML config
src/data/       # parquet loader + şema doğrulama
src/features/   # anlık + historical (PIT) feature pipeline
src/models/     # split, train, metrics, time validation
src/api/        # FastAPI servisi (app, predict, schemas)
scripts/        # headless pipeline script'leri
notebooks/      # EDA + modelleme analizi notebook'ları
artifacts/      # eğitilmiş model + feature listesi + threshold + metrikler
reports/        # tech_case_teknik_rapor.md + initial_data_profile.md
```

## Final Model

- **Tip:** CatBoostClassifier (`auto_class_weights="Balanced"`)
- **Parametreler:** iterations=600, learning_rate=0.05, depth=8, l2_leaf_reg=3.0
- **Feature set:** `selected_by_permutation` — 30 özellik
- **Demografi:** dışlandı (`demographic_excluded=True`)
- **Train range:** 2022-09-17 → 2024-03-31
- **Validation range:** 2024-04-01 → 2024-05-31
- **Test range:** 2024-06-01 → 2024-09-30
- **Val PR-AUC:** 0.9621 · **Test PR-AUC:** 0.8417 · **Test Recall@1%:** 0.9244

Tüm detaylar: `reports/tech_case_teknik_rapor.md`.
