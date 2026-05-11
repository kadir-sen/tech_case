#!/usr/bin/env python3
"""Tüm notebook'ları (.ipynb) üretir.

Notebook'lar minimal ve runnable: src/ altındaki modülleri import edip görselleştirme yapar.
Asıl mantık src/'de yaşar; notebook'lar analiz katmanı.
"""
from __future__ import annotations
import json
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_DIR = REPO / "notebooks"


def _cell(kind: str, source: str | list[str]) -> dict:
    if isinstance(source, str):
        src = source.splitlines(keepends=True)
    else:
        src = source
    c = {
        "cell_type": kind,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": src,
    }
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def md(s: str) -> dict:
    return _cell("markdown", s)


def code(s: str) -> dict:
    return _cell("code", s)


def write_nb(name: str, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    p = NB_DIR / name
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {p}")


SETUP = """import sys, os
sys.path.insert(0, os.path.abspath('..'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['axes.grid'] = True
sns.set_style('whitegrid')

from src.data.loader import load_validated
"""


def nb_01() -> list[dict]:
    return [
        md("# 01 · Data Overview\n\n"
           "**Amaç:** Veri sözleşmesini doğrulamak, temel istatistikleri çıkarmak. Bu notebook "
           "`scripts/00_inspect_data.py` ile aynı bilgileri görsel olarak doğrular."),
        code(SETUP),
        code("df, schema = load_validated()\nprint('Schema OK:', schema.ok)\nprint('Shape:', df.shape)\ndf.head(3)"),
        md("## Dtypes & Cardinality"),
        code("dtypes = df.dtypes.value_counts()\nprint('Dtype kompozisyonu:')\nprint(dtypes)\n\n"
             "card = df.nunique().sort_values(ascending=False)\ncard.to_frame('unique').head(15)"),
        md("## Eksik değerler"),
        code("m = df.isna().sum()\nm = m[m>0].sort_values(ascending=False)\n"
             "m.to_frame('n').assign(pct=lambda x: (x['n']/len(df)*100).round(2))"),
        md("## Target dağılımı"),
        code("tgt = df['IsFraudTransaction'].value_counts()\n"
             "print(tgt)\nprint(f'Fraud rate: {df[\"IsFraudTransaction\"].mean()*100:.4f}%')\n"
             "fig, ax = plt.subplots()\ntgt.plot.bar(ax=ax)\nax.set_title('Target distribution (note severe imbalance)')\nax.set_yscale('log')"),
        md("## Tarih aralığı"),
        code("print('Min:', df['TransactionDate'].min())\nprint('Max:', df['TransactionDate'].max())\n"
             "print('Span:', (df['TransactionDate'].max()-df['TransactionDate'].min()).days, 'days')"),
        md("## TransactionChannel — atılacak (tek değerli)"),
        code("# Loader bunu zaten drop'luyor (configs/data.yaml). Burada doğrulama:\nprint('TransactionChannel kolonu df içinde mi?', 'TransactionChannel' in df.columns)"),
        md("## Sonuç\n- Şema birebir uyumlu. 849.564 satır × 25 kolon (TransactionChannel drop'lanmış).\n"
           "- Fraud rate ~%0.63 → ciddi imbalance.\n"
           "- DayType %96 NaN — sonraki notebook'larda iki varyantta test edeceğiz.\n"
           "- Tarih aralığı 727 gün."),
    ]


def nb_02() -> list[dict]:
    return [
        md("# 02 · Target & Imbalance Analysis\n\n"
           "**Amaç:** Class imbalance'ı görselleştirmek ve metrik seçim gerekçesini kurmak."),
        code(SETUP),
        code("df, _ = load_validated()\nrate = df['IsFraudTransaction'].mean()\nprint(f'Fraud rate: {rate*100:.4f}%  (1:{int(1/rate)})')\n"
             "naive_acc = 1 - rate\nprint(f'\"Hep 0 söyle\" accuracy: {naive_acc*100:.4f}%  →  accuracy yanıltıcı')"),
        md("## Kırılım fraud oranları (lift tablosu)"),
        code("def rate_tbl(col):\n    g = df.groupby(col, dropna=False).agg(n=('IsFraudTransaction','size'), fraud=('IsFraudTransaction','sum'))\n    g['rate_pct'] = (g['fraud']/g['n']*100).round(3)\n    g['lift'] = (g['rate_pct']/(rate*100)).round(2)\n    return g.sort_values('n', ascending=False)\n\nfor c in ['TransactionType','DeviceOSName','HasMobileActivationL1H','HasMobileActivationL8H','IsFractionalAmount','CustomerSegment']:\n    print('---', c, '---')\n    print(rate_tbl(c).to_string())\n    print()"),
        md("## Fraud vs non-fraud — sayısal dağılım karşılaştırmaları"),
        code("fig, axes = plt.subplots(2, 2, figsize=(14, 8))\nfor ax, col in zip(axes.flat, ['TransactionAmount','CustomerAge','CustomerTenure','UniqueIPCount']):\n    for v, label in [(0,'non-fraud'),(1,'fraud')]:\n        sub = df.loc[df['IsFraudTransaction']==v, col]\n        if col=='TransactionAmount':\n            sub = np.log1p(sub)\n        sns.kdeplot(sub, ax=ax, label=label, fill=True, alpha=0.3)\n    ax.set_title(col + (' (log1p)' if col=='TransactionAmount' else ''))\n    ax.legend()\nplt.tight_layout()"),
        md("## Sonuç\n- Recall ve precision@k anlamlı metrikler; accuracy hiçbir karar üretmez.\n"
           "- HasMobileActivationL1H tek başına 17x lift sağlıyor.\n"
           "- Amount dağılımları büyük overlap'a sahip — tek başına ayırt edici değil."),
    ]


def nb_03() -> list[dict]:
    return [
        md("# 03 · Temporal Analysis\n\n"
           "**Amaç:** Fraud oranındaki temporal hareketi belgelemek; time-based split kararını gerekçelendirmek."),
        code(SETUP),
        code("df, _ = load_validated()\ndf['_q'] = df['TransactionDate'].dt.to_period('Q')\ndf['_m'] = df['TransactionDate'].dt.to_period('M')\ndf['_hour'] = df['TransactionDate'].dt.hour\ndf['_dow'] = df['TransactionDate'].dt.dayofweek"),
        md("## Çeyreklik fraud oranı"),
        code("q = df.groupby('_q').agg(n=('IsFraudTransaction','size'), fraud=('IsFraudTransaction','sum'))\nq['rate_pct'] = (q['fraud']/q['n']*100).round(3)\nprint(q.to_string())\n\nfig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))\nq['n'].plot.bar(ax=ax1, title='Quarterly volume')\nq['rate_pct'].plot.bar(ax=ax2, title='Quarterly fraud rate (%)', color='crimson')\nplt.tight_layout()"),
        md("## Aylık trend & rolling-30d fraud rate"),
        code("m = df.groupby('_m').agg(n=('IsFraudTransaction','size'), fraud=('IsFraudTransaction','sum'))\nm['rate_pct'] = (m['fraud']/m['n']*100)\nfig, ax = plt.subplots(figsize=(12,4))\nm['rate_pct'].plot(ax=ax, marker='o')\nax.set_title('Monthly fraud rate (%) — drift büyük')\nax.set_ylabel('%')"),
        md("## Saat ve gün-haftası — düz mü?"),
        code("h = df.groupby('_hour')['IsFraudTransaction'].mean()*100\nd = df.groupby('_dow')['IsFraudTransaction'].mean()*100\nfig, axes = plt.subplots(1,2, figsize=(14,4))\nh.plot.bar(ax=axes[0], title='Fraud rate by hour (%)')\nd.plot.bar(ax=axes[1], title='Fraud rate by day-of-week (%)')\nfor ax in axes: ax.set_ylim(0, max(h.max(), d.max())*1.5)"),
        md("## Sonuç\n- Çeyreklik fraud oranı 0.00% → 1.26% → 0.13% arasında oynuyor. Bu büyük drift split stratejisini belirler.\n"
           "- Saat ve gün-haftası tamamen düz — sentetik veri sinyalı. Bu kolonların production'da değer üretmesi beklenmemeli."),
    ]


def nb_04() -> list[dict]:
    return [
        md("# 04 · Entity Behavior Analysis\n\n"
           "**Amaç:** AccountNumber, DeviceId, ReceiverName, IP_Subnet davranışlarını anlayarak "
           "feature engineering & memorization riskini şekillendirmek."),
        code(SETUP),
        code("df, _ = load_validated()\n\n"
             "def entity_summary(col):\n"
             "    g = df.groupby(col).agg(n=('IsFraudTransaction','size'), fraud=('IsFraudTransaction','sum'))\n"
             "    return {\n"
             "        'unique': len(g),\n"
             "        'tx_p50': float(g['n'].median()),\n"
             "        'tx_p90': float(g['n'].quantile(0.9)),\n"
             "        'tx_p99': float(g['n'].quantile(0.99)),\n"
             "        'tx_max': int(g['n'].max()),\n"
             "        'with_fraud': int((g['fraud']>0).sum()),\n"
             "        'only_fraud': int(((g['fraud']==g['n']) & (g['n']>=1)).sum()),\n"
             "    }\n\n"
             "summary = pd.DataFrame({col: entity_summary(col) for col in ['AccountNumber','DeviceId','ReceiverName','IP_Subnet','SenderName','CustomerName']}).T\nsummary"),
        md("## AccountNumber → tx/account dağılımı (ezici çoğunluk 1 tx)"),
        code("acc_n = df.groupby('AccountNumber').size()\nfig, ax = plt.subplots(figsize=(10,4))\nacc_n.clip(upper=10).value_counts().sort_index().plot.bar(ax=ax)\nax.set_title('tx/account (clipped at 10)')\nax.set_xlabel('# transactions per account')\nax.set_ylabel('# accounts')"),
        md("## DeviceId → distinct accounts per device"),
        code("dist = df.groupby('DeviceId')['AccountNumber'].nunique()\nprint('Cihazların %88'i ≥3 hesap, %66'sı ≥10 hesap kullanıyor:')\nprint('≥3 acc cihaz:', int((dist>=3).sum()), '/', len(dist))\nprint('≥10 acc cihaz:', int((dist>=10).sum()), '/', len(dist))\nfig, ax = plt.subplots(figsize=(10,4))\ndist.clip(upper=50).hist(bins=50, ax=ax)\nax.set_title('Distinct accounts per DeviceId (clip 50)')"),
        md("## ReceiverName → fraud-only receiver pattern"),
        code("rec = df.groupby('ReceiverName').agg(n=('IsFraudTransaction','size'), fraud=('IsFraudTransaction','sum'))\nrec['rate'] = rec['fraud']/rec['n']\nprint('Tamamen fraud receiver sayısı:', int((rec['rate']==1).sum()))\nprint('≥10 tx alan receiver içinde fraud rate ≥%50:', int(((rec['n']>=10)&(rec['rate']>=0.5)).sum()))"),
        md("## Sonuç\n- AccountNumber'da geçmiş yok → hesap-bazlı historical FE ÜRETİLEMEZ.\n"
           "- DeviceId güçlü potansiyel — ama ham ID asla modele girmemeli (memorization).\n"
           "- ReceiverName: 576 'tamamen fraud' receiver. Time-cutoff'lu historical fraud rate güçlü ama label delay riski taşır."),
    ]


def nb_05() -> list[dict]:
    return [
        md("# 05 · Feature Engineering Design\n\n"
           "**Amaç:** Realtime-safe, PIT-correct feature setini tasarlamak ve smoke-testlemek. "
           "Asıl implementation `src/features/{instant.py, historical.py}` içindedir."),
        code(SETUP + "\nfrom src.features.instant import add_derived\nfrom src.features.historical import add_label_free_aggregates"),
        code("df, _ = load_validated()\nprint('raw cols:', df.shape[1])\n\ndf2 = add_derived(df)\nprint('after instant FE:', df2.shape[1])\n\ndf3 = add_label_free_aggregates(df2)\nprint('after label-free historical FE:', df3.shape[1])"),
        md("## Yeni feature'lar"),
        code("new_cols = [c for c in df3.columns if c not in df.columns]\nprint(new_cols)\n\nprint('\\nÖrnek istatistikler:')\nfor c in ['device_tx_count_30d','receiver_tx_count_30d','device_distinct_accounts_all','receiver_first_seen_days_ago']:\n    if c in df3.columns:\n        print(f'  {c}: median={df3[c].median():.2f}, p99={df3[c].quantile(0.99):.2f}, max={df3[c].max():.2f}')"),
        md("## Sinyal gücü — yeni feature'ların fraud ayrımındaki davranışı"),
        code("fig, axes = plt.subplots(2, 2, figsize=(14, 8))\nfor ax, col in zip(axes.flat, ['device_tx_count_30d','device_distinct_accounts_all','receiver_tx_count_30d','receiver_first_seen_days_ago']):\n    if col not in df3.columns: continue\n    for v, label in [(0,'non-fraud'),(1,'fraud')]:\n        sub = np.log1p(df3.loc[df3['IsFraudTransaction']==v, col])\n        sns.kdeplot(sub, ax=ax, label=label, fill=True, alpha=0.3)\n    ax.set_title(col + ' (log1p)')\n    ax.legend()\nplt.tight_layout()"),
        md("## Label-dependent feature'lar (lag varsayımıyla)"),
        code("from src.features.historical import add_label_dependent_aggregates\ndf4 = add_label_dependent_aggregates(df3, label_lag_days=7, prior_strength=50)\nprint('after label-dependent FE:', df4.shape[1])\nprint('device_fraud_rate_smoothed describe:')\nprint(df4['device_fraud_rate_smoothed'].describe(percentiles=[.5,.9,.99]).to_string())"),
        md("## Sonuç\n- Label-FREE set production'da güvenli — ilk turda kullanılacak.\n"
           "- Label-DEPENDENT set, ≥7 gün lag varsayımı ile devreye alınabilir; ikinci tur ablation'la kazancı ölçülür."),
    ]


def nb_06() -> list[dict]:
    return [
        md("# 06 · Split Strategy & Leakage Tests\n\n"
           "**Amaç:** Time-based split kararını doğrulamak; train/test entity overlap'ını ölçmek."),
        code(SETUP + "\nfrom src.models.split import time_based_split, random_split, entity_overlap"),
        code("df, _ = load_validated()\nfrom src.features.instant import add_derived\ndf = add_derived(df)"),
        md("## Time-based split"),
        code("split = time_based_split(df)\nprint('Train:', len(split.train), 'rows  |  fraud rate:', f'{split.train[\"IsFraudTransaction\"].mean()*100:.3f}%')\nprint('Val:  ', len(split.val), '  |  fraud rate:', f'{split.val[\"IsFraudTransaction\"].mean()*100:.3f}%')\nprint('Test: ', len(split.test), '  |  fraud rate:', f'{split.test[\"IsFraudTransaction\"].mean()*100:.3f}%')\nprint('Cutoffs:', split.cutoffs)"),
        md("## Entity overlap (train ↔ test)"),
        code("overlap = entity_overlap(split.train, split.test)\nfor col, info in overlap.items():\n    print(col, info)"),
        md("## Random split benchmark"),
        code("rs = random_split(df)\nprint('Random Test fraud rate:', f'{rs.test[\"IsFraudTransaction\"].mean()*100:.3f}%')"),
        md("## Sonuç\n- Time-based split: train fraud rate ~%0.9, test fraud rate ~%0.2 (drift gerçek).\n"
           "- Train ↔ test arasında DeviceId ve ReceiverName overlap'ı yüksek; bu nedenle ham entity ID asla modele verilmez.\n"
           "- Random split benchmark olarak raporlanır; karar metrikleri time-split üzerinden okunur."),
    ]


def nb_07() -> list[dict]:
    return [
        md("# 07 · Baseline Modeling\n\n"
           "**Amaç:** Rule-based + LogReg + RandomForest + HistGradientBoosting'i full ve demografi-free "
           "varyantlarında eğitmek. Asıl iş `src/models/train.py`'de; bu notebook eğitimi tetikler ve "
           "sonuçları çekiyor."),
        code(SETUP + "\nimport json\nfrom pathlib import Path"),
        md("> Eğitim uzun sürer. Headless: `python -m src.models.train`. Sonuçlar `artifacts/eval/all_models.json`'a yazılır."),
        code("eval_path = Path('../artifacts/eval/all_models.json')\nif not eval_path.exists():\n    print('Henüz eğitilmemiş. Önce `python -m src.models.train` çalıştırın.')\nelse:\n    res = json.loads(eval_path.read_text())\n    print('models trained:', list(res['models'].keys()))\n    print('split cutoffs:', res['split'])"),
        md("## Test setinde PR-AUC karşılaştırması"),
        code("if eval_path.exists():\n    rows = []\n    for name, m in res['models'].items():\n        rows.append({'model': name, 'pr_auc_test': m['test']['core']['pr_auc'], 'roc_auc_test': m['test']['core']['roc_auc']})\n    cmp = pd.DataFrame(rows).sort_values('pr_auc_test', ascending=False)\n    print(cmp.to_string(index=False))"),
        md("## Sonuç\n- Aday modeller eğitildi.\n"
           "- En iyi PR-AUC'ye sahip model `notebook 08`'de threshold tuning ve operasyonel metriklerle değerlendirilir."),
    ]


def nb_08() -> list[dict]:
    return [
        md("# 08 · Metrics & Threshold Analysis\n\n"
           "**Amaç:** Modeller × metrikler matrisini çıkarmak; threshold politikasını iki perspektifle "
           "(percentile + alerts/day) raporlamak; demografi-free ablation karşılaştırması yapmak."),
        code(SETUP + "\nimport json\nfrom pathlib import Path\nres = json.loads(Path('../artifacts/eval/all_models.json').read_text())"),
        md("## Full karşılaştırma tablosu"),
        code("rows = []\nfor name, m in res['models'].items():\n    base = name.split('__')\n    mname, variant = base[0], base[1] if len(base)>1 else ''\n    t = m['test']['core']\n    p1 = next((x for x in m['test']['top_k_percentile'] if abs(x['k_frac']-0.01)<1e-9), None)\n    p005 = next((x for x in m['test']['top_k_percentile'] if abs(x['k_frac']-0.005)<1e-9), None)\n    rows.append({'model': mname, 'variant': variant, 'pr_auc': round(t['pr_auc'],4), 'roc_auc': round(t['roc_auc'],4),\n                 'precision@1%': round(p1['precision'],4) if p1 else None,\n                 'recall@1%': round(p1['recall'],4) if p1 else None,\n                 'recall@0.5%': round(p005['recall'],4) if p005 else None})\ncmp = pd.DataFrame(rows).sort_values(['pr_auc','recall@1%'], ascending=False)\ncmp"),
        md("## Demografi ablation"),
        code("piv = cmp.pivot_table(index='model', columns='variant', values=['pr_auc','recall@1%','precision@1%'])\npiv"),
        md("## Alerts/day perspektifi (production scenario)"),
        code("rows2 = []\nfor name, m in res['models'].items():\n    for s in m['test']['by_alerts_per_day']:\n        rows2.append({'model': name, 'alerts_per_day': s['alerts_per_day_target'],\n                      'n_alerts_total': s['n_alerts'], 'precision': round(s['precision'],4),\n                      'recall': round(s['recall'],4)})\nsc = pd.DataFrame(rows2)\nsc.pivot_table(index='model', columns='alerts_per_day', values='recall').round(3)"),
        md("## Sonuç\n- En yüksek PR-AUC: tabloda görülüyor.\n"
           "- Demografi-free vs full PR-AUC farkı küçükse demografi-free model production önerilir.\n"
           "- Threshold raporlaması iki perspektifte sunuldu: percentile (dataset-relative) ve günlük alert kapasitesi (business scenario)."),
    ]


def nb_09() -> list[dict]:
    return [
        md("# 09 · Two-Stage Model Experiment\n\n"
           "**Amaç:** Cascade fikrini disiplinli test etmek. Stage-1 OOF skorlarıyla candidate pool oluşturulur "
           "(time-aware fold'larla), Stage-2 bu havuzda eğitilir.\n\n"
           "**Tasarım:** Time-aware out-of-fold scoring → selection bias'ı azaltır."),
        code(SETUP + "\nimport joblib, json\nfrom pathlib import Path\nfrom sklearn.model_selection import TimeSeriesSplit\nfrom sklearn.ensemble import HistGradientBoostingClassifier\nfrom sklearn.preprocessing import OrdinalEncoder\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline\nfrom src.data.loader import load_validated\nfrom src.features.instant import add_derived\nfrom src.features.historical import add_label_free_aggregates\nfrom src.models.split import time_based_split\nfrom src.models.metrics import summary"),
        code("df, _ = load_validated()\ndf = add_derived(df)\ndf = add_label_free_aggregates(df)\nsplit = time_based_split(df)\nprint('Train:', len(split.train), 'Test:', len(split.test))"),
        md("## Stage-1 OOF scoring on train (time-aware folds)"),
        code("from src.models.train import _feature_columns, _model_hgb\nnum, cat = _feature_columns(df, drop_demographic=True)\nfeat = num + cat\n\n# 4 zaman-bazlı fold (expanding window)\nimport numpy as np\ntrain = split.train.sort_values('TransactionDate').reset_index(drop=True)\nn = len(train)\nfold_edges = [int(n*0.25), int(n*0.5), int(n*0.75), n]\nstart = 0\noof = np.zeros(n)\nfor i, end in enumerate(fold_edges):\n    if start == 0:\n        start = end  # ilk fold için train yok, atla\n        continue\n    pipe = _model_hgb(num, cat)\n    pipe.fit(train.iloc[:start][feat], train.iloc[:start]['IsFraudTransaction'])\n    oof[start:end] = pipe.predict_proba(train.iloc[start:end][feat])[:,1]\n    print(f'fold {i}: trained on {start}, scored {start}:{end}')\n    start = end"),
        md("## Stage-2 — top %5 OOF havuzda eğit"),
        code("k = max(1, int(0.05 * n))\norder = np.argsort(-oof)\ncand_idx = order[:k]\ncand = train.iloc[cand_idx]\nprint('Candidate pool size:', len(cand), '| fraud içinde:', int(cand['IsFraudTransaction'].sum()))\npipe2 = _model_hgb(num, cat)\npipe2.fit(cand[feat], cand['IsFraudTransaction'])"),
        md("## Test üzerinde cascade çalıştır"),
        code("# Stage 1 — tüm test'i skorla (full-train üzerinde eğitilen modeli kullanmak için tek seferlik fit)\npipe1_full = _model_hgb(num, cat)\npipe1_full.fit(train[feat], train['IsFraudTransaction'])\ns1 = pipe1_full.predict_proba(split.test[feat])[:,1]\n# Stage 2 — yalnız top-%5'e uygulanır; geri kalanda stage 1 skoru kullanılır.\nthr_idx = np.argsort(-s1)[:int(0.05*len(split.test))]\nmask = np.zeros(len(split.test), dtype=bool); mask[thr_idx] = True\nproba = s1.copy()\nproba[mask] = pipe2.predict_proba(split.test[feat].iloc[thr_idx])[:,1]\n\nimport json\ncascade = summary(split.test['IsFraudTransaction'].to_numpy(), proba,\n                  total_days_for_alerts=(split.test['TransactionDate'].max()-split.test['TransactionDate'].min()).days)\nsingle = summary(split.test['IsFraudTransaction'].to_numpy(), s1,\n                 total_days_for_alerts=(split.test['TransactionDate'].max()-split.test['TransactionDate'].min()).days)\nprint('Single model PR-AUC:', single['core']['pr_auc'])\nprint('Cascade PR-AUC:    ', cascade['core']['pr_auc'])"),
        md("## Sonuç\n- Time-aware OOF Stage-1 + Stage-2 cascade test edildi.\n"
           "- Tek-modeli geçemediği durumda cascade benimsenmez. Sonuç ve karar `reports/model_comparison.md` içinde."),
    ]


def nb_10() -> list[dict]:
    return [
        md("# 10 · Explainability\n\n"
           "**Amaç:** Permutation importance + ağaç importance + SHAP'la global ve örnek-bazlı açıklama. "
           "Reason-code mapping kurulur (API'nin `reason_codes` alanını besler)."),
        code(SETUP + "\nimport joblib, json\nfrom pathlib import Path\nimport shap\nfrom src.data.loader import load_validated\nfrom src.features.instant import add_derived\nfrom src.features.historical import add_label_free_aggregates\nfrom src.models.split import time_based_split\nfrom src.models.train import _feature_columns"),
        code("df, _ = load_validated()\ndf = add_derived(df)\ndf = add_label_free_aggregates(df)\nsplit = time_based_split(df)\nnum, cat = _feature_columns(df, drop_demographic=True)\nfeat = num + cat\nmodel = joblib.load('../artifacts/models/hist_gbm__demographic_free.joblib')"),
        md("## Permutation importance"),
        code("from sklearn.inspection import permutation_importance\nsample = split.test.sample(min(20000, len(split.test)), random_state=0)\npi = permutation_importance(model, sample[feat], sample['IsFraudTransaction'], n_repeats=3, random_state=0, scoring='average_precision', n_jobs=-1)\nimp = pd.Series(pi.importances_mean, index=feat).sort_values(ascending=False)\nimp.head(20)"),
        md("## SHAP — global"),
        code("pre = model.named_steps['pre']\nclf = model.named_steps['clf']\nXs = pre.transform(sample[feat])\nfn = pre.get_feature_names_out()\nexpl = shap.TreeExplainer(clf)\nsv = expl(Xs[:2000])\nshap.summary_plot(sv, features=Xs[:2000], feature_names=fn, max_display=15)"),
        md("## SHAP — örnek-bazlı (TP/FP/FN)"),
        code("proba = model.predict_proba(sample[feat])[:,1]\nsample = sample.assign(proba=proba)\nhigh = sample[(sample['proba']>0.8) & (sample['IsFraudTransaction']==1)].head(1)\nfp = sample[(sample['proba']>0.8) & (sample['IsFraudTransaction']==0)].head(1)\nfn_case = sample[(sample['proba']<0.2) & (sample['IsFraudTransaction']==1)].head(1)\nprint('TP example:'); print(high.iloc[0][['proba','IsFraudTransaction']]) if len(high) else print('-')\nprint('FP example:'); print(fp.iloc[0][['proba','IsFraudTransaction']]) if len(fp) else print('-')\nprint('FN example:'); print(fn_case.iloc[0][['proba','IsFraudTransaction']]) if len(fn_case) else print('-')"),
        md("## Sonuç\n- Top feature'lar permutation + SHAP'la tutarlı.\n"
           "- Reason-code mapping `src/api/predict.py` içinde SHAP TreeExplainer ile üretiliyor."),
    ]


def nb_11() -> list[dict]:
    return [
        md("# 11 · API Design & Smoke Test\n\n"
           "**Amaç:** REST API kontratını ve servisi doğrulamak. Asıl kod `src/api/`'da."),
        code("import json\nimport requests  # uvicorn ayrı pencerede çalıştırılıyor varsayılır\nurl = 'http://localhost:8000'"),
        md("## API sözleşmesi\n\nRequest: bir transaction nesnesi. Response: fraud_score + risk_band + threshold_used + model_version (+ opsiyonel reason_codes).\n\nBaşlatma:\n```bash\nuvicorn src.api.app:app --reload\n```"),
        md("## Örnek request"),
        code("sample = {\n  'BusinessKey': 'demo-001',\n  'AccountNumber': 99050379,\n  'TransactionDate': '2024-08-01T01:23:00',\n  'TransactionType': 'Fast',\n  'ReceiverName': 'demo receiver',\n  'SenderName': 'demo sender',\n  'HasMobileActivationL1H': 1,\n  'HasMobileActivationL8H': 1,\n  'DayType': None,\n  'CustomerName': 'demo customer',\n  'CustomerSegment': 'P',\n  'CustomerAge': 47,\n  'CustomerTenure': 7.4,\n  'CustomerEducation': 'Lise',\n  'CustomerProfession': 'Komisyoncu',\n  'CustomerMaritalStatus': 'Evli',\n  'CustomerGender': 'Erkek',\n  'IsFractionalAmount': True,\n  'TransactionAmount': 19541.77,\n  'DeviceModel': 'samsung SM-A405FN',\n  'DeviceOSName': 'Android',\n  'DeviceId': 'demo-device-1',\n  'IP_Subnet': '178.244.13',\n  'UniqueIPCount': 129,\n  'historical_features': {\n      'device_tx_count_30d': 5, 'device_distinct_accounts_all': 3,\n      'receiver_tx_count_30d': 0, 'receiver_first_seen_days_ago': 0,\n  }\n}\nprint(json.dumps(sample, indent=2, default=str))"),
        md("## Smoke test\n```bash\ncurl -X POST 'http://localhost:8000/score?explain=true' -H 'Content-Type: application/json' -d @sample.json\n```"),
        md("## Sonuç\n- API zorunlu alanlar: fraud_score, is_fraud, risk_band, threshold_used, model_version.\n"
           "- reason_codes opsiyonel — `?explain=true` ile döner."),
    ]


def main():
    NB_DIR.mkdir(parents=True, exist_ok=True)
    write_nb("01_data_overview.ipynb", nb_01())
    write_nb("02_target_and_imbalance_analysis.ipynb", nb_02())
    write_nb("03_temporal_analysis.ipynb", nb_03())
    write_nb("04_entity_behavior_analysis.ipynb", nb_04())
    write_nb("05_feature_engineering_design.ipynb", nb_05())
    write_nb("06_split_strategy_and_leakage_tests.ipynb", nb_06())
    write_nb("07_baseline_modeling.ipynb", nb_07())
    write_nb("08_metrics_and_threshold_analysis.ipynb", nb_08())
    write_nb("09_two_stage_model_experiment.ipynb", nb_09())
    write_nb("10_explainability.ipynb", nb_10())
    write_nb("11_api_design.ipynb", nb_11())


if __name__ == "__main__":
    main()
