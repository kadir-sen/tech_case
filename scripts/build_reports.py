#!/usr/bin/env python3
"""Eğitim çıktılarından markdown raporları üretir.

Girdi: artifacts/eval/all_models.json
Çıktılar:
  reports/model_comparison.md
  reports/threshold_analysis.md
  reports/final_report.md
  artifacts/score_cutoffs.json   (API'nin band kararı için kullandığı kesim noktaları)
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def fmt(x, n=4):
    if isinstance(x, float):
        return f"{x:.{n}f}"
    return str(x)


def pct(x, n=4):
    return f"{x*100:.{n-2}f}%"


def load_eval() -> dict:
    return json.loads((REPO / "artifacts" / "eval" / "all_models.json").read_text())


def build_model_comparison(res: dict) -> str:
    lines = []
    lines.append("# Model Comparison\n")
    lines.append("> Auto-generated from `artifacts/eval/all_models.json`.\n")
    lines.append(f"## Split (time-based)\n```\n{json.dumps(res['split'], indent=2)}\n```\n")
    lines.append("## Train ↔ Test entity overlap\n")
    lines.append("| Entity | unique (train) | unique (test) | overlap | % of test seen in train |")
    lines.append("|---|---:|---:|---:|---:|")
    for c, info in res["entity_overlap_train_vs_test"].items():
        lines.append(f"| {c} | {info['n_unique_a']:,} | {info['n_unique_b']:,} | {info['n_overlap']:,} | {info['pct_of_b_seen_in_a']}% |")
    lines.append("")

    lines.append("## Test seti — tüm modeller × tüm metrikler\n")
    lines.append("| Model | Variant | PR-AUC | ROC-AUC | Recall@0.1% | Recall@0.5% | Recall@1% | Recall@5% | Precision@1% |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    def get_at_k(test_metrics, k_frac):
        for x in test_metrics["top_k_percentile"]:
            if abs(x["k_frac"] - k_frac) < 1e-9:
                return x
        return None

    rows = []
    for name, m in res["models"].items():
        parts = name.split("__")
        model_name, variant = parts[0], parts[1] if len(parts) > 1 else ""
        t = m["test"]["core"]
        r01 = get_at_k(m["test"], 0.001)
        r05 = get_at_k(m["test"], 0.005)
        r1 = get_at_k(m["test"], 0.01)
        r5 = get_at_k(m["test"], 0.05)
        rows.append((model_name, variant, t["pr_auc"], t["roc_auc"],
                     r01["recall"] if r01 else 0,
                     r05["recall"] if r05 else 0,
                     r1["recall"] if r1 else 0,
                     r5["recall"] if r5 else 0,
                     r1["precision"] if r1 else 0))
    rows.sort(key=lambda r: -r[2])
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {fmt(r[2])} | {fmt(r[3])} | "
                     f"{fmt(r[4])} | {fmt(r[5])} | {fmt(r[6])} | {fmt(r[7])} | {fmt(r[8])} |")
    lines.append("")

    lines.append("## Validation seti\n")
    lines.append("| Model | Variant | PR-AUC | ROC-AUC | Recall@1% | Precision@1% |")
    lines.append("|---|---|---:|---:|---:|---:|")
    vrows = []
    for name, m in res["models"].items():
        parts = name.split("__")
        model_name, variant = parts[0], parts[1] if len(parts) > 1 else ""
        v = m["val"]["core"]
        r1 = get_at_k(m["val"], 0.01)
        vrows.append((model_name, variant, v["pr_auc"], v["roc_auc"],
                      r1["recall"] if r1 else 0, r1["precision"] if r1 else 0))
    vrows.sort(key=lambda r: -r[2])
    for r in vrows:
        lines.append(f"| {r[0]} | {r[1]} | {fmt(r[2])} | {fmt(r[3])} | {fmt(r[4])} | {fmt(r[5])} |")
    lines.append("")

    # Demografi ablation karşılaştırması
    lines.append("## Demografi-Free vs Full ablation (test PR-AUC)\n")
    pivot = {}
    for name, m in res["models"].items():
        parts = name.split("__")
        if len(parts) < 2: continue
        mn, variant = parts[0], parts[1]
        pivot.setdefault(mn, {})[variant] = m["test"]["core"]["pr_auc"]
    lines.append("| Model | Full PR-AUC | Demografi-Free PR-AUC | Δ |")
    lines.append("|---|---:|---:|---:|")
    for mn, d in pivot.items():
        f_ = d.get("full", float("nan"))
        df_ = d.get("demographic_free", float("nan"))
        try:
            delta = df_ - f_
            lines.append(f"| {mn} | {fmt(f_)} | {fmt(df_)} | {delta:+.4f} |")
        except Exception:
            lines.append(f"| {mn} | {fmt(f_)} | {fmt(df_)} | n/a |")
    lines.append("")

    # En iyi model
    best = rows[0]
    lines.append(f"## Önerilen model\n\n**`{best[0]}__{best[1]}`** — Test PR-AUC = **{fmt(best[2])}**, "
                 f"Recall@1% = **{fmt(best[6])}** (Precision@1% = {fmt(best[8])}).\n")
    if pivot.get(best[0], {}).get("full") and pivot.get(best[0], {}).get("demographic_free"):
        delta = pivot[best[0]]["demographic_free"] - pivot[best[0]]["full"]
        if abs(delta) < 0.02:
            lines.append("Demografi-free varyant ile fark < 0.02 PR-AUC: **demografi-free model production'a önerilir** (etik/regülatör avantajı).\n")
        else:
            lines.append(f"Demografi-free varyant arasında Δ={delta:+.4f} — fark anlamlı; karar iş/etik tarafıyla netleştirilmeli.\n")
    return "\n".join(lines) + "\n"


def build_threshold_analysis(res: dict) -> str:
    lines = []
    lines.append("# Threshold Analysis\n")
    lines.append("> Auto-generated. İki perspektif: (a) dataset-relative percentile, (b) business-scenario alerts/day.\n")

    rows = []
    for name, m in res["models"].items():
        for x in m["test"]["top_k_percentile"]:
            rows.append({"model": name, "perspective": "percentile", "k": x["k_frac"],
                         "n_alerts": x["n_alerts"], "threshold": x["threshold"],
                         "precision": x["precision"], "recall": x["recall"],
                         "fraud_capture_pct": x["fraud_capture_pct"]})
        for x in m["test"]["by_alerts_per_day"]:
            rows.append({"model": name, "perspective": "alerts_per_day",
                         "k": f"{x['alerts_per_day_target']}/day",
                         "n_alerts": x["n_alerts"], "threshold": x["threshold"],
                         "precision": x["precision"], "recall": x["recall"],
                         "fraud_capture_pct": x["fraud_capture_pct"]})

    lines.append("## Perspektif 1: Dataset-relative percentile (test set)\n")
    lines.append("| Model | k | n_alerts | threshold | precision | recall | fraud_capture |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["perspective"] != "percentile": continue
        lines.append(f"| {r['model']} | {r['k']} | {r['n_alerts']:,} | {fmt(r['threshold'])} | "
                     f"{fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['fraud_capture_pct'],2)}% |")
    lines.append("")

    lines.append("## Perspektif 2: Business scenario (alerts/day) — test seti hacmine göre\n")
    lines.append("| Model | alerts/day target | n_alerts (test) | threshold | precision | recall |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in rows:
        if r["perspective"] != "alerts_per_day": continue
        lines.append(f"| {r['model']} | {r['k']} | {r['n_alerts']:,} | {fmt(r['threshold'])} | "
                     f"{fmt(r['precision'])} | {fmt(r['recall'])} |")
    lines.append("")

    lines.append("## 3-Bant politika önerisi\n"
                 "- **HIGH**: top %0.1 (otomatik blok / step-up auth).\n"
                 "- **MEDIUM**: top %0.1 - %1 arası (manuel inceleme kuyruğu).\n"
                 "- **LOW**: geri kalan (sadece logla).\n\n"
                 "Production trafik hacmi netleşince iş tarafıyla birlikte percentile vs alerts/day "
                 "perspektifinden son threshold seçilir.\n")
    return "\n".join(lines) + "\n"


def build_final_report(res: dict, comparison_md: str, threshold_md: str) -> str:
    best_name = None; best_pr = -1
    for name, m in res["models"].items():
        if m["test"]["core"]["pr_auc"] > best_pr:
            best_pr = m["test"]["core"]["pr_auc"]
            best_name = name

    lines = []
    lines.append("# Para Transferi Sahtecilik Tespiti — Final Teknik Rapor\n")
    lines.append("> Sentetik veri (`synthetic_data.parquet`) üzerinde uçtan uca proje raporu.\n")

    lines.append("## 1. Problem ve veri özeti\n"
                 "Banka Mobil & İnternet Bankacılığı para transferlerinde fraud tespiti. 849.564 işlem, "
                 "%0.63 fraud rate, 2022-09 → 2024-09 dönemi. Fraud oranı zaman içinde 30× oynuyor — "
                 "split stratejisinde ve performans yorumunda belirleyici unsur.\n")

    lines.append("## 2. Yaklaşım özeti\n"
                 "- Time-based split (train ≤ 2024-03-31, val 2024-04-05, test 2024-06-09).\n"
                 "- Feature engineering: anlık + point-in-time historical aggregate'ler (label-free öncelikli; "
                 "label-dependent ikinci turda lag varsayımıyla).\n"
                 "- 4 aday model: rule-based baseline, Logistic Regression, Random Forest, HistGradientBoosting.\n"
                 "- Her ML modeli iki varyantta: 'full' ve 'demografi-free'.\n"
                 "- Threshold analizi iki perspektifte: dataset-relative percentile + business-scenario alerts/day.\n")

    lines.append("## 3. Feature engineering kararları\n"
                 "- **Label-FREE (üretim güvenli):** device_tx_count_{1d,7d,30d}, "
                 "device_distinct_accounts, receiver_tx_count_{7d,30d}, subnet_tx_count_7d, "
                 "first_seen_days_ago feature'ları.\n"
                 "- **Label-DEPENDENT (label availability lag varsayımıyla):** "
                 "device/receiver_fraud_rate_smoothed (Bayesian smoothing, prior_strength=50, lag=7 gün).\n"
                 "- Ham ID'ler (DeviceId, ReceiverName, AccountNumber, IP_Subnet) modele girmiyor — "
                 "yalnız davranışsal türevler.\n"
                 "- DeviceModel top-100 + 'Other' bucket'landı (988 unique → 101 unique).\n"
                 "- DayType NaN → 'Unknown' kategorisi (alternatif 'Normal' yorumu da test edildi).\n")

    lines.append("## 4. Split ve leakage savunması\n"
                 f"- {comparison_md.split('## Split (time-based)')[1].split('##')[0].strip()}\n"
                 "- Hesapların %97'si tek-tx → AccountNumber kontaminasyonu doğal olarak yok.\n"
                 "- DeviceId / ReceiverName / IP_Subnet train↔test arasında ciddi overlap (tablo aşağıda) — "
                 "bu yüzden ham ID modele girmiyor; PIT historical aggregate'ler kullanılıyor.\n"
                 "- Label-dependent feature'lar yalnız ≥7 gün önceki etiketlerden — production label availability lag'ini taklit eder.\n")

    lines.append("## 5. Model karşılaştırma\n\n")
    lines.append("Detaylı tablo: `reports/model_comparison.md`. Özet:\n\n")
    lines.append(f"- **En iyi model:** `{best_name}` (Test PR-AUC = {best_pr:.4f}).\n\n")

    lines.append("## 6. Threshold politikası\n\n"
                 "Detaylı tablo: `reports/threshold_analysis.md`.\n"
                 "- HIGH = top %0.1, MEDIUM = top %0.1-%1, LOW = geri kalan.\n"
                 "- Production trafik hacmi belirsiz olduğundan eşik haritası iki perspektifte sunuldu.\n")

    lines.append("## 7. Açıklanabilirlik\n"
                 "- Permutation importance + SHAP TreeExplainer.\n"
                 "- API `?explain=true` ile top-5 reason code döner; default kapalı (latency koruma).\n")

    lines.append("## 8. Limitasyonlar — açık dille\n"
                 "1. **Sentetik veri**: Tek kaynak `synthetic_data.parquet`. Bütün performans rakamları "
                 "bu veriye özgü; gerçek banking populasyonuna transfer iddiası YAPILMIYOR.\n"
                 "2. **Saatlik / gün-haftası fraud rate'leri düz**: Gerçek hayatta beklenen örüntü yok — "
                 "sentetik jenerasyon artefaktı muhtemel.\n"
                 "3. **Hesap başına ~1 tx**: Gerçek hayatta hesap-bazlı behavioral feature'lar güçlü "
                 "olur; bu veride üretilemedi.\n"
                 "4. **Çeyreklik fraud oranındaki 30× hareket**: Nedeni veriden çıkarılamadı "
                 "(temporal drift / label policy / synthetic artifact / operasyonel müdahale ihtimalleri açık).\n"
                 "5. **Demografi**: Etik/regülatör boyutu var; demografi-free varyantı tercih edilmesi önerilir.\n"
                 "6. **Label availability**: T+N gün lag varsayımı — production'da ölçülüp doğrulanmalı.\n"
                 "7. **Cihaz/receiver overlap**: Train↔test entity overlap yüksek; ham ID modele girmemekle "
                 "kontrol edildi ama production'da yeni cihaz dağılımı modeli zorlar.\n")

    lines.append("## 9. Next steps\n"
                 "- Gerçek veride pilot çalışma.\n"
                 "- Drift monitoring (PSI, AUC drift, fraud rate trend).\n"
                 "- Graph features (receiver-sender bipartite graph).\n"
                 "- Online learning / sürekli yeniden eğitim politikası.\n"
                 "- Label availability ölçümü ve label-dependent feature'ların kazancının periyodik doğrulanması.\n"
                 "- SHAP reason-code mapping'in iş ekibiyle dilsel olarak gözden geçirilmesi.\n")

    return "\n".join(lines) + "\n"


def build_score_cutoffs(res: dict) -> dict:
    """En iyi modelin (en yüksek test PR-AUC, demografi-free öncelikli) HIGH/MEDIUM cut'larını döner."""
    candidates = {n: m for n, m in res["models"].items() if "demographic_free" in n and "rule_based" not in n}
    if not candidates:
        candidates = {n: m for n, m in res["models"].items() if "rule_based" not in n}
    best_name = max(candidates, key=lambda n: candidates[n]["test"]["core"]["pr_auc"])
    val = candidates[best_name]["val"]
    high = next((x for x in val["top_k_percentile"] if abs(x["k_frac"] - 0.001) < 1e-9), None)
    medium = next((x for x in val["top_k_percentile"] if abs(x["k_frac"] - 0.01) < 1e-9), None)
    return {
        "best_model": best_name,
        "HIGH": float(high["threshold"]) if high else 0.95,
        "MEDIUM": float(medium["threshold"]) if medium else 0.5,
    }


def main():
    res = load_eval()
    comp = build_model_comparison(res)
    thr = build_threshold_analysis(res)
    final = build_final_report(res, comp, thr)
    cuts = build_score_cutoffs(res)

    (REPO / "reports" / "model_comparison.md").write_text(comp, encoding="utf-8")
    (REPO / "reports" / "threshold_analysis.md").write_text(thr, encoding="utf-8")
    (REPO / "reports" / "final_report.md").write_text(final, encoding="utf-8")
    (REPO / "artifacts" / "score_cutoffs.json").write_text(json.dumps(cuts, indent=2))
    print("Reports + cutoffs written.")


if __name__ == "__main__":
    main()
