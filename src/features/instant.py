"""Anlık (point-in-time bağımsız) feature dönüşümleri.

Gerçek zamanlı inference'da tek bir transaction objesinden hesaplanabilen feature'lar.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


DEVICE_MODEL_TOP_N = 100
DEVICE_MODEL_OTHER = "_OTHER_"


def _bucket_high_cardinality(s: pd.Series, top_n: int, other_label: str) -> pd.Series:
    counts = s.value_counts(dropna=False)
    keep = set(counts.head(top_n).index)
    return s.where(s.isin(keep), other_label)


def _device_parent_brand(model: str) -> str:
    """DeviceModel string'inden parent brand çıkar (samsung / iPhone / Xiaomi / ...)."""
    if not isinstance(model, str):
        return "Unknown"
    m = model.lower()
    if "iphone" in m: return "iPhone"
    if "samsung" in m or m.startswith("sm-"): return "Samsung"
    if "redmi" in m or "xiaomi" in m or m.startswith("mi "): return "Xiaomi"
    if "huawei" in m or m.startswith("hua"): return "Huawei"
    if "oppo" in m: return "Oppo"
    if "vivo" in m: return "Vivo"
    if "realme" in m: return "Realme"
    if "google" in m or "pixel" in m: return "Google"
    if "oneplus" in m: return "OnePlus"
    if "lg" in m: return "LG"
    if "honor" in m: return "Honor"
    if "tcl" in m: return "TCL"
    return "Other"


def add_derived(df: pd.DataFrame, daytype_nan: str = "unknown") -> pd.DataFrame:
    """Türev anlık feature'ları ekler.

    daytype_nan: 'normal' veya 'unknown' — bkz. configs/features.yaml.
    Yan etki: DeviceModel top-N + 'Other' bucket'lanır (HGB 255 cardinality limiti için).
    """
    out = df.copy()
    if "DeviceModel" in out.columns:
        out["DeviceParentBrand"] = out["DeviceModel"].apply(_device_parent_brand).astype(str)
        out["DeviceModel"] = _bucket_high_cardinality(out["DeviceModel"], DEVICE_MODEL_TOP_N, DEVICE_MODEL_OTHER)
    out["amount_log"] = np.log1p(out["TransactionAmount"].clip(lower=0))
    # NOT (ablation_questionable kararı, 2026-05-13):
    # - hour, dow, is_weekend: bu sentetik veride saatlik/günlük fraud rate DÜZ
    #   (EDA: tüm saat ve gün-haftası fraud rate ~%0.6 ± 0.05). Ablation: Δ val PR-AUC
    #   ≈ 0. Drop edildi.
    # - amount_bucket, is_round_amount, amount_cents: IsFractionalAmount zaten kuruşlu
    #   sinyalini taşıyor; bucket amount_log'la redundant. Ablation: drop edilince TEST
    #   PR-AUC +0.0034 daha iyi. Drop edildi.
    # is_holiday + DayType_clean tatil sinyalini taşıyor (resmi tatil fraud rate %0.86)
    # — bu yüzden tutuldu.

    if daytype_nan == "normal":
        out["DayType_clean"] = out["DayType"].fillna("Normal")
    else:
        out["DayType_clean"] = out["DayType"].fillna("Unknown")
    out["is_holiday"] = (~out["DayType"].isna()).astype("int8")

    out["os_l1h"] = out["DeviceOSName"].astype(str) + "_" + out["HasMobileActivationL1H"].astype(str)
    out["os_l8h"] = out["DeviceOSName"].astype(str) + "_" + out["HasMobileActivationL8H"].astype(str)
    return out
