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
    out["amount_bucket"] = pd.cut(
        out["TransactionAmount"],
        bins=[-1, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000, np.inf],
        labels=["0-1k", "1k-5k", "5k-10k", "10k-25k", "25k-50k", "50k-100k", "100k+"],
    ).astype(str)
    out["is_round_amount"] = (out["TransactionAmount"] % 100 == 0).astype(int)
    # Cents (decimal part of amount) — IEEE-CIS "Dollars/Cents split" adaptation
    out["amount_cents"] = ((out["TransactionAmount"] * 100).round().astype("int64") % 100).astype("int16")
    out["hour"] = out["TransactionDate"].dt.hour.astype("int16")
    out["dow"] = out["TransactionDate"].dt.dayofweek.astype("int16")
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")

    if daytype_nan == "normal":
        out["DayType_clean"] = out["DayType"].fillna("Normal")
    else:
        out["DayType_clean"] = out["DayType"].fillna("Unknown")
    out["is_holiday"] = (~out["DayType"].isna()).astype("int8")

    out["os_l1h"] = out["DeviceOSName"].astype(str) + "_" + out["HasMobileActivationL1H"].astype(str)
    out["os_l8h"] = out["DeviceOSName"].astype(str) + "_" + out["HasMobileActivationL8H"].astype(str)
    return out
