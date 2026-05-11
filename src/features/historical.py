"""Historical (point-in-time correct) aggregate feature'lar.

Tüm aggregate'ler için kural: her bir satırın feature değeri, *o satırın* TransactionDate'inden
strictly ÖNCEKİ verilerden hesaplanır (current row hariç). Bu point-in-time correctness'i
sağlar; train ve inference arasındaki sapmayı önler.

İki sınıf:
  - label-FREE: tx_count, distinct_*, first_seen_days_ago — etiket gerektirmez.
  - label-DEPENDENT: fraud_rate_smoothed — label availability lag (T+N gün) uygulanır.

Input dataframe TransactionDate'e göre artan sıralı varsayılır (loader bunu sağlar).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _group_indices(by_vals: np.ndarray) -> dict[object, np.ndarray]:
    s = pd.Series(np.arange(len(by_vals)))
    return {k: np.asarray(v) for k, v in s.groupby(by_vals).groups.items()}


def _rolling_count_prev(times: np.ndarray, by_vals: np.ndarray, window_days: int) -> np.ndarray:
    """Per-group, current row hariç, son `window_days` günde gerçekleşen tx sayısı."""
    out = np.zeros(len(by_vals), dtype=np.int32)
    window = np.timedelta64(window_days, "D")
    for v, idx in _group_indices(by_vals).items():
        t = times[idx]
        lo = np.searchsorted(t, t - window, side="left")
        hi = np.searchsorted(t, t, side="left")
        out[idx] = hi - lo
    return out


def _expanding_count_prev(by_vals: np.ndarray) -> np.ndarray:
    """Per-group, current row hariç, o ana kadar görülen tx sayısı."""
    return pd.Series(np.zeros(len(by_vals))).groupby(by_vals).cumcount().to_numpy().astype(np.int32)


def _distinct_count_prev(by_vals: np.ndarray, tgt_vals: np.ndarray) -> np.ndarray:
    """Per-group, current row hariç, o ana kadar görülen distinct `target` sayısı."""
    out = np.zeros(len(by_vals), dtype=np.int32)
    for v, idx in _group_indices(by_vals).items():
        seen: set = set()
        for i in idx:
            out[i] = len(seen)
            seen.add(tgt_vals[i])
    return out


def _first_seen_days_ago(times: np.ndarray, by_vals: np.ndarray) -> np.ndarray:
    """Per-group, current row için ilk görülme tarihinden geçen gün sayısı (ilk için 0)."""
    out = np.zeros(len(by_vals), dtype=np.float32)
    for v, idx in _group_indices(by_vals).items():
        t = times[idx]
        first = t[0]
        out[idx] = ((t - first).astype("timedelta64[s]").astype(np.int64) / 86400.0).astype(np.float32)
    return out


def add_label_free_aggregates(
    df: pd.DataFrame,
    device_windows: tuple[int, ...] = (1, 7, 30),
    receiver_windows: tuple[int, ...] = (7, 30),
    subnet_windows: tuple[int, ...] = (7,),
) -> pd.DataFrame:
    out = df.copy()
    times = out["TransactionDate"].to_numpy().astype("datetime64[ns]")

    dev = out["DeviceId"].to_numpy()
    out["device_tx_count_all"] = _expanding_count_prev(dev)
    for w in device_windows:
        out[f"device_tx_count_{w}d"] = _rolling_count_prev(times, dev, w)
    out["device_distinct_accounts_all"] = _distinct_count_prev(dev, out["AccountNumber"].to_numpy())
    out["device_distinct_receivers_all"] = _distinct_count_prev(dev, out["ReceiverName"].to_numpy())
    out["device_first_seen_days_ago"] = _first_seen_days_ago(times, dev)

    rec = out["ReceiverName"].to_numpy()
    out["receiver_tx_count_all"] = _expanding_count_prev(rec)
    for w in receiver_windows:
        out[f"receiver_tx_count_{w}d"] = _rolling_count_prev(times, rec, w)
    out["receiver_distinct_senders_all"] = _distinct_count_prev(rec, out["SenderName"].to_numpy())
    out["receiver_distinct_devices_all"] = _distinct_count_prev(rec, dev)
    out["receiver_first_seen_days_ago"] = _first_seen_days_ago(times, rec)

    sub = out["IP_Subnet"].to_numpy()
    out["subnet_tx_count_all"] = _expanding_count_prev(sub)
    for w in subnet_windows:
        out[f"subnet_tx_count_{w}d"] = _rolling_count_prev(times, sub, w)
    out["subnet_distinct_devices_all"] = _distinct_count_prev(sub, dev)

    acc = out["AccountNumber"].to_numpy()
    out["account_tx_count_all"] = _expanding_count_prev(acc)
    out["account_first_seen_days_ago"] = _first_seen_days_ago(times, acc)

    out["device_is_first_seen"] = (out["device_tx_count_all"] == 0).astype("int8")
    out["receiver_is_first_seen"] = (out["receiver_tx_count_all"] == 0).astype("int8")
    out["account_is_first_seen"] = (out["account_tx_count_all"] == 0).astype("int8")
    return out


def _smoothed_fraud_rate_with_lag(
    times: np.ndarray,
    by_vals: np.ndarray,
    labels: np.ndarray,
    lag_days: int,
    prior_strength: float,
    p_global: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-group, t - lag_days'ten ÖNCEKİ tx'lerin etiketleri üzerinden smoothed fraud rate.

    Returns: (rate_smoothed, n_lagged_observations)
    """
    rate = np.full(len(by_vals), p_global, dtype=np.float32)
    n_arr = np.zeros(len(by_vals), dtype=np.int32)
    lag = np.timedelta64(lag_days, "D")
    for v, idx in _group_indices(by_vals).items():
        t = times[idx]
        y = labels[idx]
        cum_y = np.concatenate([[0], np.cumsum(y)])
        cum_n = np.arange(len(idx) + 1)
        cutoff = t - lag
        upper = np.searchsorted(t, cutoff, side="left")
        n = cum_n[upper]
        k = cum_y[upper]
        rate[idx] = (k + prior_strength * p_global) / (n + prior_strength)
        n_arr[idx] = n
    return rate, n_arr


def add_label_dependent_aggregates(
    df: pd.DataFrame,
    label_lag_days: int = 7,
    prior_strength: float = 50.0,
) -> pd.DataFrame:
    out = df.copy()
    times = out["TransactionDate"].to_numpy().astype("datetime64[ns]")
    labels = out["IsFraudTransaction"].to_numpy().astype(np.int8)
    p_global = float(labels.mean())

    for col, prefix in (("DeviceId", "device"), ("ReceiverName", "receiver")):
        rate, n = _smoothed_fraud_rate_with_lag(
            times, out[col].to_numpy(), labels, label_lag_days, prior_strength, p_global
        )
        out[f"{prefix}_fraud_rate_smoothed"] = rate
        out[f"{prefix}_label_n"] = n
    return out
