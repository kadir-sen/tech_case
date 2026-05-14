"""Time-aware validation utilities.

Leakage-free fold üreten helper'lar. Random shuffle ASLA kullanılmaz.

Kullanım:
    from src.models.time_validation import (
        make_expanding_time_folds, make_walk_forward_folds,
        validate_time_order, summarize_folds, assert_no_future_leakage,
    )
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator
import pandas as pd


@dataclass(frozen=True)
class TimeFold:
    fold_index: int
    train: pd.DataFrame
    val: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def validate_time_order(train_df: pd.DataFrame, val_df: pd.DataFrame,
                         date_col: str = "TransactionDate") -> bool:
    """Train'in max tarihi val'in min tarihinden küçük olmalı (strict)."""
    if len(train_df) == 0 or len(val_df) == 0:
        return False
    return train_df[date_col].max() < val_df[date_col].min()


def assert_no_future_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame,
                              date_col: str = "TransactionDate") -> None:
    """Train ile val arasında zaman overlap'ı varsa hata fırlat."""
    if not validate_time_order(train_df, val_df, date_col):
        train_max = train_df[date_col].max()
        val_min = val_df[date_col].min()
        raise AssertionError(
            f"Future leakage: train max ({train_max}) >= val min ({val_min}). "
            f"Time-aware split bozulmuş."
        )


def make_expanding_time_folds(
    df: pd.DataFrame,
    n_splits: int = 4,
    date_col: str = "TransactionDate",
    validation_period_days: int | None = None,
    min_train_period_days: int | None = None,
) -> list[TimeFold]:
    """Expanding window CV: train başlangıçtan başlar, her fold için val penceresi ilerler.

    Args:
        df: zaman-sıralı DataFrame.
        n_splits: kaç fold üretilecek.
        validation_period_days: val penceresinin uzunluğu. None ise df aralığını eşit böler.
        min_train_period_days: ilk fold'un en az kaç günlük train içermesi gerektiği.

    Returns:
        list[TimeFold] — her birinde train_df, val_df ve tarihsel sınırlar.

    Random shuffle YOK. Her fold'da train.max < val.min garantisi (assert_no_future_leakage).
    """
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    total_days = (df_sorted[date_col].max() - df_sorted[date_col].min()).days

    if validation_period_days is None:
        # df aralığını yaklaşık n_splits+1 eşit parçaya böl; her val ~aynı uzunluk
        validation_period_days = max(1, total_days // (n_splits + 1))
    if min_train_period_days is None:
        min_train_period_days = max(1, total_days // (n_splits + 2))

    start_date = df_sorted[date_col].min()
    folds: list[TimeFold] = []
    for i in range(n_splits):
        val_end_day = min_train_period_days + (i + 1) * validation_period_days
        val_start_day = min_train_period_days + i * validation_period_days
        val_start = start_date + pd.Timedelta(days=val_start_day)
        val_end = start_date + pd.Timedelta(days=val_end_day)
        train_df = df_sorted[df_sorted[date_col] < val_start]
        val_df = df_sorted[(df_sorted[date_col] >= val_start) & (df_sorted[date_col] < val_end)]
        if len(train_df) == 0 or len(val_df) == 0:
            continue
        assert_no_future_leakage(train_df, val_df, date_col)
        folds.append(TimeFold(
            fold_index=i,
            train=train_df, val=val_df,
            train_start=train_df[date_col].min(),
            train_end=train_df[date_col].max(),
            val_start=val_df[date_col].min(),
            val_end=val_df[date_col].max(),
        ))
    return folds


def make_walk_forward_folds(
    df: pd.DataFrame,
    n_splits: int = 4,
    date_col: str = "TransactionDate",
    train_window_days: int | None = None,
    validation_window_days: int | None = None,
) -> list[TimeFold]:
    """Sliding window CV: train penceresi sabit boyutlu, her fold için ilerler.

    Expanding'in aksine train pencere boyutu sabittir; eski veriyi unuttuğumuzu varsayar.
    """
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    total_days = (df_sorted[date_col].max() - df_sorted[date_col].min()).days

    if validation_window_days is None:
        validation_window_days = max(1, total_days // (2 * n_splits))
    if train_window_days is None:
        train_window_days = max(1, validation_window_days * 3)

    start_date = df_sorted[date_col].min()
    folds: list[TimeFold] = []
    for i in range(n_splits):
        train_start = start_date + pd.Timedelta(days=i * validation_window_days)
        train_end = train_start + pd.Timedelta(days=train_window_days)
        val_start = train_end
        val_end = val_start + pd.Timedelta(days=validation_window_days)
        train_df = df_sorted[(df_sorted[date_col] >= train_start) & (df_sorted[date_col] < train_end)]
        val_df = df_sorted[(df_sorted[date_col] >= val_start) & (df_sorted[date_col] < val_end)]
        if len(train_df) == 0 or len(val_df) == 0:
            continue
        assert_no_future_leakage(train_df, val_df, date_col)
        folds.append(TimeFold(
            fold_index=i,
            train=train_df, val=val_df,
            train_start=train_df[date_col].min(),
            train_end=train_df[date_col].max(),
            val_start=val_df[date_col].min(),
            val_end=val_df[date_col].max(),
        ))
    return folds


def summarize_folds(folds: list[TimeFold], target_col: str = "IsFraudTransaction") -> pd.DataFrame:
    """Her fold için satır sayısı, tarih aralığı, fraud rate özet tablosu."""
    rows = []
    for f in folds:
        train_rate = float(f.train[target_col].mean())
        val_rate = float(f.val[target_col].mean())
        rows.append({
            "fold_index": f.fold_index,
            "train_start": f.train_start,
            "train_end": f.train_end,
            "val_start": f.val_start,
            "val_end": f.val_end,
            "train_rows": len(f.train),
            "val_rows": len(f.val),
            "train_fraud_rate": round(train_rate, 5),
            "val_fraud_rate": round(val_rate, 5),
            "train_fraud_count": int(f.train[target_col].sum()),
            "val_fraud_count": int(f.val[target_col].sum()),
        })
    return pd.DataFrame(rows)


def iter_folds(folds: list[TimeFold]) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Convenience iterator: yields (train, val) pairs."""
    for f in folds:
        yield f.train, f.val
