from __future__ import annotations
from pathlib import Path
import pandas as pd
import yaml

from .schemas import check_schema, SchemaCheckResult

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str = "data") -> dict:
    cfg_path = REPO_ROOT / "configs" / f"{name}.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        cfg = load_config("data")
        path = REPO_ROOT / cfg["data"]["path"]
    df = pd.read_parquet(path)
    if "TransactionDate" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["TransactionDate"]):
        df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    return df


def load_validated() -> tuple[pd.DataFrame, SchemaCheckResult]:
    df = load_raw()
    result = check_schema(df)
    if not result.ok:
        raise ValueError(
            f"Schema mismatch. missing={result.missing} extra={result.extra}"
        )
    cfg = load_config("data")
    drop_cols = cfg.get("drop_columns") or []
    drop_cols = [c for c in drop_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = df.sort_values("TransactionDate", kind="mergesort").reset_index(drop=True)
    return df, result
