"""Train/Val/Test split — time-based (default) + random (benchmark)."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import yaml

from ..data.loader import REPO_ROOT


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    strategy: str
    cutoffs: dict


def _cfg() -> dict:
    with open(REPO_ROOT / "configs" / "split.yaml") as f:
        return yaml.safe_load(f)


def time_based_split(df: pd.DataFrame) -> SplitResult:
    cfg = _cfg()["split"]
    train_end = pd.Timestamp(cfg["train_end"])
    val_start = pd.Timestamp(cfg["val_start"])
    val_end = pd.Timestamp(cfg["val_end"])
    test_start = pd.Timestamp(cfg["test_start"])
    test_end = pd.Timestamp(cfg["test_end"])
    train = df[df["TransactionDate"] <= train_end].copy()
    val = df[(df["TransactionDate"] >= val_start) & (df["TransactionDate"] <= val_end)].copy()
    test = df[(df["TransactionDate"] >= test_start) & (df["TransactionDate"] <= test_end)].copy()
    return SplitResult(
        train=train, val=val, test=test, strategy="time_based",
        cutoffs={"train_end": str(train_end), "val": [str(val_start), str(val_end)],
                 "test": [str(test_start), str(test_end)]}
    )


def random_split(df: pd.DataFrame) -> SplitResult:
    cfg = _cfg().get("random_benchmark", {})
    test_size = float(cfg.get("test_size", 0.2))
    rs = int(cfg.get("random_state", 42))
    shuffled = df.sample(frac=1.0, random_state=rs).reset_index(drop=True)
    n_test = int(len(shuffled) * test_size)
    n_val = int(len(shuffled) * test_size / 2)
    test = shuffled.iloc[:n_test]
    val = shuffled.iloc[n_test:n_test + n_val]
    train = shuffled.iloc[n_test + n_val:]
    return SplitResult(train=train, val=val, test=test, strategy="random",
                       cutoffs={"test_size": test_size, "random_state": rs})


def entity_overlap(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Train vs Test arasında entity overlap yüzdeleri."""
    out = {}
    for col in ("AccountNumber", "DeviceId", "ReceiverName", "IP_Subnet"):
        sa = set(a[col].unique())
        sb = set(b[col].unique())
        inter = sa & sb
        out[col] = {
            "n_unique_a": len(sa),
            "n_unique_b": len(sb),
            "n_overlap": len(inter),
            "pct_of_b_seen_in_a": round(len(inter) / max(len(sb), 1) * 100, 2),
        }
    return out
