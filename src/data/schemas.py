from __future__ import annotations
from dataclasses import dataclass

EXPECTED_COLUMNS: tuple[str, ...] = (
    "BusinessKey",
    "AccountNumber",
    "TransactionDate",
    "TransactionType",
    "TransactionChannel",
    "ReceiverName",
    "SenderName",
    "IsFraudTransaction",
    "HasMobileActivationL1H",
    "HasMobileActivationL8H",
    "DayType",
    "CustomerName",
    "CustomerSegment",
    "CustomerAge",
    "CustomerTenure",
    "CustomerEducation",
    "CustomerProfession",
    "CustomerMaritalStatus",
    "CustomerGender",
    "IsFractionalAmount",
    "TransactionAmount",
    "DeviceModel",
    "DeviceOSName",
    "DeviceId",
    "IP_Subnet",
    "UniqueIPCount",
)

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "TransactionType",
    "TransactionChannel",
    "DayType",
    "CustomerSegment",
    "CustomerEducation",
    "CustomerProfession",
    "CustomerMaritalStatus",
    "CustomerGender",
    "DeviceModel",
    "DeviceOSName",
)

ENTITY_COLUMNS: tuple[str, ...] = (
    "AccountNumber",
    "DeviceId",
    "ReceiverName",
    "SenderName",
    "CustomerName",
    "IP_Subnet",
)

PII_COLUMNS: tuple[str, ...] = (
    "ReceiverName",
    "SenderName",
    "CustomerName",
)


@dataclass(frozen=True)
class SchemaCheckResult:
    ok: bool
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    n_rows: int
    n_cols: int


def check_schema(df) -> SchemaCheckResult:
    cols = set(df.columns)
    missing = tuple(c for c in EXPECTED_COLUMNS if c not in cols)
    extra = tuple(c for c in df.columns if c not in EXPECTED_COLUMNS)
    return SchemaCheckResult(
        ok=(not missing and not extra),
        missing=missing,
        extra=extra,
        n_rows=len(df),
        n_cols=df.shape[1],
    )
