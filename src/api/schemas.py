"""API request/response Pydantic şemaları."""
from __future__ import annotations
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    BusinessKey: str
    AccountNumber: int
    TransactionDate: datetime
    TransactionType: Literal["Fast", "Havale", "Eft"]
    ReceiverName: str
    SenderName: str
    HasMobileActivationL1H: int = Field(ge=0, le=1)
    HasMobileActivationL8H: int = Field(ge=0, le=1)
    DayType: str | None = None
    CustomerName: str
    CustomerSegment: str
    CustomerAge: int = Field(ge=0, le=120)
    CustomerTenure: float = Field(ge=0)
    CustomerEducation: str | None = None
    CustomerProfession: str | None = None
    CustomerMaritalStatus: str | None = None
    CustomerGender: str
    IsFractionalAmount: bool
    TransactionAmount: float = Field(ge=0)
    DeviceModel: str
    DeviceOSName: str
    DeviceId: str
    IP_Subnet: str
    UniqueIPCount: int = Field(ge=0)
    # Historical aggregate'ler request ile birlikte gelir (feature store / online feature service'ten)
    # In real production'da bu API içinde sorgulanırdı; bu case study için client gönderiyor.
    historical_features: dict | None = Field(
        default=None,
        description="device_tx_count_30d, receiver_tx_count_30d, ... PIT-correct olarak feature "
                    "store'dan gelir. Eksikse model imputation uygular.",
    )


class ReasonCode(BaseModel):
    feature: str
    contribution: float
    direction: Literal["increases_risk", "decreases_risk"]


class ScoreResponse(BaseModel):
    fraud_score: float = Field(ge=0, le=1)
    risk_band: Literal["LOW", "MEDIUM", "HIGH"]
    is_fraud: bool
    threshold_used: float
    model_version: str
    score_calculated_at: datetime
    reason_codes: list[ReasonCode] | None = None
