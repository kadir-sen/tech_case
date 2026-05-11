"""FastAPI servisi.

Endpoint:
  GET  /health
  POST /score?explain={true|false}
"""
from __future__ import annotations
from fastapi import FastAPI, Query
from .schemas import TransactionInput, ScoreResponse
from .predict import predict, MODEL_NAME

app = FastAPI(
    title="Fraud Detection API",
    description="Para transferi sahtecilik tespiti. Tek-tx skorlama.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/score", response_model=ScoreResponse)
def score(tx: TransactionInput, explain: bool = Query(default=False)) -> ScoreResponse:
    return predict(tx, explain=explain)
