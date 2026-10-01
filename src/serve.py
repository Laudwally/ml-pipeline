"""FastAPI service that serves predictions from the latest registered MLflow model."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "churn-classifier"

model_holder: dict[str, Pipeline] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    try:
        model_holder["model"] = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
    except Exception as exc:  # noqa: BLE001 - surface a clear startup error either way
        raise RuntimeError(
            f"Could not load '{MODEL_NAME}' from the MLflow registry. Run `python src/train.py` first."
        ) from exc
    yield
    model_holder.clear()


app = FastAPI(title="Churn Prediction Service", lifespan=lifespan)


class PredictionRequest(BaseModel):
    tenure_months: int
    monthly_charges: float
    total_charges: float
    support_tickets: int
    plan_type: Literal["basic", "standard", "premium"]


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": "model" in model_holder}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    model = model_holder.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    frame = pd.DataFrame([request.model_dump()])
    proba = float(model.predict_proba(frame)[0, 1])
    return PredictionResponse(churn_probability=proba, churn_prediction=int(proba >= 0.5))
