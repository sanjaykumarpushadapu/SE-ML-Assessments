"""Prediction microservice. It only loads the registered champion model; it never trains."""
import logging
import time
import uuid
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from pydantic import BaseModel, ConfigDict, Field, create_model

import config
from api.prediction_log import log_prediction

log = logging.getLogger("api")

# Request schema: one required float for each of the 30 model inputs.
# allow_inf_nan=False rejects NaN and infinity; Amount must also be zero or more.
_fields = {c: (float, Field(..., allow_inf_nan=False)) for c in config.FEATURES}
_fields["Amount"] = (float, Field(..., ge=0, allow_inf_nan=False))
# extra="forbid" rejects unknown fields (for example a card number) with HTTP 422.
Transaction = create_model("Transaction", __config__=ConfigDict(extra="forbid"), **_fields)


class Prediction(BaseModel):
    """What the service sends back for one transaction."""
    request_id: str
    probability: float
    is_fraud: bool
    threshold: float
    model_version: str
    latency_ms: float


def load_champion():
    """Load models:/fraud-model@champion, its version and its threshold (BR-002, BR-009)."""
    mlflow.set_tracking_uri(config.TRACKING_URI)
    # Only the registered model with the champion alias is served, never a file or a run.
    model = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}@{config.ALIAS}")
    client = MlflowClient()
    mv = client.get_model_version_by_alias(config.MODEL_NAME, config.ALIAS)
    # The threshold was chosen during evaluation and stored as a run parameter,
    # so the service reads it instead of having its own copy.
    threshold = float(client.get_run(mv.run_id).data.params["threshold"])
    return model, str(mv.version), threshold


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the service starts: load the model into memory (keeps latency low)."""
    app.state.model, app.state.version, app.state.threshold = load_champion()
    log.info("Loaded %s version %s, threshold %.4f", config.MODEL_NAME, app.state.version,
             app.state.threshold)
    yield


app = FastAPI(title="Credit Card Fraud Detection Service", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Return 422 without echoing the input (a NaN value cannot be serialised back)."""
    errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health() -> dict:
    """Report that the service is up and which model version it serves."""
    return {"status": "ok", "model": config.MODEL_NAME, "model_version": app.state.version,
            "threshold": app.state.threshold}


@app.post("/predict", response_model=Prediction)
def predict(tx: Transaction) -> Prediction:
    """Score one transaction; the label is probability >= threshold and nothing else (BR-001)."""
    start = time.perf_counter()
    # Invalid requests never reach this point, FastAPI has already answered with 422.
    row = pd.DataFrame([tx.model_dump()], columns=config.FEATURES)
    # The saved model scales the input itself, so no separate preprocessing is done here (BR-004).
    probability = float(app.state.model.predict_proba(row)[0, 1])
    is_fraud = probability >= app.state.threshold
    latency_ms = (time.perf_counter() - start) * 1000
    request_id = str(uuid.uuid4())  # unique id so a log row can be matched to a call
    try:
        log_prediction(request_id, probability, is_fraud, latency_ms, app.state.version)
    except OSError as exc:
        # A failed log write is reported as an error, never ignored (BR-008).
        raise HTTPException(status_code=500, detail=f"Prediction log failed: {exc}") from exc
    return Prediction(request_id=request_id, probability=probability, is_fraud=is_fraud,
                      threshold=app.state.threshold, model_version=app.state.version,
                      latency_ms=round(latency_ms, 3))
