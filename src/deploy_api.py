"""
deploy_api.py — "Deploy Models" + "Online Feature Store" + "Run Inference" steps

A FastAPI service that:
  1. Loads the current 'champion' model from the MLflow Model Registry
  2. On each request, fetches that applicant's features live from the Feast
     online store (sub-millisecond SQLite lookup, same pattern as a
     production Redis/DynamoDB online store)
  3. Returns a prediction + the model version that served it (for traceability)

Run:
    venv\\Scripts\\python -m uvicorn src.deploy_api:app --port 8001 --reload
Then open http://localhost:8001/docs
"""

from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from feast import FeatureStore
from pydantic import BaseModel
from sklearn.preprocessing import OrdinalEncoder

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "loan_approval_model"
FEATURE_REPO_DIR = Path(__file__).resolve().parent / "feature_store" / "feature_repo"

FEATURE_COLUMNS = [
    "annual_income", "credit_score", "loan_amount", "employment_years",
    "debt_to_income", "previous_defaults", "income_to_loan_ratio",
    "credit_score_bucket", "age",
]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
_model = None
_model_version = None
_store = FeatureStore(repo_path=str(FEATURE_REPO_DIR))
# bucket boundaries must match prepare_features.py
_bucket_encoder = OrdinalEncoder(categories=[["poor", "fair", "good", "excellent"]])
_bucket_encoder.fit([["poor"], ["fair"], ["good"], ["excellent"]])


def _load_champion():
    global _model, _model_version
    client = mlflow.tracking.MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    _model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@champion")
    _model_version = mv.version
    print(f"[deploy_api] loaded champion model version {_model_version}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_champion()
    yield


app = FastAPI(title="Loan Approval Inference API", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    applicant_id: int


class PredictResponse(BaseModel):
    applicant_id: int
    approved: bool
    approval_probability: float
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok", "model_version": _model_version}


@app.post("/reload_model")
def reload_model():
    """Lets Jenkins (or you, live in the demo) hot-swap in a newly retrained champion without restarting the API."""
    _load_champion()
    return {"reloaded": True, "model_version": _model_version}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    resp = _store.get_online_features(
        features=[f"loan_features:{c}" for c in FEATURE_COLUMNS],
        entity_rows=[{"applicant_id": req.applicant_id}],
    ).to_dict()

    if resp["annual_income"][0] is None:
        raise HTTPException(status_code=404, detail=f"No features found for applicant_id={req.applicant_id}")

    row = {c: resp[c][0] for c in FEATURE_COLUMNS}
    X = pd.DataFrame([row])
    X["credit_score_bucket"] = _bucket_encoder.transform(X[["credit_score_bucket"]])

    proba = _model.predict_proba(X)[0, 1]
    approved = bool(proba > 0.5)

    return PredictResponse(
        applicant_id=req.applicant_id,
        approved=approved,
        approval_probability=round(float(proba), 4),
        model_version=str(_model_version),
    )
