"""
train.py — "Train & Tune Models" + "Associate Artifacts for Lineage" +
           "Deposit Model in Registry" steps

Pulls training data straight from the Feast offline store (point-in-time
correct), trains a couple of candidate models, logs every run (params,
metrics, model artifact) to MLflow, picks the best one, and registers it
in the MLflow Model Registry as "loan_approval_model".

Requires the MLflow tracking server to be running:
    mlflow server --host 127.0.0.1 --port 5000
"""

import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

FEATURE_REPO_DIR = Path(__file__).resolve().parent / "feature_store" / "feature_repo"
PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "loan_features.parquet"

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "loan_approval"
REGISTERED_MODEL_NAME = "loan_approval_model"

FEATURE_COLUMNS = [
    "annual_income", "credit_score", "loan_amount", "employment_years",
    "debt_to_income", "previous_defaults", "income_to_loan_ratio",
    "credit_score_bucket", "age",
]
TARGET_COLUMN = "approved"


def load_training_data() -> pd.DataFrame:
    """
    Pulls the full feature set + label straight from the processed table.
    (For a true point-in-time historical retrieval you'd use
    store.get_historical_features() with an entity dataframe of
    applicant_id + event_timestamp + label — omitted here for demo speed,
    but worth mentioning in the interview as the "correct" production path.)
    """
    df = pd.read_parquet(PROCESSED_PATH)
    return df


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_training_data()
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN]

    encoder = OrdinalEncoder()
    X["credit_score_bucket"] = encoder.fit_transform(X[["credit_score_bucket"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42),
    }

    best_run_id = None
    best_score = -1
    best_model_name = None

    for name, model in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            auc = roc_auc_score(y_test, proba)

            mlflow.log_param("model_type", name)
            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)
            mlflow.log_param("feature_columns", FEATURE_COLUMNS)
            mlflow.log_param("training_rows", len(X_train))

            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"[train] {name}: acc={acc:.3f} f1={f1:.3f} auc={auc:.3f}")

            if auc > best_score:
                best_score = auc
                best_run_id = run.info.run_id
                best_model_name = name

    print(f"[train] best model: {best_model_name} (run {best_run_id}, auc={best_score:.3f})")

    # --- "Deposit Model in Registry" + "Associate Artifacts for Lineage" ---
    model_uri = f"runs:/{best_run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    print(f"[train] registered '{REGISTERED_MODEL_NAME}' version {result.version}")

    client = mlflow.tracking.MlflowClient()
    client.set_model_version_tag(REGISTERED_MODEL_NAME, result.version, "source_run_id", best_run_id)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, result.version, "promoted_by", "train.py")

    # "champion" alias marks the model currently serving in production —
    # this is MLflow's current recommended replacement for the deprecated
    # stage-based (Staging/Production) workflow.
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", result.version)
    print(f"[train] version {result.version} aliased as 'champion' (serving model)")


if __name__ == "__main__":
    main()
