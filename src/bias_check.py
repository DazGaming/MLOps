"""
bias_check.py — "Detect and Mitigate Bias" step

Loads the current champion model from the MLflow registry, evaluates it for
fairness across the sensitive attribute 'gender' using Fairlearn, logs the
fairness report as an MLflow artifact attached to the training run, and
demonstrates one mitigation technique (threshold optimization) so you can
show a before/after in the interview.

Run this AFTER train.py.
"""

from pathlib import Path

import mlflow
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "loan_approval_model"
PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "loan_features.parquet"

FEATURE_COLUMNS = [
    "annual_income", "credit_score", "loan_amount", "employment_years",
    "debt_to_income", "previous_defaults", "income_to_loan_ratio",
    "credit_score_bucket", "age",
]
SENSITIVE_COLUMN = "gender"
TARGET_COLUMN = "approved"

# fairness thresholds we'd flag as "needs mitigation" in a real gate
DEMOGRAPHIC_PARITY_THRESHOLD = 0.10
EQUALIZED_ODDS_THRESHOLD = 0.10


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    df = pd.read_parquet(PROCESSED_PATH)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN]
    sensitive = df[SENSITIVE_COLUMN]

    encoder = OrdinalEncoder()
    X["credit_score_bucket"] = encoder.fit_transform(X[["credit_score_bucket"]])

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=42, stratify=y
    )

    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@champion")
    preds = model.predict(X_test)

    # --- fairness metrics on the champion model, before mitigation ---
    mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
        y_true=y_test, y_pred=preds, sensitive_features=s_test,
    )
    dp_diff = demographic_parity_difference(y_test, preds, sensitive_features=s_test)
    eo_diff = equalized_odds_difference(y_test, preds, sensitive_features=s_test)

    print("[bias_check] per-group metrics (BEFORE mitigation):")
    print(mf.by_group)
    print(f"[bias_check] demographic parity difference: {dp_diff:.3f}")
    print(f"[bias_check] equalized odds difference:     {eo_diff:.3f}")

    flagged = dp_diff > DEMOGRAPHIC_PARITY_THRESHOLD or eo_diff > EQUALIZED_ODDS_THRESHOLD
    print(f"[bias_check] FLAGGED FOR BIAS: {flagged}")

    # --- mitigation: threshold optimizer post-processing, per sensitive group ---
    mitigator = ThresholdOptimizer(
        estimator=model, constraints="demographic_parity", predict_method="predict_proba"
    )
    mitigator.fit(X_train, y_train, sensitive_features=s_train)
    mitigated_preds = mitigator.predict(X_test, sensitive_features=s_test)

    mf_after = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
        y_true=y_test, y_pred=mitigated_preds, sensitive_features=s_test,
    )
    dp_diff_after = demographic_parity_difference(y_test, mitigated_preds, sensitive_features=s_test)

    print("\n[bias_check] per-group metrics (AFTER mitigation):")
    print(mf_after.by_group)
    print(f"[bias_check] demographic parity difference after mitigation: {dp_diff_after:.3f}")

    # log everything as a run attached to the same experiment for traceability
    mlflow.set_experiment("loan_approval")
    with mlflow.start_run(run_name="bias_audit"):
        mlflow.log_metric("demographic_parity_diff_before", dp_diff)
        mlflow.log_metric("equalized_odds_diff_before", eo_diff)
        mlflow.log_metric("demographic_parity_diff_after", dp_diff_after)
        mlflow.log_metric("bias_flagged", int(flagged))
        mf.by_group.to_csv("/tmp/fairness_by_group_before.csv")
        mf_after.by_group.to_csv("/tmp/fairness_by_group_after.csv")
        mlflow.log_artifact("/tmp/fairness_by_group_before.csv")
        mlflow.log_artifact("/tmp/fairness_by_group_after.csv")

    print("\n[bias_check] fairness report logged to MLflow run 'bias_audit'.")


if __name__ == "__main__":
    main()
