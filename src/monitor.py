"""
monitor.py — "Monitor Models" step

Simulates a batch of new incoming loan applications (with a deliberate
shift in the income/credit-score distribution — pretend the economy
shifted, or a new marketing channel brought in a different applicant
pool), compares it against the original training data using Evidently,
and writes an HTML drift report you open in the browser.

In the "Automated Workflow," Jenkins would run this on a schedule and
fail the pipeline / trigger retrain.py if drift is detected.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "loan_features.parquet"
REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "monitoring" / "drift_report.html"

MONITORED_COLUMNS = [
    "annual_income", "credit_score", "loan_amount", "employment_years",
    "debt_to_income", "previous_defaults", "income_to_loan_ratio", "age",
]


def simulate_new_batch(reference: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """
    Stands in for "new production traffic pulled from logs/feature store".
    Shifts income down and credit_score down slightly to simulate a
    realistic drift scenario (e.g. broader/riskier applicant pool).
    """
    rng = np.random.default_rng(seed)
    current = reference.sample(n=1500, random_state=seed).copy()
    current["annual_income"] = (current["annual_income"] * 0.78 + rng.normal(0, 2000, len(current))).clip(15000, None)
    current["credit_score"] = (current["credit_score"] - 45 + rng.normal(0, 10, len(current))).clip(300, 850)
    current["debt_to_income"] = (current["debt_to_income"] * 1.25).clip(0, 0.95)
    return current


def main():
    reference = pd.read_parquet(PROCESSED_PATH)[MONITORED_COLUMNS]
    current = simulate_new_batch(pd.read_parquet(PROCESSED_PATH))[MONITORED_COLUMNS]

    report = Report([DataDriftPreset()])
    result = report.run(current_data=current, reference_data=reference)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(REPORT_PATH))

    result_dict = result.dict()
    print(f"[monitor] drift report saved to {REPORT_PATH}")
    print("[monitor] open that file in your browser to view the full report.")
    print(f"[monitor] raw summary keys available: {list(result_dict.keys())[:5]} ...")


if __name__ == "__main__":
    main()
