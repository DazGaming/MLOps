"""
ingest.py — "Ingest Data" step

Generates a synthetic loan-approval dataset and writes it to data/raw/.

Why synthetic data: it's fully offline (no network dependency during your
interview demo), reproducible (fixed seed), and lets us deliberately bake in
a historical bias pattern (against 'gender' == 'Female') so the bias-detection
step later in the pipeline has something real to catch.

In a real project this file would instead pull from a DB, API, or data lake.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "loan_applications.csv"
N_SAMPLES = 6000
RANDOM_SEED = 42


def generate_loan_data(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 65, n_samples)
    gender = rng.choice(["Male", "Female"], n_samples, p=[0.55, 0.45])
    annual_income = rng.normal(58000, 22000, n_samples).clip(15000, 250000)
    credit_score = rng.normal(680, 65, n_samples).clip(300, 850)
    loan_amount = rng.normal(18000, 9000, n_samples).clip(1000, 80000)
    employment_years = rng.integers(0, 35, n_samples)
    debt_to_income = rng.normal(0.32, 0.12, n_samples).clip(0.0, 0.9)
    previous_defaults = rng.choice([0, 1, 2], n_samples, p=[0.82, 0.14, 0.04])

    # --- ground-truth approval logic based on legitimate financial factors ---
    score = (
        (credit_score - 300) / 550 * 0.45
        + (1 - debt_to_income) * 0.25
        + (annual_income / 250000) * 0.15
        + (employment_years / 35) * 0.10
        - (previous_defaults * 0.15)
    )

    # --- intentionally injected historical bias: female applicants penalized ---
    # this simulates biased legacy underwriting data that a fairness audit
    # would be expected to catch — this is the whole point of the demo.
    bias_penalty = np.where(gender == "Female", 0.06, 0.0)
    score = score - bias_penalty

    noise = rng.normal(0, 0.05, n_samples)
    final_score = score + noise
    approved = (final_score > np.quantile(final_score, 0.42)).astype(int)

    df = pd.DataFrame({
        "applicant_id": np.arange(1, n_samples + 1),
        "age": age,
        "gender": gender,
        "annual_income": annual_income.round(2),
        "credit_score": credit_score.round(0).astype(int),
        "loan_amount": loan_amount.round(2),
        "employment_years": employment_years,
        "debt_to_income": debt_to_income.round(3),
        "previous_defaults": previous_defaults,
        "approved": approved,
    })
    return df


def main():
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_loan_data()
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"[ingest] wrote {len(df)} rows to {RAW_DATA_PATH}")
    print(f"[ingest] approval rate overall: {df['approved'].mean():.3f}")
    print(df.groupby("gender")["approved"].mean())


if __name__ == "__main__":
    main()
