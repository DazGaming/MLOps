"""
prepare_features.py — "Prepare Data, Transform Features" + "Process Data" steps

Reads raw data, cleans it, engineers a couple of derived features, and writes
a processed table to data/processed/. This processed table is what gets
registered into the feature store (Feast) in the next step.
"""

import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "loan_applications.csv"
PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "loan_features.parquet"


def main():
    df = pd.read_csv(RAW_PATH)

    # basic cleaning
    df = df.dropna()
    df = df[(df["age"] >= 18) & (df["age"] <= 100)]

    # feature engineering
    df["income_to_loan_ratio"] = (df["annual_income"] / df["loan_amount"]).round(3)
    df["credit_score_bucket"] = pd.cut(
        df["credit_score"],
        bins=[0, 580, 670, 740, 850],
        labels=["poor", "fair", "good", "excellent"],
    ).astype(str)

    # Feast requires an event timestamp column for point-in-time joins
    df["event_timestamp"] = pd.Timestamp.now()

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    print(f"[prepare_features] wrote {len(df)} rows to {PROCESSED_PATH}")
    print(df.dtypes)


if __name__ == "__main__":
    main()
