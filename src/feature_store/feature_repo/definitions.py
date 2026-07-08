"""
Feast feature definitions for the loan approval project.

Run from inside this directory:
    feast apply
    feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)   (Linux/Mac)
On Windows (PowerShell), see materialize.py instead - it calls the Feast
Python SDK directly so you don't need to fight with date formatting in cmd.
"""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String, ValueType

PROCESSED_DATA_PATH = str(
    (Path(__file__).resolve().parent.parent.parent.parent / "data" / "processed" / "loan_features.parquet")
)

applicant = Entity(name="applicant_id", join_keys=["applicant_id"], value_type=ValueType.INT64)

loan_source = FileSource(
    path=PROCESSED_DATA_PATH,
    timestamp_field="event_timestamp",
)

loan_features_view = FeatureView(
    name="loan_features",
    entities=[applicant],
    ttl=timedelta(days=365),
    schema=[
        Field(name="annual_income", dtype=Float64),
        Field(name="credit_score", dtype=Int64),
        Field(name="loan_amount", dtype=Float64),
        Field(name="employment_years", dtype=Int64),
        Field(name="debt_to_income", dtype=Float64),
        Field(name="previous_defaults", dtype=Int64),
        Field(name="income_to_loan_ratio", dtype=Float64),
        Field(name="credit_score_bucket", dtype=String),
        Field(name="age", dtype=Int64),
        Field(name="gender", dtype=String),
    ],
    source=loan_source,
    online=True,
)
