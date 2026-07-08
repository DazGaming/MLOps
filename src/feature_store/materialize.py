"""
materialize.py — "Store Data in Feature Store" + "Online Feature Store" steps

Applies the Feast feature definitions and materializes them from the
offline store (the parquet file) into the online store (SQLite), so that
deploy_api.py can fetch features by applicant_id at low latency at
inference time — same pattern as a production online feature store.

Run this from the project root:
    venv\\Scripts\\python src\\feature_store\\materialize.py
"""

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

FEATURE_REPO_DIR = Path(__file__).resolve().parent / "feature_repo"


def run(cmd: list[str]):
    print(f"[materialize] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=FEATURE_REPO_DIR)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    feast_bin = str(Path(sys.executable).parent / "feast")
    run([feast_bin, "apply"])

    end = datetime.utcnow()
    start = end - timedelta(days=3650)  # cover all synthetic data regardless of when ingest ran
    run([
        feast_bin, "materialize",
        start.strftime("%Y-%m-%dT%H:%M:%S"),
        end.strftime("%Y-%m-%dT%H:%M:%S"),
    ])
    print("[materialize] done — online store populated.")


if __name__ == "__main__":
    main()
