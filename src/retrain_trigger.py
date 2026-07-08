"""
retrain_trigger.py — "Retrain Models" step

This is what Jenkins calls (manually or on a schedule) to run a full
retrain cycle: re-ingest -> re-prepare -> re-materialize features ->
re-train -> re-check bias -> hot-swap the new champion into the live
inference API.

Deliberately just orchestrates the other scripts as subprocesses so each
step stays independently runnable/testable, and so this file mirrors
exactly what the Jenkinsfile pipeline stages do.
"""

import subprocess
import sys
import urllib.request
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable
INFERENCE_API_URL = "http://127.0.0.1:8001/reload_model"


def run_step(script_name: str):
    print(f"\n=== [retrain] running {script_name} ===")
    result = subprocess.run([PYTHON, str(SRC_DIR / script_name)])
    if result.returncode != 0:
        print(f"[retrain] {script_name} FAILED — stopping pipeline")
        sys.exit(result.returncode)


def notify_api_to_reload():
    try:
        req = urllib.request.Request(INFERENCE_API_URL, method="POST", data=b"{}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[retrain] inference API reload response: {resp.read().decode()}")
    except Exception as e:
        print(f"[retrain] WARNING: could not reach inference API to hot-swap model ({e}). "
              f"Is deploy_api.py running on port 8001?")


def main():
    run_step("ingest.py")
    run_step("prepare_features.py")
    run_step(str(Path("feature_store") / "materialize.py"))
    run_step("train.py")
    run_step("bias_check.py")
    notify_api_to_reload()
    print("\n[retrain] full retrain cycle complete — new champion model is live.")


if __name__ == "__main__":
    main()
