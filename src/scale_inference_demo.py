"""
scale_inference_demo.py — "Scale Inference" step

Not a production load balancer — a small, honest demo of the *concept*.
Run two copies of deploy_api.py on different ports (8001 and 8002), then
run this script: it round-robins prediction requests across both and
prints which replica served each one, so you can show horizontal scaling
live without needing Docker/K8s/nginx installed.

Before running this:
    Terminal 1: venv\\Scripts\\python -m uvicorn src.deploy_api:app --port 8001
    Terminal 2: venv\\Scripts\\python -m uvicorn src.deploy_api:app --port 8002
Then:
    venv\\Scripts\\python src\\scale_inference_demo.py
"""

import itertools
import json
import time
import urllib.request

REPLICAS = ["http://127.0.0.1:8001", "http://127.0.0.1:8002"]
SAMPLE_APPLICANT_IDS = [1, 2, 3, 4, 5, 6, 7, 8]


def call_predict(base_url: str, applicant_id: int):
    data = json.dumps({"applicant_id": applicant_id}).encode()
    req = urllib.request.Request(
        f"{base_url}/predict", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read().decode())
    elapsed_ms = (time.perf_counter() - start) * 1000
    return body, elapsed_ms


def main():
    replica_cycle = itertools.cycle(REPLICAS)
    print(f"[scale_demo] round-robining {len(SAMPLE_APPLICANT_IDS)} requests across {len(REPLICAS)} replicas\n")
    for applicant_id in SAMPLE_APPLICANT_IDS:
        replica = next(replica_cycle)
        try:
            body, elapsed_ms = call_predict(replica, applicant_id)
            print(f"[{replica}] applicant={applicant_id} -> "
                  f"approved={body['approved']} prob={body['approval_probability']} "
                  f"({elapsed_ms:.1f} ms)")
        except Exception as e:
            print(f"[{replica}] applicant={applicant_id} -> ERROR: {e}")


if __name__ == "__main__":
    main()
