# MLOps
# Loan Approval MLOps Demo

An end-to-end MLOps pipeline for loan approval, built to mirror a standard
"manual exploratory workflow -> automated pipeline" architecture:

**Manual/Exploratory:** Ingest -> Prepare/Transform -> Process -> Feature Store
-> Bias Detection -> Train & Tune -> Register Model -> Deploy -> Online
Feature Store -> Run Inference

**Automated:** Jenkins Pipeline -> Monitor (drift) -> Retrain -> Scale Inference

Everything runs locally with `pip` + `venv`. No Docker, no WSL, no cloud
account required.

---

## 1. One-time setup (do this today, not Sunday night)

```powershell
cd C:\
git clone <your-repo-or-just-copy-this-folder>   REM or just copy the folder to C:\ml-mlops-demo
cd ml-mlops-demo

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Install Jenkins separately: download the Windows installer from
https://www.jenkins.io/download/ , run it (it installs as a Windows
service on **port 8080**), then open http://localhost:8080 and finish the
setup wizard (it'll give you an initial admin password from a file path
it shows you).

---

## 2. Run order (do this once now to make sure it all works on YOUR machine)

Open **5 separate PowerShell windows**, `cd C:\ml-mlops-demo` and
`venv\Scripts\activate` in each one, then:

**Window 1 — MLflow tracking server (leave running)**
```powershell
mlflow server --host 127.0.0.1 --port 5000
```
Open http://localhost:5000 in a browser — you should see the MLflow UI.

**Window 2 — run the manual/exploratory pipeline once**
```powershell
python src\ingest.py
python src\prepare_features.py
python src\feature_store\materialize.py
python src\train.py
python src\bias_check.py
```
Check MLflow UI: Experiments tab shows `loan_approval` with 3 runs
(random_forest, gradient_boosting, bias_audit). Models tab shows
`loan_approval_model` with a `champion` alias.

**Window 2 — start the inference API (leave running)**
```powershell
python -m uvicorn src.deploy_api:app --host 127.0.0.1 --port 8001
```
Open http://localhost:8001/docs — try `POST /predict` with
`{"applicant_id": 5}`.

**Window 3 — generate a monitoring report**
```powershell
python src\monitor.py
```
Open `reports\monitoring\drift_report.html` in a browser — this is your
Evidently drift dashboard.

**Window 3 — trigger a full retrain (this is your "automated" story)**
```powershell
python src\retrain_trigger.py
```
Watch it re-ingest, re-train, re-check bias, and hot-swap the running API
in Window 2 to the new model — call `/health` on port 8001 again and
watch `model_version` increment with zero downtime.

**Window 4 & 5 — scale inference demo**
```powershell
REM Window 4
python -m uvicorn src.deploy_api:app --host 127.0.0.1 --port 8002
```
```powershell
REM Window 5
python src\scale_inference_demo.py
```
Shows requests round-robining across two replicas.

---

## 3. Wire up Jenkins (the "automated pipeline" box)

1. In Jenkins, **New Item -> Pipeline**, name it `loan-approval-pipeline`
2. Under Pipeline, choose "Pipeline script", paste the contents of
   `jenkins\Jenkinsfile`
3. Edit the `PROJECT_DIR` env var at the top of the Jenkinsfile to match
   your actual path (e.g. `C:\ml-mlops-demo`)
4. Click **Build Now** — watch each stage run in the Jenkins UI
   (Ingest -> Prepare -> Feature Store -> Train -> Bias Check -> Deploy -> Monitor)
5. For the "Retrain" story: duplicate the job, point it at a Jenkinsfile
   that just runs `retrain_trigger.py`, and set a **Build Trigger ->
   Build periodically** schedule (e.g. `H */6 * * *`) to show you
   understand scheduled retraining

---

## 4. What to say in the interview (mapping back to the diagram)

| Diagram box | What you show | Where |
|---|---|---|
| Ingest / Prepare / Process | `ingest.py`, `prepare_features.py` | Terminal |
| Store Data in Feature Store | Feast local repo, `feast apply` output | Terminal + `src/feature_store/` |
| Detect and Mitigate Bias | Fairlearn demographic parity/equalized odds, before/after mitigation | `bias_check.py` output + MLflow run |
| Train & Tune Models | Two candidate models, best picked by AUC | MLflow experiment UI |
| Associate Lineage / Deposit in Registry | `mlflow.register_model`, version tags, `champion` alias | MLflow Models UI |
| Deploy Models / Run Inference | FastAPI `/predict`, live Feast online lookup | http://localhost:8001/docs |
| Online Feature Store | Feast SQLite online store, materialized data | `materialize.py` |
| Build Pipeline that integrates Steps | Jenkins pipeline running all stages | http://localhost:8080 |
| Monitor Models | Evidently drift report | `reports/monitoring/drift_report.html` |
| Retrain Models | `retrain_trigger.py` + hot-swap, zero-downtime | Terminal, then `/health` |
| Scale Inference | Two replicas + round-robin script | `scale_inference_demo.py` |

**Honest framing for the interviewer** (use this if asked "is this
production-grade?"): this is a local, single-node reference
implementation of the pattern — in production you'd swap SQLite/file
stores for Redis/S3/a real data warehouse, Jenkins for a managed
orchestrator or Airflow-on-k8s, and add authentication, autoscaling, and
canary deployments. The point was to demonstrate you understand every
stage of the lifecycle and how they connect, not to reinvent SageMaker.

---

## 5. Known rough edges (so you're not surprised live)

- `feast`, `mlflow`, and the API all show harmless `DeprecationWarning`
  lines on some versions — cosmetic, doesn't affect functionality
- The bias-mitigation step (`ThresholdOptimizer`) needs the sensitive
  attribute (`gender`) passed at prediction time too — that's realistic:
  fairness-aware post-processing generally requires it, and it's a good
  talking point about the practical trade-offs of different mitigation
  techniques (pre-processing vs. in-processing vs. post-processing)
- If port 5000/8001/8002/8080 are already in use on your machine, change
  the port in the relevant command AND in `deploy_api.py` /
  `retrain_trigger.py` (`MLFLOW_TRACKING_URI`, `INFERENCE_API_URL`)
