# End-to-End MLOps Infrastructure Project

An industry-style ML engineering and MLOps platform built from scratch, covering the complete lifecycle:

**Data -> Feature Engineering -> Model Training -> Experiment Tracking -> Model Registry -> API Serving -> Monitoring/Alerting -> Automatic Retraining -> Kubernetes Orchestration**

This repository is implemented according to `roadmap.md` and is designed to demonstrate both algorithmic depth and production deployment capability for ML Engineer / MLOps Engineer roles.

---

## Table of Contents

- [Business Goal](#business-goal)
- [Project Scope and Tech Stack](#project-scope-and-tech-stack)
- [Architecture and End-to-End Workflow](#architecture-and-end-to-end-workflow)
- [Phase-by-Phase Implementation](#phase-by-phase-implementation)
- [Algorithms Built from Scratch](#algorithms-built-from-scratch)
- [Detailed Experiment Results](#detailed-experiment-results)
- [MLflow Tracking and Model Registry](#mlflow-tracking-and-model-registry)
- [FastAPI Model Serving](#fastapi-model-serving)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoring, Drift Detection, and Alerting](#monitoring-drift-detection-and-alerting)
- [Automatic Retraining with Prefect](#automatic-retraining-with-prefect)
- [Docker and Kubernetes Deployment](#docker-and-kubernetes-deployment)
- [How to Run This Project](#how-to-run-this-project)
- [Screenshot Walkthrough](#screenshot-walkthrough)
- [What Recruiters Should Notice](#what-recruiters-should-notice)

---

## Business Goal

Build a robust ML platform where models are not only trained, but also:
- benchmarked against standard baselines
- tracked and versioned
- served as APIs
- monitored for reliability and drift
- retrained automatically under defined quality triggers
- deployed on container and cluster environments with production patterns

---

## Project Scope and Tech Stack

### ML / Data
- Python, NumPy, Pandas, scikit-learn
- Custom algorithms from scratch:
  - Logistic Regression
  - K-Means
  - Isolation Forest

### MLOps / Platform
- MLflow (tracking + model registry)
- FastAPI + Uvicorn
- Docker + Docker Compose
- GitHub Actions (CI)
- Prometheus + Grafana + Alertmanager
- Prefect (orchestration/retraining)
- Kubernetes (`kind`, `kubectl`, `kustomize`)

---

## Architecture and End-to-End Workflow

```mermaid
flowchart LR
  A[Raw Datasets] --> B[Data Pipeline]
  B --> C[Scratch + sklearn Experiments]
  C --> D[MLflow Tracking]
  D --> E[Model Registry]
  E --> F[FastAPI Service]
  F --> G[Prometheus Metrics]
  G --> H[Grafana Dashboards]
  G --> I[Alertmanager]
  G --> J[Prefect Retraining Flow]
  J --> B
  J --> C
  J --> E
  J --> F
```

Operational flow:
1. preprocess data for each task
2. train scratch and sklearn variants
3. log params/metrics/artifacts to MLflow
4. register models and move versions across stages
5. serve models via FastAPI endpoints
6. collect runtime + drift metrics
7. trigger retraining when drift/performance thresholds are breached

---

## Phase-by-Phase Implementation

### Phase 1-3: Environment, Data Setup, and ETL

Implemented modules in `data_pipeline/`:
- `ingestion.py`
- `cleaning.py`
- `feature_engineering.py`
- `split_data.py`
- `run_pipeline.py`

What the pipeline does:
- reads raw datasets from `datasets/`
- removes duplicates and checks missing values
- applies dataset-specific transformations:
  - spam: TF-IDF feature generation
  - customers: categorical encoding + scaling
  - fraud: normalization and balanced split checks
- saves processed outputs to `data_pipeline/processed_data/`

### Phase 4-7: Scratch Algorithms + Evaluation + Experiment Framework

Built custom implementations and comparison scripts:
- `experiments/exp_logistic_vs_sklearn.py`
- `experiments/exp_kmeans_vs_sklearn.py`
- `experiments/exp_isolation_forest_vs_sklearn.py`

Evaluation supports:
- classification metrics (accuracy, precision, recall, F1, AUC)
- clustering metrics (inertia, silhouette, Davies-Bouldin, Calinski-Harabasz)
- anomaly metrics (ROC-AUC, average precision, precision, recall, F1)
- cross-validation
- plot artifact generation

### Phase 8: FastAPI Model Serving

Implemented `api/app.py` with:
- `POST /predict/spam`
- `POST /predict/cluster`
- `POST /predict/fraud`
- `POST /predict/batch/spam`
- `GET /health`
- `GET /metrics`

Model loading strategy:
- registry-first loading from MLflow Model Registry
- fallback to recent experiment artifacts if needed
- dtype-safe conversion (`float64`) for sklearn compatibility

### Phase 9: Containerization

Implemented:
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

Compose services:
- API
- MLflow
- Prometheus
- Grafana
- Alertmanager

### Phase 10: CI/CD (GitHub Actions)

Workflow file: `.github/workflows/ci.yml`

CI pipeline includes:
- dependency installation
- syntax/import sanity checks
- smoke-level unit import tests
- config auto-adjustment for faster CI runs
- generation of dummy processed datasets for reproducible CI
- full experiment execution
- API startup and endpoint tests (`api/test_api.py`)

This proves reproducible automation of both training and serving checks.

### Phase 11: Monitoring and Alerting

Implemented:
- Prometheus scrape + rules in `monitoring/prometheus/`
- Grafana datasource + dashboard provisioning in `monitoring/grafana/`
- Alertmanager routing in `monitoring/alertmanager/`

Tracked metrics:
- API request count/rate
- latency histogram and p95
- prediction distribution
- data drift score
- model availability state

### Phase 12: Automatic Retraining

Flow implementation: `retraining/prefect_retraining_flow.py`

Trigger logic:
- drift trigger from Prometheus query
- accuracy trigger from latest MLflow runs

Automated steps:
1. load retraining config
2. check trigger conditions
3. run ETL pipeline
4. retrain selected models
5. register models
6. redeploy API via configured command

### Phase 13: Kubernetes Production-Style Deployment

Manifest layout:
- `k8s/base/`
- `k8s/overlays/dev/`

Implemented capabilities:
- namespace separation (`mlops`)
- deployment/service for API, MLflow, Prometheus, Grafana, Alertmanager
- ingress routing
- ConfigMap and Secret management
- persistent volume claims
- liveness/readiness/startup probes
- resource requests/limits
- HPA (`hpa-api.yaml`)
- PodDisruptionBudget (`pdb-api.yaml`)
- NetworkPolicy
- retraining CronJob

### Phase 14: Reporting and Demonstration

Final reports in `results/`:
- `experiment_report_logistic.md`
- `experiment_report_kmean.md`
- `experiment_report_isolation.md`

These reports provide the quantitative basis for model comparison and design decisions.

---

## Algorithms Built from Scratch

### 1) Logistic Regression (Spam Classification)

Core engineering features:
- custom sigmoid + cross-entropy optimization
- adaptive learning rate and early stopping
- regularization tuning and threshold optimization
- high-dimensional sparse input handling (TF-IDF)

Why this implementation is strong:
- significantly improves recall and F1 on spam class
- maintains high AUC close to sklearn baseline
- reproducible convergence with configuration-driven hyperparameters

### 2) K-Means (Customer Segmentation)

Core engineering features:
- KMeans++ style initialization
- multi-restart optimization (`n_init`)
- convergence checks and inertia history tracking
- auto-k analysis support (elbow/gap style search)

Why this implementation is strong:
- clustering quality is near sklearn on major metrics
- very fast training in this benchmark setup
- transparent cluster diagnostics and centroid analysis

### 3) Isolation Forest (Fraud Detection)

Core engineering features:
- custom isolation tree building
- extended split strategy
- threshold optimization for recall-driven anomaly detection
- production-oriented contamination strategy and large dataset handling

Why this implementation is strong:
- better ROC-AUC, average precision, and recall than sklearn in this setup
- substantially faster training time in benchmark results
- intentionally tuned for fraud catch-rate (business-priority recall)

---

## Detailed Experiment Results

All values below are from `results/*.md`.

### Logistic Regression (Spam)

| Metric | Scratch | sklearn |
|---|---:|---:|
| Accuracy | 0.9725 | 0.9617 |
| Precision | 0.9208 | 0.9634 |
| Recall | 0.8611 | 0.7315 |
| F1 | 0.8900 | 0.8316 |
| AUC | 0.9867 | 0.9922 |
| Training Time | 0.48s | 0.04s |

Cross-validation:
- scratch mean accuracy: 0.9734 +- 0.0029
- sklearn mean accuracy: 0.9533 +- 0.0032

Interpretation:
- scratch model improves recall/F1 and overall accuracy
- sklearn remains stronger in precision and slightly AUC
- clear precision-recall tradeoff with business-relevant uplift on missed spam reduction

### K-Means (Store Customers)

| Metric | Scratch | sklearn |
|---|---:|---:|
| Inertia | 601.36 | 600.56 |
| Silhouette | 0.3063 | 0.3076 |
| Davies-Bouldin | 1.1079 | 1.1098 |
| Calinski-Harabasz | 1095.33 | 1097.14 |
| Training Time | 0.1719s | 8.1511s |

Cross-validation silhouette:
- scratch: 0.3045 +- 0.0057
- sklearn: 0.3022 +- 0.0089

Interpretation:
- model quality is highly competitive
- scratch shows clear speed advantage in this configuration

### Isolation Forest (Credit Card Fraud)

| Metric | Scratch | sklearn |
|---|---:|---:|
| ROC-AUC | 0.9409 | 0.9364 |
| Avg Precision | 0.1705 | 0.1224 |
| Precision | 0.0356 | 0.0628 |
| Recall | 0.8105 | 0.6632 |
| F1 | 0.0682 | 0.1148 |
| Training Time | 10.97s | 36.22s |

Cross-validation:
- scratch ROC-AUC mean: 0.9521 +- 0.0163
- sklearn ROC-AUC mean: 0.9475 +- 0.0161

Interpretation:
- scratch is optimized for detection coverage (high recall/AP/AUC)
- sklearn gives higher precision/F1 at a stricter detection boundary
- this is a deliberate operating-point choice for fraud risk control

---

## MLflow Tracking and Model Registry

Tracking utility: `tracking/mlflow_tracker.py`

Logged content:
- hyperparameters and config context
- scalar metrics and fold metrics
- artifacts (plots, confusion matrices, curve files, dataset metadata)
- model signatures and input examples
- model versions in registry

Model lifecycle:
1. run training experiments
2. log model artifacts and metadata
3. register versions via `api/register_models.py`
4. transition versions (Staging -> Production)
5. API serves model by stage/name

---

## FastAPI Model Serving

Service file: `api/app.py`

Design highlights:
- Pydantic request/response schemas
- explicit error handling and health diagnostics
- registry-aware model load with fallback
- endpoint-level instrumentation through monitoring manager
- batch and single inference support

Operational value:
- clean interface for product integration
- observable and testable inference service
- decoupled from training through registry-driven loading

---

## CI/CD Pipeline

Workflow: `.github/workflows/ci.yml`

What runs automatically:
- checkout + Python setup
- install dependencies
- syntax/import checks with `compileall`
- smoke imports for core algorithm classes
- speed-tuned CI config rewrite
- synthetic processed dataset generation
- run all three experiment scripts
- start API and run endpoint tests

This pipeline validates that:
- code imports are healthy
- training scripts remain executable
- API comes up and responds correctly
- project stays reproducible in clean environments

---

## Monitoring, Drift Detection, and Alerting

Instrumentation module: `api/monitoring.py`

Prometheus:
- request counters by endpoint/method/status
- latency histograms
- model/drift gauges

Grafana:
- live dashboard panels for traffic, latency, drift, prediction distribution

Alerting:
- high latency
- high error rate
- model unavailable
- drift warning and drift critical

Operational outcome:
- real-time visibility into model-serving behavior
- measurable SLA/SLO signals
- fast incident detection capability

---

## Automatic Retraining with Prefect

Flow: `retraining/prefect_retraining_flow.py`

Retraining trigger policy:
- drift score query exceeds threshold
- latest logistic model accuracy drops under minimum threshold

Pipeline actions:
1. evaluate monitoring + performance signals
2. execute data pipeline
3. retrain requested model set
4. register model versions
5. trigger redeploy command (local Docker or Kubernetes rollout restart)

Can run:
- manually (demo)
- via scheduled Kubernetes CronJob (production-like)

---

## Docker and Kubernetes Deployment

### Docker

Run entire local stack with:
- API
- MLflow
- Prometheus
- Grafana
- Alertmanager

### Kubernetes

Production-style deployment with:
- Kustomize overlays
- persistent storage
- health probes
- autoscaling controls
- network policy
- ingress routing
- scheduled retraining jobs

This setup demonstrates resilient, observable, and scalable ML operations.

---

## How to Run This Project

### 1) Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Run data pipeline

```bash
python data_pipeline/run_pipeline.py
```

### 3) Run experiments

```bash
python experiments/exp_logistic_vs_sklearn.py
python experiments/exp_kmeans_vs_sklearn.py
python experiments/exp_isolation_forest_vs_sklearn.py
```

### 4) Register models

```bash
python api/register_models.py
```

### 5) Start local MLOps stack

```bash
docker compose up -d --build
```

### 6) Run API tests

```bash
python api/test_api.py
```

### 7) Run Prefect retraining flow

```bash
python retraining/prefect_retraining_flow.py
```

### 8) Kubernetes deployment

```bash
kubectl apply -k k8s/overlays/dev
kubectl get pods -n mlops
```

### 9) Useful port-forwards (Kubernetes)

```bash
kubectl port-forward -n mlops svc/mlops-api 8001:8000 --address 127.0.0.1
kubectl port-forward -n mlops svc/mlflow 5001:5000 --address 127.0.0.1
kubectl port-forward -n mlops svc/grafana 3001:3000 --address 127.0.0.1
kubectl port-forward -n mlops svc/prometheus 9091:9090 --address 127.0.0.1
kubectl port-forward -n mlops svc/alertmanager 9094:9093 --address 127.0.0.1
```

---

## Screenshot Walkthrough

### MLflow (Experiments + Registry + Stages)

![MLflow Overview](images/mlflow-overview.PNG)
![MLflow Experiments](images/mlflow-experiments.PNG)
![MLflow Comparison](images/mlflow-comparison.PNG)
![MLflow Experiment Comparison 2](images/exp-comaprison-mlflow2.PNG)
![MLflow Staging Models](images/mlflow-staging-model.PNG)
![MLflow Versions](images/mlflow-versionofmodels-show.PNG)
![Register Models to Staging/Production](images/registermodelsstagingproduction-mlflow.PNG)

### FastAPI Health, Inference, Metrics

![API Health](images/apihealth.PNG)
![API Inference Result](images/apiresult8080.PNG)
![API Metrics Endpoint](images/8080ort-metrics.PNG)

### Monitoring and Alerts

![Prometheus Main](images/prometheus.PNG)
![Prometheus Query](images/prometheous2.PNG)
![Grafana Dashboard](images/grafana.PNG)
![Alertmanager](images/alert-manager.PNG)

### Prefect Retraining

![Prefect Run 1](images/prefect-showing.PNG)
![Prefect Run 2](images/prefect2.PNG)

### Kubernetes

![Kubernetes Running Stack](images/kubernate-running.PNG)
![Ingress and Autoscaling](images/ingress+autoscaling.PNG)

### Algorithm Result Screenshots (from project images)

![Logistic Result 1](images/logisticalgo-results.PNG)
![Logistic Result 2](images/logisticalgo-results2.PNG)

### Algorithm Plot Artifacts (from `algorithms/`)

#### Logistic Regression plots
![Logistic ROC Curve](algorithms/logistic_regression/roc_curve_comparison.png)
![Logistic Training Curve](algorithms/logistic_regression/training_curve.png)

#### K-Means plots
![KMeans Elbow](algorithms/kmean/elbow_curve.png)
![KMeans Scratch Clusters](algorithms/kmean/scratch_clusters.png)
![KMeans Sklearn Clusters](algorithms/kmean/sklearn_clusters.png)
![KMeans Inertia History](algorithms/kmean/inertia_history.png)

#### Isolation Forest plots
![Isolation ROC Curve](algorithms/isolation_forest/roc_curve_comparison.png)
![Isolation PR Curve](algorithms/isolation_forest/pr_curve_comparison.png)
![Isolation Score Distribution](algorithms/isolation_forest/score_distributions.png)

---

## What Recruiters Should Notice

- end-to-end system ownership from algorithm internals to platform operations
- practical decisions around recall/precision tradeoffs for business risk contexts
- reproducible MLOps stack with experiment governance and deployment lifecycle
- strong operational maturity: observability, alerting, retraining, and orchestration
- production-minded Kubernetes design (health, scaling, persistence, scheduling)

This project is intentionally built as a portfolio-grade demonstration of both ML depth and MLOps execution capability.
