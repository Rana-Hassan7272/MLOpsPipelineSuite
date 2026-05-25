# MLOpsPipelineSuite — End-to-End MLOps Infrastructure

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Production--Style-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)
![AWS Lambda](https://img.shields.io/badge/AWS_Lambda-Serverless-FF9900?style=for-the-badge&logo=awslambda&logoColor=white)
![Terraform](https://img.shields.io/badge/Terraform-IaC-7B42BC?style=for-the-badge&logo=terraform&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-Dashboards-F46800?style=for-the-badge&logo=grafana&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)
![Prefect](https://img.shields.io/badge/Prefect-Orchestration-3D3D3D?style=for-the-badge&logo=prefect&logoColor=white)

**A production-grade ML platform built from the ground up — covering the complete ML lifecycle from raw data ingestion to Kubernetes-orchestrated, auto-retraining model serving with real-time observability and live AWS serverless deployment.**

*Designed as a capstone demonstrating ML engineering depth, MLOps platform breadth, and end-to-end production deployment capability.*

</div>

---

## What Makes This Project Different

Most ML projects end at model training. This one starts there.

This platform implements what a real ML engineering team builds and maintains in production:

- **3 custom algorithms** built from mathematical foundations — not sklearn wrappers — benchmarked against sklearn under a strict, reproducible evaluation protocol
- **Full MLflow lifecycle**: experiment tracking → model registry → staged promotion → API serving, wired together end-to-end
- **Observable, self-healing API service** instrumented with Prometheus metrics, Grafana dashboards, and Alertmanager rules covering drift, latency, and availability
- **Automatic retraining pipeline** orchestrated by Prefect, triggered by live drift signals from the Prometheus stack
- **Kubernetes production deployment** with HPA autoscaling, PodDisruptionBudget, NetworkPolicy, liveness/readiness probes, and a scheduled retraining CronJob
- **A/B testing infrastructure** with champion/challenger traffic splitting, shadow deployment mode, and automated rollback on error rate or latency breach
- **Live AWS serverless deployment** via Terraform IaC — Lambda + API Gateway + S3 + DynamoDB + ECR + SNS + EventBridge + CloudWatch, fully provisioned with a single command
- **Reproducible CI/CD** via GitHub Actions — every push trains models and validates the full serving stack; every merge to `main` auto-deploys to AWS Lambda

---

## Table of Contents

1. [Business Goal](#business-goal)
2. [Tech Stack](#tech-stack)
3. [System Architecture](#system-architecture)
4. [Repository Structure](#repository-structure)
5. [Phase-by-Phase Implementation](#phase-by-phase-implementation)
6. [Algorithms Built From Scratch — Deep Dive](#algorithms-built-from-scratch--deep-dive)
7. [Detailed Experiment Results & Analysis](#detailed-experiment-results--analysis)
8. [Benchmark Protocol](#benchmark-protocol)
9. [MLflow Tracking & Model Registry](#mlflow-tracking--model-registry)
10. [FastAPI Model Serving](#fastapi-model-serving)
11. [A/B Testing & Experiment Management](#ab-testing--experiment-management)
12. [CI/CD Pipeline](#cicd-pipeline)
13. [Monitoring, Drift Detection & Alerting](#monitoring-drift-detection--alerting)
14. [Automatic Retraining with Prefect](#automatic-retraining-with-prefect)
15. [Docker & Kubernetes Deployment](#docker--kubernetes-deployment)
16. [AWS Production Deployment](#aws-production-deployment)
17. [How to Run](#how-to-run)
18. [Screenshot Walkthrough](#screenshot-walkthrough)

---

## Business Goal

Build a robust ML platform where models are not only trained, but continuously tracked, served, monitored, and retrained in production.

| Capability | Implementation |
|---|---|
| Benchmarked vs. baselines | Custom scratch vs. sklearn comparison across 3 algorithms |
| Tracked and versioned | MLflow experiment tracking + model registry with staged promotion |
| Served as APIs | FastAPI with registry-first model loading, batch support, health probes |
| Experiment-managed | Champion/challenger A/B testing with shadow mode and auto-rollback |
| Monitored in production | Prometheus + Grafana + Alertmanager with drift and SLO alerts |
| Auto-retrained | Prefect flow triggered by live drift and accuracy signals |
| Cluster-deployed | Kubernetes with HPA, PDB, NetworkPolicy, CronJob retraining |
| Cloud-deployed (production) | AWS Lambda + API Gateway + Terraform IaC — live serverless endpoint |

---

## Tech Stack

### ML / Data

| Tool | Usage |
|---|---|
| Python 3.10+ | Core language |
| NumPy / Pandas | Data processing |
| scikit-learn | Baseline comparison + preprocessing utilities |
| TF-IDF (custom) | Spam feature engineering |

### Custom Algorithms (From Scratch)

| Algorithm | Task |
|---|---|
| Logistic Regression | Spam classification |
| K-Means Clustering | Customer segmentation |
| Isolation Forest | Credit card fraud detection |

### MLOps / Platform

| Tool | Role |
|---|---|
| MLflow | Experiment tracking + model registry |
| FastAPI + Uvicorn | Model serving REST API |
| Docker + Docker Compose | Local containerized stack |
| GitHub Actions | CI/CD automation |
| Prometheus | Metrics collection |
| Grafana | Dashboard visualization |
| Alertmanager | Alert routing (drift, latency, availability) |
| Prefect | Retraining orchestration |
| Kubernetes (kind + kustomize) | Production-style cluster deployment |

### Cloud Infrastructure (AWS)

| Tool | Role |
|---|---|
| AWS Lambda | Serverless model serving (container image, linux/amd64) |
| AWS API Gateway | HTTPS REST endpoint with stage routing |
| AWS ECR | Docker container registry |
| AWS S3 | Model artifact storage |
| AWS DynamoDB | Per-request logging (pay-per-request billing) |
| AWS SNS | Email alerting on error rate breaches |
| AWS CloudWatch | Log aggregation + 5xx alarm |
| AWS EventBridge | Daily scheduled drift check trigger |
| AWS IAM | Least-privilege Lambda execution role |
| Terraform | Full infrastructure as code |

---

## System Architecture

### End-to-End Data & Model Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA & TRAINING PLANE                              │
│                                                                             │
│  Raw Datasets                                                               │
│  (spam / customers / fraud)                                                 │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │              Data Pipeline (data_pipeline/)          │                   │
│  │  ingestion → cleaning → feature engineering → split  │                   │
│  └─────────────────────────┬───────────────────────────┘                   │
│                             │                                               │
│                             ▼                                               │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │         Experiment Framework (experiments/)           │                  │
│  │  Scratch Algorithm  ◄──────────►  sklearn Baseline   │                  │
│  │  (custom optimizer)              (reference impl)     │                  │
│  └──────────────────────┬───────────────────────────────┘                  │
│                          │                                                  │
│                          ▼                                                  │
│  ┌────────────────────────────────────────────────────────┐                │
│  │              MLflow Tracking (tracking/)                │                │
│  │   params → metrics → artifacts → model signatures      │                │
│  └──────────────────────┬─────────────────────────────────┘                │
│                          │                                                  │
│                          ▼                                                  │
│  ┌────────────────────────────────────────────────────────┐                │
│  │           MLflow Model Registry (api/register)          │                │
│  │      None ──► Staging ──► Production ──► Archived       │                │
│  └──────────────────────┬─────────────────────────────────┘                │
└─────────────────────────│───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SERVING PLANE                                      │
│                                                                             │
│  ┌────────────────────────────────────────────────────────┐                │
│  │                  FastAPI Service (api/)                  │                │
│  │                                                          │                │
│  │  POST /predict/spam      POST /predict/cluster           │                │
│  │  POST /predict/fraud     POST /predict/batch/spam        │                │
│  │  GET  /health            GET  /metrics                   │                │
│  │  GET  /experiment/status POST /experiment/configure      │                │
│  │                                                          │                │
│  │  registry-first model loading  →  fallback artifacts     │                │
│  │  champion/challenger A/B split →  shadow deployment mode │                │
│  └──────────────────────┬─────────────────────────────────┘                │
└─────────────────────────│───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OBSERVABILITY PLANE                                  │
│                                                                             │
│  ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────────┐   │
│  │   Prometheus    │───►│     Grafana       │    │    Alertmanager      │   │
│  │                 │    │                   │    │                      │   │
│  │  • request rate │    │  • traffic panels │    │  • high latency      │   │
│  │  • p95 latency  │    │  • latency charts │    │  • drift warning     │   │
│  │  • drift score  │    │  • drift gauges   │    │  • drift critical    │   │
│  │  • pred distrib │    │  • pred distrib   │    │  • model unavailable │   │
│  │  • model state  │    │  • model health   │    │  • high error rate   │   │
│  └────────┬────────┘    └──────────────────┘    └──────────────────────┘   │
└───────────│─────────────────────────────────────────────────────────────────┘
            │  drift threshold breach / accuracy drop
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       RETRAINING PLANE (Prefect)                            │
│                                                                             │
│  Trigger Check (Prometheus + MLflow)                                        │
│       │                                                                     │
│       ▼                                                                     │
│  ETL Pipeline ──► Retrain Models ──► Register Versions ──► Redeploy API    │
│                                                                             │
│  Runs: manually / Docker exec / Kubernetes CronJob (scheduled)              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### AWS Serverless Architecture

```
User Request
     │
     ▼
API Gateway (HTTPS) ──► Lambda (Container Image, linux/amd64)
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
           Load models    Log request    Check drift
           from S3        to DynamoDB    via EventBridge
               │              │              │
               ▼              ▼              ▼
          Predict &      Store metadata   SNS Alert
          return result  (pay-per-req)    (if errors)
```

### Kubernetes Cluster Layout

```
Cluster (kind)
└── Namespace: mlops
    ├── Deployments
    │   ├── mlops-api          (FastAPI + Uvicorn)
    │   ├── mlflow             (Tracking Server)
    │   ├── prometheus         (Metrics Collection)
    │   ├── grafana            (Visualization)
    │   └── alertmanager       (Alert Routing)
    ├── Services (ClusterIP / NodePort)
    ├── Ingress (multi-service routing)
    ├── ConfigMaps + Secrets
    ├── PersistentVolumeClaims (MLflow + Grafana state)
    ├── HPA  →  mlops-api (horizontal autoscaling)
    ├── PDB  →  mlops-api (disruption budget)
    ├── NetworkPolicy (ingress/egress rules)
    └── CronJob → retraining/prefect_retraining_flow.py
```

---

## Repository Structure

```
MLOpsPipelineSuite/
├── algorithms/                   # Custom ML implementations (from scratch)
│   ├── logistic_regression.py
│   ├── kmeans.py
│   └── isolation_forest.py
├── api/                          # FastAPI serving layer
│   ├── app.py                    # Endpoints, model loading, health
│   ├── monitoring.py             # Prometheus instrumentation
│   ├── experiment_manager.py     # A/B testing: champion/challenger, shadow mode, auto-rollback
│   ├── register_models.py        # MLflow registry promotion
│   ├── lambda_handler.py         # Mangum ASGI adapter for Lambda
│   └── test_api.py               # Integration tests
├── configs/
│   └── config.yaml               # Central hyperparameter config
├── data_pipeline/                # ETL pipeline
│   ├── ingestion.py
│   ├── cleaning.py
│   ├── feature_engineering.py
│   ├── split_data.py
│   └── run_pipeline.py
├── evaluation/                   # Metric computation modules
├── experiments/                  # Scratch vs. sklearn benchmark scripts
│   ├── exp_logistic_vs_sklearn.py
│   ├── exp_kmeans_vs_sklearn.py
│   └── exp_isolation_forest_vs_sklearn.py
├── images/                       # Screenshot artifacts (CI + demo)
├── k8s/
│   ├── base/                     # Reusable Kubernetes manifests
│   └── overlays/dev/             # Kustomize dev overlay
├── monitoring/
│   ├── prometheus/               # Scrape config + alert rules
│   ├── grafana/                  # Dashboard + datasource provisioning
│   └── alertmanager/             # Alert routing config
├── results/                      # Experiment reports
│   ├── experiment_report_logistic.md
│   ├── experiment_report_kmean.md
│   └── experiment_report_isolation.md
├── retraining/
│   └── prefect_retraining_flow.py
├── tracking/
│   └── mlflow_tracker.py
├── training/                     # Training entry points
├── infrastructure/
│   └── terraform/
│       ├── main.tf               # All AWS resources (Lambda, API GW, S3, DynamoDB, ECR, SNS, CW, EB)
│       └── variables.tf          # Configurable deployment variables
├── .github/
│   └── workflows/
│       ├── ci.yml                # CI: test + train + API integration (every push)
│       └── deploy.yml            # CD: build Docker → push ECR → update Lambda (every merge to main)
├── Dockerfile
├── Dockerfile.lambda             # Lambda-specific container image (linux/amd64)
├── docker-compose.yml
└── requirements.txt
```

---

## Phase-by-Phase Implementation

### Phase 1–3: Environment, Data Setup & ETL

A modular data pipeline under `data_pipeline/` handles the full ingestion-to-split workflow:

| Module | Responsibility |
|---|---|
| `ingestion.py` | Reads raw datasets from `datasets/` |
| `cleaning.py` | Deduplication, missing value checks |
| `feature_engineering.py` | Dataset-specific transforms (TF-IDF, encoding, scaling) |
| `split_data.py` | Stratified train/test splitting |
| `run_pipeline.py` | Orchestrates the full ETL run |

**Dataset-specific engineering:**
- **Spam**: TF-IDF with 5,000 features on raw message text — high-dimensional sparse matrix
- **Customers**: Categorical encoding + standard scaling for segmentation readiness
- **Fraud**: Normalization + balanced split validation for heavily imbalanced classes

---

### Phase 4–7: Scratch Algorithms, Evaluation & Experiment Framework

Three head-to-head benchmark scripts in `experiments/` run scratch implementations against sklearn baselines under an identical data and preprocessing protocol.

**Evaluation depth per task:**

| Task Type | Metrics |
|---|---|
| Classification | Accuracy, Precision, Recall, F1, AUC, cross-validation |
| Clustering | Inertia, Silhouette, Davies-Bouldin, Calinski-Harabasz, CV silhouette |
| Anomaly Detection | ROC-AUC, Avg Precision, Precision, Recall, F1, CV AUC |

All plots and artifacts are logged to MLflow automatically.

---

### Phase 8: FastAPI Model Serving

`api/app.py` exposes a production-ready inference API:

| Endpoint | Method | Description |
|---|---|---|
| `/predict/spam` | POST | Single spam classification |
| `/predict/cluster` | POST | Customer segment assignment |
| `/predict/fraud` | POST | Fraud anomaly score + decision |
| `/predict/batch/spam` | POST | Bulk spam inference |
| `/experiment/status` | GET | Current A/B experiment configuration and traffic split |
| `/experiment/configure` | POST | Configure champion/challenger split or shadow mode |
| `/health` | GET | Runtime readiness + model load state |
| `/metrics` | GET | Prometheus scrape target |

**Model loading strategy:**
1. **Primary**: MLflow Model Registry (by stage: `Production`)
2. **Fallback**: Latest experiment artifact run
3. **Defensive dtype casting** (float64) to prevent sklearn runtime errors at inference time

---

### Phase 9: Containerization

```bash
docker compose up -d --build
```

Spins up the full observability stack:

| Service | Port | Purpose |
|---|---|---|
| API (FastAPI) | 8000 | Inference endpoints |
| MLflow | 5000 | Tracking UI + artifact store |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Dashboard visualization |
| Alertmanager | 9093 | Alert routing |

**Docker design notes:**
- API container runs as non-root for security
- Filesystem permissions hardened for matplotlib/temp artifact writing
- Compose networking uses stable service names for inter-service DNS
- `.dockerignore` excludes datasets, model artifacts, and dev files to minimize build context

---

### Phase 10: CI/CD (GitHub Actions)

Two workflows in `.github/workflows/`:

**`ci.yml` — Continuous Integration (every push):**

```
checkout + setup Python
      │
      ▼
install requirements
      │
      ▼
compileall syntax check + smoke imports
      │
      ▼
pytest -v tests/  (contracts, drift logic, retraining triggers, A/B manager)
      │
      ▼
rewrite config.yaml (CI-fast hyperparams)
      │
      ▼
generate synthetic processed datasets
      │
      ▼
run all 3 experiment scripts (training + MLflow logging)
      │
      ▼
start FastAPI + wait for /health
      │
      ▼
run api/test_api.py endpoint integration tests
```

**`deploy.yml` — Continuous Deployment (push to main):**

```
Run tests (ci job)
      │
      ▼
Configure AWS credentials
      │
      ▼
Login to ECR + build Docker image (linux/amd64, Dockerfile.lambda)
      │
      ▼
Push image to ECR (tagged: latest + commit SHA)
      │
      ▼
Update Lambda function code with new image URI
      │
      ▼
Wait for Lambda update + health check
```

Every commit proves the code is healthy, training works, and the API serves correctly. Every merge to `main` auto-deploys to AWS Lambda.

---

### Phase 11: Monitoring & Alerting

`api/monitoring.py` instruments every API request with Prometheus metrics:

| Metric | Type | Labels |
|---|---|---|
| `api_requests_total` | Counter | endpoint, method, status code |
| `api_request_latency_seconds` | Histogram | endpoint (p50, p95, p99) |
| `prediction_distribution` | Gauge | model, label |
| `data_drift_score` | Gauge | model |
| `model_available` | Gauge | model (0 or 1) |

**Grafana panels:** live traffic rate, p95 latency trend, prediction class distribution, drift score timeline with threshold reference lines, model availability state per model.

**Alertmanager rules:**

| Alert | Condition | Severity |
|---|---|---|
| `HighLatency` | p95 > configured threshold | warning |
| `HighErrorRate` | 5xx rate spike | critical |
| `ModelUnavailable` | `model_available` == 0 | critical |
| `DriftWarning` | drift_score > warning threshold | warning |
| `DriftCritical` | drift_score > critical threshold | critical |

---

### Phase 12: Automatic Retraining with Prefect

`retraining/prefect_retraining_flow.py` implements a production-style triggered retraining pipeline.

**Trigger conditions evaluated on each run:**
1. Drift score queried from live Prometheus exceeds the configured threshold
2. Latest logistic model accuracy from MLflow drops below the minimum accuracy threshold

**Automated pipeline steps:**

```
Check triggers (Prometheus + MLflow)
        │
        ▼ (if triggered)
Run ETL pipeline (data_pipeline/run_pipeline.py)
        │
        ▼
Retrain selected models (scratch + sklearn)
        │
        ▼
Log to MLflow + register new model versions
        │
        ▼
Redeploy API (Docker exec or kubectl rollout restart)
```

Can be run manually for demonstration or scheduled via Kubernetes CronJob in production.

---

### Phase 13: Kubernetes Production Deployment

Organized with Kustomize for clean environment separation:

```
k8s/
├── base/               ← reusable manifests (all services)
└── overlays/dev/       ← dev-specific patches
```

**Production-grade capabilities implemented:**

| Feature | Manifest |
|---|---|
| Namespace isolation (`mlops`) | `namespace.yaml` |
| API horizontal autoscaling | `hpa-api.yaml` |
| Pod disruption budget | `pdb-api.yaml` |
| Network ingress/egress rules | `networkpolicy.yaml` |
| Liveness / readiness / startup probes | all Deployments |
| Persistent volumes (MLflow + Grafana) | `pvc-*.yaml` |
| ConfigMap + Secret separation | `configmap.yaml`, `secret.yaml` |
| Ingress routing | `ingress.yaml` |
| Scheduled retraining | `cronjob-retraining.yaml` |
| Resource requests + limits | all Deployments |

---

### Phase 14: AWS Production Deployment

A fully serverless production environment provisioned with Terraform — one command deploys everything.

See the [AWS Production Deployment](#aws-production-deployment) section for full details.

---

## Algorithms Built From Scratch — Deep Dive

> These are not sklearn wrappers. Each algorithm was implemented from mathematical foundations and benchmarked against sklearn baselines under a reproducible evaluation protocol.

---

### 1. Logistic Regression — Spam Classification

#### Design Choices

| Design Choice | What It Does | Why It Matters |
|---|---|---|
| Adaptive learning rate decay | Reduces LR as training progresses | Avoids oscillation near convergence; sklearn's LBFGS uses a fixed line-search |
| Early stopping with patience control | Halts when validation loss plateaus | Prevents overfitting; configurable without re-wrapping |
| Custom threshold calibration | Post-training PR curve scan to select operating threshold | Directly optimizes the precision-recall tradeoff — sklearn defaults to 0.5 |
| Numeric-safe sigmoid | Clips extreme values before exponentiation | Prevents NaN gradients on high-dimensional sparse inputs |
| Sparsity-aware gradient pass | Sparse matrix multiplication in gradient updates | Stable training on 5,000-feature TF-IDF without dense conversion |

#### Mathematical Core

```
Forward pass:    p = σ(Xw + b),   where σ(z) = 1 / (1 + e^(-z))

Regularized loss: L = -(1/N) Σ [y·log(p) + (1-y)·log(1-p)] + λ||w||²

Gradient updates:
    ∂L/∂w = (1/N) Xᵀ(p - y) + 2λw
    ∂L/∂b = (1/N) Σ(p - y)

Learning rate schedule:  lr_t = lr_0 / (1 + decay * t)
```

#### Operating Point Caveat

Spam detection is an asymmetric cost problem — a missed spam (false negative) costs more than a false alarm (false positive). The reported recall uplift depends on threshold choice. Fair comparison requires threshold tuning for both scratch and sklearn models on the same validation protocol (see [Benchmark Protocol](#benchmark-protocol)).

---

### 2. K-Means — Customer Segmentation

#### Design Choices

| Design Choice | What It Does | Why It Matters |
|---|---|---|
| KMeans++ initialization | Probabilistic centroid seeding proportional to distance² | Avoids poor local minima from random init; exposes the full seeding trajectory |
| Multi-restart with best-inertia selection | Runs n_init independent restarts, keeps lowest inertia | Full diagnostic output per restart |
| Dual convergence criterion | Stops on both centroid drift AND assignment stability | More principled than a single centroid-only check |
| Auto-k elbow analysis | Trains across k range and identifies elbow before final fit | Built-in model selection without manual wrapping |
| Full diagnostics | Centroid evolution, cluster size distribution, distance statistics logged per iteration | Enables segmentation explainability |

#### Mathematical Core

```
Assignment step:   c(xᵢ) = argmin_j ||xᵢ - μⱼ||²

Update step:       μⱼ = (1/|Cⱼ|) Σᵢ∈Cⱼ xᵢ

Objective:         minimize J = Σᵢ ||xᵢ - μ_{c(xᵢ)}||²  (within-cluster SSQ)

KMeans++ seeding:  P(xᵢ chosen) ∝ min_j ||xᵢ - μⱼ||²
```

---

### 3. Isolation Forest — Credit Card Fraud Detection

#### Design Choices

| Design Choice | What It Does | Why It Matters |
|---|---|---|
| Extended split geometry | Uses random hyperplane splits, not just axis-aligned cuts | Captures anomalies in correlated feature spaces — critical for financial fraud where features co-vary |
| Subset sampling with contamination alignment | Subsample size tuned to realistic fraud prevalence | Avoids over-isolating normal points in extreme class imbalance scenarios |
| Post-score threshold optimization | Scans score thresholds to maximize recall | Fraud detection is recall-first: a missed fraud has far higher business cost than a false alert |
| Path-length normalization | Normalizes isolation depth by expected depth for sample size | Correct anomaly score interpretation independent of tree depth and sample count |

#### Mathematical Core

```
Anomaly score:  s(x, n) = 2^( -E[h(x)] / c(n) )

where:
  h(x)   = path length to isolate x in a tree
  E[h(x)]= expected path length across all trees
  c(n)   = average path length for a dataset of size n
           c(n) = 2·H(n-1) - (2(n-1)/n),  H = harmonic number

Extended split: split on  wᵀx ≤ threshold  (random weight vector w)
                vs. axis-aligned  xⱼ ≤ threshold
```

---

## Detailed Experiment Results & Analysis

### Logistic Regression (Spam Classification)

| Metric | Scratch | sklearn | Delta |
|---|---|---|---|
| **Accuracy** | **0.9725** | 0.9617 | +1.1% |
| Precision | 0.9208 | **0.9634** | -4.2% |
| **Recall** | **0.8611** | 0.7315 | **+17.7%** |
| **F1** | **0.8900** | 0.8316 | **+7.0%** |
| AUC | 0.9867 | **0.9922** | -0.6% |
| Training Time | 0.48s | **0.04s** | ~12× slower |

**Cross-Validation:**

| | Mean Accuracy | Std Dev |
|---|---|---|
| Scratch | **0.9734** | ±0.0029 |
| sklearn | 0.9533 | ±0.0032 |

**Key insight:** The recall uplift is meaningful for spam, but only when both models are evaluated at comparable operating points. Threshold-tuned results are the primary comparison; default-threshold results serve as secondary reference.

![Logistic Result 1](images/logisticalgo-results.PNG)
![Logistic Result 2](images/logisticalgo-results2.PNG)

---

### K-Means (Customer Segmentation)

| Metric | Scratch | sklearn | Delta |
|---|---|---|---|
| Inertia | 601.36 | **600.56** | ~same |
| **Silhouette** | 0.3063 | **0.3076** | ~same |
| Davies-Bouldin | 1.1079 | 1.1098 | ~same |
| Calinski-Harabasz | 1095.33 | **1097.14** | ~same |
| **Training Time** | **0.1719s** | 8.1511s | **47× faster** |

**Cross-Validation Silhouette:**

| | Mean | Std Dev |
|---|---|---|
| Scratch | **0.3045** | ±0.0057 |
| sklearn | 0.3022 | ±0.0089 |

**Key insight:** Statistically identical clustering quality at 47× the speed on this dataset and hardware. Lower CV variance suggests more stable cluster assignments across folds.

---

### Isolation Forest (Credit Card Fraud Detection)

| Metric | Scratch | sklearn | Delta |
|---|---|---|---|
| **ROC-AUC** | **0.9409** | 0.9364 | +0.5% |
| **Avg Precision** | **0.1705** | 0.1224 | **+39.3%** |
| Precision | 0.0356 | **0.0628** | -43% |
| **Recall** | **0.8105** | 0.6632 | **+22.2%** |
| F1 | 0.0682 | **0.1148** | -41% |
| **Training Time** | **10.97s** | 36.22s | **3.3× faster** |

**Cross-Validation ROC-AUC:**

| | Mean | Std Dev |
|---|---|---|
| Scratch | **0.9521** | ±0.0163 |
| sklearn | 0.9475 | ±0.0161 |

**Key insight:** The precision/F1 tradeoff is deliberate — this operating point prioritizes recall for fraud detection. The precision gap reflects an asymmetric cost function, not a general quality deficit. Report both threshold-tuned and default-threshold comparisons to avoid overstating general superiority.

---

### Algorithm Design Summary

| Algorithm | Business Metric | Scratch vs. sklearn |
|---|---|---|
| Logistic Regression | Recall (catch more spam) | **+17.7% recall, +7.0% F1** |
| K-Means | Training efficiency | **47× faster, equal clustering quality** |
| Isolation Forest | Recall (catch more fraud) | **+22.2% recall, +39.3% avg precision** |

---

## Benchmark Protocol

All scratch vs. sklearn benchmarks follow a strict reproducibility protocol to prevent optimistic or apples-to-oranges comparisons.

**1. Same data split**
Use identical train/validation/test partitions and random seeds for both models. Keep a fixed holdout test set that is never used for threshold tuning.

**2. Same preprocessing**
Fit preprocessing objects (vectorizer, encoders, scalers) on training data only. Apply identical transformed features to both implementations.

**3. Threshold tuning for both models**
For classification and anomaly tasks, tune decision threshold on validation data for both models under the same objective (F1, recall@precision>=X, or cost-weighted utility). Report two views:
- `Default threshold` — reference only
- `Tuned threshold` — primary decision-quality comparison

**4. Uncertainty and confidence intervals**
Run repeated evaluation (repeated stratified CV or multiple seeds). Report mean ± 95% CI for each key metric.

**5. Compute fairness**
Compare training and inference times on the same machine, same thread settings, and same input dimensionality. Include hardware and software metadata with each benchmark run.

**6. Claim discipline**
Prefer wording like "in this dataset/run" rather than universal superiority claims. Treat differences as provisional unless confidence intervals are clearly separated.

---

## MLflow Tracking & Model Registry

**Tracking utility:** `tracking/mlflow_tracker.py`

**Logged per run:**
- All hyperparameters and config context
- Scalar metrics (per split + per CV fold)
- Artifacts: ROC curves, confusion matrices, PR curves, training loss histories, dataset metadata
- Model signatures and input examples (for API compatibility)
- Model versions registered to the registry

**Model lifecycle:**

```
Training Run
    │
    ▼
MLflow Experiment Log (params + metrics + artifacts)
    │
    ▼
Model Registry Registration (api/register_models.py)
    │
    ▼
  None ──► Staging ──► Production ──► Archived
                           │
                           ▼
                    FastAPI loads by stage name
```

![MLflow Overview](images/mlflow-overview.PNG)
![MLflow Experiments](images/mlflow-experiments.PNG)
![MLflow Comparison](images/mlflow-comparison.PNG)
![MLflow Staging Models](images/mlflow-staging-model.PNG)
![MLflow Versions](images/mlflow-versionofmodels-show.PNG)
![MLflow Register Staging Production](images/registermodelsstagingproduction-mlflow.PNG)

---

## FastAPI Model Serving

**Service:** `api/app.py`

| Feature | Implementation |
|---|---|
| Request validation | Pydantic schemas with typed fields |
| Model loading | Registry-first with experiment artifact fallback |
| Dtype safety | Explicit float64 casting before sklearn inference |
| Observability | Every request emits Prometheus metrics via `monitoring.py` |
| Batch support | `/predict/batch/spam` for bulk inference |
| Health probes | `/health` returns model load state + readiness for Kubernetes probes |
| A/B routing | `experiment_manager.py` routes traffic per configured split |

**Startup sequence:**

```
API boot
  → initialize ModelLoader (registry-first)
  → initialize ExperimentManager (champion/challenger config)
  → initialize MonitoringManager (Prometheus)
  → register /metrics scrape endpoint
  → expose /health for orchestration probes
  → ready to serve
```

![API Health](images/apihealth.PNG)
![API Inference Result](images/apiresult8080.PNG)
![API Metrics Endpoint](images/8080ort-metrics.PNG)

---

## A/B Testing & Experiment Management

**Service:** `api/experiment_manager.py`

Production ML systems require controlled rollout mechanisms before committing to a new model version. The `ExperimentManager` provides:

| Feature | Behavior |
|---|---|
| Champion/challenger traffic split | Configurable percentage of live traffic routed to challenger model |
| Shadow deployment mode | 100% of traffic served by champion; challenger runs in background for evaluation without user impact |
| Automatic rollback | Rolls back to champion if challenger error rate exceeds 5% or p95 latency breaches the configured threshold |
| Full test coverage | `pytest` suite covering split logic, shadow mode, rollback triggers, and configuration validation |

**Configuration example:**

```python
# Route 10% of traffic to challenger, shadow mode off
POST /experiment/configure
{
  "champion": "logistic_v3",
  "challenger": "logistic_v4",
  "traffic_split": 0.10,
  "shadow_mode": false
}

# Enable shadow mode — challenger receives all traffic but responses are discarded
POST /experiment/configure
{
  "shadow_mode": true
}
```

**Auto-rollback triggers:**
- Challenger error rate > 5% over a rolling window
- Challenger p95 latency breach above the configured SLO threshold

---

## CI/CD Pipeline

**Files:** `.github/workflows/ci.yml` · `.github/workflows/deploy.yml`

### Continuous Integration (every push)

```yaml
jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - checkout + Python 3.10 setup
      - pip install -r requirements.txt
      - python -m compileall .
      - smoke import tests for all algorithm classes
      - pytest -v tests/  (contracts, drift, retraining triggers, A/B manager)
      - rewrite configs/config.yaml (CI-fast epochs)
      - generate synthetic processed datasets (schema-consistent)
      - python experiments/exp_logistic_vs_sklearn.py
      - python experiments/exp_kmeans_vs_sklearn.py
      - python experiments/exp_isolation_forest_vs_sklearn.py
      - uvicorn api/app.py --background
      - wait for /health readiness (200 only when models + MLflow are ready)
      - python api/test_api.py
```

**What this guards against:** broken imports or syntax regressions, config changes that break training pipelines, local-only file assumptions, API startup failures or contract breakages, accidental model format incompatibilities.

CI also generates benchmark artifacts (`results/benchmark_ci_report.md` and `results/benchmark_ci_report.json`) uploaded as GitHub Actions artifacts on every run.

### Continuous Deployment (push to main)

```yaml
jobs:
  deploy:
    needs: ci
    steps:
      - Configure AWS credentials
      - Login to ECR + build Dockerfile.lambda (linux/amd64)
      - Push image to ECR (tagged: latest + commit SHA)
      - Update Lambda function code with new image URI
      - Wait for Lambda update + health check
```

---

## Monitoring, Drift Detection & Alerting

### Prometheus Metrics

```
api_requests_total{endpoint, method, status}     Counter
api_request_latency_seconds{endpoint}            Histogram (p50, p95, p99)
prediction_distribution{model, label}            Gauge
data_drift_score{model}                          Gauge
model_available{model}                           Gauge (0 or 1)
```

### Grafana Dashboard Panels

- Request rate per endpoint (last 5 min)
- p95 latency trend (rolling window)
- Prediction class distribution (spam/not-spam, fraud/normal)
- Drift score over time with threshold reference lines
- Model availability state (green/red per model)

### Alertmanager Rules

| Alert | Condition | Severity |
|---|---|---|
| `HighLatency` | p95 > configured threshold | warning |
| `HighErrorRate` | 5xx rate > threshold | critical |
| `ModelUnavailable` | `model_available` == 0 | critical |
| `DriftWarning` | drift_score > warning threshold | warning |
| `DriftCritical` | drift_score > critical threshold | critical |

![Prometheus Main](images/prometheus.PNG)
![Prometheus Query](images/prometheous2.PNG)
![Grafana Dashboard](images/grafana.PNG)
![Alertmanager](images/alert-manager.PNG)

---

## Automatic Retraining with Prefect

**Flow:** `retraining/prefect_retraining_flow.py`

### Trigger Conditions

```python
# Condition 1: Drift trigger
drift_score = query_prometheus("data_drift_score")
if drift_score > config["drift_threshold"]:
    trigger_retraining()

# Condition 2: Accuracy trigger
latest_accuracy = query_mlflow_latest_run("logistic_accuracy")
if latest_accuracy < config["min_accuracy_threshold"]:
    trigger_retraining()
```

### Automated Steps

```
1. Evaluate trigger conditions (Prometheus query + MLflow API)
2. Run ETL pipeline (data_pipeline/run_pipeline.py)
3. Retrain selected models (scratch + sklearn comparison)
4. Log new runs to MLflow
5. Register new model versions → promote to Staging
6. Redeploy API:
   - Local:      docker compose restart api
   - Kubernetes: kubectl rollout restart deployment/mlops-api -n mlops
```

**Deployment modes:**
- `python retraining/prefect_retraining_flow.py` — manual / demonstration
- Kubernetes CronJob — production scheduled (configurable interval)

![Prefect Run 1](images/prefect-showing.PNG)
![Prefect Run 2](images/prefect2.PNG)

---

## Docker & Kubernetes Deployment

### Docker (Local Full Stack)

```bash
docker compose up -d --build
```

| Service | Port | URL |
|---|---|---|
| FastAPI | 8000 | http://localhost:8000 |
| MLflow | 5000 | http://localhost:5000 |
| Prometheus | 9090 | http://localhost:9090 |
| Grafana | 3000 | http://localhost:3000 |
| Alertmanager | 9093 | http://localhost:9093 |

### Kubernetes (Production-Style)

```bash
kubectl apply -k k8s/overlays/dev
kubectl get pods -n mlops
```

| Manifest | Purpose |
|---|---|
| `hpa-api.yaml` | Scale API pods on CPU/memory load |
| `pdb-api.yaml` | Maintain minimum available replicas during disruptions |
| `networkpolicy.yaml` | Restrict pod-to-pod communication to necessary paths |
| `pvc-mlflow.yaml` | Persistent MLflow artifact + run storage |
| `pvc-grafana.yaml` | Persistent Grafana dashboard state |
| `ingress.yaml` | Route external traffic to all services |
| `cronjob-retraining.yaml` | Scheduled Prefect retraining flow |

![Kubernetes Running Stack](images/kubernate-running.PNG)
![Ingress and Autoscaling](images/ingress+autoscaling.PNG)

---

## AWS Production Deployment

Full serverless production infrastructure provisioned with Terraform — **one command deploys everything**.

### AWS Services Deployed

| Service | Resource | Purpose |
|---|---|---|
| Lambda | `mlops-hassan-api` | Serverless inference — 512MB, 60s timeout, container image |
| API Gateway | `mlops-hassan-api` | Public HTTPS endpoint |
| ECR | `mlops-hassan-api` | Docker image registry (linux/amd64 Lambda image) |
| S3 | `mlops-hassan-artifacts-*` | Model `.pkl` files |
| DynamoDB | `mlops-hassan-requests` | Per-request logging, pay-per-request billing |
| SNS | `mlops-hassan-alerts` | Email alerts on API 5xx errors |
| CloudWatch | `mlops-hassan-api-5xx` | Alarm triggers SNS when error rate spikes |
| EventBridge | `mlops-hassan-daily-drift` | Triggers Lambda daily for drift detection |
| IAM | `mlops-hassan-lambda-role` | Least-privilege role: S3, DynamoDB, ECR, SNS, CloudWatch |

### Deploy

```bash
cd infrastructure/terraform
terraform init
terraform apply -auto-approve
```

### Model Loading on Lambda

Lambda bypasses the MLflow registry on cold start — models are loaded directly from S3 as `.pkl` files via joblib, eliminating the MLflow dependency in the serverless environment:

```
Lambda cold start
     │
     ▼
Download logistic_regression.pkl from S3
Download kmeans.pkl from S3
Download isolation_forest.pkl from S3
     │
     ▼
All 3 models in memory — ready to serve
```

### Live Endpoint

**Base URL:** `https://f3frttac81.execute-api.us-east-1.amazonaws.com/dev`

```bash
# Health check
curl https://f3frttac81.execute-api.us-east-1.amazonaws.com/dev/health

# Spam detection
curl -X POST https://f3frttac81.execute-api.us-east-1.amazonaws.com/dev/predict/spam \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.0, 0.5, 0.2, 0.0]}'

# Fraud detection
curl -X POST https://f3frttac81.execute-api.us-east-1.amazonaws.com/dev/predict/fraud \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5]}'

# Customer clustering
curl -X POST https://f3frttac81.execute-api.us-east-1.amazonaws.com/dev/predict/cluster \
  -H "Content-Type: application/json" \
  -d '{"features": [25.0, 50000.0, 2.0, 1.0]}'
```

![Lambda Function](public/images/mlops-hassan-api%20_%20Functions%20_%20Lambda%20-%20Google%20Chrome%205_25_2026%208_01_15%20PM.png)
![Lambda Configuration](public/images/mlops-hassan-api%20_%20Functions%20_%20Lambda%20-%20Google%20Chrome%205_25_2026%208_01_48%20PM.png)
![Lambda Environment](public/images/mlops-hassan-api%20_%20Functions%20_%20Lambda%20-%20Google%20Chrome%205_25_2026%208_02_55%20PM.png)
![S3 Buckets](public/images/S3%20buckets%20_%20S3%20_%20us-east-1%20-%20Google%20Chrome%205_25_2026%208_03_26%20PM.png)
![S3 Models Folder](public/images/mlops-hassan-artifacts-kq6qw72q%20-%20S3%20bucket%20_%20S3%20_%20us-east-1%20-%20Google%20Chrome%205_25_2026%209_06_45%20PM.png)
![ECR Repository](public/images/Elastic%20Container%20Registry%20-%20Private%20repositories%20-%20Google%20Chrome%205_25_2026%208_04_22%20PM.png)
![CloudWatch Alarm](public/images/CloudWatch%20_%20us-east-1%20-%20Google%20Chrome%205_25_2026%208_04_57%20PM.png)
![IAM Role](public/images/Roles%20_%20IAM%20_%20Global%20-%20Google%20Chrome%205_25_2026%209_11_11%20PM.png)
![Terraform Apply](public/images/MINGW64__c_Users_PMY_Desktop_ARTIFICIAL%20INTELLIGENCE_MachineLearningProjects_mlops-infrastructure_infrastructure_terraform%205_25_2026%208_00_23%20PM.png)
![Health Check — All Models Loaded](public/images/MINGW64__c_Users_PMY_Desktop_ARTIFICIAL%20INTELLIGENCE_MachineLearningProjects_mlops-infrastructure%205_25_2026%209_14_35%20PM.png)

---

## How to Run

### 1. Install

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Data Pipeline

```bash
python data_pipeline/run_pipeline.py
```

### 3. Run Experiments (Training + MLflow Logging)

```bash
python experiments/exp_logistic_vs_sklearn.py
python experiments/exp_kmeans_vs_sklearn.py
python experiments/exp_isolation_forest_vs_sklearn.py
```

### 4. Register Models to MLflow Registry

```bash
python api/register_models.py
```

### 5. Start Full Local Stack

```bash
docker compose up -d --build
```

### 6. Run Automated Test Suite

```bash
pytest -v tests/
```

Coverage includes: model loading fallback behavior, endpoint contracts and validation, drift metric computation and feature-shape mismatch handling, retraining trigger behavior (skip, drift-triggered, force mode), and A/B experiment manager logic.

### 7. Run API Integration Tests

```bash
python api/test_api.py
```

### 8. Run Prefect Retraining Flow

```bash
python retraining/prefect_retraining_flow.py
```

### 9. Deploy to AWS (Terraform)

```bash
# Upload models to S3
BUCKET=$(aws s3 ls | grep mlops-hassan-artifacts | awk '{print $3}')
aws s3 cp experiments/mlruns/1/models/.../model.pkl s3://$BUCKET/models/logistic_regression.pkl
aws s3 cp experiments/mlruns/2/models/.../model.pkl s3://$BUCKET/models/kmeans.pkl
aws s3 cp experiments/mlruns/4/models/.../model.pkl s3://$BUCKET/models/isolation_forest.pkl

# Provision infrastructure
cd infrastructure/terraform
terraform init
terraform apply -auto-approve
```

### 10. Deploy to Kubernetes

```bash
kubectl apply -k k8s/overlays/dev
kubectl get pods -n mlops
```

### 11. Kubernetes Port-Forwards

```bash
kubectl port-forward -n mlops svc/mlops-api     8001:8000 --address 127.0.0.1
kubectl port-forward -n mlops svc/mlflow        5001:5000 --address 127.0.0.1
kubectl port-forward -n mlops svc/grafana       3001:3000 --address 127.0.0.1
kubectl port-forward -n mlops svc/prometheus    9091:9090 --address 127.0.0.1
kubectl port-forward -n mlops svc/alertmanager  9094:9093 --address 127.0.0.1
```

---

## Screenshot Walkthrough

### MLflow — Experiments, Comparison & Registry

| | |
|---|---|
| ![MLflow Overview](images/mlflow-overview.PNG) | ![MLflow Experiments](images/mlflow-experiments.PNG) |
| ![MLflow Comparison](images/mlflow-comparison.PNG) | ![MLflow Staging](images/mlflow-staging-model.PNG) |

### FastAPI — Health, Inference & Metrics

| | |
|---|---|
| ![API Health](images/apihealth.PNG) | ![API Inference](images/apiresult8080.PNG) |
| ![API Metrics](images/8080ort-metrics.PNG) | |

### Monitoring — Prometheus, Grafana & Alertmanager

| | |
|---|---|
| ![Prometheus](images/prometheus.PNG) | ![Prometheus Query](images/prometheous2.PNG) |
| ![Grafana](images/grafana.PNG) | ![Alertmanager](images/alert-manager.PNG) |

### Prefect Retraining

| | |
|---|---|
| ![Prefect Run 1](images/prefect-showing.PNG) | ![Prefect Run 2](images/prefect2.PNG) |

### Kubernetes

| | |
|---|---|
| ![Kubernetes Stack](images/kubernate-running.PNG) | ![HPA + Ingress](images/ingress+autoscaling.PNG) |

### Algorithm Results

| | |
|---|---|
| ![Logistic Result 1](images/logisticalgo-results.PNG) | ![Logistic Result 2](images/logisticalgo-results2.PNG) |

### AWS Production Deployment

| | |
|---|---|
| ![Lambda](public/images/mlops-hassan-api%20_%20Functions%20_%20Lambda%20-%20Google%20Chrome%205_25_2026%208_01_15%20PM.png) | ![Lambda Config](public/images/mlops-hassan-api%20_%20Functions%20_%20Lambda%20-%20Google%20Chrome%205_25_2026%208_01_48%20PM.png) |
| ![Lambda Environment](public/images/mlops-hassan-api%20_%20Functions%20_%20Lambda%20-%20Google%20Chrome%205_25_2026%208_02_55%20PM.png) | ![Terraform Apply](public/images/MINGW64__c_Users_PMY_Desktop_ARTIFICIAL%20INTELLIGENCE_MachineLearningProjects_mlops-infrastructure_infrastructure_terraform%205_25_2026%208_00_23%20PM.png) |
| ![S3 Bucket](public/images/S3%20buckets%20_%20S3%20_%20us-east-1%20-%20Google%20Chrome%205_25_2026%208_03_26%20PM.png) | ![S3 Models](public/images/mlops-hassan-artifacts-kq6qw72q%20-%20S3%20bucket%20_%20S3%20_%20us-east-1%20-%20Google%20Chrome%205_25_2026%209_06_45%20PM.png) |
| ![ECR](public/images/Elastic%20Container%20Registry%20-%20Private%20repositories%20-%20Google%20Chrome%205_25_2026%208_04_22%20PM.png) | ![CloudWatch](public/images/CloudWatch%20_%20us-east-1%20-%20Google%20Chrome%205_25_2026%208_04_57%20PM.png) |
| ![IAM Role](public/images/Roles%20_%20IAM%20_%20Global%20-%20Google%20Chrome%205_25_2026%209_11_11%20PM.png) | ![Health Check — All Models Loaded](public/images/MINGW64__c_Users_PMY_Desktop_ARTIFICIAL%20INTELLIGENCE_MachineLearningProjects_mlops-infrastructure%205_25_2026%209_14_35%20PM.png) |

---

<div align="center">

**Built by Muhammad Hassan Shahbaz**

*ML Engineer · MLOps Engineer · Platform Engineer*

[![GitHub](https://img.shields.io/badge/GitHub-Rana--Hassan7272-181717?style=flat-square&logo=github)](https://github.com/Rana-Hassan7272)

</div>