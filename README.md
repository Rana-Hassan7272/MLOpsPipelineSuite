# MLOpsPipelineSuite — End-to-End MLOps Infrastructure

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Production--Style-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-Dashboards-F46800?style=for-the-badge&logo=grafana&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)
![Prefect](https://img.shields.io/badge/Prefect-Orchestration-3D3D3D?style=for-the-badge&logo=prefect&logoColor=white)

**A production-grade ML platform built entirely from scratch — covering the full ML lifecycle from raw data ingestion to Kubernetes-orchestrated, auto-retraining model serving with real-time observability.**

*Designed as a capstone demonstrating ML engineering depth, MLOps platform breadth, and production deployment capability.*

</div>

---

## What Makes This Project Different

Most ML projects end at model training. This one starts there.

This platform implements what a real ML engineering team builds and maintains in production:

- **3 custom algorithms** built from mathematical foundations — not just wrapped sklearn — benchmarked against sklearn under explicit evaluation protocol
- **Full MLflow lifecycle**: experiment tracking → model registry → staged promotion → API serving, all wired together end-to-end
- **Observable, self-healing API service** with Prometheus metrics, Grafana dashboards, and Alertmanager rules for drift and latency
- **Automatic retraining pipeline** orchestrated by Prefect, triggered by drift signals from the live Prometheus stack
- **Kubernetes production deployment** with HPA autoscaling, PodDisruptionBudget, NetworkPolicy, liveness/readiness probes, and a scheduled retraining CronJob
- **Reproducible CI/CD** via GitHub Actions — every commit trains models and tests the full serving stack

---

## Table of Contents

1. [Business Goal](#business-goal)
2. [Tech Stack](#tech-stack)
3. [System Architecture](#system-architecture)
4. [Repository Structure](#repository-structure)
5. [Phase-by-Phase Implementation](#phase-by-phase-implementation)
6. [Algorithms Built From Scratch — Deep Dive](#algorithms-built-from-scratch--deep-dive)
7. [Detailed Experiment Results & Analysis](#detailed-experiment-results--analysis)
8. [Benchmark Protocol (Fairness & Reproducibility)](#benchmark-protocol-fairness--reproducibility)
9. [MLflow Tracking & Model Registry](#mlflow-tracking--model-registry)
10. [FastAPI Model Serving](#fastapi-model-serving)
11. [CI/CD Pipeline](#cicd-pipeline)
12. [Monitoring, Drift Detection & Alerting](#monitoring-drift-detection--alerting)
13. [Automatic Retraining with Prefect](#automatic-retraining-with-prefect)
14. [Docker & Kubernetes Deployment](#docker--kubernetes-deployment)
15. [How to Run](#how-to-run)
16. [Screenshot Walkthrough](#screenshot-walkthrough)

---

## Business Goal

Build a robust ML platform where models are not only trained, but:

| Capability | Implementation |
|---|---|
| Benchmarked vs. baselines | Custom scratch vs. sklearn comparison across 3 algorithms |
| Tracked and versioned | MLflow experiment tracking + model registry with staged promotion |
| Served as APIs | FastAPI with registry-first model loading, batch support, health probes |
| Monitored in production | Prometheus + Grafana + Alertmanager with drift and SLO alerts |
| Auto-retrained | Prefect flow triggered by live drift/accuracy signals |
| Cluster-deployed | Kubernetes with HPA, PDB, NetworkPolicy, CronJob retraining |

---

## Tech Stack

### ML / Data
| Tool | Usage |
|---|---|
| Python 3.10+ | Core language |
| NumPy / Pandas | Data processing |
| scikit-learn | Baseline comparison + preprocessing utilities |
| TF-IDF (custom) | Spam feature engineering |

### Custom Algorithms (Scratch)
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
│  │                                                          │                │
│  │  registry-first model loading  →  fallback artifacts     │                │
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
│   ├── register_models.py        # MLflow registry promotion
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
├── .github/workflows/ci.yml      # GitHub Actions CI
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── roadmap.md
```

---

## Phase-by-Phase Implementation

### Phase 1–3: Environment, Data Setup & ETL

Implemented a modular data pipeline under `data_pipeline/`:

| Module | Responsibility |
|---|---|
| `ingestion.py` | Reads raw datasets from `datasets/` |
| `cleaning.py` | Deduplication, missing value checks |
| `feature_engineering.py` | Dataset-specific transforms (TF-IDF, encoding, scaling) |
| `split_data.py` | Stratified train/test splitting |
| `run_pipeline.py` | Orchestrates full ETL run |

**Dataset-specific engineering:**
- **Spam**: TF-IDF with 5,000 features on raw message text — high-dimensional sparse matrix
- **Customers**: Categorical encoding + standard scaling for segmentation readiness
- **Fraud**: Normalization + balanced split validation for heavily imbalanced classes

---

### Phase 4–7: Scratch Algorithms, Evaluation & Experiment Framework

Three head-to-head benchmark scripts in `experiments/`:

- `exp_logistic_vs_sklearn.py`
- `exp_kmeans_vs_sklearn.py`
- `exp_isolation_forest_vs_sklearn.py`

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
| `/health` | GET | Runtime readiness + model load state |
| `/metrics` | GET | Prometheus scrape target |

**Model loading strategy:**
1. Primary: MLflow Model Registry (by stage: `Production`)
2. Fallback: Latest experiment artifact run
3. Defensive dtype casting (float64) to prevent sklearn runtime errors

---

### Phase 9: Containerization

```
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

---

### Phase 10: CI/CD (GitHub Actions)

Every push triggers `.github/workflows/ci.yml`:

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

This pipeline ensures every commit proves: code is healthy, training works, API serves correctly — in a clean environment with no local dependencies.

---

### Phase 11: Monitoring & Alerting

`api/monitoring.py` instruments every API request:

**Prometheus metrics collected:**
- `api_requests_total` — by endpoint, method, status code
- `api_request_latency_seconds` — histogram with p95 tracking
- `prediction_distribution` — class label distribution over time
- `data_drift_score` — feature distribution shift gauge
- `model_available` — binary availability state

**Grafana panels:**
- Live traffic rate
- p95 latency trend
- Prediction class distribution
- Drift score timeline
- Model health state

**Alertmanager rules:**
| Alert | Condition |
|---|---|
| `HighLatency` | p95 > threshold |
| `HighErrorRate` | 5xx rate spike |
| `ModelUnavailable` | model_available = 0 |
| `DriftWarning` | drift_score > warning threshold |
| `DriftCritical` | drift_score > critical threshold |

---

### Phase 12: Automatic Retraining with Prefect

`retraining/prefect_retraining_flow.py` implements a production-style triggered retraining pipeline:

**Trigger conditions checked on each run:**
1. Drift score from live Prometheus query exceeds configured threshold
2. Latest logistic model accuracy from MLflow drops below minimum threshold

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

Can be run manually for demo or scheduled via Kubernetes CronJob in production.

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

### Phase 14: Experiment Reports

Quantitative analysis reports in `results/`:
- `experiment_report_logistic.md` — full classification analysis with interpretation
- `experiment_report_kmean.md` — clustering quality + speed comparison
- `experiment_report_isolation.md` — anomaly detection benchmark + business justification

---

## Algorithms Built From Scratch — Deep Dive

> These are not sklearn wrappers. Each algorithm was implemented from mathematical foundations and benchmarked against sklearn baselines. Results are presented with operating-point caveats and a reproducible protocol.

---

### 1. Logistic Regression — Spam Classification

#### What Was Built

A custom binary classifier trained on high-dimensional sparse TF-IDF representations (5,000 features), with several implementation choices that go beyond sklearn defaults:

| Design Choice | What It Does | Why It Matters |
|---|---|---|
| Adaptive learning rate decay | Reduces LR as training progresses | Avoids oscillation near convergence; sklearn's LBFGS uses a fixed line-search |
| Early stopping with patience control | Halts when validation loss plateaus | Prevents overfitting; configurable without re-wrapping |
| Custom threshold calibration | Post-training PR curve scan to select operating threshold | Directly optimizes the precision-recall tradeoff for the spam use case — sklearn defaults to 0.5 |
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

#### Observed Result and Caveat

The core insight: this project prioritizes recall-oriented operating points for spam detection.

Spam detection is an asymmetric cost problem — a missed spam (false negative) costs more than a false alarm (false positive). The reported uplift depends on threshold choice, so fair comparison requires threshold tuning for both scratch and sklearn models on the same validation protocol (see benchmark protocol section).

---

### 2. K-Means — Customer Segmentation

#### What Was Built

A custom clustering implementation with initialization, convergence, and diagnostic behaviors that exceed sklearn's defaults:

| Design Choice | What It Does | Why It Matters |
|---|---|---|
| KMeans++ initialization | Probabilistic centroid seeding proportional to distance² | Avoids poor local minima that random init falls into; sklearn uses KMeans++ too but this implementation exposes the full seeding trajectory |
| Multi-restart with best-inertia selection | Runs n_init independent restarts, keeps lowest inertia | Same as sklearn, but with full diagnostic output per restart |
| Dual convergence criterion | Stops on both centroid drift AND assignment stability | More principled than sklearn's single centroid-only check |
| Auto-k elbow analysis | Trains across k range and identifies elbow before final fit | Built-in model selection; sklearn requires manual wrapping |
| Full diagnostics: centroid evolution, cluster size distribution, distance statistics | Logged per iteration | Enables segmentation explainability that sklearn's fit() hides |

#### Mathematical Core

```
Assignment step:   c(xᵢ) = argmin_j ||xᵢ - μⱼ||²

Update step:       μⱼ = (1/|Cⱼ|) Σᵢ∈Cⱼ xᵢ

Objective:         minimize J = Σᵢ ||xᵢ - μ_{c(xᵢ)}||²  (within-cluster SSQ)

KMeans++ seeding:  P(xᵢ chosen) ∝ min_j ||xᵢ - μⱼ||²
```

#### Observed Result and Caveat

On the current dataset and hardware, this run shows **47x lower training time** (0.17s vs 8.15s) at comparable clustering quality. Treat this as dataset/hardware-specific evidence, not a universal claim.

---

### 3. Isolation Forest — Credit Card Fraud Detection

#### What Was Built

A custom anomaly detector with several key innovations over the standard Isolation Forest algorithm:

| Design Choice | What It Does | Why It Matters |
|---|---|---|
| Extended split geometry | Uses random hyperplane splits, not just axis-aligned cuts | Captures anomalies in correlated feature spaces that axis-aligned trees miss — critical for financial fraud where features co-vary |
| Subset sampling with contamination alignment | Subsample size tuned to realistic fraud prevalence | sklearn's default contamination often over-isolates normal points in extreme class imbalance scenarios |
| Post-score threshold optimization | Scans score thresholds to maximize recall | Fraud detection is recall-first: a missed fraud (FN) has far higher business cost than a false alert (FP). sklearn defaults to a fixed percentile cutoff |
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

#### Observed Result and Caveat

The fraud detection operating point is asymmetric. In this run, the implementation achieves **ROC-AUC 0.9409 vs. sklearn's 0.9364** and **recall 0.8105 vs. 0.6632**. As with spam, recall differences are threshold-sensitive; fair interpretation requires matched threshold tuning and uncertainty reporting for both models.

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

**Key Insight:** Recall uplift is meaningful for spam, but only when both models are evaluated at comparable operating points. Use threshold-tuned results for both models as the main comparison, and keep default-threshold results as a secondary reference.

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

**Key Insight:** Statistically identical clustering quality at 47× the speed. The scratch implementation demonstrates that a well-vectorized custom K-Means with tight convergence criteria can match sklearn's heavily optimized implementation while being substantially faster in this data profile. Lower CV variance also suggests more stable cluster assignments across folds.

_Visualization note:_ dedicated K-Means plots are not yet exported separately in this repository. The numeric metrics table above is the primary evidence until model-specific figures are added.

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

**Key Insight:** The precision/F1 tradeoff is deliberate for recall-first fraud detection, but claims should be read as operating-point dependent. Report both threshold-tuned and default-threshold comparisons to avoid overstating general superiority.

_Visualization note:_ dedicated Isolation Forest plots are not yet exported separately in this repository. The numeric metrics table above is the primary evidence until model-specific figures are added.

---

### Algorithm Design Summary

| Algorithm | Business Metric | Scratch vs. sklearn |
|---|---|---|
| Logistic Regression | Recall (catch more spam) | **+17.7% recall, +7.0% F1** |
| K-Means | Training efficiency | **47× faster, equal quality** |
| Isolation Forest | Recall (catch more fraud) | **+22.2% recall, +39.3% avg precision** |

---

## Benchmark Protocol (Fairness & Reproducibility)

To prevent optimistic or apples-to-oranges comparisons, all scratch vs sklearn benchmarks should follow this strict protocol.

1. **Same data split**
   - Use identical train/validation/test partitions and identical random seeds for both models.
   - Keep a fixed holdout test set that is never used for threshold tuning.

2. **Same preprocessing**
   - Fit preprocessing objects (tokenizer/vectorizer, encoders, scalers, feature selectors) on training data only.
   - Apply the same transformed features to both scratch and sklearn implementations.

3. **Threshold tuning for both models**
   - For classification/anomaly tasks, tune decision threshold on validation data for both models under the same objective (for example F1, recall@precision>=X, or cost-weighted utility).
   - Report two views:
     - `Default threshold` (reference only)
     - `Tuned threshold` (primary decision-quality comparison)

4. **Uncertainty and confidence intervals**
   - Run repeated evaluation (for example repeated stratified CV or multiple seeds).
   - Report mean +- 95% confidence interval for each key metric:
     - Classification: accuracy, precision, recall, F1, ROC-AUC, PR-AUC
     - Clustering: silhouette, Davies-Bouldin, Calinski-Harabasz, inertia
     - Anomaly detection: ROC-AUC, PR-AUC, recall, precision, F1

5. **Compute fairness**
   - Compare training/inference times on the same machine, same thread settings, and same input dimensionality.
   - Include hardware and software metadata with each benchmark run.

6. **Claim discipline**
   - Prefer wording like "in this dataset/run" instead of universal superiority claims.
   - Treat differences as provisional unless confidence intervals are clearly separated.

This protocol is used to keep reported wins technically credible and reviewer-trustworthy.

---

## MLflow Tracking & Model Registry

**Tracking utility:** `tracking/mlflow_tracker.py`

**Logged per run:**
- All hyperparameters and config context
- Scalar metrics (per split + per CV fold)
- Artifacts: ROC curves, confusion matrices, PR curves, training loss histories, dataset metadata
- Model signatures and input examples (for API compatibility)
- Model versions in the registry

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

**Screenshots:**

![MLflow Overview](images/mlflow-overview.PNG)
![MLflow Experiments](images/mlflow-experiments.PNG)
![MLflow Comparison](images/mlflow-comparison.PNG)
![MLflow Staging Models](images/mlflow-staging-model.PNG)
![MLflow Versions](images/mlflow-versionofmodels-show.PNG)

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

**Startup sequence:**
```
API boot
  → initialize ModelLoader (registry-first)
  → initialize MonitoringManager (Prometheus)
  → register /metrics scrape endpoint
  → expose /health for orchestration probes
  → ready to serve
```

**Screenshots:**

![API Health](images/apihealth.PNG)
![API Inference Result](images/apiresult8080.PNG)
![API Metrics Endpoint](images/8080ort-metrics.PNG)

---

## CI/CD Pipeline

**File:** `.github/workflows/ci.yml`

Every push to the repository runs the full pipeline automatically:

```yaml
jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - checkout + Python 3.10 setup
      - pip install -r requirements.txt
      - python -m compileall .  (syntax check)
      - smoke import tests for all algorithm classes
      - pytest -v tests/  (contracts + drift + retraining triggers)
      - rewrite configs/config.yaml (CI-fast epochs)
      - generate synthetic processed datasets (schema-consistent)
      - python experiments/exp_logistic_vs_sklearn.py
      - python experiments/exp_kmeans_vs_sklearn.py
      - python experiments/exp_isolation_forest_vs_sklearn.py
      - uvicorn api/app.py --background
      - wait for /health readiness (200 only when models + MLflow are ready)
      - python api/test_api.py
```

**What this guards against:**
- Broken imports or syntax regressions
- Config changes that break training pipelines
- Local-only file assumptions
- API startup failures or contract breakages
- Accidental model format incompatibilities

**Validation snapshot (local):**
- `pytest -v` passing on full suite (API contracts, model loading, drift logic, retraining trigger behavior)
- integration endpoint checks passing via `api/test_api.py`
- CI also generates benchmark artifacts:
  - `results/benchmark_ci_report.md`
  - `results/benchmark_ci_report.json`
  - uploaded in GitHub Actions as artifact `benchmark-ci-report`

---

## Monitoring, Drift Detection & Alerting

### Prometheus Metrics (`api/monitoring.py`)

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
| HighLatency | p95 > configured_threshold | warning |
| HighErrorRate | 5xx rate > threshold | critical |
| ModelUnavailable | model_available{model} == 0 | critical |
| DriftWarning | drift_score > warning_threshold | warning |
| DriftCritical | drift_score > critical_threshold | critical |

**Screenshots:**

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
2. Ran ETL pipeline (data_pipeline/run_pipeline.py)
3. Retrain selected models (scratch + sklearn comparison)
4. Log new runs to MLflow
5. Register new model versions → promote to Staging
6. Redeploy API:
   - Local: docker compose restart api
   - Kubernetes: kubectl rollout restart deployment/mlops-api -n mlops
```

**Deployment modes:**
- `python retraining/prefect_retraining_flow.py` — manual / demo
- Kubernetes CronJob — production scheduled (configurable interval)

**Screenshots:**

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

**Docker design details:**
- API container runs as non-root style for security
- Filesystem permissions hardened for matplotlib/temp artifact writing
- Compose networking uses stable service names for inter-service DNS
- `.dockerignore` excludes datasets, model artifacts, and dev files to minimize build context

### Kubernetes (Production-Style)

```bash
kubectl apply -k k8s/overlays/dev
kubectl get pods -n mlops
```

| Component | Purpose |
|---|---|
| `hpa-api.yaml` | Scale API pods on CPU/memory load |
| `pdb-api.yaml` | Maintain minimum available replicas during disruptions |
| `networkpolicy.yaml` | Restrict pod-to-pod communication to necessary paths |
| `pvc-mlflow.yaml` | Persistent MLflow artifact + run storage |
| `pvc-grafana.yaml` | Persistent Grafana dashboard state |
| `ingress.yaml` | Route external traffic to all services |
| `cronjob-retraining.yaml` | Scheduled Prefect retraining flow |

**Screenshots:**

![Kubernetes Running Stack](images/kubernate-running.PNG)
![Ingress and Autoscaling](images/ingress+autoscaling.PNG)

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

### 6. Run Automated Test Suite (pytest)

```bash
pytest -v tests/
```

Coverage includes:
- model loading fallback behavior (registry then experiment run)
- endpoint contracts and validation behavior
- drift metric computation and feature-shape mismatch handling
- retraining trigger behavior (skip, drift-triggered, force mode)

### 7. Run API Integration Tests

```bash
python api/test_api.py
```

### 8. Run Prefect Retraining Flow

```bash
python retraining/prefect_retraining_flow.py
```

### 9. Kubernetes Deployment

```bash
kubectl apply -k k8s/overlays/dev
kubectl get pods -n mlops
```

### 10. Kubernetes Port-Forwards

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

### Algorithm Plot Status

- Current repository contains two generic algorithm result images used in the logistic section.
- Dedicated K-Means and Isolation Forest plots are pending export with model-specific filenames.
- Until then, rely on numeric benchmark tables and experiment reports in `results/` as primary evidence.

---

<div align="center">

**Built by Muhammad Hassan Shahbaz**

*ML Engineer · MLOps Engineer · Platform Engineer*

[![GitHub](https://img.shields.io/badge/GitHub-Rana--Hassan7272-181717?style=flat-square&logo=github)](https://github.com/Rana-Hassan7272)

</div>