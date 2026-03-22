"""
Monitoring utilities for production API.

Tracks:
- endpoint request count and latency
- prediction distribution and score histograms
- online data drift score against training baseline statistics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram
from scipy.sparse import load_npz


REQUEST_COUNT = Counter(
    "mlops_api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "http_status"],
)

REQUEST_LATENCY = Histogram(
    "mlops_api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

PREDICTION_COUNT = Counter(
    "mlops_predictions_total",
    "Total number of predictions emitted",
    ["model", "prediction"],
)

PREDICTION_SCORE = Histogram(
    "mlops_prediction_score",
    "Distribution of model output scores",
    ["model", "score_type"],
    buckets=(0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0),
)

DRIFT_SCORE = Gauge(
    "mlops_data_drift_score",
    "Online drift score based on normalized distance from training baseline",
    ["model"],
)

DRIFT_ALERTS = Counter(
    "mlops_data_drift_alerts_total",
    "Count of high data drift events",
    ["model", "severity"],
)

MODEL_LOADED = Gauge(
    "mlops_model_loaded",
    "Model loaded state (1 = loaded, 0 = not loaded)",
    ["model"],
)


@dataclass
class BaselineStats:
    mean: np.ndarray
    std: np.ndarray


class MonitoringManager:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.baselines: Dict[str, BaselineStats] = {}

    def load_baselines(self) -> Dict[str, bool]:
        """Load baseline feature statistics from training datasets."""
        loaded = {"logistic_regression": False, "kmeans": False, "isolation_forest": False}
        data_dir = self.project_root / "data_pipeline" / "processed_data"

        # Logistic Regression baseline (sparse npz)
        try:
            x_train_sparse = load_npz(data_dir / "spam_X_train.npz")
            x_train = x_train_sparse.toarray()
            self.baselines["logistic_regression"] = BaselineStats(
                mean=x_train.mean(axis=0), std=x_train.std(axis=0) + 1e-8
            )
            loaded["logistic_regression"] = True
        except Exception:
            pass

        # KMeans baseline (tabular csv)
        try:
            kmeans_df = pd.read_csv(data_dir / "store_customers_processed.csv")
            kmeans_values = kmeans_df.select_dtypes(include=["number"]).to_numpy(dtype=np.float64)
            self.baselines["kmeans"] = BaselineStats(
                mean=kmeans_values.mean(axis=0), std=kmeans_values.std(axis=0) + 1e-8
            )
            loaded["kmeans"] = True
        except Exception:
            pass

        # Isolation Forest baseline (tabular csv)
        try:
            fraud_df = pd.read_csv(data_dir / "creditcard_train.csv")
            numeric = fraud_df.select_dtypes(include=["number"])
            if "Class" in numeric.columns:
                numeric = numeric.drop(columns=["Class"])
            fraud_values = numeric.to_numpy(dtype=np.float64)
            self.baselines["isolation_forest"] = BaselineStats(
                mean=fraud_values.mean(axis=0), std=fraud_values.std(axis=0) + 1e-8
            )
            loaded["isolation_forest"] = True
        except Exception:
            pass

        return loaded

    def set_model_loaded(self, model_name: str, loaded: bool) -> None:
        MODEL_LOADED.labels(model=model_name).set(1 if loaded else 0)

    def observe_prediction(self, model_name: str, prediction: int, score: Optional[float] = None, score_type: str = "score") -> None:
        PREDICTION_COUNT.labels(model=model_name, prediction=str(prediction)).inc()
        if score is not None:
            PREDICTION_SCORE.labels(model=model_name, score_type=score_type).observe(float(np.clip(score, 0.0, 1.0)))

    def compute_and_record_drift(self, model_name: str, features: np.ndarray) -> Optional[float]:
        baseline = self.baselines.get(model_name)
        if baseline is None:
            return None

        x = np.asarray(features, dtype=np.float64).reshape(-1)
        if x.shape[0] != baseline.mean.shape[0]:
            # Feature-shape mismatch itself indicates strong drift
            DRIFT_SCORE.labels(model=model_name).set(10.0)
            DRIFT_ALERTS.labels(model=model_name, severity="critical").inc()
            return 10.0

        z_score = np.abs(x - baseline.mean) / baseline.std
        drift = float(np.mean(z_score))
        DRIFT_SCORE.labels(model=model_name).set(drift)

        if drift >= 4.0:
            DRIFT_ALERTS.labels(model=model_name, severity="critical").inc()
        elif drift >= 2.0:
            DRIFT_ALERTS.labels(model=model_name, severity="warning").inc()

        return drift
