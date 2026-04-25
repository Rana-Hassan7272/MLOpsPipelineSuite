import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure project root is importable when pytest starts from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import api.app as app_module


class DummyLogisticModel:
    def predict_proba(self, X):
        probs = np.full((len(X), 2), 0.0, dtype=np.float64)
        probs[:, 1] = 0.8
        probs[:, 0] = 0.2
        return probs


class DummyKMeansModel:
    cluster_centers_ = np.array([[0.0, 0.0, 0.0, 0.0], [25.0, 50000.0, 2.0, 1.0]], dtype=np.float64)

    def predict(self, X):
        return np.array([1] * len(X), dtype=np.int32)


class DummyIsolationModel:
    def predict(self, X):
        return np.array([-1] * len(X), dtype=np.int32)

    def score_samples(self, X):
        return np.array([-0.3] * len(X), dtype=np.float64)


@pytest.fixture
def client_ready(monkeypatch):
    app_module.models.clear()
    app_module.scalers.clear()

    def fake_load_models():
        app_module.models["logistic_regression"] = DummyLogisticModel()
        app_module.models["logistic_regression_version"] = "test-v1"
        app_module.models["kmeans"] = DummyKMeansModel()
        app_module.models["kmeans_version"] = "test-v1"
        app_module.models["isolation_forest"] = DummyIsolationModel()
        app_module.models["isolation_forest_version"] = "test-v1"

    monkeypatch.setattr(app_module, "load_models", fake_load_models)
    monkeypatch.setattr(app_module.monitor, "load_baselines", lambda: {"ok": True})
    monkeypatch.setattr(app_module.monitor, "set_model_loaded", lambda model_name, loaded: None)
    monkeypatch.setattr(app_module.monitor, "observe_prediction", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_module.monitor, "compute_and_record_drift", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr("mlflow.tracking.MlflowClient.search_experiments", lambda self, max_results=1: [])

    with TestClient(app_module.app) as test_client:
        yield test_client


@pytest.fixture
def client_not_ready(monkeypatch):
    app_module.models.clear()
    app_module.scalers.clear()

    monkeypatch.setattr(app_module, "load_models", lambda: None)
    monkeypatch.setattr(app_module.monitor, "load_baselines", lambda: {"ok": False})
    monkeypatch.setattr(app_module.monitor, "set_model_loaded", lambda model_name, loaded: None)
    monkeypatch.setattr("mlflow.tracking.MlflowClient.search_experiments", lambda self, max_results=1: (_ for _ in ()).throw(RuntimeError("mlflow down")))

    with TestClient(app_module.app) as test_client:
        yield test_client
