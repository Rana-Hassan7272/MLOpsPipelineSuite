import api.app as app_module


def test_load_models_uses_registry_then_fallback(monkeypatch):
    app_module.models.clear()
    app_module.scalers.clear()

    class SentinelModel:
        pass

    registry_calls = []
    fallback_calls = []

    def fake_registry(model_name, stage="Staging"):
        registry_calls.append((model_name, stage))
        if model_name == "KMeans_sklearn":
            return SentinelModel(), "reg-kmeans-v1"
        return None, None

    def fake_experiment(experiment_name, run_name):
        fallback_calls.append((experiment_name, run_name))
        return SentinelModel(), f"run-{run_name}"

    monkeypatch.setattr(app_module, "load_model_from_registry", fake_registry)
    monkeypatch.setattr(app_module, "load_model_from_experiment", fake_experiment)
    monkeypatch.setattr(app_module, "project_root", app_module.project_root)

    app_module.load_models()

    assert "logistic_regression" in app_module.models
    assert "kmeans" in app_module.models
    assert "isolation_forest" in app_module.models
    assert app_module.models["kmeans_version"] == "reg-kmeans-v1"
    assert app_module.models["logistic_regression_version"].startswith("run-")
    assert app_module.models["isolation_forest_version"].startswith("run-")
    assert len(registry_calls) == 3
    assert len(fallback_calls) == 2
