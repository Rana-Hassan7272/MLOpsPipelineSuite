import retraining.prefect_retraining_flow as retraining_module


class _DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


def _base_cfg():
    return {
        "monitoring": {
            "prometheus_url": "http://localhost:9090",
            "drift_query": "max(mlops_data_drift_score)",
            "drift_threshold": 2.0,
        },
        "performance": {"logistic_min_accuracy": 0.95},
        "orchestration": {
            "default_models": ["logistic_regression", "kmeans", "isolation_forest"],
            "redeploy_after_retrain": True,
            "redeploy_command": ["docker", "compose", "restart", "api"],
            "run_data_pipeline": True,
        },
    }


def test_retraining_skips_when_no_trigger(monkeypatch):
    monkeypatch.setattr(retraining_module, "get_run_logger", lambda: _DummyLogger())
    monkeypatch.setattr(retraining_module, "load_retraining_config", lambda config_path: _base_cfg())
    monkeypatch.setattr(retraining_module, "query_prometheus", lambda prometheus_url, promql_query: 0.1)
    monkeypatch.setattr(retraining_module, "get_latest_logistic_accuracy", lambda: 0.99)

    result = retraining_module.automatic_retraining_flow.fn(force_retrain=False)

    assert result["status"] == "skipped"
    assert "No retraining trigger fired" in result["message"]


def test_retraining_runs_when_drift_trigger_fires(monkeypatch):
    monkeypatch.setattr(retraining_module, "get_run_logger", lambda: _DummyLogger())
    monkeypatch.setattr(retraining_module, "load_retraining_config", lambda config_path: _base_cfg())
    monkeypatch.setattr(retraining_module, "query_prometheus", lambda prometheus_url, promql_query: 3.0)
    monkeypatch.setattr(retraining_module, "get_latest_logistic_accuracy", lambda: 0.99)
    monkeypatch.setattr(retraining_module, "run_data_pipeline", lambda: (True, "pipeline ok"))
    monkeypatch.setattr(retraining_module, "run_model_retraining", lambda model_name: (True, f"{model_name} ok"))
    monkeypatch.setattr(retraining_module, "register_models", lambda: (True, "registry ok"))
    monkeypatch.setattr(retraining_module, "redeploy_api", lambda command: (True, "redeploy ok"))

    result = retraining_module.automatic_retraining_flow.fn(force_retrain=False)

    assert result["status"] == "success"
    assert any("drift_detected" in reason for reason in result["reasons"])
    assert result["pipeline"]["ok"] is True
    assert result["registry"]["ok"] is True
    assert result["redeploy"]["ok"] is True


def test_retraining_runs_when_forced(monkeypatch):
    monkeypatch.setattr(retraining_module, "get_run_logger", lambda: _DummyLogger())
    monkeypatch.setattr(retraining_module, "load_retraining_config", lambda config_path: _base_cfg())
    monkeypatch.setattr(retraining_module, "query_prometheus", lambda prometheus_url, promql_query: 0.0)
    monkeypatch.setattr(retraining_module, "get_latest_logistic_accuracy", lambda: 1.0)
    monkeypatch.setattr(retraining_module, "run_data_pipeline", lambda: (True, "pipeline ok"))
    monkeypatch.setattr(retraining_module, "run_model_retraining", lambda model_name: (True, f"{model_name} ok"))
    monkeypatch.setattr(retraining_module, "register_models", lambda: (True, "registry ok"))
    monkeypatch.setattr(retraining_module, "redeploy_api", lambda command: (True, "redeploy ok"))

    result = retraining_module.automatic_retraining_flow.fn(force_retrain=True)

    assert result["status"] == "success"
    assert "force_retrain=true" in result["reasons"]
