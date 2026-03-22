"""
Phase 12 - Automatic Retraining with Prefect

Pipeline:
monitor -> retrain -> redeploy

Triggers:
- data drift detected from Prometheus
- accuracy drop detected from MLflow
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlencode
from urllib.request import urlopen
import json

import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, get_run_logger, task
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_SCRIPT_BY_MODEL = {
    "logistic_regression": PROJECT_ROOT / "experiments" / "exp_logistic_vs_sklearn.py",
    "kmeans": PROJECT_ROOT / "experiments" / "exp_kmeans_vs_sklearn.py",
    "isolation_forest": PROJECT_ROOT / "experiments" / "exp_isolation_forest_vs_sklearn.py",
}


@task
def load_retraining_config(config_path: str = str(DEFAULT_CONFIG_PATH)) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config.get("automatic_retraining", {})


@task
def query_prometheus(prometheus_url: str, promql_query: str) -> float:
    params = urlencode({"query": promql_query})
    query_url = f"{prometheus_url.rstrip('/')}/api/v1/query?{params}"
    with urlopen(query_url, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if payload.get("status") != "success":
        return 0.0

    result = payload.get("data", {}).get("result", [])
    if not result:
        return 0.0

    value = result[0].get("value", [None, "0"])[1]
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@task
def get_latest_logistic_accuracy() -> float:
    experiment = mlflow.get_experiment_by_name("Logistic_Regression_Comparison")
    if experiment is None:
        return 0.0

    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'Logistic_Regression_sklearn'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return 0.0

    metrics = runs[0].data.metrics
    return float(metrics.get("accuracy", 0.0))


def _run_python_script(script_path: Path) -> Tuple[bool, str]:
    command = [sys.executable, str(script_path)]
    child_env = os.environ.copy()
    # Force UTF-8 in child processes to avoid Windows cp1252 decode/encode crashes.
    child_env.setdefault("PYTHONIOENCODING", "utf-8")
    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
    )
    output = f"{result.stdout}\n{result.stderr}".strip()
    return result.returncode == 0, output


def _run_command(command: List[str]) -> Tuple[bool, str]:
    child_env = os.environ.copy()
    child_env.setdefault("PYTHONIOENCODING", "utf-8")
    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
    )
    output = f"{result.stdout}\n{result.stderr}".strip()
    return result.returncode == 0, output


@task
def run_data_pipeline() -> Tuple[bool, str]:
    return _run_python_script(PROJECT_ROOT / "data_pipeline" / "run_pipeline.py")


@task
def run_model_retraining(model_name: str) -> Tuple[bool, str]:
    script_path = EXPERIMENT_SCRIPT_BY_MODEL[model_name]
    return _run_python_script(script_path)


@task
def register_models() -> Tuple[bool, str]:
    return _run_python_script(PROJECT_ROOT / "api" / "register_models.py")


@task
def redeploy_api(redeploy_command: List[str]) -> Tuple[bool, str]:
    return _run_command(redeploy_command)


@flow(name="automatic-retraining-flow")
def automatic_retraining_flow(
    selected_models: List[str] | None = None,
    force_retrain: bool = False,
    skip_redeploy: bool = False,
    config_path: str = str(DEFAULT_CONFIG_PATH),
) -> Dict:
    logger = get_run_logger()
    cfg = load_retraining_config(config_path)

    monitoring_cfg = cfg.get("monitoring", {})
    perf_cfg = cfg.get("performance", {})
    orch_cfg = cfg.get("orchestration", {})

    # Runtime overrides allow the same flow to run in local Docker Compose and Kubernetes.
    monitoring_cfg["prometheus_url"] = os.getenv("PROMETHEUS_URL", monitoring_cfg.get("prometheus_url", "http://localhost:9090"))

    models = selected_models or orch_cfg.get("default_models", ["logistic_regression", "kmeans", "isolation_forest"])
    models = [m for m in models if m in EXPERIMENT_SCRIPT_BY_MODEL]

    drift_score = 0.0
    latest_accuracy = 0.0
    drift_reason = False
    accuracy_reason = False
    reasons: List[str] = []

    if not force_retrain:
        try:
            drift_score = query_prometheus(
                monitoring_cfg.get("prometheus_url", "http://localhost:9090"),
                monitoring_cfg.get("drift_query", "max(mlops_data_drift_score)"),
            )
            drift_threshold = float(monitoring_cfg.get("drift_threshold", 2.0))
            drift_reason = drift_score >= drift_threshold
            if drift_reason:
                reasons.append(f"drift_detected(score={drift_score:.4f}, threshold={drift_threshold:.4f})")
        except Exception as e:
            logger.warning("Prometheus drift check failed: %s", e)

        try:
            latest_accuracy = get_latest_logistic_accuracy()
            min_accuracy = float(perf_cfg.get("logistic_min_accuracy", 0.95))
            accuracy_reason = latest_accuracy < min_accuracy
            if accuracy_reason:
                reasons.append(f"accuracy_drop(latest={latest_accuracy:.4f}, min={min_accuracy:.4f})")
        except Exception as e:
            logger.warning("MLflow performance check failed: %s", e)

    if force_retrain:
        reasons.append("force_retrain=true")

    if not reasons:
        msg = "No retraining trigger fired. Exiting without changes."
        logger.info(msg)
        return {
            "status": "skipped",
            "message": msg,
            "drift_score": drift_score,
            "latest_accuracy": latest_accuracy,
            "models": models,
        }

    logger.info("Retraining triggered: %s", "; ".join(reasons))
    results: Dict[str, Dict] = {"reasons": reasons, "models": {}, "pipeline": {}, "registry": {}, "redeploy": {}}

    if bool(orch_cfg.get("run_data_pipeline", True)):
        ok, output = run_data_pipeline()
        results["pipeline"] = {"ok": ok, "output": output}
        if not ok:
            return {"status": "failed", "step": "data_pipeline", **results}

    for model_name in models:
        ok, output = run_model_retraining(model_name)
        results["models"][model_name] = {"ok": ok, "output": output}
        if not ok:
            return {"status": "failed", "step": f"retrain_{model_name}", **results}

    ok, output = register_models()
    results["registry"] = {"ok": ok, "output": output}
    if not ok:
        return {"status": "failed", "step": "register_models", **results}

    should_redeploy = bool(orch_cfg.get("redeploy_after_retrain", True)) and not skip_redeploy
    if should_redeploy:
        redeploy_command = orch_cfg.get("redeploy_command", ["docker", "compose", "restart", "api"])
        ok, output = redeploy_api(redeploy_command)
        results["redeploy"] = {"ok": ok, "output": output}
        if not ok:
            return {"status": "failed", "step": "redeploy_api", **results}

    return {
        "status": "success",
        "drift_score": drift_score,
        "latest_accuracy": latest_accuracy,
        **results,
    }


def _parse_models(raw_models: str | None) -> List[str] | None:
    if not raw_models:
        return None
    parsed = [m.strip() for m in raw_models.split(",") if m.strip()]
    return parsed or None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic retraining flow (Phase 12) using Prefect.")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated models: logistic_regression,kmeans,isolation_forest",
    )
    parser.add_argument("--force", action="store_true", help="Force retraining even if no trigger fires.")
    parser.add_argument("--skip-redeploy", action="store_true", help="Skip docker API restart.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to config.yaml")
    args = parser.parse_args()

    result = automatic_retraining_flow(
        selected_models=_parse_models(args.models),
        force_retrain=args.force,
        skip_redeploy=args.skip_redeploy,
        config_path=args.config,
    )
    print(json.dumps(result, indent=2, default=str))
