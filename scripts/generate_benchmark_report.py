"""
Generate CI benchmark artifact from latest MLflow runs.

Output:
- results/benchmark_ci_report.md
- results/benchmark_ci_report.json
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import yaml
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_MD_PATH = RESULTS_DIR / "benchmark_ci_report.md"
REPORT_JSON_PATH = RESULTS_DIR / "benchmark_ci_report.json"


def _default_tracking_uri() -> str:
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    return f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


def _load_config() -> dict:
    cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _latest_run_metrics(
    client: MlflowClient, experiment_name: str, run_name: str
) -> Tuple[Optional[dict], Optional[str]]:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if not exp:
        return None, f"Experiment not found: {experiment_name}"

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None, f"Run not found: {run_name}"

    return runs[0].data.metrics, None


def _ci95_from_mean_std(mean: Optional[float], std: Optional[float], n: int) -> Optional[Tuple[float, float]]:
    if mean is None or std is None or n <= 1:
        return None
    half_width = 1.96 * (std / math.sqrt(float(n)))
    return float(mean - half_width), float(mean + half_width)


def _format_ci(ci: Optional[Tuple[float, float]]) -> str:
    if ci is None:
        return "n/a"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def _value(metrics: dict, key: str) -> Optional[float]:
    val = metrics.get(key)
    return float(val) if val is not None else None


def _section_rows(
    scratch_metrics: dict,
    sklearn_metrics: dict,
    fold_count: int,
    specs: List[Tuple[str, str, Optional[str]]],
) -> List[dict]:
    rows = []
    for label, point_key, cv_std_key in specs:
        s_point = _value(scratch_metrics, point_key)
        k_point = _value(sklearn_metrics, point_key)
        s_ci = None
        k_ci = None
        if cv_std_key:
            s_ci = _ci95_from_mean_std(s_point, _value(scratch_metrics, cv_std_key), fold_count)
            k_ci = _ci95_from_mean_std(k_point, _value(sklearn_metrics, cv_std_key), fold_count)

        rows.append(
            {
                "metric": label,
                "scratch": s_point,
                "sklearn": k_point,
                "scratch_ci95": s_ci,
                "sklearn_ci95": k_ci,
            }
        )
    return rows


def _md_table(rows: List[dict]) -> str:
    lines = [
        "| Metric | Scratch | Scratch 95% CI | sklearn | sklearn 95% CI |",
        "|---|---:|---|---:|---|",
    ]
    for r in rows:
        s = "n/a" if r["scratch"] is None else f"{r['scratch']:.4f}"
        k = "n/a" if r["sklearn"] is None else f"{r['sklearn']:.4f}"
        lines.append(
            f"| {r['metric']} | {s} | {_format_ci(r['scratch_ci95'])} | {k} | {_format_ci(r['sklearn_ci95'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    tracking_uri = _default_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    cfg = _load_config()
    lr_folds = int(cfg.get("logistic_regression", {}).get("evaluation", {}).get("cv_folds", 2))
    km_folds = int(cfg.get("kmeans", {}).get("evaluation", {}).get("cv_folds", 2))
    if_folds = int(cfg.get("isolation_forest", {}).get("evaluation", {}).get("cv_folds", 2))

    errors: List[str] = []

    lr_s, err = _latest_run_metrics(client, "Logistic_Regression_Comparison", "Logistic_Regression_Scratch")
    if err:
        errors.append(err)
    lr_k, err = _latest_run_metrics(client, "Logistic_Regression_Comparison", "Logistic_Regression_sklearn")
    if err:
        errors.append(err)

    km_s, err = _latest_run_metrics(client, "KMeans_Comparison", "KMeans_Scratch")
    if err:
        errors.append(err)
    km_k, err = _latest_run_metrics(client, "KMeans_Comparison", "KMeans_sklearn")
    if err:
        errors.append(err)

    if_s, err = _latest_run_metrics(client, "Isolation_Forest_Comparison", "Isolation_Forest_Scratch")
    if err:
        errors.append(err)
    if_k, err = _latest_run_metrics(client, "Isolation_Forest_Comparison", "Isolation_Forest_sklearn")
    if err:
        errors.append(err)

    report = {
        "tracking_uri": tracking_uri,
        "errors": errors,
        "sections": {},
    }

    md_parts = [
        "# CI Benchmark Report (Auto-Generated)",
        "",
        "This report is generated in CI from latest MLflow runs.",
        "",
        "- Point estimates come from latest run metrics.",
        "- 95% CI is approximated as `mean +- 1.96 * (cv_std / sqrt(cv_folds))` when CV std is available.",
        "- For threshold-sensitive tasks (spam/fraud), interpret results with operating-point caveats.",
        "",
    ]

    if lr_s and lr_k:
        lr_rows = _section_rows(
            lr_s,
            lr_k,
            lr_folds,
            [
                ("Accuracy", "accuracy", "cv_std_accuracy"),
                ("Precision", "precision", None),
                ("Recall", "recall", None),
                ("F1", "f1_score", None),
                ("AUC", "auc", None),
                ("Training Time (s)", "training_time_seconds", None),
            ],
        )
        report["sections"]["logistic_regression"] = {"cv_folds": lr_folds, "rows": lr_rows}
        md_parts.extend(["## Logistic Regression (Spam)", "", _md_table(lr_rows), ""])

    if km_s and km_k:
        km_rows = _section_rows(
            km_s,
            km_k,
            km_folds,
            [
                ("Inertia", "inertia", None),
                ("Silhouette", "silhouette", "cv_std_silhouette"),
                ("Davies-Bouldin", "davies_bouldin", None),
                ("Calinski-Harabasz", "calinski_harabasz", None),
                ("Training Time (s)", "training_time_seconds", None),
            ],
        )
        report["sections"]["kmeans"] = {"cv_folds": km_folds, "rows": km_rows}
        md_parts.extend(["## K-Means (Store Customers)", "", _md_table(km_rows), ""])

    if if_s and if_k:
        if_rows = _section_rows(
            if_s,
            if_k,
            if_folds,
            [
                ("ROC-AUC", "roc_auc", "cv_std_roc_auc"),
                ("Avg Precision", "average_precision", "cv_std_ap"),
                ("Precision", "precision", None),
                ("Recall", "recall", None),
                ("F1", "f1", None),
                ("Training Time (s)", "training_time_seconds", None),
            ],
        )
        report["sections"]["isolation_forest"] = {"cv_folds": if_folds, "rows": if_rows}
        md_parts.extend(["## Isolation Forest (Fraud)", "", _md_table(if_rows), ""])

    if errors:
        md_parts.extend(["## Missing Data Warnings", ""])
        md_parts.extend([f"- {e}" for e in errors])
        md_parts.append("")

    REPORT_MD_PATH.write_text("\n".join(md_parts), encoding="utf-8")
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"Generated: {REPORT_MD_PATH}")
    print(f"Generated: {REPORT_JSON_PATH}")


if __name__ == "__main__":
    main()

