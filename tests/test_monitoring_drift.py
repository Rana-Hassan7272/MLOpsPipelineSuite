from pathlib import Path

import numpy as np

from api.monitoring import MonitoringManager


def test_compute_and_record_drift_returns_expected_value(tmp_path: Path):
    monitor = MonitoringManager(tmp_path)
    monitor.baselines["logistic_regression"] = type(
        "BaselineStatsStub",
        (),
        {"mean": np.array([0.0, 0.0]), "std": np.array([1.0, 2.0])},
    )()

    # z = [|2-0|/1, |4-0|/2] = [2, 2], mean => 2.0
    drift = monitor.compute_and_record_drift("logistic_regression", np.array([2.0, 4.0]))
    assert drift == 2.0


def test_compute_and_record_drift_feature_mismatch_returns_critical(tmp_path: Path):
    monitor = MonitoringManager(tmp_path)
    monitor.baselines["kmeans"] = type(
        "BaselineStatsStub",
        (),
        {"mean": np.array([0.0, 0.0, 0.0]), "std": np.array([1.0, 1.0, 1.0])},
    )()

    drift = monitor.compute_and_record_drift("kmeans", np.array([1.0, 2.0]))
    assert drift == 10.0


def test_compute_and_record_drift_returns_none_when_baseline_missing(tmp_path: Path):
    monitor = MonitoringManager(tmp_path)
    drift = monitor.compute_and_record_drift("isolation_forest", np.array([1.0, 2.0, 3.0]))
    assert drift is None
