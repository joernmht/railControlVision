"""NullTracker bookkeeping, the tracker factory and the MLflow stub."""

from __future__ import annotations

from pathlib import Path

import pytest

from rail_vision_bench.config import TrackerKind
from rail_vision_bench.settings import Settings
from rail_vision_bench.tracking.base import NullTracker, Tracker, get_tracker
from rail_vision_bench.tracking.mlflow_tracker import MlflowTracker


def test_null_tracker_records_calls_in_order():
    tracker = NullTracker()
    assert tracker.start_run("r", {"seed": 1}) == "null-run"
    tracker.log_metrics({"f1": 0.5}, step=2)
    tracker.log_artifact(Path("report.html"))
    tracker.end_run()
    assert tracker.calls == [
        ("start_run", {"name": "r", "params": {"seed": 1}}),
        ("log_metrics", {"metrics": {"f1": 0.5}, "step": 2}),
        ("log_artifact", {"path": Path("report.html")}),
        ("end_run", {}),
    ]


def test_null_tracker_copies_mappings():
    tracker = NullTracker()
    params = {"a": 1}
    tracker.start_run("r", params)
    params["a"] = 2
    assert tracker.calls[0][1]["params"] == {"a": 1}


def test_get_tracker_kinds():
    assert isinstance(get_tracker(TrackerKind.NULL), NullTracker)
    settings = Settings(_env_file=None)
    mlflow = get_tracker(TrackerKind.MLFLOW, settings)
    assert isinstance(mlflow, MlflowTracker)
    assert mlflow.settings is settings
    assert isinstance(get_tracker(TrackerKind.MLFLOW), MlflowTracker)


def test_trackers_satisfy_protocol():
    null_tracker: Tracker = NullTracker()
    mlflow_tracker: Tracker = MlflowTracker(Settings(_env_file=None))
    assert null_tracker is not mlflow_tracker


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("start_run", ("x", {})),
        ("log_metrics", ({"f1": 1.0},)),
        ("log_artifact", (Path("x"),)),
        ("end_run", ()),
    ],
)
def test_mlflow_tracker_methods_raise(method: str, args: tuple[object, ...]):
    tracker = MlflowTracker(Settings(_env_file=None))
    with pytest.raises(NotImplementedError, match="not implemented in the skeleton"):
        getattr(tracker, method)(*args)
