"""NullTracker bookkeeping, the tracker factory and the MLflow tracker against a local store."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from rail_vision_bench import __version__
from rail_vision_bench.config import TrackerKind
from rail_vision_bench.settings import Settings
from rail_vision_bench.tracking.base import NullTracker, Tracker, get_tracker
from rail_vision_bench.tracking.mlflow_tracker import (
    PARAM_VALUE_MAX,
    VERSION_TAG,
    MlflowTracker,
    flatten_params,
)


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


@pytest.fixture(scope="module")
def tracking_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    """One local SQLite tracking store per module (creating its tables takes seconds)."""
    return f"sqlite:///{tmp_path_factory.mktemp('mlflow') / 'mlflow.db'}"


@pytest.fixture
def mlflow_tracker(
    tracking_uri: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[MlflowTracker]:
    # chdir: whatever MLflow might create relative to the cwd lands in tmp_path, not the repo.
    monkeypatch.chdir(tmp_path)
    tracker = MlflowTracker(
        Settings(_env_file=None, mlflow_tracking_uri=tracking_uri),
        experiment=f"test-{tmp_path.name}",
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )
    yield tracker
    tracker.end_run()


def test_mlflow_tracker_is_lazy():
    tracker = MlflowTracker(Settings(_env_file=None))
    assert tracker.run_id is None
    assert tracker._client is None


def test_flatten_params():
    params = {"a": 1, "b": {"c": None, "d": [1, "x"]}, "e": True, "f": Path("p/q"), "g": "s" * 7000}
    assert dict(flatten_params(params)) == {
        "a": "1",
        "b.c": "None",
        "b.d": '[1,"x"]',
        "e": "True",
        "f": "p/q",
        "g": "s" * PARAM_VALUE_MAX,
    }


def test_mlflow_tracker_logs_params_metrics_artifacts(
    mlflow_tracker: MlflowTracker, tracking_uri: str, tmp_path: Path
):
    artifact = tmp_path / "report.html"
    artifact.write_text("<html></html>", encoding="utf-8")
    run_id = mlflow_tracker.start_run("run-1", {"seed": 7, "model": {"name": "m", "t": 0.0}})
    mlflow_tracker.log_metrics({"f1": 0.5, "cer": 1}, step=2)
    mlflow_tracker.log_metrics({"f1": 0.75}, step=3)
    mlflow_tracker.log_metrics({"f1": 0.25})  # step 0
    mlflow_tracker.log_artifact(artifact)
    mlflow_tracker.end_run()
    assert mlflow_tracker.run_id is None

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.info.run_name == "run-1"
    assert run.data.tags[VERSION_TAG] == __version__
    assert run.data.params == {"seed": "7", "model.name": "m", "model.t": "0.0"}
    assert run.data.metrics == {"f1": 0.75, "cer": 1.0}
    history = sorted((m.step, m.value) for m in client.get_metric_history(run_id, "f1"))
    assert history == [(0, 0.25), (2, 0.5), (3, 0.75)]
    assert [f.path for f in client.list_artifacts(run_id)] == ["report.html"]
    assert (tmp_path / "artifacts").is_dir()


def test_mlflow_tracker_logs_directory_artifact(
    mlflow_tracker: MlflowTracker, tracking_uri: str, tmp_path: Path
):
    folder = tmp_path / "plots"
    folder.mkdir()
    (folder / "a.txt").write_text("a", encoding="utf-8")
    run_id = mlflow_tracker.start_run("run-dir", {})
    mlflow_tracker.log_artifact(folder)
    client = MlflowClient(tracking_uri=tracking_uri)
    assert [f.path for f in client.list_artifacts(run_id, "plots")] == ["plots/a.txt"]


def test_mlflow_tracker_reuses_experiment(mlflow_tracker: MlflowTracker, tracking_uri: str):
    first = mlflow_tracker.start_run("a", {})
    mlflow_tracker.end_run()
    second = mlflow_tracker.start_run("b", {})
    client = MlflowClient(tracking_uri=tracking_uri)
    assert client.get_run(first).info.experiment_id == client.get_run(second).info.experiment_id


def test_mlflow_tracker_run_state_errors(mlflow_tracker: MlflowTracker, tmp_path: Path):
    with pytest.raises(RuntimeError, match="no open run"):
        mlflow_tracker.log_metrics({"f1": 1.0})
    with pytest.raises(RuntimeError, match="no open run"):
        mlflow_tracker.log_artifact(tmp_path)
    mlflow_tracker.end_run()  # no-op without a run
    run_id = mlflow_tracker.start_run("x", {})
    assert mlflow_tracker.run_id == run_id
    with pytest.raises(RuntimeError, match="still open"):
        mlflow_tracker.start_run("y", {})
    with pytest.raises(FileNotFoundError):
        mlflow_tracker.log_artifact(tmp_path / "missing.txt")
