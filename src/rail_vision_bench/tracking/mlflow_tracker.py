"""MLflow-backed tracker: the ``Tracker`` protocol of ``tracking.base`` over MLflow.

``mlflow`` is imported lazily on the first :meth:`MlflowTracker.start_run`, so
building a tracker (``get_tracker``) stays cheap and touches neither the network
nor the file system. The tracker talks to MLflow through an explicit
``MlflowClient`` instead of the fluent ``mlflow.*`` API, so it never changes
process-global MLflow state (tracking URI, active run) and several trackers can
coexist in one process.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import orjson

from rail_vision_bench import __version__
from rail_vision_bench.settings import Settings

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient

DEFAULT_EXPERIMENT: Final = "rail-vision-bench"
"""Experiment the runs are filed under unless the constructor names another one."""

PARAM_VALUE_MAX: Final = 6000
"""MLflow's limit on the length of a parameter value; longer values are truncated."""

VERSION_TAG: Final = "rail_vision_bench.version"
"""Run tag carrying the package version that produced the run."""


def flatten_params(params: Mapping[str, Any], prefix: str = "") -> Iterator[tuple[str, str]]:
    """Flatten nested parameters into dotted keys with string values.

    Nested mappings become ``parent.child`` keys; ``None``, scalars and
    strings are stringified (``None`` as ``"None"``, booleans as ``True``/``False``),
    every other value (lists, pydantic dumps, paths inside lists) is encoded as
    JSON. Values are cut to :data:`PARAM_VALUE_MAX` characters.

    Args:
        params: The parameters, possibly nested.
        prefix: Key prefix of the enclosing mapping (``""`` at the top level).

    Yields:
        ``(key, value)`` pairs in mapping order.
    """
    for key, value in params.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, Mapping):
            yield from flatten_params(value, prefix=f"{full_key}.")
            continue
        if value is None or isinstance(value, str | int | float | bool | Path):
            text = str(value)
        else:
            text = orjson.dumps(value, default=str).decode("utf-8")
        yield full_key, text[:PARAM_VALUE_MAX]


class MlflowTracker:
    """Tracker that logs to the MLflow tracking server of ``settings.mlflow_tracking_uri``.

    With ``mlflow_tracking_uri`` unset, MLflow's own resolution applies
    (``MLFLOW_TRACKING_URI``, else its local default store). The
    ``docker/compose.yaml`` MLflow service (``http://localhost:5000``) is the
    intended local target. One run is open at a time.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        experiment: str = DEFAULT_EXPERIMENT,
        artifact_location: str | None = None,
    ) -> None:
        self.settings = settings
        self.experiment = experiment
        """Name of the MLflow experiment; created on first use when missing."""
        self.artifact_location = artifact_location
        """Artifact root for a newly created experiment; the server default when ``None``."""
        self.run_id: str | None = None
        """Id of the open run, ``None`` between runs."""
        self._client: MlflowClient | None = None

    @property
    def client(self) -> MlflowClient:
        """The MLflow client bound to the configured tracking URI (created on first access)."""
        if self._client is None:
            from mlflow.tracking import MlflowClient

            self._client = MlflowClient(tracking_uri=self.settings.mlflow_tracking_uri)
        return self._client

    def _experiment_id(self) -> str:
        experiment = self.client.get_experiment_by_name(self.experiment)
        if experiment is not None:
            return str(experiment.experiment_id)
        return self.client.create_experiment(
            self.experiment, artifact_location=self.artifact_location
        )

    def _require_run(self) -> str:
        if self.run_id is None:
            raise RuntimeError("MlflowTracker has no open run; call start_run first")
        return self.run_id

    def start_run(self, name: str, params: Mapping[str, Any]) -> str:
        """Open an MLflow run named ``name`` and log ``params``.

        The run is created in :attr:`experiment` (created when missing), tagged
        with the package version, and ``params`` are logged in one batch after
        :func:`flatten_params`.

        Args:
            name: Run name.
            params: Run parameters; nested mappings are flattened to dotted keys.

        Returns:
            The MLflow run id.

        Raises:
            RuntimeError: When a run is already open.
        """
        from mlflow.entities import Param

        if self.run_id is not None:
            raise RuntimeError(f"MlflowTracker run {self.run_id} is still open; call end_run")
        run = self.client.create_run(
            self._experiment_id(), run_name=name, tags={VERSION_TAG: __version__}
        )
        run_id: str = run.info.run_id
        self.run_id = run_id
        flat = [Param(key, value) for key, value in flatten_params(params)]
        if flat:
            self.client.log_batch(run_id, params=flat)
        return run_id

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Log metric values to the open MLflow run in one batch.

        Args:
            metrics: Metric values by name; values are converted with ``float``.
            step: Optional step index; MLflow's step ``0`` when omitted.

        Raises:
            RuntimeError: When no run is open.
        """
        from mlflow.entities import Metric

        run_id = self._require_run()
        timestamp = int(time.time() * 1000)
        batch = [
            Metric(key, float(value), timestamp, step if step is not None else 0)
            for key, value in metrics.items()
        ]
        if batch:
            self.client.log_batch(run_id, metrics=batch)

    def log_artifact(self, path: Path) -> None:
        """Upload a file (or a directory, recursively) to the open MLflow run's artifact root.

        Args:
            path: The artifact file or directory.

        Raises:
            RuntimeError: When no run is open.
            FileNotFoundError: When ``path`` does not exist.
        """
        run_id = self._require_run()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            self.client.log_artifacts(run_id, str(path), artifact_path=path.name)
        else:
            self.client.log_artifact(run_id, str(path))

    def end_run(self) -> None:
        """Mark the open MLflow run ``FINISHED``; a no-op when no run is open."""
        if self.run_id is None:
            return
        self.client.set_terminated(self.run_id, status="FINISHED")
        self.run_id = None
