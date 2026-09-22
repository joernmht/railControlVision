"""MLflow-backed tracker (stub; ``mlflow`` is never imported in the skeleton)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rail_vision_bench.settings import Settings


class MlflowTracker:
    """Tracker that logs to the MLflow server of ``settings.mlflow_tracking_uri``.

    Intended implementation: ``mlflow.set_tracking_uri(settings.mlflow_tracking_uri)``
    at construction, then ``mlflow.start_run`` / ``log_params``, ``log_metrics``,
    ``log_artifact`` and ``end_run`` mapped one to one onto the protocol; the
    ``docker/compose.yaml`` MLflow service is the local default target.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def start_run(self, name: str, params: Mapping[str, Any]) -> str:
        """Open an MLflow run named ``name`` and log ``params``.

        Args:
            name: Run name.
            params: Run parameters.

        Returns:
            The MLflow run id.

        Raises:
            NotImplementedError: Always, in the skeleton.
        """
        raise NotImplementedError(
            "rail_vision_bench.tracking.mlflow_tracker.MlflowTracker.start_run is not "
            "implemented in the skeleton: mlflow.start_run plus log_params"
        )

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Log metric values to the open MLflow run.

        Args:
            metrics: Metric values by name.
            step: Optional step index.

        Raises:
            NotImplementedError: Always, in the skeleton.
        """
        raise NotImplementedError(
            "rail_vision_bench.tracking.mlflow_tracker.MlflowTracker.log_metrics is not "
            "implemented in the skeleton: mlflow.log_metrics"
        )

    def log_artifact(self, path: Path) -> None:
        """Upload a file to the open MLflow run.

        Args:
            path: The artifact file.

        Raises:
            NotImplementedError: Always, in the skeleton.
        """
        raise NotImplementedError(
            "rail_vision_bench.tracking.mlflow_tracker.MlflowTracker.log_artifact is not "
            "implemented in the skeleton: mlflow.log_artifact"
        )

    def end_run(self) -> None:
        """Close the open MLflow run.

        Raises:
            NotImplementedError: Always, in the skeleton.
        """
        raise NotImplementedError(
            "rail_vision_bench.tracking.mlflow_tracker.MlflowTracker.end_run is not "
            "implemented in the skeleton: mlflow.end_run"
        )
