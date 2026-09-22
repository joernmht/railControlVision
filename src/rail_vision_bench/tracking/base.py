"""The tracker protocol, the in-memory null tracker and the factory."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from rail_vision_bench.config import TrackerKind
from rail_vision_bench.settings import Settings, get_settings
from rail_vision_bench.tracking.mlflow_tracker import MlflowTracker


class Tracker(Protocol):
    """What the runner needs from an experiment-tracking backend."""

    def start_run(self, name: str, params: Mapping[str, Any]) -> str:
        """Open a run and return its backend id."""

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Log metric values, optionally at a step."""

    def log_artifact(self, path: Path) -> None:
        """Attach a file to the open run."""

    def end_run(self) -> None:
        """Close the open run."""


class NullTracker:
    """A tracker that only records what was asked of it (default; used in tests)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        """Every call as ``(method, kwargs)`` in call order."""

    def start_run(self, name: str, params: Mapping[str, Any]) -> str:
        """Record the call and return the fixed id ``null-run``.

        Args:
            name: Run name.
            params: Run parameters.

        Returns:
            ``"null-run"``.
        """
        self.calls.append(("start_run", {"name": name, "params": dict(params)}))
        return "null-run"

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Record the call.

        Args:
            metrics: Metric values by name.
            step: Optional step index.
        """
        self.calls.append(("log_metrics", {"metrics": dict(metrics), "step": step}))

    def log_artifact(self, path: Path) -> None:
        """Record the call.

        Args:
            path: The artifact file.
        """
        self.calls.append(("log_artifact", {"path": path}))

    def end_run(self) -> None:
        """Record the call."""
        self.calls.append(("end_run", {}))


def get_tracker(kind: TrackerKind, settings: Settings | None = None) -> Tracker:
    """Build the tracker of a run.

    Args:
        kind: The configured backend.
        settings: Runtime settings for backends that need them; ``get_settings()`` when omitted.

    Returns:
        A :class:`NullTracker` or an :class:`MlflowTracker`.
    """
    if kind is TrackerKind.MLFLOW:
        return MlflowTracker(settings if settings is not None else get_settings())
    return NullTracker()
