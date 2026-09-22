"""Run, task and model-catalogue configuration: models, YAML loaders and resolution.

Nothing here resolves repository-relative resources: the paths inside a
:class:`RunConfig` are interpreted relative to the current working directory
when the run is resolved, so ``bench run --config PATH`` works from any
checkout layout.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rail_vision_bench.schema.models import SourceKind

ProviderName = Literal["anthropic", "openai", "gemini", "mistral", "ollama", "litellm"]
PROVIDER_NAMES: Final[tuple[ProviderName, ...]] = (
    "anthropic",
    "openai",
    "gemini",
    "mistral",
    "ollama",
    "litellm",
)
"""Registry keys shared by the model catalogue and the provider registry."""


class Mode(StrEnum):
    """How a model is driven over one scene."""

    SINGLE_SHOT = "single_shot"
    AGENTIC = "agentic"


class TrackerKind(StrEnum):
    """Experiment-tracking backend of a run."""

    NULL = "null"
    MLFLOW = "mlflow"


class ModelConfig(BaseModel):
    """One catalogue entry: how to reach a model and what it costs."""

    model_config = ConfigDict(extra="forbid")

    name: str
    provider: ProviderName
    model_id: str
    supports_tools: bool = False
    cost_per_1k_in: float | None = None
    cost_per_1k_out: float | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class ModelCatalogue(BaseModel):
    """The models a run may reference by name (``configs/models.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    models: list[ModelConfig]

    def names(self) -> list[str]:
        """Return the catalogue names in file order.

        Returns:
            The ``name`` of every entry.
        """
        return [model.name for model in self.models]

    def get(self, name: str) -> ModelConfig:
        """Look up a catalogue entry by name.

        Args:
            name: The catalogue name.

        Returns:
            The matching entry.

        Raises:
            KeyError: If no entry has that name; the message lists the known names.
        """
        for model in self.models:
            if model.name == name:
                return model
        msg = f"unknown model {name!r}; known: {self.names()}"
        raise KeyError(msg)


class TaskConfig(BaseModel):
    """What is benchmarked: a dataset split, a prompt and the metrics to report."""

    model_config = ConfigDict(extra="forbid")

    name: str
    source_kind: SourceKind
    split: str
    prompt: str = "single_shot"
    metrics: list[str] = Field(default_factory=list)
    description: str | None = None


class RunConfig(BaseModel):
    """One benchmark run (``bench run --config``).

    ``task`` and ``catalogue`` are file paths interpreted relative to the current
    working directory at resolution time; a copy of the resolved configuration is
    written into the run directory.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    task: Path
    catalogue: Path = Path("configs/models.yaml")
    models: list[str] = Field(min_length=1)
    mode: Mode = Mode.SINGLE_SHOT
    limit: int | None = None
    seed: int = 0
    max_rounds: int = 3
    out_dir: Path = Path("runs")
    tracker: TrackerKind = TrackerKind.NULL


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file that must hold a mapping at the top level.

    Args:
        path: The YAML file.

    Returns:
        The parsed mapping.

    Raises:
        ValueError: If the document is not a mapping (including an empty file).
    """
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = f"{path}: expected a mapping"
        raise ValueError(msg)
    return loaded


def load_run_config(path: Path) -> RunConfig:
    """Load and validate a :class:`RunConfig` from YAML.

    Args:
        path: The run configuration file.

    Returns:
        The validated configuration.
    """
    return RunConfig.model_validate(load_yaml(path))


def load_model_catalogue(path: Path) -> ModelCatalogue:
    """Load and validate a :class:`ModelCatalogue` from YAML.

    Args:
        path: The catalogue file.

    Returns:
        The validated catalogue.
    """
    return ModelCatalogue.model_validate(load_yaml(path))


def load_task_config(path: Path) -> TaskConfig:
    """Load and validate a :class:`TaskConfig` from YAML.

    Args:
        path: The task file.

    Returns:
        The validated task.
    """
    return TaskConfig.model_validate(load_yaml(path))


def apply_overrides(
    config: RunConfig,
    *,
    models: Sequence[str] | None = None,
    mode: Mode | None = None,
    limit: int | None = None,
    seed: int | None = None,
    out_dir: Path | None = None,
) -> RunConfig:
    """Return a copy of ``config`` with the command-line overrides applied.

    Only the arguments that are not ``None`` replace the corresponding field.

    Args:
        config: The configuration loaded from file.
        models: Catalogue names replacing ``config.models``.
        mode: Replacement for ``config.mode``.
        limit: Replacement for ``config.limit``.
        seed: Replacement for ``config.seed``.
        out_dir: Replacement for ``config.out_dir``.

    Returns:
        The updated copy (``config`` itself is left untouched).
    """
    update: dict[str, Any] = {}
    if models is not None:
        update["models"] = list(models)
    if mode is not None:
        update["mode"] = mode
    if limit is not None:
        update["limit"] = limit
    if seed is not None:
        update["seed"] = seed
    if out_dir is not None:
        update["out_dir"] = out_dir
    return config.model_copy(update=update)


def resolve_run(config: RunConfig) -> tuple[RunConfig, TaskConfig, ModelCatalogue]:
    """Cross-check a run configuration against the files it references.

    The task and catalogue paths are resolved relative to the current working
    directory, and every entry of ``config.models`` must exist in the catalogue.

    Args:
        config: The configuration to resolve.

    Returns:
        The configuration together with the loaded task and catalogue.

    Raises:
        FileNotFoundError: If the task or catalogue file does not exist.
        ValueError: If a referenced file is not a mapping or fails validation.
        KeyError: If a model name is not in the catalogue.
    """
    cwd = Path.cwd()
    task = load_task_config(cwd / config.task)
    catalogue = load_model_catalogue(cwd / config.catalogue)
    for name in config.models:
        catalogue.get(name)
    return config, task, catalogue
