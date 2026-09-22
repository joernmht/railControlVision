"""The ``manifest.jsonl`` contract: one row per scene, JSON Lines on disk."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import orjson
from pydantic import BaseModel, ConfigDict, Field

from rail_vision_bench.dataset.splits import Difficulty, Partition
from rail_vision_bench.schema.models import ElementId, SourceKind


class ManifestRow(BaseModel):
    """One scene of a dataset split; paths are relative to the repository root."""

    model_config = ConfigDict(extra="forbid")

    scene_id: ElementId
    split: str = Field(description="Named subset, e.g. synthetic_clean or panel_photo_v1.")
    partition: Partition = Field(description="dev or test, from assign_partition(scene_id).")
    difficulty: Difficulty | None = None
    source_kind: SourceKind
    image: str = Field(description="Image path relative to the repository root.")
    gt: str = Field(description="Ground-truth SceneAnnotation JSON path, repository-relative.")
    width: Annotated[int, Field(ge=1)]
    height: Annotated[int, Field(ge=1)]
    license: str | None = Field(default=None, description="SPDX id or free text; null if unknown.")
    quality: dict[str, float] | None = Field(
        default=None, description="QualityMetrics fields when measured."
    )


def read_manifest(path: Path) -> list[ManifestRow]:
    """Read a JSON Lines manifest.

    Blank lines are skipped so hand-edited files with trailing newlines load.

    Args:
        path: The ``manifest.jsonl`` file.

    Returns:
        The rows in file order.
    """
    return [
        ManifestRow.model_validate(orjson.loads(line))
        for line in path.read_bytes().splitlines()
        if line.strip()
    ]


def write_manifest(rows: Sequence[ManifestRow], path: Path) -> None:
    """Write rows as JSON Lines, one object per line with a trailing newline.

    Args:
        rows: The rows to write.
        path: Destination file; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(orjson.dumps(row.model_dump(mode="json")) + b"\n" for row in rows))
