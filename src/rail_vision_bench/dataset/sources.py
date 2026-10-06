"""Provenance of third-party source images: ``sources.jsonl`` and the attribution list.

A ``SourceRecord`` describes one downloaded image before it has ground truth: where it came
from, who made it, under which licence, and whether a human accepted it for the benchmark.
Once a scene has a ground-truth document, its ``ManifestRow`` carries the same author,
licence and source fields, so attribution survives into every split.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import orjson
from pydantic import BaseModel, ConfigDict, Field

from rail_vision_bench.dataset.manifest import ManifestRow
from rail_vision_bench.schema.models import ElementId, SourceKind


class CurationStatus(StrEnum):
    """Where an image stands in human curation."""

    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class SourceRecord(BaseModel):
    """One third-party source image with its provenance; paths are repository-relative."""

    model_config = ConfigDict(extra="forbid")

    source_id: ElementId = Field(description="Stable id, e.g. commons-<first 12 hex of sha256>.")
    status: CurationStatus = CurationStatus.CANDIDATE
    image: str = Field(description="Image path relative to the repository root.")
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    width: Annotated[int, Field(ge=1)]
    height: Annotated[int, Field(ge=1)]
    title: str = Field(description="Title at the source, e.g. the Commons 'File:...' name.")
    source_url: str = Field(description="Human-visitable page documenting the image.")
    file_url: str | None = Field(default=None, description="The URL the bytes were fetched from.")
    author: str | None = None
    credit: str | None = None
    license: str = Field(description="SPDX id, or a LicenseRef- id for PD / Commons Attribution.")
    license_name: str = Field(description="Licence short name as the source states it.")
    license_url: str | None = None
    description: str | None = None
    search_term: str | None = Field(default=None, description="The query that surfaced it.")
    display_title: str | None = Field(
        default=None,
        description="Title used in public-facing lists instead of `title` (e.g. without names).",
    )
    display_url: str | None = Field(
        default=None, description="Source link used in public-facing lists instead of source_url."
    )
    panel_group: ElementId | None = Field(
        default=None,
        description="Physical panel or site shown; dev/test splits must keep a group together.",
    )
    curation_note: str | None = Field(default=None, description="Why it was accepted/rejected.")
    modification: str | None = Field(
        default=None, description="How the file differs from the source, e.g. 'faces blurred'."
    )
    blur_regions: list[tuple[int, int, int, int]] = Field(
        default_factory=list, description="Blurred pixel boxes [x0, y0, x1, y1]."
    )
    ingested_at: str = Field(description="UTC timestamp of the download, ISO 8601.")


def read_sources(path: Path) -> list[SourceRecord]:
    """Read a ``sources.jsonl`` file; a missing file is an empty list.

    Args:
        path: The JSON Lines file.

    Returns:
        The records in file order.
    """
    if not path.is_file():
        return []
    return [
        SourceRecord.model_validate(orjson.loads(line))
        for line in path.read_bytes().splitlines()
        if line.strip()
    ]


def write_sources(records: Iterable[SourceRecord], path: Path) -> None:
    """Write records as JSON Lines sorted by ``source_id``, so the file diffs stably.

    Args:
        records: The records to write.
        path: Destination file; parent directories are created.
    """
    ordered = sorted(records, key=lambda record: record.source_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(
            orjson.dumps(record.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS) + b"\n"
            for record in ordered
        )
    )


_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>|])")


def _md(text: str) -> str:
    """Escape Markdown control characters and collapse whitespace."""
    return _MD_SPECIAL.sub(r"\\\1", " ".join(text.split()))


def attribution_markdown(records: Sequence[SourceRecord]) -> str:
    """Render the attribution list (title, author, source, licence) for accepted images.

    Args:
        records: Source records; only ``accepted`` ones are listed, sorted by ``source_id``.

    Returns:
        A Markdown document ending in a newline.
    """
    accepted = sorted(
        (record for record in records if record.status is CurationStatus.ACCEPTED),
        key=lambda record: record.source_id,
    )
    lines = [
        "# Image attribution",
        "",
        "Third-party images in the rail-vision-bench data plane, with the licence each one is",
        "used under. Benchmark tiles, crops and rescaled or perspective-corrected versions are",
        "adaptations of these images (cropping, rescaling, perspective correction). Images",
        "marked *Modified* were changed before inclusion, as stated. Generated by",
        "`bench commons attribution` from `sources.jsonl`; do not edit by hand.",
        "",
        f"{len(accepted)} image(s).",
        "",
    ]
    for record in accepted:
        title = _md(record.display_title or record.title.removeprefix("File:"))
        author = _md(record.author) if record.author else "unknown author"
        licence = (
            f"[{_md(record.license_name)}](<{record.license_url}>)"
            if record.license_url
            else _md(record.license_name)
        )
        url = record.display_url or record.source_url
        entry = f"- `{record.source_id}`: [{title}](<{url}>) by {author}, {licence}."
        if record.credit and record.credit.strip().lower() != "own work":
            entry += f" Credit: {_md(record.credit).rstrip('.')}."
        if record.modification:
            entry += f" Modified ({_md(record.modification)})."
        lines.append(entry)
    return "\n".join(lines) + "\n"


def manifest_rows(
    records: Iterable[SourceRecord], split: str, *, data_dir: Path, source_kind: SourceKind
) -> list[ManifestRow]:
    """Build the manifest rows of a split from the accepted source records.

    The partition is drawn from the ``panel_group`` (the ``source_id`` when a record has no
    group), so all images of one physical panel land in the same partition.

    Args:
        records: Source records; only ``accepted`` ones become rows.
        split: The split name, e.g. ``panel_photo_v1``.
        data_dir: The data root, for the ground-truth paths.
        source_kind: The source kind of every row.

    Returns:
        The rows, sorted by ``scene_id``.
    """
    from rail_vision_bench.dataset.manifest import gt_path
    from rail_vision_bench.dataset.splits import assign_partition

    rows = [
        ManifestRow(
            scene_id=record.source_id,
            split=split,
            partition=assign_partition(record.panel_group or record.source_id),
            source_kind=source_kind,
            image=record.image,
            gt=gt_path(data_dir, split, record.source_id).as_posix(),
            width=record.width,
            height=record.height,
            license=record.license,
            license_url=record.license_url,
            author=record.author,
            source_url=record.display_url or record.source_url,
            panel_group=record.panel_group,
        )
        for record in records
        if record.status is CurationStatus.ACCEPTED
    ]
    return sorted(rows, key=lambda row: row.scene_id)
