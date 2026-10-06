"""Files of the ground-truth review and the step from a reviewed draft to ground truth.

Layout under the data root (DVC-tracked), per split::

    review/<split>/drafts/<scene_id>.json   consensus draft (never edited)
    review/<split>/work/<scene_id>.json     the reviewer's current version
    gt/<split>/<scene_id>.json              finalized ground truth (strict-valid)

:func:`finalize` stamps every element of the ground truth with where it came from, by
comparing the reviewed document with the draft (``meta.provenance``), and records the
counts (``meta.review_summary``) from which the share of human-edited elements is reported.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from rail_vision_bench.dataset.manifest import ManifestRow, gt_path, manifest_path, read_manifest
from rail_vision_bench.eval.records import PredictionRecord
from rail_vision_bench.graph.validator import validate_scene
from rail_vision_bench.review.consensus import lenient_document, merge
from rail_vision_bench.schema.issues import ValidationReport
from rail_vision_bench.schema.models import SceneAnnotation

PROVENANCE_VALUES: Final = (
    "prefill_accepted",
    "suggestion_accepted",
    "human_edited",
    "human_added",
)
"""Per-element provenance in ground truth: unchanged consensus draft element, adopted
single-model suggestion left unchanged, model element changed by the reviewer, element the
reviewer drew."""

_STATE_FAMILY: Final = {
    "nodes": "switches",
    "edges": "tracks",
    "signals": "signals",
    "derailers": "derailers",
    "routes": "routes",
}


def drafts_dir(data_dir: Path, split: str) -> Path:
    """Return ``<data_dir>/review/<split>/drafts``."""
    return data_dir / "review" / split / "drafts"


def work_dir(data_dir: Path, split: str) -> Path:
    """Return ``<data_dir>/review/<split>/work``."""
    return data_dir / "review" / split / "work"


def read_document(path: Path) -> SceneAnnotation:
    """Read one scene document."""
    return SceneAnnotation.model_validate_json(path.read_bytes())


def write_document(doc: SceneAnnotation, path: Path) -> None:
    """Write one scene document as indented JSON (stable diffs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(doc.model_dump(mode="json", exclude_none=True), indent=1, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def documents_for_scene(
    records: list[PredictionRecord], row: ManifestRow, models: list[str]
) -> list[tuple[str, SceneAnnotation]]:
    """The usable document of every model for one scene, in ``models`` order.

    The last attempt per model counts. An answer that failed the strict schema is recovered
    leniently (invalid elements dropped) so that one bad value does not cost the whole
    answer.
    """
    from rail_vision_bench.agents.single_shot import prepare_image

    last: dict[str, PredictionRecord] = {}
    for candidate in records:
        if candidate.scene_id == row.scene_id:
            last[candidate.model] = candidate
    scale: float | None = None
    out: list[tuple[str, SceneAnnotation]] = []
    for model in models:
        record = last.get(model)
        if record is None:
            continue
        if record.annotation is not None:
            out.append((model, record.annotation))
            continue
        if not record.raw_text:
            continue
        if scale is None:
            scale = prepare_image(Path(row.image)).scale
        base = SceneAnnotation.model_validate(
            {
                "schema_version": "v0",
                "scene_id": row.scene_id,
                "source": {
                    "kind": row.source_kind,
                    "image": row.image,
                    "width": row.width,
                    "height": row.height,
                },
                "provenance": {"kind": "prediction", "model_id": model, "run_id": record.run_id},
                "topology": {},
                "state": {},
            }
        )
        recovered = lenient_document(record.raw_text, scale, base)
        if recovered is not None:
            out.append((model, recovered))
    return out


def build_drafts(
    records: list[PredictionRecord],
    *,
    data_dir: Path,
    split: str,
    models: list[str],
    run_id: str,
    overwrite: bool = False,
) -> list[Path]:
    """Write one consensus draft per manifest scene.

    Args:
        records: The pre-fill run's predictions.
        data_dir: The data root.
        split: The split whose manifest lists the scenes.
        models: Merge order (ties go to earlier models).
        run_id: Recorded in each draft.
        overwrite: Replace existing drafts (never touches the reviewer's work files).

    Returns:
        The written draft paths.
    """
    written: list[Path] = []
    for row in read_manifest(manifest_path(data_dir, split)):
        path = drafts_dir(data_dir, split) / f"{row.scene_id}.json"
        if path.exists() and not overwrite:
            continue
        docs = documents_for_scene(records, row, models)
        if not docs:
            continue
        write_document(merge(docs, run_id=run_id), path)
        written.append(path)
    return written


def _strip(element: dict[str, Any]) -> dict[str, Any]:
    """An element without the fields that do not count as content."""
    return {key: value for key, value in element.items() if key not in {"confidence"}}


def _content(doc: dict[str, Any]) -> dict[str, tuple[dict[str, Any], dict[str, Any] | None]]:
    """``id -> (element, state entry)`` over every family of a JSON document."""
    families = {
        "nodes": doc["topology"].get("nodes", []),
        "edges": doc["topology"].get("edges", []),
        "signals": doc.get("signals", []),
        "derailers": doc.get("derailers", []),
        "routes": doc.get("routes", []),
    }
    out: dict[str, tuple[dict[str, Any], dict[str, Any] | None]] = {}
    for family, items in families.items():
        states = doc["state"].get(_STATE_FAMILY[family], {})
        for item in items:
            state = states.get(item["id"])
            out[item["id"]] = (
                _strip(item),
                None if state is None else _strip(state),
            )
    return out


def finalize(
    reviewed: SceneAnnotation, draft: SceneAnnotation, *, annotator: str
) -> tuple[SceneAnnotation, ValidationReport]:
    """Turn a reviewed document into ground truth with per-element provenance.

    Args:
        reviewed: The reviewer's version.
        draft: The consensus draft it started from.
        annotator: Who reviewed it.

    Returns:
        The ground-truth document and its strict validation report; the caller writes it
        only when the report is ok.
    """
    reviewed_json = reviewed.model_dump(mode="json", exclude_none=True)
    draft_json = draft.model_dump(mode="json", exclude_none=True)
    final = _content(reviewed_json)
    seeded = _content(draft_json)
    review_meta = draft.meta.get("review", {})
    suggestions = review_meta.get("suggestions", {})
    provenance: dict[str, str] = {}
    for element_id, content in final.items():
        if element_id in seeded:
            provenance[element_id] = (
                "prefill_accepted" if content == seeded[element_id] else "human_edited"
            )
        elif element_id in suggestions:
            suggested = (
                _strip(suggestions[element_id]["element"]),
                None
                if not suggestions[element_id].get("state")
                else _strip(next(iter(suggestions[element_id]["state"].values()))),
            )
            provenance[element_id] = (
                "suggestion_accepted" if content == suggested else "human_edited"
            )
        else:
            provenance[element_id] = "human_added"
    counts = Counter(provenance.values())
    checked = reviewed.meta.get("review", {}).get("checked", [])
    summary = {
        "draft_elements": len(seeded),
        "final_elements": len(final),
        "deleted_from_draft": len(set(seeded) - set(final)),
        **{value: counts.get(value, 0) for value in PROVENANCE_VALUES},
        "checked": len(set(checked) & set(final)),
        "human_share": round(
            (counts.get("human_edited", 0) + counts.get("human_added", 0)) / max(1, len(final)), 3
        ),
    }
    gt_json = {
        **reviewed_json,
        "provenance": {
            "kind": "ground_truth",
            "annotator": annotator,
            "run_id": review_meta.get("prefill_run"),
            "tool": "rail_vision_bench.review",
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "meta": {
            **{k: v for k, v in reviewed_json.get("meta", {}).items() if k != "review"},
            "provenance": provenance,
            "review_summary": summary,
            "prefill": {
                "run": review_meta.get("prefill_run"),
                "models": review_meta.get("models", []),
            },
        },
    }
    gt = SceneAnnotation.model_validate(gt_json)
    return gt, validate_scene(gt, strict=True)


def gt_file(data_dir: Path, split: str, scene_id: str) -> Path:
    """Where the ground truth of a scene is written."""
    return gt_path(data_dir, split, scene_id)
