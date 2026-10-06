"""The ground-truth review app: a phone-friendly page over the drafts of one split.

Serve it on localhost only (``bench review serve``); it is reached over the private tailnet,
never the public internet, because the drafts are unpublished data. The page
(``review/static/index.html``) talks to the JSON API below.
"""

from __future__ import annotations

import io
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, ValidationError

from rail_vision_bench import __version__
from rail_vision_bench.dataset.manifest import ManifestRow, manifest_path, read_manifest
from rail_vision_bench.graph.validator import validate_scene
from rail_vision_bench.review.store import (
    drafts_dir,
    finalize,
    gt_file,
    read_document,
    work_dir,
    write_document,
)
from rail_vision_bench.schema.issues import ValidationReport
from rail_vision_bench.schema.models import SceneAnnotation

IMAGE_MAX_SIDE = 2400
"""Longest side of the image sent to the page; the overlay works in original pixels."""


class SaveResponse(BaseModel):
    """Answer to a save or finalize: the validation report and, on finalize, the summary."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    report: ValidationReport
    summary: dict[str, Any] | None = None
    written: str | None = None


def create_review_app(data_dir: Path, split: str, *, annotator: str) -> FastAPI:
    """Build the review application for one split.

    Args:
        data_dir: The data root (manifest, drafts, work files, ground truth).
        split: The split under review.
        annotator: Recorded as the ground truth's annotator.

    Returns:
        The FastAPI application.
    """
    app = FastAPI(title="rail-vision-bench review", version=__version__)

    def rows() -> dict[str, ManifestRow]:
        return {row.scene_id: row for row in read_manifest(manifest_path(data_dir, split))}

    def row_for(scene_id: str) -> ManifestRow:
        row = rows().get(scene_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"unknown scene {scene_id!r}")
        return row

    def draft_for(scene_id: str) -> SceneAnnotation:
        path = drafts_dir(data_dir, split) / f"{scene_id}.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"no draft for {scene_id!r}")
        return read_document(path)

    @lru_cache(maxsize=16)
    def image_bytes(path: str) -> bytes:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        image.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        return buffer.getvalue()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """The single-page review UI."""
        return (
            files("rail_vision_bench")
            .joinpath("review", "static", "index.html")
            .read_text(encoding="utf-8")
        )

    @app.get("/api/scenes")
    def scenes() -> list[dict[str, Any]]:
        """Every scene of the split with its review status and draft counts."""
        out = []
        for scene_id, row in sorted(rows().items()):
            draft_path = drafts_dir(data_dir, split) / f"{scene_id}.json"
            elements: dict[str, Any] = {}
            if draft_path.is_file():
                elements = read_document(draft_path).meta.get("review", {}).get("elements", {})
            status = (
                "done"
                if gt_file(data_dir, split, scene_id).is_file()
                else "in_progress"
                if (work_dir(data_dir, split) / f"{scene_id}.json").is_file()
                else "todo"
                if draft_path.is_file()
                else "no_draft"
            )
            out.append(
                {
                    "scene_id": scene_id,
                    "panel_group": row.panel_group,
                    "partition": row.partition,
                    "status": status,
                    "elements": len(elements),
                    "disputed": sum(1 for e in elements.values() if e["status"] == "disputed"),
                    "width": row.width,
                    "height": row.height,
                }
            )
        return out

    @app.get("/api/scenes/{scene_id}")
    def scene(scene_id: str) -> dict[str, Any]:
        """The current document (work file, else draft), the draft and the manifest row."""
        row = row_for(scene_id)
        draft = draft_for(scene_id)
        work_path = work_dir(data_dir, split) / f"{scene_id}.json"
        current = read_document(work_path) if work_path.is_file() else draft
        return {
            "row": row.model_dump(mode="json"),
            "draft": draft.model_dump(mode="json", exclude_none=True),
            "document": current.model_dump(mode="json", exclude_none=True),
            "report": validate_scene(current, strict=True).model_dump(mode="json"),
            "done": gt_file(data_dir, split, scene_id).is_file(),
        }

    @app.get("/api/scenes/{scene_id}/image")
    def image(scene_id: str) -> Response:
        """The scene image, downscaled for the phone (the page scales the overlay)."""
        return Response(content=image_bytes(row_for(scene_id).image), media_type="image/jpeg")

    def parse(scene_id: str, document: dict[str, Any]) -> SceneAnnotation:
        try:
            doc = SceneAnnotation.model_validate(document)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()[:5]) from None
        if doc.scene_id != scene_id:
            raise HTTPException(status_code=422, detail="scene_id does not match the URL")
        return doc

    @app.put("/api/scenes/{scene_id}/work")
    def save(scene_id: str, document: dict[str, Any]) -> SaveResponse:
        """Save the reviewer's version; always stored, validation is reported."""
        row_for(scene_id)
        doc = parse(scene_id, document)
        write_document(doc, work_dir(data_dir, split) / f"{scene_id}.json")
        report = validate_scene(doc, strict=True)
        return SaveResponse(ok=report.ok, report=report)

    @app.post("/api/scenes/{scene_id}/finalize")
    def finalize_scene(scene_id: str, document: dict[str, Any]) -> SaveResponse:
        """Save, then write the ground truth if it is strict-valid."""
        row_for(scene_id)
        doc = parse(scene_id, document)
        write_document(doc, work_dir(data_dir, split) / f"{scene_id}.json")
        gt, report = finalize(doc, draft_for(scene_id), annotator=annotator)
        if not report.ok:
            return SaveResponse(ok=False, report=report)
        path = gt_file(data_dir, split, scene_id)
        write_document(gt, path)
        return SaveResponse(
            ok=True, report=report, summary=gt.meta["review_summary"], written=path.as_posix()
        )

    return app
