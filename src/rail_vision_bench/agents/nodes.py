"""The six node functions of the agentic loop and the critic's router.

Every node has the same shape: it is a coroutine that takes the current
:class:`AgentState`, uses the keyword-only ``provider`` for model calls (with
the SDK model id from ``state["model"]``), and returns the keys it updates.
The prompt in ``prompts/agentic/<node>.md`` is the node's specification and is
sent as the system prompt; the user turn carries the state the node reads and
the JSON shape of its answer.

Nodes never raise on a bad model answer: they fall back to a conservative
result, append a message to ``errors`` and let the critic judge the outcome.
Token usage of every call is added to ``usage``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal

from rail_vision_bench.agents.assembly import assemble_candidate
from rail_vision_bench.agents.documents import (
    add_usage,
    default_source,
    stamp_document,
)
from rail_vision_bench.agents.prompts import load_prompt
from rail_vision_bench.agents.state import AgentState
from rail_vision_bench.providers.base import ImageInput, Usage, VisionProvider, VisionRequest
from rail_vision_bench.providers.parsing import extract_json
from rail_vision_bench.schema.issues import IssueCode
from rail_vision_bench.schema.models import Source
from rail_vision_bench.tools.tools import crop_tool, read_tool, render_tool, validate_tool

READ_ZOOM: Final = 2.0
"""Magnification of every crop the reader sends to the model."""
MAX_REGIONS: Final = 24
"""Upper bound on the planner's regions per round (one reader call each)."""
MAX_CRITIQUE_ISSUES: Final = 30
"""Validator issues quoted in one critique; the rest are summarised by count."""

_JSON_ONLY: Final = (
    "Answer with exactly one JSON object and nothing else: no prose, no Markdown fences."
)
_PLAN_FORMAT: Final = (
    '{"image_type": "<Stelltisch panel | ESTW screen | CTC screen | stream frame>", '
    '"plan": [{"item": "<instruction for the reader>", '
    '"bbox": [x0, y0, x1, y1]}]} with every bbox in pixels of the full image, '
    "ordered left to right and top to bottom"
)
_READ_FORMAT: Final = (
    '{"readings": [{"text": "<literal transcription>", "what": "<what it is or looks like>", '
    '"bbox": [x0, y0, x1, y1], "confidence": 0.0}]} with every bbox in pixels of THIS crop '
    "(an empty list when nothing is legible)"
)
_INTERPRET_FORMAT: Final = (
    '{"nodes": [{"id", "kind", "label", "half_labels", "bbox", '
    '"state": {"position" | "active_paths", "raw"}}], '
    '"signals": [{"id", "kind", "system", "label", "bbox", "state": {"aspect", "raw"}}], '
    '"derailers": [{"id", "label", "bbox", "state": {"position"}}], "legend": {}}'
)
_GEOMETRY_FORMAT: Final = (
    '{"nodes": [{"id", "kind": "boundary" | "joint", "point": [x, y]}] (only nodes you add), '
    '"edges": [{"id", "kind", "a": {"node", "port"}, "b": {"node", "port"}, '
    '"polyline": [[x, y], ...]}], '
    '"attachments": {"<signal or derailer id>": {"edge", "offset", "direction"}}, '
    '"tracks": {"<edge id>": {"occupancy", "route_set"}}}'
)
_COMPARE_FORMAT: Final = (
    '{"matches": true | false, "critique": ["<one concrete correction>", ...]} where the '
    "first image is the original and the second the re-rendering of the candidate"
)


def _require_image(state: AgentState) -> ImageInput:
    """Return the scene image of the state.

    Raises:
        ValueError: If the state carries no image.
    """
    image = state.get("image")
    if image is None:
        msg = "the agent state has no 'image'"
        raise ValueError(msg)
    return image


def _require_model(state: AgentState) -> str:
    """Return the SDK model id of the state.

    Raises:
        ValueError: If the state carries no model id.
    """
    model = state.get("model")
    if not model:
        msg = "the agent state has no 'model'"
        raise ValueError(msg)
    return model


def _source(state: AgentState) -> Source:
    """Return the scene's source block, derived from the image when the caller gave none."""
    source = state.get("source")
    if source:
        return Source.model_validate(source)
    image = _require_image(state)
    return default_source(image, scene_id=state.get("scene_id") or "scene")


def _dump(value: object) -> str:
    """Serialise state for a prompt: compact, stable and readable."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


async def _ask_json(
    provider: VisionProvider,
    state: AgentState,
    *,
    node: str,
    context: Mapping[str, Any],
    answer_format: str,
    images: Sequence[ImageInput] = (),
) -> tuple[dict[str, Any] | None, str, Usage]:
    """Send the node's specification, its inputs and the answer format; recover the JSON.

    Args:
        provider: The model backend.
        state: The loop state (for the model id).
        node: The node name; ``prompts/agentic/<node>.md`` is the system prompt.
        context: The state keys the node reads, serialised into the user turn.
        answer_format: The JSON shape the node expects back.
        images: Images sent with the request.

    Returns:
        The recovered object (None when the answer holds none), the raw text and the usage.
    """
    request = VisionRequest(
        model=_require_model(state),
        system=load_prompt(f"agentic/{node}"),
        prompt=f"Inputs:\n{_dump(context)}\n\nAnswer format: {answer_format}\n\n{_JSON_ONLY}",
        images=list(images),
    )
    response = await provider.complete(request)
    parsed = response.parsed if response.parsed is not None else extract_json(response.text)
    return parsed, response.text, response.usage


def _bookkeeping(state: AgentState, usage: Usage, errors: Sequence[str]) -> AgentState:
    """Return the ``usage`` and ``errors`` updates after one node."""
    total = state.get("usage") or Usage()
    return {"usage": add_usage(total, usage), "errors": [*state.get("errors", []), *errors]}


def _usage_of(value: object) -> Usage:
    """Read a usage record out of a tool payload (a model or its JSON dump)."""
    if isinstance(value, Usage):
        return value
    if isinstance(value, Mapping):
        return Usage.model_validate({key: value[key] for key in Usage.model_fields if key in value})
    return Usage()


def _bbox(value: object, width: float, height: float) -> tuple[float, float, float, float] | None:
    """Parse ``[x0, y0, x1, y1]``, order it, clamp it into the frame; None when degenerate."""
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 4:
        return None
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in value):
        return None
    x0, x1 = sorted((min(width, max(0.0, float(value[0]))), min(width, max(0.0, float(value[2])))))
    y0, y1 = sorted(
        (min(height, max(0.0, float(value[1]))), min(height, max(0.0, float(value[3]))))
    )
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    return (x0, y0, x1, y1)


def _plan_regions(
    parsed: Mapping[str, Any] | None, width: int, height: int
) -> list[dict[str, Any]]:
    """Turn the planner's answer into ``{"item", "bbox"}`` regions inside the image."""
    if parsed is None:
        return []
    raw_plan = parsed.get("plan")
    if not isinstance(raw_plan, list):
        return []
    full = (0.0, 0.0, float(width), float(height))
    regions: list[dict[str, Any]] = []
    for entry in raw_plan[:MAX_REGIONS]:
        if isinstance(entry, str) and entry.strip():
            regions.append({"item": entry.strip(), "bbox": list(full)})
        elif isinstance(entry, Mapping) and str(entry.get("item") or "").strip():
            bbox = _bbox(entry.get("bbox"), width, height) or full
            regions.append({"item": str(entry["item"]).strip(), "bbox": list(bbox)})
    return regions


async def planner(state: AgentState, *, provider: VisionProvider) -> AgentState:
    """Plan the round: which regions to read and what the previous critique asks for.

    Reads ``scene_id``, ``image``, ``critique``, ``round`` and ``max_rounds``;
    writes ``plan`` (ordered work items), ``regions`` (each item with its bbox
    in full-image pixels), ``source`` and ``round`` (incremented by one). When
    the model's answer holds no usable plan, a single item covering the whole
    image is planned and the failure is recorded in ``errors``.

    Args:
        state: The loop state.
        provider: The model used to inspect the overview image.

    Returns:
        The updated keys.
    """
    image = _require_image(state)
    source = _source(state)
    round_no = state.get("round", 0) + 1
    context = {
        "scene_id": state.get("scene_id"),
        "image": {"kind": source.kind.value, "width": source.width, "height": source.height},
        "round": round_no,
        "max_rounds": state.get("max_rounds"),
        "critique": state.get("critique"),
    }
    parsed, _, usage = await _ask_json(
        provider, state, node="planner", context=context, answer_format=_PLAN_FORMAT, images=[image]
    )
    regions = _plan_regions(parsed, source.width, source.height)
    errors = []
    if not regions:
        errors.append(f"planner round {round_no}: no usable plan, reading the whole image")
        regions = [
            {
                "item": "read every label, indication and legend in the whole image",
                "bbox": [0.0, 0.0, float(source.width), float(source.height)],
            }
        ]
    return {
        "plan": [region["item"] for region in regions],
        "regions": regions,
        "round": round_no,
        "source": source.model_dump(mode="json"),
        **_bookkeeping(state, usage, errors),
    }


def _readings_of(
    payload: Mapping[str, Any], *, crop_index: int, bbox: Sequence[float], zoom: float
) -> list[dict[str, Any]]:
    """Convert the read tool's answer into readings with full-image bboxes."""
    text = str(payload.get("text") or "")
    parsed = extract_json(text)
    x0, y0, x1, y1 = bbox
    entries = parsed.get("readings") if parsed is not None else None
    if not isinstance(entries, list):
        if not text.strip():
            return []
        confidence = payload.get("confidence")
        return [
            {
                "crop": crop_index,
                "text": text.strip(),
                "what": "unstructured answer for the whole crop",
                "bbox": list(bbox),
                "confidence": confidence if isinstance(confidence, int | float) else None,
            }
        ]
    readings = []
    for entry in entries:
        if not isinstance(entry, Mapping) or entry.get("text") is None:
            continue
        local = _bbox(entry.get("bbox"), (x1 - x0) * zoom, (y1 - y0) * zoom)
        full = (
            [x0 + local[0] / zoom, y0 + local[1] / zoom, x0 + local[2] / zoom, y0 + local[3] / zoom]
            if local is not None
            else list(bbox)
        )
        confidence = entry.get("confidence")
        readings.append(
            {
                "crop": crop_index,
                "text": str(entry["text"]),
                "what": str(entry.get("what") or ""),
                "bbox": full,
                "confidence": (
                    min(1.0, max(0.0, float(confidence)))
                    if isinstance(confidence, int | float) and not isinstance(confidence, bool)
                    else None
                ),
            }
        )
    return readings


async def reader(state: AgentState, *, provider: VisionProvider) -> AgentState:
    """Transcribe labels, legends and indications from crops of the planned regions.

    Reads ``image``, ``model`` and ``regions`` (falling back to ``plan`` over
    the whole image); writes ``crops`` and ``readings``. Every region is cut
    out with the crop tool at :data:`READ_ZOOM` and sent to the read tool; the
    regions are read concurrently. Bboxes the model reports in crop pixels are
    mapped back to full-image pixels. The read tool wraps its answer as
    ``{"text", "confidence"}``; the reader asks for its readings object inside
    ``text`` and falls back to one reading per crop when ``text`` holds none.

    Args:
        state: The loop state.
        provider: The model used to read each crop.

    Returns:
        The updated keys.
    """
    image = _require_image(state)
    model = _require_model(state)
    source = _source(state)
    regions = state.get("regions") or [
        {"item": item, "bbox": [0.0, 0.0, float(source.width), float(source.height)]}
        for item in state.get("plan", [])
    ]
    instructions = load_prompt("agentic/reader")

    async def read_region(
        index: int, region: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], Usage, str | None]:
        bbox = tuple(float(v) for v in region["bbox"])
        x0, y0, x1, y1 = bbox
        try:
            crop = crop_tool(image.data, (x0, y0, x1, y1), zoom=READ_ZOOM)
        except (ValueError, OSError) as exc:
            return [], Usage(), f"reader crop {index}: {exc}"
        question = (
            f"{instructions}\n\nPlan item: {region['item']}\n"
            f"The image is the region {list(bbox)} of the full image, magnified "
            f"{READ_ZOOM:g}x.\n\nThe `text` of your answer is itself the JSON object "
            f"{_READ_FORMAT}"
        )
        result = await read_tool(crop, question, provider=provider, model=model)
        usage = _usage_of(result.payload.get("usage"))
        if not result.ok:
            return [], usage, f"reader crop {index}: {'; '.join(result.errors) or 'read failed'}"
        readings = _readings_of(result.payload, crop_index=index, bbox=bbox, zoom=READ_ZOOM)
        return readings, usage, None

    outcomes = await asyncio.gather(
        *(read_region(index, region) for index, region in enumerate(regions))
    )
    usage = Usage()
    readings: list[dict[str, Any]] = []
    errors: list[str] = []
    for found, used, error in outcomes:
        readings.extend(found)
        usage = add_usage(usage, used)
        if error is not None:
            errors.append(error)
    crops = [{"bbox": list(region["bbox"]), "item": region["item"]} for region in regions]
    return {"crops": crops, "readings": readings, **_bookkeeping(state, usage, errors)}


def _keep_mappings(parsed: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    """Keep ``keys`` of ``parsed`` whose values are lists (of objects) or objects."""
    kept: dict[str, Any] = {}
    for key in keys:
        value = parsed.get(key)
        if isinstance(value, list):
            kept[key] = [item for item in value if isinstance(item, Mapping)]
        elif isinstance(value, Mapping):
            kept[key] = dict(value)
    return kept


async def interpreter(state: AgentState, *, provider: VisionProvider) -> AgentState:
    """Map transcribed symbols and literals to element kinds and states.

    Reads ``readings``, ``crops`` and ``image`` (the symbols themselves are
    only visible in the image); writes ``interpretation`` with ``nodes``,
    ``signals``, ``derailers`` and ``legend``. An unusable answer keeps the
    previous round's interpretation (or an empty one) and records an error.

    Args:
        state: The loop state.
        provider: The model used to classify symbols.

    Returns:
        The updated keys.
    """
    image = _require_image(state)
    context = {"readings": state.get("readings", []), "crops": state.get("crops", [])}
    parsed, _, usage = await _ask_json(
        provider,
        state,
        node="interpreter",
        context=context,
        answer_format=_INTERPRET_FORMAT,
        images=[image],
    )
    errors = []
    if parsed is None:
        errors.append(f"interpreter round {state.get('round', 0)}: no JSON object in the answer")
        interpretation = state.get("interpretation") or {}
    else:
        interpretation = {
            "nodes": [],
            "signals": [],
            "derailers": [],
            "legend": {},
            **_keep_mappings(parsed, ("nodes", "signals", "derailers", "legend")),
        }
    return {"interpretation": interpretation, **_bookkeeping(state, usage, errors)}


async def geometer(state: AgentState, *, provider: VisionProvider) -> AgentState:
    """Connect the interpreted elements into edges with node ports and attachments.

    Reads ``image``, ``interpretation`` and ``readings``; writes ``geometry``
    with ``edges``, ``attachments``, ``tracks`` and the ``nodes`` the
    geometer added (boundaries, joints). An unusable answer keeps the previous
    geometry (or an empty one) and records an error.

    Args:
        state: The loop state.
        provider: The model used to trace track lines.

    Returns:
        The updated keys.
    """
    image = _require_image(state)
    source = _source(state)
    context = {
        "image": {"width": source.width, "height": source.height},
        "interpretation": state.get("interpretation", {}),
        "readings": state.get("readings", []),
    }
    parsed, _, usage = await _ask_json(
        provider,
        state,
        node="geometer",
        context=context,
        answer_format=_GEOMETRY_FORMAT,
        images=[image],
    )
    errors = []
    if parsed is None:
        errors.append(f"geometer round {state.get('round', 0)}: no JSON object in the answer")
        geometry = state.get("geometry") or {}
    else:
        geometry = {
            "nodes": [],
            "edges": [],
            "attachments": {},
            "tracks": {},
            **_keep_mappings(parsed, ("nodes", "edges", "attachments", "tracks")),
        }
    return {"geometry": geometry, **_bookkeeping(state, usage, errors)}


def _error_count(document: Mapping[str, Any]) -> int:
    """Number of error-level validator issues of a candidate."""
    return len(validate_tool(document).errors)


async def builder(state: AgentState, *, provider: VisionProvider) -> AgentState:
    """Assemble one schema v0 document from the interpretation and the geometry.

    Reads ``scene_id``, ``model``, ``source``, ``interpretation`` and
    ``geometry``; writes ``candidate``. Code first assembles a draft
    deterministically (:func:`~rail_vision_bench.agents.assembly.assemble_candidate`),
    then the model receives the draft with the builder specification and
    returns the complete document, adding routes and fixing the draft where
    the rules demand. Both are stamped with ``schema_version``, ``scene_id``,
    ``source`` and ``provenance`` by code; the model's document is kept unless
    it has more error-level validator issues than the draft.

    Args:
        state: The loop state.
        provider: The model used to emit the JSON document.

    Returns:
        The updated keys.
    """
    model = _require_model(state)
    source = _source(state)
    scene_id = state.get("scene_id") or "scene"
    interpretation = state.get("interpretation") or {}
    geometry = state.get("geometry") or {}
    draft = stamp_document(
        assemble_candidate(interpretation, geometry, width=source.width, height=source.height),
        scene_id=scene_id,
        source=source,
        model=model,
    )
    context = {
        "scene_id": scene_id,
        "source": source.model_dump(mode="json", exclude_none=True),
        "interpretation": interpretation,
        "geometry": geometry,
        "draft": draft,
    }
    parsed, _, usage = await _ask_json(
        provider,
        state,
        node="builder",
        context=context,
        answer_format=(
            "the complete schema v0 document; start from `draft`, which code assembled from "
            "the interpretation and the geometry, add the routes and fix what the rules demand"
        ),
    )
    errors = []
    candidate = draft
    if parsed is None:
        errors.append(f"builder round {state.get('round', 0)}: no JSON object, keeping the draft")
    else:
        proposed = stamp_document(parsed, scene_id=scene_id, source=source, model=model)
        if _error_count(proposed) <= _error_count(draft):
            candidate = proposed
        else:
            errors.append(
                f"builder round {state.get('round', 0)}: the model's document has more "
                "validation errors than the draft, keeping the draft"
            )
    return {"candidate": candidate, **_bookkeeping(state, usage, errors)}


def _issue_line(issue: Mapping[str, Any]) -> str:
    """Format one validator issue as a critique item."""
    where = f" at {issue['path']}" if issue.get("path") else ""
    element = f" ({issue['element_id']})" if issue.get("element_id") else ""
    return f"{issue['severity']} {issue['code']}{where}{element}: {issue['message']}"


async def critic(state: AgentState, *, provider: VisionProvider) -> AgentState:
    """Validate the candidate, compare its re-rendering with the image and judge.

    Reads ``candidate``, ``image``, ``readings``, ``round`` and ``max_rounds``;
    writes ``validation`` (the validate tool's report), ``critique`` and
    ``done``. The render tool redraws a schema-valid candidate and the model
    compares it with the original; the candidate is accepted (``done``) when
    the report has no error-level issue and the comparison found no mismatch.
    When no re-rendering is possible or the comparison answer is unusable,
    the decision rests on the report alone and the reason is recorded in
    ``errors``.

    Args:
        state: The loop state.
        provider: The model used to compare the re-rendering with the image.

    Returns:
        The updated keys.
    """
    image = _require_image(state)
    candidate = state.get("candidate")
    round_no = state.get("round", 0)
    if candidate is None:
        return {
            "validation": None,
            "critique": "- no candidate could be assembled; read and trace the panel again",
            "done": False,
        }
    result = validate_tool(candidate)
    report = result.payload
    issues = [issue for issue in report.get("issues", []) if isinstance(issue, Mapping)]
    items = [_issue_line(issue) for issue in issues[:MAX_CRITIQUE_ISSUES]]
    if len(issues) > MAX_CRITIQUE_ISSUES:
        items.append(f"... and {len(issues) - MAX_CRITIQUE_ISSUES} more validator issues")

    usage = Usage()
    errors: list[str] = []
    matches: bool | None = None
    if any(issue.get("code") == IssueCode.SCHEMA_INVALID.value for issue in issues):
        errors.append(f"critic round {round_no}: candidate fails schema v0, not re-rendered")
    else:
        try:
            rendering = render_tool(candidate)
        except (ValueError, OSError) as exc:
            errors.append(f"critic round {round_no}: render failed: {exc}")
        else:
            context = {
                "round": round_no,
                "max_rounds": state.get("max_rounds"),
                "validation": report,
                "readings": state.get("readings", []),
                "candidate": candidate,
            }
            parsed, _, usage = await _ask_json(
                provider,
                state,
                node="critic",
                context=context,
                answer_format=_COMPARE_FORMAT,
                images=[image, ImageInput(data=rendering, media_type="image/png")],
            )
            verdict = parsed.get("matches") if parsed is not None else None
            if isinstance(verdict, bool):
                matches = verdict
                remarks = parsed.get("critique") if parsed is not None else None
                if isinstance(remarks, list):
                    items.extend(str(remark) for remark in remarks if str(remark).strip())
                elif isinstance(remarks, str) and remarks.strip():
                    items.append(remarks.strip())
            else:
                errors.append(f"critic round {round_no}: no usable comparison verdict")

    done = result.ok and matches is not False
    critique = None if done else "\n".join(f"- {item}" for item in items) or None
    if not done and critique is None:
        critique = "- the re-rendering does not match the image; re-read the panel"
    return {
        "validation": report,
        "critique": critique,
        "done": done,
        **_bookkeeping(state, usage, errors),
    }


def critic_should_continue(state: AgentState) -> Literal["planner", "__end__"]:
    """Route after the critic: stop when it is satisfied or the round budget is spent.

    Args:
        state: The loop state after the critic ran.

    Returns:
        ``"__end__"`` when ``done`` is set or ``round`` reached ``max_rounds``,
        ``"planner"`` for another round.
    """
    if state.get("done") or state.get("round", 0) >= state.get("max_rounds", 1):
        return "__end__"
    return "planner"
