"""Merge several models' documents of one scene into a pre-filled review draft.

Elements are clustered across models family by family with the same cost and Hungarian
assignment the metrics use (:mod:`rail_vision_bench.eval.matching`), nodes first, then edges
(which additionally compare the node clusters at their ends), then signals, derailers and
routes. Each cluster becomes one draft element whose attributes are the majority of its
members'; its *support* is the number of models that produced it.

A cluster enters the draft when at least ``min_support`` models produced it, or when a draft
element references it. Clusters with less support become *suggestions* the reviewer can adopt
with one tap. Everything the reviewer needs to judge the draft is in ``meta.review``: which
models were merged, and per element its support, the supporting models, its status
(:func:`status`: ``consensus``, ``contested`` or ``minority``) and the attributes the models
disagreed on.

Model-seeded ground truth is biased towards the seeding models wherever the reviewer misses
an error; per-element support and the later share of human-edited elements make that bias
measurable (see ``data/README.md``).
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, TypeVar

from pydantic import BaseModel, ValidationError

from rail_vision_bench.agents.single_shot import extract_json, rescale_geometry
from rail_vision_bench.eval.matching import (
    UNMATCHABLE,
    Element,
    Family,
    anchor,
    assign,
    elements,
    geometry_distance,
    pair_cost,
)
from rail_vision_bench.schema.models import (
    Derailer,
    DerailerState,
    EdgeAttachment,
    Geometry,
    Node,
    NodeKind,
    PortRef,
    Route,
    RouteState,
    SceneAnnotation,
    Signal,
    SignalState,
    State,
    SwitchState,
    Track,
    TrackState,
)

MIN_SUPPORT: Final = 2
"""Models that must agree on an element before it enters the draft unasked."""

THRESHOLD: Final = 0.6
"""Maximum assignment cost of two elements that are taken to be the same."""

SWITCHING_KINDS: Final = frozenset({NodeKind.SWITCH, NodeKind.DKW, NodeKind.EKW})
_PREFIX: Final[dict[Family, str]] = {
    "nodes": "n",
    "edges": "e",
    "signals": "s",
    "derailers": "d",
    "routes": "r",
}
_FAMILIES: Final[tuple[Family, ...]] = ("nodes", "edges", "signals", "derailers", "routes")

T = TypeVar("T")


@dataclass
class Member:
    """One model's element inside a cluster."""

    model: str
    doc: SceneAnnotation
    element: Element


@dataclass
class Cluster:
    """Elements of one family that several models produced for the same thing."""

    family: Family
    members: list[Member] = field(default_factory=list)
    draft_id: str = ""

    @property
    def models(self) -> list[str]:
        """The supporting models, in merge order."""
        return [member.model for member in self.members]


def majority(values: Sequence[T], key: Callable[[T], Hashable] = lambda v: v) -> tuple[T, float]:
    """Return the most common value (first seen wins ties) and the share that agrees with it."""
    counts = Counter(key(value) for value in values)
    best = max(counts.values())
    winner = next(value for value in values if counts[key(value)] == best)
    return winner, best / len(values)


def repair_truncated_json(text: str) -> dict[str, Any] | None:
    """Recover the complete part of a JSON object that was cut off mid-way.

    Answers that hit the output-token limit stop in the middle of an element. The text is
    scanned (strings and escapes respected) and cut right after the last value that was
    closed completely; the brackets still open at that point are then closed.

    Args:
        text: The model output.

    Returns:
        The recovered object, or ``None`` when no prefix can be closed into an object.
    """
    start = text.find("{")
    if start < 0:
        return None
    body = text[start:]
    stack: list[str] = []
    in_string = escaped = False
    cut: tuple[int, str] | None = None
    for index, char in enumerate(body):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                break
            stack.pop()
            cut = (index + 1, "".join(stack))
            if not stack:
                break
    if cut is None:
        return None
    position, still_open = cut
    closing = "".join("}" if bracket == "{" else "]" for bracket in reversed(still_open))
    try:
        value = json.loads(body[:position] + closing)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def lenient_document(raw_text: str, scale: float, base: SceneAnnotation) -> SceneAnnotation | None:
    """Recover what is valid from an answer that failed the strict schema.

    Each element and state entry is validated on its own and dropped when invalid, so one bad
    enum value does not discard a whole panel. An answer cut off at the output-token limit is
    recovered up to its last complete value. ``base`` supplies the header fields.

    Returns:
        The recovered document, or ``None`` when the answer holds no JSON object.
    """
    try:
        decoded: dict[str, Any] | None = extract_json(raw_text)
    except ValueError:
        decoded = repair_truncated_json(raw_text)
    if decoded is None:
        return None
    raw = rescale_geometry(decoded, scale)

    def keep(items: Any, model: type[BaseModel]) -> list[dict[str, Any]]:
        out = []
        for item in items if isinstance(items, list) else []:
            try:
                out.append(model.model_validate(item).model_dump(mode="json"))
            except ValidationError:
                continue
        return out

    def keep_state(entries: Any, model: type[BaseModel]) -> dict[str, Any]:
        out = {}
        for key, item in (entries if isinstance(entries, dict) else {}).items():
            try:
                out[str(key)] = model.model_validate(item).model_dump(mode="json")
            except ValidationError:
                continue
        return out

    topology = raw.get("topology") if isinstance(raw.get("topology"), dict) else {}
    state = raw.get("state") if isinstance(raw.get("state"), dict) else {}
    document = {
        **base.model_dump(mode="json", include={"schema_version", "scene_id", "source"}),
        "provenance": base.provenance.model_dump(mode="json"),
        "topology": {
            "nodes": keep(topology.get("nodes"), Node),
            "edges": keep(topology.get("edges"), Track),
        },
        "signals": keep(raw.get("signals"), Signal),
        "derailers": keep(raw.get("derailers"), Derailer),
        "routes": keep(raw.get("routes"), Route),
        "state": {
            "switches": keep_state(state.get("switches"), SwitchState),
            "signals": keep_state(state.get("signals"), SignalState),
            "tracks": keep_state(state.get("tracks"), TrackState),
            "derailers": keep_state(state.get("derailers"), DerailerState),
            "routes": keep_state(state.get("routes"), RouteState),
        },
        "meta": raw.get("meta") if isinstance(raw.get("meta"), dict) else {},
    }
    try:
        return SceneAnnotation.model_validate(document)
    except ValidationError:
        return None


class _Merger:
    """Clusters the documents of one scene and builds the draft."""

    def __init__(self, docs: Sequence[tuple[str, SceneAnnotation]], min_support: int) -> None:
        self.docs = docs
        self.min_support = min_support
        reference = docs[0][1]
        self.diagonal = math.hypot(reference.source.width, reference.source.height)
        self.clusters: dict[Family, list[Cluster]] = {}
        # (model, element id) -> cluster, per family, to resolve references between families
        self.owner: dict[tuple[str, str], Cluster] = {}

    # -- clustering ------------------------------------------------------------------------

    def _cost(self, family: Family, rep: Member, candidate: Member) -> float:
        if family != "edges":
            return pair_cost(rep.element, candidate.element, self.diagonal)
        first, second = rep.element, candidate.element
        assert isinstance(first, Track)
        assert isinstance(second, Track)
        ends_first = {
            id(c)
            for node in (first.a.node, first.b.node)
            if (c := self._node_cluster(rep.model, node)) is not None
        }
        ends_second = {
            id(c)
            for node in (second.a.node, second.b.node)
            if (c := self._node_cluster(candidate.model, node)) is not None
        }
        shared = len(ends_first & ends_second)
        geometry = geometry_distance(first.geometry, second.geometry, self.diagonal)
        if shared == 0 and geometry >= 1.0:
            return UNMATCHABLE
        return 0.3 * (1 - shared / 2) + 0.7 * geometry

    def _node_cluster(self, model: str, node_id: str) -> Cluster | None:
        return self.owner.get((model, f"nodes/{node_id}"))

    def cluster(self, family: Family) -> None:
        clusters: list[Cluster] = []
        for model, doc in self.docs:
            members = [Member(model, doc, element) for element in elements(doc, family)]
            if family == "routes":
                matched = self._match_routes(clusters, members)
            else:
                cost = [[self._cost(family, c.members[0], m) for m in members] for c in clusters]
                matched = {col: row for row, col, _ in assign(cost, THRESHOLD)} if members else {}
            for col, member in enumerate(members):
                target = clusters[matched[col]] if col in matched else None
                if target is None:
                    target = Cluster(family)
                    clusters.append(target)
                target.members.append(member)
                self.owner[(model, f"{family}/{member.element.id}")] = target
        self.clusters[family] = clusters

    def _route_key(self, member: Member) -> tuple[int, ...]:
        route = member.element
        assert isinstance(route, Route)
        start = self.owner.get((member.model, f"signals/{route.start}"))
        path = [self.owner.get((member.model, f"edges/{edge}")) for edge in route.path]
        return (id(start), *(id(c) for c in path))

    def _match_routes(self, clusters: list[Cluster], members: list[Member]) -> dict[int, int]:
        keys = {self._route_key(c.members[0]): row for row, c in enumerate(clusters)}
        return {
            col: keys[self._route_key(m)]
            for col, m in enumerate(members)
            if self._route_key(m) in keys
        }

    # -- ids -------------------------------------------------------------------------------

    def assign_ids(self) -> None:
        """Give every cluster a deterministic id, ordered top-to-bottom, left-to-right."""
        for family in _FAMILIES:

            def position(cluster: Cluster) -> tuple[float, float]:
                points = [
                    p
                    for m in cluster.members
                    if not isinstance(m.element, Route)
                    and (p := anchor(m.element.geometry)) is not None
                ]
                if not points:
                    return (math.inf, math.inf)
                return (
                    statistics.median(p[1] for p in points),
                    statistics.median(p[0] for p in points),
                )

            ordered = sorted(self.clusters[family], key=position)
            for index, cluster in enumerate(ordered, start=1):
                cluster.draft_id = f"{_PREFIX[family]}{index:03d}"

    # -- building --------------------------------------------------------------------------

    def ref(self, member: Member, family: Family, element_id: str) -> str:
        """Translate a member's reference into the draft id space."""
        cluster = self.owner.get((member.model, f"{family}/{element_id}"))
        return cluster.draft_id if cluster is not None else element_id

    def build(self, cluster: Cluster) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the merged element (as JSON) and its review record."""
        members = cluster.members
        disagree: list[str] = []

        def pick(name: str, values: Sequence[T], key: Callable[[T], Hashable] = lambda v: v) -> T:
            value, share = majority(values, key)
            if share < 1.0:
                disagree.append(name)
            return value

        support = len(members)
        confidence = round(support / len(self.docs), 3)
        element: dict[str, Any]
        if cluster.family == "nodes":
            nodes = [m.element for m in members if isinstance(m.element, Node)]
            kind = pick("kind", [n.kind for n in nodes])
            points = [p for n in nodes if (p := anchor(n.geometry)) is not None]
            element = Node(
                id=cluster.draft_id,
                kind=kind,
                label=pick("label", [n.label for n in nodes]),
                half_labels=pick("half_labels", [n.half_labels for n in nodes], tuple)
                if kind in {NodeKind.DKW, NodeKind.EKW}
                else [],
                geometry=Geometry(point=_median_point(points)) if points else None,
                confidence=confidence,
            ).model_dump(mode="json", exclude_none=True)
        elif cluster.family == "edges":
            tracks = [(m, m.element) for m in members if isinstance(m.element, Track)]
            ends = pick(
                "ends",
                [
                    (
                        self.ref(m, "nodes", t.a.node),
                        t.a.port,
                        self.ref(m, "nodes", t.b.node),
                        t.b.port,
                    )
                    for m, t in tracks
                ],
                lambda e: frozenset({(e[0], e[1]), (e[2], e[3])}),
            )
            element = Track(
                id=cluster.draft_id,
                kind=pick("kind", [t.kind for _, t in tracks]),
                label=pick("label", [t.label for _, t in tracks]),
                a=PortRef(node=ends[0], port=ends[1]),
                b=PortRef(node=ends[2], port=ends[3]),
                geometry=_medoid_geometry([t.geometry for _, t in tracks], self.diagonal),
                confidence=confidence,
            ).model_dump(mode="json", exclude_none=True)
        elif cluster.family in {"signals", "derailers"}:
            items = [(m, m.element) for m in members if isinstance(m.element, Signal | Derailer)]
            attachment = pick(
                "at",
                [(self.ref(m, "edges", e.at.edge), e.at.direction) for m, e in items],
            )
            at = EdgeAttachment(
                edge=attachment[0],
                offset=round(statistics.median(e.at.offset for _, e in items), 3),
                direction=attachment[1],
            )
            points = [p for _, e in items if (p := anchor(e.geometry)) is not None]
            geometry = Geometry(point=_median_point(points)) if points else None
            label = pick("label", [e.label for _, e in items])
            if cluster.family == "signals":
                signals = [e for _, e in items if isinstance(e, Signal)]
                element = Signal(
                    id=cluster.draft_id,
                    kind=pick("kind", [s.kind for s in signals]),
                    label=label,
                    system=pick("system", [s.system for s in signals]),
                    at=at,
                    geometry=geometry,
                    confidence=confidence,
                ).model_dump(mode="json", exclude_none=True)
            else:
                element = Derailer(
                    id=cluster.draft_id,
                    label=label,
                    at=at,
                    geometry=geometry,
                    confidence=confidence,
                ).model_dump(mode="json", exclude_none=True)
        else:
            first = members[0]
            route = first.element
            assert isinstance(route, Route)
            element = Route(
                id=cluster.draft_id,
                label=pick("label", [m.element.label for m in members]),
                start=self.ref(first, "signals", route.start),
                end=self.ref(first, "signals", route.end)
                if (first.model, f"signals/{route.end}") in self.owner
                else self.ref(first, "nodes", route.end),
                path=[self.ref(first, "edges", edge) for edge in route.path],
                switch_positions={
                    self.ref(first, "nodes", node): setting
                    for node, setting in route.switch_positions.items()
                },
                confidence=confidence,
            ).model_dump(mode="json", exclude_none=True)
        review = {
            "family": cluster.family,
            "support": support,
            "of": len(self.docs),
            "models": cluster.models,
            "disagree": disagree,
            "status": status(support, len(self.docs), disagree),
        }
        return element, review

    def state(self, cluster: Cluster) -> tuple[str, dict[str, Any], list[str]] | None:
        """Return (state family, merged state entry, disagreeing fields) for a cluster."""
        table: dict[Family, tuple[str, Callable[[State], dict[str, Any]]]] = {
            "nodes": ("switches", lambda s: s.switches),
            "edges": ("tracks", lambda s: s.tracks),
            "signals": ("signals", lambda s: s.signals),
            "derailers": ("derailers", lambda s: s.derailers),
            "routes": ("routes", lambda s: s.routes),
        }
        name, getter = table[cluster.family]
        entries = [
            entry.model_dump(mode="json", exclude={"confidence"})
            for m in cluster.members
            if (entry := getter(m.doc.state).get(m.element.id)) is not None
        ]
        if cluster.family == "nodes":
            node = cluster.members[0].element
            kinds = {m.element.kind for m in cluster.members if isinstance(m.element, Node)}
            if not isinstance(node, Node) or not kinds & SWITCHING_KINDS:
                return None
        if not entries:
            return None
        merged, share = majority(entries, lambda e: repr(sorted(e.items())))
        return name, merged, ([] if share == 1.0 else [f"state.{name}"])


def status(support: int, of: int, disagree: Sequence[str]) -> str:
    """Review status of a merged element.

    ``consensus``: more than half of the models produced it and they agree on every attribute;
    ``contested``: more than half produced it but they disagree on an attribute;
    ``minority``: at most half of the models produced it.
    """
    if support * 2 <= of:
        return "minority"
    return "contested" if disagree else "consensus"


def references(element: dict[str, Any]) -> list[str]:
    """The ids a draft element (as JSON) points at: edge ends, attachments, route parts."""
    refs = [element[end]["node"] for end in ("a", "b") if isinstance(element.get(end), dict)]
    if isinstance(element.get("at"), dict):
        refs.append(element["at"]["edge"])
    if "path" in element:
        refs.extend([element["start"], element["end"], *element["path"]])
    return refs


def _median_point(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    return (
        round(statistics.median(p[0] for p in points), 1),
        round(statistics.median(p[1] for p in points), 1),
    )


def _medoid_geometry(geometries: Sequence[Geometry | None], diagonal: float) -> Geometry | None:
    """The geometry closest to all others (the most central model's polyline)."""
    known = [g for g in geometries if g is not None]
    if not known:
        return None
    return min(known, key=lambda g: sum(geometry_distance(g, o, diagonal) for o in known))


def merge(
    docs: Sequence[tuple[str, SceneAnnotation]],
    *,
    min_support: int = MIN_SUPPORT,
    run_id: str | None = None,
) -> SceneAnnotation:
    """Merge the models' documents of one scene into a review draft.

    Args:
        docs: ``(model name, document)`` pairs in merge order (ties go to earlier models).
        min_support: Models that must agree before an element enters the draft; capped at the
            number of documents, so a single-model pre-fill puts every element in the draft.
        run_id: The pre-fill run, recorded in ``meta.review``.

    Returns:
        The draft: a ``prediction``-kind document whose ``meta.review`` describes per element
        its support and status and lists the suggestions.

    Raises:
        ValueError: If ``docs`` is empty.
    """
    if not docs:
        msg = "nothing to merge"
        raise ValueError(msg)
    min_support = min(min_support, len(docs))  # a single-model pre-fill keeps everything
    merger = _Merger(docs, min_support)
    for family in _FAMILIES:
        merger.cluster(family)
    merger.assign_ids()

    built: dict[
        str,
        tuple[Family, dict[str, Any], dict[str, Any], tuple[str, dict[str, Any], list[str]] | None],
    ] = {}
    for family in _FAMILIES:
        for cluster in merger.clusters[family]:
            element, review = merger.build(cluster)
            merged_state = merger.state(cluster)
            if merged_state is not None and merged_state[2]:
                review["disagree"] += merged_state[2]
                review["status"] = status(review["support"], review["of"], review["disagree"])
            built[cluster.draft_id] = (family, element, review, merged_state)

    chosen: set[str] = set()
    pending = [key for key, (_, _, review, _) in built.items() if review["support"] >= min_support]
    while pending:
        key = pending.pop()
        if key in chosen or key not in built:
            continue
        chosen.add(key)
        pending.extend(references(built[key][1]))

    families: dict[Family, list[dict[str, Any]]] = {family: [] for family in _FAMILIES}
    state: dict[str, dict[str, Any]] = {
        "switches": {},
        "signals": {},
        "tracks": {},
        "derailers": {},
        "routes": {},
    }
    records: dict[str, Any] = {}
    suggestions: dict[str, Any] = {}
    for key in sorted(built):
        family, element, review, merged_state = built[key]
        if key in chosen:
            families[family].append(element)
            if merged_state is not None:
                state[merged_state[0]][key] = merged_state[1]
            records[key] = review
        else:
            suggestions[key] = {
                **review,
                "element": element,
                "state": None if merged_state is None else {merged_state[0]: merged_state[1]},
                "requires": [ref for ref in references(element) if ref not in chosen],
            }
    reference = docs[0][1]
    draft = {
        "schema_version": "v0",
        "scene_id": reference.scene_id,
        "source": reference.source.model_dump(mode="json"),
        "provenance": {
            "kind": "prediction",
            "run_id": run_id,
            "tool": "rail_vision_bench.review.consensus",
        },
        "topology": {"nodes": families["nodes"], "edges": families["edges"]},
        "signals": families["signals"],
        "derailers": families["derailers"],
        "routes": families["routes"],
        "state": state,
        "meta": {
            "review": {
                "prefill_run": run_id,
                "models": [model for model, _ in docs],
                "min_support": min_support,
                "elements": records,
                "suggestions": suggestions,
                "notes": {
                    model: doc.meta.get("notes") for model, doc in docs if doc.meta.get("notes")
                },
            }
        },
    }
    return SceneAnnotation.model_validate(draft)
