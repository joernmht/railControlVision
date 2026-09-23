"""Deterministic assembly of a schema v0 document from the loop's intermediate results.

The builder node starts from :func:`assemble_candidate`: it maps the
interpreter's elements and the geometer's edges and attachments onto the
schema, coerces every enum to its vocabulary (``"unknown"`` when the value is
not one), clamps coordinates into the image, drops elements without a valid
id or attachment and repairs nodes whose ports are not exactly connected. The
result is always shaped like schema v0, so the critic's validator reports on
the graph rather than on JSON syntax.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Final, TypeVar

from rail_vision_bench.graph.rules import PORTS_BY_KIND, SLIP_KINDS
from rail_vision_bench.schema.models import (
    DerailerPosition,
    Direction,
    NodeKind,
    Occupancy,
    Port,
    SignalAspect,
    SignalKind,
    SwitchPosition,
    TrackKind,
)

_ID: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_E = TypeVar("_E", bound=StrEnum)


def _as_list(value: object) -> list[Mapping[str, Any]]:
    """Return the mapping entries of ``value`` when it is a list, else nothing."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _as_mapping(value: object) -> Mapping[str, Any]:
    """Return ``value`` when it is a mapping, else an empty one."""
    return value if isinstance(value, Mapping) else {}


def _enum(value: object, enum: type[_E], default: _E) -> _E:
    """Coerce ``value`` into ``enum``, falling back to ``default``."""
    try:
        return enum(str(value).strip().lower()) if value is not None else default
    except ValueError:
        return default


def _port(value: object) -> Port | None:
    """Parse a port name; ``toe``/``straight``/``diverging`` are lower case, ``A``-``D`` upper."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    for candidate in (text, text.upper(), text.lower()):
        try:
            return Port(candidate)
        except ValueError:
            continue
    return None


def _number(value: object) -> float | None:
    """Return ``value`` as a float when it is a real number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _confidence(entry: Mapping[str, Any]) -> float | None:
    """Return the entry's confidence clamped to ``[0, 1]``."""
    value = _number(entry.get("confidence"))
    return None if value is None else min(1.0, max(0.0, value))


def _label(value: object) -> str | None:
    """Return a label as text (numbers painted on a panel may come back as ints)."""
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


class _Frame:
    """Clamps coordinates into ``[0, width] x [0, height]``."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def point(self, value: object) -> list[float] | None:
        """Parse and clamp an ``[x, y]`` point."""
        if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
            return None
        x, y = _number(value[0]), _number(value[1])
        if x is None or y is None:
            return None
        return [min(float(self.width), max(0.0, x)), min(float(self.height), max(0.0, y))]

    def bbox(self, value: object) -> list[float] | None:
        """Parse, order and clamp an ``[x0, y0, x1, y1]`` box."""
        if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 4:
            return None
        first, second = self.point(value[:2]), self.point(value[2:])
        if first is None or second is None:
            return None
        return [
            min(first[0], second[0]),
            min(first[1], second[1]),
            max(first[0], second[0]),
            max(first[1], second[1]),
        ]

    def polyline(self, value: object) -> list[list[float]] | None:
        """Parse and clamp a polyline; None when fewer than two points survive."""
        if not isinstance(value, Sequence) or isinstance(value, str):
            return None
        points = [point for point in (self.point(item) for item in value) if point is not None]
        return points if len(points) >= 2 else None

    def geometry(self, entry: Mapping[str, Any]) -> dict[str, Any] | None:
        """Collect ``bbox``/``point``/``polyline`` from an entry or its ``geometry`` block."""
        nested = _as_mapping(entry.get("geometry"))
        geometry: dict[str, Any] = {}
        for key, parse in (("bbox", self.bbox), ("point", self.point), ("polyline", self.polyline)):
            parsed = parse(entry.get(key, nested.get(key)))
            if parsed is not None:
                geometry[key] = parsed
        return geometry or None


class _Ids:
    """Hands out ids that are valid and unique across every element family."""

    def __init__(self) -> None:
        self.used: set[str] = set()
        self._fresh = 0

    def claim(self, value: object) -> str | None:
        """Claim ``value`` as an id; None when it is invalid or already taken."""
        text = _label(value)
        if text is None or not _ID.match(text) or text in self.used:
            return None
        self.used.add(text)
        return text

    def fresh(self, prefix: str) -> str:
        """Create a new unused id ``<prefix><n>``."""
        while True:
            self._fresh += 1
            candidate = f"{prefix}{self._fresh}"
            if candidate not in self.used:
                self.used.add(candidate)
                return candidate


def _attachment(value: object, edges: set[str]) -> dict[str, Any] | None:
    """Parse an ``{edge, offset, direction}`` attachment onto an existing edge."""
    at = _as_mapping(value)
    edge = _label(at.get("edge"))
    if edge is None or edge not in edges:
        return None
    try:
        direction = Direction(str(at.get("direction")).strip().lower())
    except ValueError:
        return None
    offset = _number(at.get("offset"))
    return {
        "edge": edge,
        "offset": 0.5 if offset is None else min(1.0, max(0.0, offset)),
        "direction": direction.value,
    }


def _infer_kind(ports: Counter[Port]) -> NodeKind | None:
    """Kind of a node the geometer referenced without declaring it, from its used ports."""
    if ports == Counter({Port.A: 1}):
        return NodeKind.BOUNDARY
    if ports == Counter({Port.A: 1, Port.B: 1}):
        return NodeKind.JOINT
    return None


def _switch_state(kind: NodeKind, state: Mapping[str, Any]) -> dict[str, Any]:
    """Build the switch state of a switch (``position``) or a dkw/ekw (``active_paths``)."""
    entry: dict[str, Any] = {"raw": _label(state.get("raw")), "confidence": _confidence(state)}
    if kind in SLIP_KINDS:
        paths = state.get("active_paths")
        if isinstance(paths, list):
            parsed = []
            for pair in paths:
                if isinstance(pair, Sequence) and not isinstance(pair, str) and len(pair) == 2:
                    first, second = _port(pair[0]), _port(pair[1])
                    if first is not None and second is not None:
                        parsed.append([first.value, second.value])
            entry["active_paths"] = parsed
    else:
        entry["position"] = _enum(
            state.get("position"), SwitchPosition, SwitchPosition.UNKNOWN
        ).value
    return {key: value for key, value in entry.items() if value is not None}


def assemble_candidate(
    interpretation: Mapping[str, Any], geometry: Mapping[str, Any], *, width: int, height: int
) -> dict[str, Any]:
    """Assemble the body of a schema v0 document from the interpretation and the geometry.

    Nodes come from ``interpretation.nodes`` plus any ``geometry.nodes`` the
    geometer added (boundaries, joints); a node referenced by an edge but
    declared nowhere is inferred as a boundary (one ``A`` end) or a joint
    (``A`` and ``B``). A node whose used ports differ from its kind's ports is
    dropped and every edge end at it is terminated at a new ``boundary`` node,
    so a single misread node cannot take its neighbours down with it. Signals
    and derailers need a valid attachment; every remaining element gets a
    state entry. What was dropped or repaired is listed in ``meta.notes``.

    Args:
        interpretation: The interpreter's ``{"nodes", "signals", "derailers", "legend"}``.
        geometry: The geometer's ``{"edges", "attachments", "tracks", "nodes"?}``.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        ``topology``, ``signals``, ``derailers``, ``routes`` (empty), ``state``
        and ``meta``; the caller adds the bookkeeping fields.
    """
    frame = _Frame(width, height)
    ids = _Ids()
    notes: list[str] = []

    nodes: dict[str, dict[str, Any]] = {}
    node_states: dict[str, Mapping[str, Any]] = {}
    declared = [*_as_list(interpretation.get("nodes")), *_as_list(geometry.get("nodes"))]
    for entry in declared:
        try:
            kind = NodeKind(str(entry.get("kind")).strip().lower())
        except ValueError:
            notes.append(f"node {entry.get('id')!r}: unknown kind {entry.get('kind')!r}, dropped")
            continue
        node_id = ids.claim(entry.get("id"))
        if node_id is None:
            notes.append(f"node {entry.get('id')!r}: invalid or duplicate id, dropped")
            continue
        node: dict[str, Any] = {"id": node_id, "kind": kind.value}
        if (label := _label(entry.get("label"))) is not None:
            node["label"] = label
        halves = entry.get("half_labels")
        if kind in SLIP_KINDS and isinstance(halves, list):
            node["half_labels"] = [text for text in map(_label, halves) if text is not None]
        geom = frame.geometry(entry) or {}
        if node_geometry := {key: geom[key] for key in ("bbox", "point") if key in geom}:
            node["geometry"] = node_geometry
        if (confidence := _confidence(entry)) is not None:
            node["confidence"] = confidence
        nodes[node_id] = node
        node_states[node_id] = _as_mapping(entry.get("state"))

    edges: list[dict[str, Any]] = []
    for entry in _as_list(geometry.get("edges")):
        ends = []
        for key in ("a", "b"):
            end = _as_mapping(entry.get(key))
            node_ref, port = _label(end.get("node")), _port(end.get("port"))
            if node_ref is None or port is None:
                break
            ends.append({"node": node_ref, "port": port.value})
        if len(ends) != 2:
            notes.append(f"edge {entry.get('id')!r}: an end lacks a node or a valid port, dropped")
            continue
        if ends[0]["node"] == ends[1]["node"]:
            notes.append(f"edge {entry.get('id')!r}: connects a node to itself, dropped")
            continue
        edge_id = ids.claim(entry.get("id"))
        if edge_id is None:
            notes.append(f"edge {entry.get('id')!r}: invalid or duplicate id, dropped")
            continue
        edge: dict[str, Any] = {
            "id": edge_id,
            "kind": _enum(entry.get("kind"), TrackKind, TrackKind.UNKNOWN).value,
            "a": ends[0],
            "b": ends[1],
        }
        if (label := _label(entry.get("label"))) is not None:
            edge["label"] = label
        if (polyline := frame.polyline(entry.get("polyline"))) is not None:
            edge["geometry"] = {"polyline": polyline}
        if (confidence := _confidence(entry)) is not None:
            edge["confidence"] = confidence
        edges.append(edge)

    used: dict[str, Counter[Port]] = {}
    for edge in edges:
        for key in ("a", "b"):
            used.setdefault(edge[key]["node"], Counter())[Port(edge[key]["port"])] += 1
    for node_ref, ports in used.items():
        if node_ref in nodes:
            continue
        inferred = _infer_kind(ports)
        if inferred is not None and ids.claim(node_ref) is not None:
            nodes[node_ref] = {"id": node_ref, "kind": inferred.value}
            node_states[node_ref] = {}
            notes.append(f"node {node_ref!r}: undeclared, inferred as {inferred.value}")

    broken = {
        node_id
        for node_id, node in nodes.items()
        if used.get(node_id, Counter()) != Counter(PORTS_BY_KIND[NodeKind(node["kind"])])
    }
    broken |= {node_ref for node_ref in used if node_ref not in nodes}
    for node_id in sorted(broken):
        if node_id in nodes:
            notes.append(f"node {node_id!r}: ports not exactly connected, dropped")
            del nodes[node_id]
            node_states.pop(node_id, None)
        else:
            notes.append(f"node {node_id!r}: referenced by an edge but unknown, dropped")
    for edge in edges:
        for key, index in (("a", 0), ("b", -1)):
            if edge[key]["node"] in broken:
                boundary = ids.fresh("bnd")
                node = {"id": boundary, "kind": NodeKind.BOUNDARY.value}
                polyline = edge.get("geometry", {}).get("polyline")
                if polyline:
                    node["geometry"] = {"point": polyline[index]}
                nodes[boundary] = node
                node_states[boundary] = {}
                notes.append(
                    f"edge {edge['id']!r}: end {key} re-attached from dropped node "
                    f"{edge[key]['node']!r} to new boundary {boundary!r}"
                )
                edge[key] = {"node": boundary, "port": Port.A.value}

    edge_ids = {edge["id"] for edge in edges}
    attachments = _as_mapping(geometry.get("attachments"))

    def trackside(family: str) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
        elements: list[dict[str, Any]] = []
        states: dict[str, Mapping[str, Any]] = {}
        for entry in _as_list(interpretation.get(family)):
            raw_id = _label(entry.get("id"))
            at = _attachment(attachments.get(raw_id or "", entry.get("at")), edge_ids)
            if at is None:
                notes.append(f"{family[:-1]} {raw_id!r}: no valid attachment, dropped")
                continue
            element_id = ids.claim(raw_id)
            if element_id is None:
                notes.append(f"{family[:-1]} {raw_id!r}: invalid or duplicate id, dropped")
                continue
            element: dict[str, Any] = {"id": element_id, "at": at}
            if family == "signals":
                element["kind"] = _enum(entry.get("kind"), SignalKind, SignalKind.UNKNOWN).value
                if (system := _label(entry.get("system"))) is not None:
                    element["system"] = system
            if (label := _label(entry.get("label"))) is not None:
                element["label"] = label
            if (geom := frame.geometry(entry)) is not None and "bbox" in geom:
                element["geometry"] = {"bbox": geom["bbox"]}
            if (confidence := _confidence(entry)) is not None:
                element["confidence"] = confidence
            elements.append(element)
            states[element_id] = _as_mapping(entry.get("state"))
        return elements, states

    signals, signal_states = trackside("signals")
    derailers, derailer_states = trackside("derailers")

    track_states = _as_mapping(geometry.get("tracks"))
    state: dict[str, Any] = {
        "switches": {
            node_id: _switch_state(NodeKind(node["kind"]), node_states.get(node_id, {}))
            for node_id, node in nodes.items()
            if node["kind"] in {NodeKind.SWITCH.value, NodeKind.EKW.value, NodeKind.DKW.value}
        },
        "signals": {
            signal_id: {
                key: value
                for key, value in {
                    "aspect": _enum(entry.get("aspect"), SignalAspect, SignalAspect.UNKNOWN).value,
                    "raw": _label(entry.get("raw")),
                    "confidence": _confidence(entry),
                }.items()
                if value is not None
            }
            for signal_id, entry in signal_states.items()
        },
        "tracks": {},
        "derailers": {
            derailer_id: {
                "position": _enum(
                    entry.get("position"), DerailerPosition, DerailerPosition.UNKNOWN
                ).value
            }
            for derailer_id, entry in derailer_states.items()
        },
        "routes": {},
    }
    for edge in edges:
        observed = _as_mapping(track_states.get(edge["id"]))
        route_set = observed.get("route_set")
        state["tracks"][edge["id"]] = {
            "occupancy": _enum(observed.get("occupancy"), Occupancy, Occupancy.UNKNOWN).value,
            "route_set": route_set if isinstance(route_set, bool) else None,
        }

    meta: dict[str, Any] = {}
    if notes:
        meta["notes"] = notes
    legend = interpretation.get("legend")
    if isinstance(legend, Mapping) and legend:
        meta["legend"] = dict(legend)
    return {
        "topology": {"nodes": list(nodes.values()), "edges": edges},
        "signals": signals,
        "derailers": derailers,
        "routes": [],
        "state": state,
        "meta": meta,
    }
