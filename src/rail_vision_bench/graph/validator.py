"""Semantic validation of schema v0 documents (rules 1-15 of the schema specification).

Validation is total: :func:`validate_scene` never raises on any
:class:`SceneAnnotation`. Every lookup is guarded, a dangling reference yields
an issue and the check that depends on it is skipped. Issues are sorted, so the
result is deterministic and independent of element order.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from rail_vision_bench.graph.build import endpoint_ports
from rail_vision_bench.graph.rules import (
    DEGREE_BY_KIND,
    PORTS_BY_KIND,
    SLIP_KINDS,
    SWITCHING_KINDS,
    as_path,
    expected_setting,
    is_path_allowed,
    is_path_set_allowed,
)
from rail_vision_bench.schema.issues import (
    STRICT_PROMOTED,
    IssueCode,
    Severity,
    ValidationIssue,
    ValidationReport,
)
from rail_vision_bench.schema.models import (
    Derailer,
    Direction,
    Geometry,
    NodeKind,
    Port,
    Route,
    SceneAnnotation,
    Signal,
    SwitchPosition,
    Track,
)
from rail_vision_bench.schema.validate import schema_issues


def _issue(
    code: IssueCode,
    message: str,
    *,
    element_id: str | None = None,
    path: str | None = None,
    severity: Severity = "error",
) -> ValidationIssue:
    return ValidationIssue(
        code=code, severity=severity, message=message, element_id=element_id, path=path
    )


def _fmt_path(path: frozenset[Port]) -> str:
    return "-".join(sorted(port.value for port in path))


def _fmt_setting(setting: SwitchPosition | tuple[Port, Port] | frozenset[Port] | None) -> str:
    if setting is None:
        return "none"
    if isinstance(setting, SwitchPosition):
        return setting.value
    return _fmt_path(frozenset(setting))


# ----------------------------------------------------------------------------- topology


def _iter_ids(doc: SceneAnnotation) -> Iterator[tuple[str, str]]:
    """Yield (id, location) for every element of the document."""
    families: tuple[tuple[str, list[Any]], ...] = (
        ("topology/nodes", doc.topology.nodes),
        ("topology/edges", doc.topology.edges),
        ("signals", doc.signals),
        ("derailers", doc.derailers),
        ("routes", doc.routes),
    )
    for prefix, elements in families:
        for index, element in enumerate(elements):
            yield element.id, f"{prefix}/{index}"


def _iter_geometries(doc: SceneAnnotation) -> Iterator[tuple[str, str, Geometry]]:
    """Yield (element id, location, geometry) for every element carrying geometry."""
    families: tuple[tuple[str, list[Any]], ...] = (
        ("topology/nodes", doc.topology.nodes),
        ("topology/edges", doc.topology.edges),
        ("signals", doc.signals),
        ("derailers", doc.derailers),
    )
    for prefix, elements in families:
        for index, element in enumerate(elements):
            if element.geometry is not None:
                yield element.id, f"{prefix}/{index}/geometry", element.geometry


def _geometry_out_of_bounds(geometry: Geometry, width: int, height: int) -> str | None:
    """Return the name of the first geometry field with a coordinate outside the image."""
    points: list[tuple[str, float, float]] = []
    if geometry.bbox is not None:
        x0, y0, x1, y1 = geometry.bbox
        points.extend([("bbox", x0, y0), ("bbox", x1, y1)])
    if geometry.polyline is not None:
        points.extend(("polyline", x, y) for x, y in geometry.polyline)
    if geometry.point is not None:
        points.append(("point", *geometry.point))
    for name, x, y in points:
        if not (0.0 <= x <= width and 0.0 <= y <= height):
            return name
    return None


def validate_topology(doc: SceneAnnotation) -> list[ValidationIssue]:
    """Check ids, references, ports, degrees, self loops, half labels and geometry.

    Args:
        doc: The document (already structurally valid).

    Returns:
        Issues found by rules 1-8; unsorted.
    """
    issues: list[ValidationIssue] = []
    nodes = doc.node_map()
    edges = doc.edge_map()

    # Rule 1: every id is unique across all element families; reported once per id.
    id_counts = Counter(element_id for element_id, _ in _iter_ids(doc))
    reported: set[str] = set()
    for element_id, location in _iter_ids(doc):
        if id_counts[element_id] > 1 and element_id not in reported:
            reported.add(element_id)
            issues.append(
                _issue(
                    IssueCode.ID_DUPLICATE,
                    f"id {element_id!r} is used {id_counts[element_id]} times",
                    element_id=element_id,
                    path=location,
                )
            )

    # Rules 2, 3 and 5 on edge endpoints.
    for index, edge in enumerate(doc.topology.edges):
        for end, ref in (("a", edge.a), ("b", edge.b)):
            location = f"topology/edges/{index}/{end}"
            node = nodes.get(ref.node)
            if node is None:
                issues.append(
                    _issue(
                        IssueCode.TOPO_DANGLING_REF,
                        f"edge {edge.id!r} end {end} references unknown node {ref.node!r}",
                        element_id=edge.id,
                        path=location,
                    )
                )
            elif ref.port not in PORTS_BY_KIND[node.kind]:
                issues.append(
                    _issue(
                        IssueCode.TOPO_PORT_UNKNOWN,
                        f"node {ref.node!r} of kind {node.kind.value} has no port"
                        f" {ref.port.value!r}",
                        element_id=edge.id,
                        path=location,
                    )
                )
        if edge.a.node == edge.b.node:
            issues.append(
                _issue(
                    IssueCode.TOPO_SELF_LOOP,
                    f"edge {edge.id!r} connects node {edge.a.node!r} to itself",
                    element_id=edge.id,
                    path=f"topology/edges/{index}",
                )
            )

    # Rule 2 on trackside attachments.
    attached: tuple[tuple[str, Sequence[Signal | Derailer]], ...] = (
        ("signals", doc.signals),
        ("derailers", doc.derailers),
    )
    for prefix, elements in attached:
        issues.extend(
            _issue(
                IssueCode.TOPO_DANGLING_REF,
                f"{element.id!r} is attached to unknown edge {element.at.edge!r}",
                element_id=element.id,
                path=f"{prefix}/{index}/at/edge",
            )
            for index, element in enumerate(elements)
            if element.at.edge not in edges
        )

    # Rules 4 and 6 from the endpoint table (both ends of every edge).
    endpoints = endpoint_ports(doc.topology)
    for node_id, node_endpoints in endpoints.items():
        port_counts = Counter(port for port, _ in node_endpoints)
        for port, count in port_counts.items():
            if count > 1:
                users = [edge_id for p, edge_id in node_endpoints if p == port]
                issues.append(
                    _issue(
                        IssueCode.TOPO_PORT_DUP,
                        f"port {port.value!r} of node {node_id!r} is used by {count} edge"
                        f" endpoints: {', '.join(users)}",
                        element_id=node_id,
                        path=f"topology/nodes/{node_id}/{port.value}",
                    )
                )
    for index, node in enumerate(doc.topology.nodes):
        degree = len(endpoints.get(node.id, []))
        expected = DEGREE_BY_KIND[node.kind]
        if degree != expected:
            issues.append(
                _issue(
                    IssueCode.TOPO_DEGREE,
                    f"node {node.id!r} of kind {node.kind.value} has {degree} edge endpoints,"
                    f" expected {expected}",
                    element_id=node.id,
                    path=f"topology/nodes/{index}",
                )
            )
        # Rule 7: half labels only make sense on the two halves of a slip switch.
        if node.half_labels and node.kind not in SLIP_KINDS:
            issues.append(
                _issue(
                    IssueCode.NODE_HALF_LABELS_UNEXPECTED,
                    f"node {node.id!r} of kind {node.kind.value} carries half_labels",
                    element_id=node.id,
                    path=f"topology/nodes/{index}/half_labels",
                    severity="warning",
                )
            )

    # Rule 8: geometry inside the source image.
    for element_id, location, geometry in _iter_geometries(doc):
        field = _geometry_out_of_bounds(geometry, doc.source.width, doc.source.height)
        if field is not None:
            issues.append(
                _issue(
                    IssueCode.GEOM_OUT_OF_BOUNDS,
                    f"{element_id!r} {field} lies outside the {doc.source.width}x"
                    f"{doc.source.height} image",
                    element_id=element_id,
                    path=f"{location}/{field}",
                    severity="warning",
                )
            )
    return issues


# ----------------------------------------------------------------------------- state


def validate_state(doc: SceneAnnotation) -> list[ValidationIssue]:
    """Check state keys, kind-specific fields, active path sets and missing entries.

    Args:
        doc: The document (already structurally valid).

    Returns:
        Issues found by rules 9, 10, 11 and 14; unsorted.
    """
    issues: list[ValidationIssue] = []
    nodes = doc.node_map()
    families: tuple[tuple[str, Mapping[str, Any], set[str]], ...] = (
        ("switches", doc.state.switches, set(nodes)),
        ("signals", doc.state.signals, set(doc.signal_map())),
        ("tracks", doc.state.tracks, set(doc.edge_map())),
        ("derailers", doc.state.derailers, set(doc.derailer_map())),
        ("routes", doc.state.routes, set(doc.route_map())),
    )

    # Rule 9: state keys reference elements of the matching family.
    for family, entries, known in families:
        issues.extend(
            _issue(
                IssueCode.STATE_UNKNOWN_ELEMENT,
                f"state.{family} entry {key!r} does not match any {family[:-1]}",
                element_id=key,
                path=f"state/{family}/{key}",
            )
            for key in entries
            if key not in known
        )

    # Rules 10 and 11 on switch state.
    for node_id, switch_state in doc.state.switches.items():
        node = nodes.get(node_id)
        if node is None:
            continue  # reported by rule 9
        location = f"state/switches/{node_id}"
        if node.kind not in SWITCHING_KINDS:
            issues.append(
                _issue(
                    IssueCode.STATE_WRONG_FIELD_FOR_KIND,
                    f"node {node_id!r} of kind {node.kind.value} carries no switch state",
                    element_id=node_id,
                    path=location,
                )
            )
            continue
        if node.kind == NodeKind.SWITCH:
            if switch_state.active_paths is not None:
                issues.append(
                    _issue(
                        IssueCode.STATE_WRONG_FIELD_FOR_KIND,
                        f"switch {node_id!r} must use position, not active_paths",
                        element_id=node_id,
                        path=f"{location}/active_paths",
                    )
                )
            continue
        if switch_state.position is not None:
            issues.append(
                _issue(
                    IssueCode.STATE_WRONG_FIELD_FOR_KIND,
                    f"{node.kind.value} {node_id!r} must use active_paths, not position",
                    element_id=node_id,
                    path=f"{location}/position",
                )
            )
        if switch_state.active_paths is not None:
            paths = [as_path(pair) for pair in switch_state.active_paths]
            if not is_path_set_allowed(node.kind, paths):
                rendered = ", ".join(_fmt_path(path) for path in paths)
                issues.append(
                    _issue(
                        IssueCode.STATE_PATH_NOT_ALLOWED,
                        f"{node.kind.value} {node_id!r} cannot connect [{rendered}] at once",
                        element_id=node_id,
                        path=f"{location}/active_paths",
                    )
                )

    # Rule 14: every stateful element has a state entry.
    expected: tuple[tuple[str, list[str], Mapping[str, Any]], ...] = (
        (
            "switches",
            [node.id for node in doc.topology.nodes if node.kind in SWITCHING_KINDS],
            doc.state.switches,
        ),
        ("tracks", [edge.id for edge in doc.topology.edges], doc.state.tracks),
        ("signals", [signal.id for signal in doc.signals], doc.state.signals),
        ("derailers", [derailer.id for derailer in doc.derailers], doc.state.derailers),
        ("routes", [route.id for route in doc.routes], doc.state.routes),
    )
    for family, ids, entries in expected:
        issues.extend(
            _issue(
                IssueCode.STATE_MISSING,
                f"state.{family} has no entry for {element_id!r}",
                element_id=element_id,
                path=f"state/{family}/{element_id}",
                severity="warning",
            )
            for element_id in ids
            if element_id not in entries
        )
    return issues


# ----------------------------------------------------------------------------- routes


def _port_at(edge: Track, node_id: str) -> tuple[Port, str, Port] | None:
    """Return (port at node_id, other node, port at other node) or None if not incident."""
    if edge.a.node == node_id:
        return edge.a.port, edge.b.node, edge.b.port
    if edge.b.node == node_id:
        return edge.b.port, edge.a.node, edge.a.port
    return None


def _route_references(
    doc: SceneAnnotation, index: int, route: Route, all_ids: set[str]
) -> tuple[list[ValidationIssue], bool]:
    """Check rules 2 and 12 for one route; the flag says whether the walk can run."""
    issues: list[ValidationIssue] = []
    signals = doc.signal_map()
    nodes = doc.node_map()
    edges = doc.edge_map()
    location = f"routes/{index}"
    walkable = True

    if route.start not in all_ids:
        walkable = False
        issues.append(
            _issue(
                IssueCode.TOPO_DANGLING_REF,
                f"route {route.id!r} starts at unknown element {route.start!r}",
                element_id=route.id,
                path=f"{location}/start",
            )
        )
    elif route.start not in signals:
        walkable = False
        issues.append(
            _issue(
                IssueCode.ROUTE_START_NOT_SIGNAL,
                f"route {route.id!r} starts at {route.start!r}, which is not a signal",
                element_id=route.id,
                path=f"{location}/start",
            )
        )
    elif signals[route.start].at.edge not in edges:
        walkable = False  # the dangling attachment is reported by validate_topology
    if route.end not in signals and route.end not in nodes:
        issues.append(
            _issue(
                IssueCode.TOPO_DANGLING_REF,
                f"route {route.id!r} ends at {route.end!r}, which is neither a signal nor a node",
                element_id=route.id,
                path=f"{location}/end",
            )
        )
    for position, edge_id in enumerate(route.path):
        if edge_id not in edges:
            walkable = False
            issues.append(
                _issue(
                    IssueCode.TOPO_DANGLING_REF,
                    f"route {route.id!r} path references unknown edge {edge_id!r}",
                    element_id=route.id,
                    path=f"{location}/path/{position}",
                )
            )
    issues.extend(
        _issue(
            IssueCode.TOPO_DANGLING_REF,
            f"route {route.id!r} declares a position for unknown node {key!r}",
            element_id=route.id,
            path=f"{location}/switch_positions/{key}",
        )
        for key in route.switch_positions
        if key not in nodes
    )
    return issues, walkable


def _walk_route(doc: SceneAnnotation, index: int, route: Route) -> list[ValidationIssue]:
    """Rule 13: walk the route edge by edge, checking contiguity, traversal and switches.

    Precondition: the start is a signal on a known edge and every path entry
    is a known edge (see :func:`_route_references`).
    """
    issues: list[ValidationIssue] = []
    signals = doc.signal_map()
    nodes = doc.node_map()
    edges = doc.edge_map()
    location = f"routes/{index}"
    start = signals[route.start]

    if route.path[0] != start.at.edge:
        issues.append(
            _issue(
                IssueCode.ROUTE_DISCONTIGUOUS,
                f"route {route.id!r} must start on edge {start.at.edge!r} (signal"
                f" {route.start!r}), not {route.path[0]!r}",
                element_id=route.id,
                path=f"{location}/path/0",
            )
        )
        return issues

    # Leave the first edge at its far end, as seen from the signal's direction.
    first = edges[route.path[0]]
    far = first.b if start.at.direction == Direction.A_TO_B else first.a
    current, entry_port = far.node, far.port
    walked_switches: set[str] = set()
    complete = True

    for position, edge_id in enumerate(route.path[1:], start=1):
        node = nodes.get(current)
        if node is None:
            complete = False  # dangling endpoint, reported by validate_topology
            break
        incidence = _port_at(edges[edge_id], current)
        if incidence is None:
            complete = False
            issues.append(
                _issue(
                    IssueCode.ROUTE_DISCONTIGUOUS,
                    f"route {route.id!r}: edge {edge_id!r} is not incident to node {current!r}",
                    element_id=route.id,
                    path=f"{location}/path/{position}",
                )
            )
            break
        port_out, next_node, next_port = incidence
        pair = as_path((entry_port, port_out))
        if not is_path_allowed(node.kind, pair):
            issues.append(
                _issue(
                    IssueCode.ROUTE_INVALID_TRAVERSAL,
                    f"route {route.id!r} cannot traverse {node.kind.value} {current!r}"
                    f" from {entry_port.value!r} to {port_out.value!r}",
                    element_id=current,
                    path=f"{location}/path/{position}",
                )
            )
        elif node.kind in SWITCHING_KINDS:
            walked_switches.add(current)
            issues.extend(_check_switch_setting(route, location, node.kind, current, pair))
        current, entry_port = next_node, next_port

    if not complete:
        return issues

    # End check: a signal must sit on the last edge, a node must be where the walk ended.
    if route.end in signals:
        if signals[route.end].at.edge != route.path[-1]:
            issues.append(
                _issue(
                    IssueCode.ROUTE_DISCONTIGUOUS,
                    f"route {route.id!r} ends at signal {route.end!r}, which is not on"
                    f" its last edge {route.path[-1]!r}",
                    element_id=route.id,
                    path=f"{location}/end",
                )
            )
    elif route.end in nodes and route.end != current:
        issues.append(
            _issue(
                IssueCode.ROUTE_DISCONTIGUOUS,
                f"route {route.id!r} ends at node {route.end!r} but its path ends at {current!r}",
                element_id=route.id,
                path=f"{location}/end",
            )
        )
    # Declared settings for nodes the route never switches through.
    issues.extend(
        _issue(
            IssueCode.ROUTE_SWITCH_MISMATCH,
            f"route {route.id!r} declares a setting for {key!r}, which is not a"
            " switching node on its path",
            element_id=key,
            path=f"{location}/switch_positions/{key}",
        )
        for key in route.switch_positions
        if key in nodes and key not in walked_switches
    )
    return issues


def _check_switch_setting(
    route: Route, location: str, kind: NodeKind, node_id: str, pair: frozenset[Port]
) -> list[ValidationIssue]:
    """Compare the declared switch setting at ``node_id`` with what ``pair`` requires."""
    expected = expected_setting(kind, pair)
    declared = route.switch_positions.get(node_id)
    if declared is None:
        return [
            _issue(
                IssueCode.ROUTE_SWITCH_MISSING,
                f"route {route.id!r} declares no setting for {kind.value} {node_id!r}",
                element_id=node_id,
                path=f"{location}/switch_positions/{node_id}",
                severity="warning",
            )
        ]
    if isinstance(declared, SwitchPosition):
        matches = kind == NodeKind.SWITCH and declared == expected
    else:
        matches = kind in SLIP_KINDS and as_path(declared) == expected
    if matches:
        return []
    return [
        _issue(
            IssueCode.ROUTE_SWITCH_MISMATCH,
            f"route {route.id!r} declares {_fmt_setting(declared)} for {kind.value}"
            f" {node_id!r}, expected {_fmt_setting(expected)}",
            element_id=node_id,
            path=f"{location}/switch_positions/{node_id}",
        )
    ]


def validate_routes(doc: SceneAnnotation) -> list[ValidationIssue]:
    """Check route references, start signals and walk every route (rules 2, 12, 13).

    Args:
        doc: The document (already structurally valid).

    Returns:
        Issues found; unsorted.
    """
    issues: list[ValidationIssue] = []
    all_ids = {element_id for element_id, _ in _iter_ids(doc)}
    for index, route in enumerate(doc.routes):
        reference_issues, walkable = _route_references(doc, index, route, all_ids)
        issues.extend(reference_issues)
        if walkable:
            issues.extend(_walk_route(doc, index, route))
    return issues


# ----------------------------------------------------------------------------- entry points


def validate_scene(doc: SceneAnnotation, *, strict: bool = False) -> ValidationReport:
    """Run every semantic rule on a structurally valid document.

    Args:
        doc: The document.
        strict: Promote ``STRICT_PROMOTED`` warnings to errors.

    Returns:
        The report, issues sorted by (path, code, element id, message).
    """
    issues = validate_topology(doc) + validate_state(doc) + validate_routes(doc)
    if strict:
        issues = [
            issue.model_copy(update={"severity": "error"})
            if issue.code in STRICT_PROMOTED
            else issue
            for issue in issues
        ]
    issues.sort(key=lambda i: (i.path or "", i.code, i.element_id or "", i.message))
    return ValidationReport.from_issues(issues)


def validate_document(
    data: SceneAnnotation | Mapping[str, Any],
    *,
    strict: bool = False,
    schema: Mapping[str, Any] | None = None,
) -> ValidationReport:
    """Validate a raw document (schema, then models, then semantics) or a parsed one.

    Args:
        data: A raw JSON document or an already parsed :class:`SceneAnnotation`.
        strict: Promote ``STRICT_PROMOTED`` warnings to errors.
        schema: A JSON Schema to use instead of the in-memory export.

    Returns:
        The report; when the structural passes fail, only ``SCHEMA_INVALID``
        issues are reported and no semantic rule runs.
    """
    if isinstance(data, SceneAnnotation):
        return validate_scene(data, strict=strict)
    structural = schema_issues(data, schema)
    if structural:
        return ValidationReport.from_issues(structural)
    try:
        doc = SceneAnnotation.model_validate(dict(data))
    except ValidationError as exc:
        pydantic_issues = [
            _issue(
                IssueCode.SCHEMA_INVALID,
                error["msg"],
                path="/".join(str(part) for part in error["loc"]),
            )
            for error in exc.errors()
        ]
        pydantic_issues.sort(key=lambda i: (i.path or "", i.message))
        return ValidationReport.from_issues(pydantic_issues)
    return validate_scene(doc, strict=strict)
