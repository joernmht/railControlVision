"""Rules tables, path-set semantics, the route walker and the validator entry points."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from rail_vision_bench.graph.build import endpoint_ports, to_networkx
from rail_vision_bench.graph.generate import DEFAULT_KINDS, generate_scene, generate_topology
from rail_vision_bench.graph.rules import (
    ALLOWED_PATHS_BY_KIND,
    DEGREE_BY_KIND,
    PATH_FAMILIES_BY_KIND,
    PORTS_BY_KIND,
    SLIP_KINDS,
    SWITCHING_KINDS,
    TERMINAL_KINDS,
    as_path,
    expected_setting,
    is_path_allowed,
    is_path_set_allowed,
    switch_position_for,
)
from rail_vision_bench.graph.validator import (
    validate_document,
    validate_routes,
    validate_scene,
    validate_state,
    validate_topology,
)
from rail_vision_bench.schema.issues import IssueCode, ValidationIssue
from rail_vision_bench.schema.models import (
    Derailer,
    Direction,
    EdgeAttachment,
    Node,
    NodeKind,
    Port,
    PortRef,
    Provenance,
    ProvenanceKind,
    Route,
    SceneAnnotation,
    Signal,
    SignalKind,
    SignalState,
    Source,
    SourceKind,
    State,
    SwitchPosition,
    SwitchState,
    Topology,
    Track,
    TrackKind,
    TrackState,
)
from rail_vision_bench.schema.validate import load_schema
from tests.conftest import FIXTURES_DIR, REPO_ROOT, load_json

AC, BD, AD, BC = (
    as_path(p) for p in ((Port.A, Port.C), (Port.B, Port.D), (Port.A, Port.D), (Port.B, Port.C))
)
AB, CD = as_path((Port.A, Port.B)), as_path((Port.C, Port.D))


def codes(issues: list[ValidationIssue]) -> list[str]:
    return sorted(issue.code.value for issue in issues)


# ----------------------------------------------------------------------------- rules tables


def test_ports_and_degrees():
    assert [DEGREE_BY_KIND[k] for k in NodeKind] == [1, 1, 2, 3, 4, 4, 4]
    assert all(DEGREE_BY_KIND[k] == len(PORTS_BY_KIND[k]) for k in NodeKind)
    assert PORTS_BY_KIND[NodeKind.SWITCH] == {Port.TOE, Port.STRAIGHT, Port.DIVERGING}
    assert {NodeKind.SWITCH, NodeKind.EKW, NodeKind.DKW} == SWITCHING_KINDS
    assert SLIP_KINDS < SWITCHING_KINDS
    assert {NodeKind.BUFFER_STOP, NodeKind.BOUNDARY} == TERMINAL_KINDS


def test_allowed_paths_and_families():
    assert [len(ALLOWED_PATHS_BY_KIND[k]) for k in NodeKind] == [0, 0, 1, 2, 2, 3, 4]
    for kind, paths in ALLOWED_PATHS_BY_KIND.items():
        assert all(len(p) == 2 and p <= PORTS_BY_KIND[kind] for p in paths), kind
    assert PATH_FAMILIES_BY_KIND[NodeKind.CROSSING] == (frozenset({AC, BD}),)
    assert PATH_FAMILIES_BY_KIND[NodeKind.EKW] == (frozenset({AC, BD}), frozenset({AD}))
    assert PATH_FAMILIES_BY_KIND[NodeKind.DKW] == (frozenset({AC, BD}), frozenset({AD, BC}))
    for kind, families in PATH_FAMILIES_BY_KIND.items():
        assert frozenset().union(*families) == ALLOWED_PATHS_BY_KIND[kind], kind
    assert is_path_allowed(NodeKind.JOINT, AB)
    assert not is_path_allowed(NodeKind.CROSSING, AD)


def test_switch_position_for_and_expected_setting():
    straight, diverging = as_path((Port.TOE, Port.STRAIGHT)), as_path((Port.TOE, Port.DIVERGING))
    assert switch_position_for(straight) is SwitchPosition.STRAIGHT
    assert switch_position_for(diverging) is SwitchPosition.DIVERGING
    assert switch_position_for(as_path((Port.STRAIGHT, Port.DIVERGING))) is None
    assert switch_position_for(AC) is None
    assert expected_setting(NodeKind.SWITCH, diverging) is SwitchPosition.DIVERGING
    assert expected_setting(NodeKind.DKW, BC) == BC
    assert expected_setting(NodeKind.EKW, AD) == AD
    assert expected_setting(NodeKind.JOINT, AB) is None
    assert expected_setting(NodeKind.CROSSING, AC) is None


PATH_SET_TABLE = [
    (NodeKind.DKW, [], True),
    (NodeKind.DKW, [AC], True),
    (NodeKind.DKW, [BD], True),
    (NodeKind.DKW, [AC, BD], True),
    (NodeKind.DKW, [AD], True),
    (NodeKind.DKW, [BC], True),
    (NodeKind.DKW, [AD, BC], True),
    (NodeKind.DKW, [AB], False),
    (NodeKind.DKW, [CD], False),
    (NodeKind.DKW, [AC, AD], False),
    (NodeKind.DKW, [AC, BC], False),
    (NodeKind.DKW, [AD, AC], False),
    (NodeKind.EKW, [], True),
    (NodeKind.EKW, [AC], True),
    (NodeKind.EKW, [BD], True),
    (NodeKind.EKW, [AC, BD], True),
    (NodeKind.EKW, [AD], True),
    (NodeKind.EKW, [BC], False),
    (NodeKind.EKW, [AD, AC], False),
    (NodeKind.CROSSING, [], True),
    (NodeKind.CROSSING, [AC, BD], True),
    (NodeKind.CROSSING, [AD], False),
    (NodeKind.SWITCH, [], True),
    (NodeKind.SWITCH, [as_path((Port.TOE, Port.STRAIGHT))], False),
    (NodeKind.JOINT, [AB], True),
    (NodeKind.JOINT, [AB, AB], False),
]


@pytest.mark.parametrize(("kind", "paths", "allowed"), PATH_SET_TABLE)
def test_is_path_set_allowed(kind: NodeKind, paths: list[frozenset[Port]], allowed: bool):
    assert is_path_set_allowed(kind, paths) is allowed


# ----------------------------------------------------------------------------- state path sets


def _station_with(doc: dict[str, Any], kind: str, active_paths: list[list[str]] | None):
    """station_dkw with dkw1 re-typed and its active paths replaced (None removes the entry)."""
    mutated = copy.deepcopy(doc)
    node = next(n for n in mutated["topology"]["nodes"] if n["id"] == "dkw1")
    node["kind"] = kind
    if active_paths is None:
        del mutated["state"]["switches"]["dkw1"]
    else:
        mutated["state"]["switches"]["dkw1"]["active_paths"] = active_paths
    if kind == "crossing":
        node["half_labels"] = []
        for route in mutated["routes"]:
            route["switch_positions"].pop("dkw1", None)
    return SceneAnnotation.model_validate(mutated)


STATE_TABLE = [
    ("dkw", [], True),
    ("dkw", [["A", "C"]], True),
    ("dkw", [["B", "D"]], True),
    ("dkw", [["A", "C"], ["B", "D"]], True),
    ("dkw", [["C", "A"], ["D", "B"]], True),
    ("dkw", [["A", "D"]], True),
    ("dkw", [["B", "C"]], True),
    ("dkw", [["A", "D"], ["B", "C"]], True),
    ("dkw", [["A", "B"]], False),
    ("dkw", [["C", "D"]], False),
    ("dkw", [["A", "A"]], False),
    ("dkw", [["A", "C"], ["A", "D"]], False),
    ("dkw", [["A", "C"], ["B", "C"]], False),
    ("dkw", [["A", "D"], ["A", "C"]], False),
    ("ekw", [], True),
    ("ekw", [["A", "C"]], True),
    ("ekw", [["B", "D"]], True),
    ("ekw", [["A", "C"], ["B", "D"]], True),
    ("ekw", [["A", "D"]], True),
    ("ekw", [["B", "C"]], False),
    ("ekw", [["A", "D"], ["A", "C"]], False),
]


@pytest.mark.parametrize(("kind", "active_paths", "accepted"), STATE_TABLE)
def test_state_path_sets(
    station_dkw_doc: dict[str, Any], kind: str, active_paths: list[list[str]], accepted: bool
):
    report = validate_scene(_station_with(station_dkw_doc, kind, active_paths), strict=True)
    if accepted:
        assert report.ok, report.issues
    else:
        assert codes(report.errors) == ["STATE_PATH_NOT_ALLOWED"]
        assert report.errors[0].element_id == "dkw1"
        assert report.errors[0].path == "state/switches/dkw1/active_paths"


def test_crossing_has_fixed_paths_and_no_state():
    station = load_json(REPO_ROOT / "schema" / "examples" / "v0" / "station_dkw.json")
    without_state = _station_with(station, "crossing", None)
    assert validate_scene(without_state, strict=True).ok
    with_state = _station_with(station, "crossing", [["A", "C"], ["B", "D"]])
    assert codes(validate_scene(with_state).errors) == ["STATE_WRONG_FIELD_FOR_KIND"]


def test_switch_uses_position_not_paths(station_dkw_scene: SceneAnnotation):
    scene = station_dkw_scene.model_copy(deep=True)
    scene.state.switches["sw1"] = SwitchState(active_paths=[(Port.TOE, Port.STRAIGHT)])
    scene.state.switches["dkw1"] = SwitchState(position=SwitchPosition.STRAIGHT)
    issues = validate_state(scene)
    assert codes(issues) == ["STATE_WRONG_FIELD_FOR_KIND", "STATE_WRONG_FIELD_FOR_KIND"]
    assert {issue.element_id for issue in issues} == {"sw1", "dkw1"}


# ----------------------------------------------------------------------------- route walker


def test_station_routes_walk_cleanly(station_dkw_scene: SceneAnnotation):
    assert validate_routes(station_dkw_scene) == []
    assert validate_topology(station_dkw_scene) == []
    assert validate_state(station_dkw_scene) == []


def test_swapped_switch_positions_mismatch(station_dkw_scene: SceneAnnotation):
    scene = station_dkw_scene.model_copy(deep=True)
    route = scene.route_map()["rt_A_1"]
    route.switch_positions["sw1"], route.switch_positions["dkw1"] = (
        route.switch_positions["dkw1"],
        route.switch_positions["sw1"],
    )
    issues = validate_routes(scene)
    assert codes(issues) == ["ROUTE_SWITCH_MISMATCH", "ROUTE_SWITCH_MISMATCH"]
    assert {issue.element_id for issue in issues} == {"sw1", "dkw1"}


def test_switch_position_missing_is_promoted_in_strict(station_dkw_scene: SceneAnnotation):
    scene = station_dkw_scene.model_copy(deep=True)
    del scene.route_map()["rt_F_main"].switch_positions["sw2"]
    lenient = validate_scene(scene)
    assert lenient.ok
    assert codes(lenient.warnings) == ["ROUTE_SWITCH_MISSING"]
    strict = validate_scene(scene, strict=True)
    assert not strict.ok
    assert codes(strict.errors) == ["ROUTE_SWITCH_MISSING"]


def test_extra_switch_position_key_is_a_mismatch(station_dkw_scene: SceneAnnotation):
    scene = station_dkw_scene.model_copy(deep=True)
    scene.route_map()["rt_A_1"].switch_positions["sw2"] = SwitchPosition.STRAIGHT
    issues = validate_routes(scene)
    assert codes(issues) == ["ROUTE_SWITCH_MISMATCH"]
    assert issues[0].element_id == "sw2"


def test_route_end_checks(station_dkw_scene: SceneAnnotation):
    scene = station_dkw_scene.model_copy(deep=True)
    scene.route_map()["rt_F_main"].end = "n_e"  # the walk ends at n_w
    assert codes(validate_routes(scene)) == ["ROUTE_DISCONTIGUOUS"]
    scene = station_dkw_scene.model_copy(deep=True)
    scene.route_map()["rt_A_1"].end = "sig_F"  # sits on e7, not on the last edge e6
    assert codes(validate_routes(scene)) == ["ROUTE_DISCONTIGUOUS"]
    scene = station_dkw_scene.model_copy(deep=True)
    scene.route_map()["rt_A_1"].end = "der_1"  # neither a signal nor a node
    assert codes(validate_routes(scene)) == ["TOPO_DANGLING_REF"]


def _parallel_scene() -> SceneAnnotation:
    """Two switches joined by two parallel edges (straight-straight and diverging-diverging)."""
    topology = Topology(
        nodes=[
            Node(id="n_w", kind=NodeKind.BOUNDARY),
            Node(id="sw1", kind=NodeKind.SWITCH),
            Node(id="sw2", kind=NodeKind.SWITCH),
            Node(id="n_e", kind=NodeKind.BOUNDARY),
        ],
        edges=[
            Track(
                id="e_in", a=PortRef(node="n_w", port=Port.A), b=PortRef(node="sw1", port=Port.TOE)
            ),
            Track(
                id="p_straight",
                a=PortRef(node="sw1", port=Port.STRAIGHT),
                b=PortRef(node="sw2", port=Port.STRAIGHT),
            ),
            Track(
                id="p_diverging",
                a=PortRef(node="sw2", port=Port.DIVERGING),
                b=PortRef(node="sw1", port=Port.DIVERGING),
            ),
            Track(
                id="e_out", a=PortRef(node="sw2", port=Port.TOE), b=PortRef(node="n_e", port=Port.A)
            ),
        ],
    )
    return SceneAnnotation(
        schema_version="v0",
        scene_id="parallel",
        source=Source(kind=SourceKind.SYNTHETIC, image="p.png", width=10, height=10),
        provenance=Provenance(kind=ProvenanceKind.GROUND_TRUTH),
        topology=topology,
        signals=[
            Signal(
                id="sig_in",
                kind=SignalKind.MAIN,
                at=EdgeAttachment(edge="e_in", direction=Direction.A_TO_B),
            )
        ],
        routes=[
            Route(
                id="via_straight",
                start="sig_in",
                end="n_e",
                path=["e_in", "p_straight", "e_out"],
                switch_positions={"sw1": SwitchPosition.STRAIGHT, "sw2": SwitchPosition.STRAIGHT},
            ),
            Route(
                id="via_diverging",
                start="sig_in",
                end="n_e",
                path=["e_in", "p_diverging", "e_out"],
                switch_positions={
                    "sw1": SwitchPosition.DIVERGING,
                    "sw2": SwitchPosition.DIVERGING,
                },
            ),
        ],
        state=State(
            switches={
                "sw1": SwitchState(position=SwitchPosition.STRAIGHT),
                "sw2": SwitchState(position=SwitchPosition.STRAIGHT),
            },
            signals={"sig_in": SignalState()},
            tracks={e.id: TrackState() for e in topology.edges},
            routes={"via_straight": {}, "via_diverging": {}},  # type: ignore[dict-item]
        ),
    )


def test_route_walk_distinguishes_parallel_edges():
    scene = _parallel_scene()
    assert validate_scene(scene, strict=True).ok
    graph = to_networkx(scene.topology)
    assert graph.number_of_edges("sw1", "sw2") == 2
    wrong = scene.model_copy(deep=True)
    wrong.route_map()["via_diverging"].switch_positions["sw2"] = SwitchPosition.STRAIGHT
    issues = validate_routes(wrong)
    assert codes(issues) == ["ROUTE_SWITCH_MISMATCH"]
    assert issues[0].element_id == "sw2"
    assert issues[0].path == "routes/1/switch_positions/sw2"


def test_invalid_traversal_names_the_node(station_dkw_scene: SceneAnnotation):
    scene = station_dkw_scene.model_copy(deep=True)
    route = scene.route_map()["rt_A_1"]
    route.path = ["e1", "e2", "e5", "e6"]  # enters dkw1 at A, leaves at D: a slip path
    route.switch_positions = {"sw1": SwitchPosition.STRAIGHT, "dkw1": (Port.A, Port.D)}
    assert validate_routes(scene) == []
    route.path = ["e1", "e2", "e3", "e5", "e6"]  # A -> B is not a path through a dkw
    issues = [i for i in validate_routes(scene) if i.code is IssueCode.ROUTE_INVALID_TRAVERSAL]
    assert [i.element_id for i in issues] == ["dkw1"]


# ----------------------------------------------------------------------------- entry points


def test_strict_promotion_only_touches_promoted_codes():
    doc = load_json(FIXTURES_DIR / "strict" / "state_missing.json")
    doc["topology"]["nodes"][1]["half_labels"] = ["1a", "1b"]  # sw1 is a plain switch
    lenient = validate_document(doc)
    assert lenient.ok
    assert codes(lenient.warnings) == ["NODE_HALF_LABELS_UNEXPECTED", "STATE_MISSING"]
    strict = validate_document(doc, strict=True)
    assert codes(strict.errors) == ["STATE_MISSING"]
    assert codes(strict.warnings) == ["NODE_HALF_LABELS_UNEXPECTED"]


def test_validation_is_deterministic_and_order_independent():
    doc = load_json(FIXTURES_DIR / "invalid" / "self_loop.json")
    first = validate_document(doc)
    assert first == validate_document(doc)
    assert first.issues == sorted(
        first.issues, key=lambda i: (i.path or "", i.code, i.element_id or "", i.message)
    )
    reordered = copy.deepcopy(doc)
    reordered["topology"]["nodes"].reverse()
    reordered["topology"]["edges"].reverse()
    key = sorted((i.code, i.element_id) for i in first.issues)
    assert sorted((i.code, i.element_id) for i in validate_document(reordered).issues) == key


def test_validate_document_inputs(minimal_doc: dict[str, Any], minimal_scene: SceneAnnotation):
    assert validate_document(minimal_scene, strict=True).ok
    assert validate_document(minimal_doc, schema=load_schema(REPO_ROOT / "schema" / "v0.json")).ok
    # JSON Schema failure: nothing else runs.
    report = validate_document({**minimal_doc, "extra": 1})
    assert not report.ok
    assert codes(report.issues) == ["SCHEMA_INVALID"]
    # Pydantic-only failure: an id-keyed dict with a key that misses the pattern
    # passes patternProperties (unmatched keys are ignored) but not the model.
    doc = copy.deepcopy(minimal_doc)
    doc["state"]["tracks"]["bad key!"] = {"occupancy": "free"}
    report = validate_document(doc)
    assert not report.ok
    assert codes(report.issues) == ["SCHEMA_INVALID"]
    assert (report.issues[0].path or "").startswith("state/tracks/bad key!")


def test_validate_scene_never_raises_when_everything_dangles():
    scene = SceneAnnotation(
        schema_version="v0",
        scene_id="dangling",
        source=Source(kind=SourceKind.SYNTHETIC, image="x.png", width=1, height=1),
        provenance=Provenance(kind=ProvenanceKind.PREDICTION),
        topology=Topology(
            nodes=[Node(id="lonely", kind=NodeKind.DKW, geometry={"point": [5, 5]})],  # type: ignore[arg-type]
            edges=[
                Track(
                    id="e",
                    a=PortRef(node="ghost_a", port=Port.A),
                    b=PortRef(node="ghost_b", port=Port.A),
                )
            ],
        ),
        signals=[
            Signal(
                id="s",
                kind=SignalKind.MAIN,
                at=EdgeAttachment(edge="ghost_e", direction=Direction.A_TO_B),
            )
        ],
        derailers=[Derailer(id="d", at=EdgeAttachment(edge="ghost_e", direction=Direction.B_TO_A))],
        routes=[
            Route(
                id="r1",
                start="ghost_s",
                end="ghost_n",
                path=["ghost_e"],
                switch_positions={"ghost_sw": SwitchPosition.STRAIGHT},
            ),
            Route(id="r2", start="s", end="lonely", path=["e"]),
            Route(id="r3", start="d", end="s", path=["e"]),
        ],
        state=State(
            switches={
                "ghost_sw": SwitchState(),
                "lonely": SwitchState(active_paths=[(Port.A, Port.A)]),
            },
            signals={"ghost_sig": SignalState()},
            tracks={"ghost_e": TrackState()},
            derailers={"ghost_d": {}},  # type: ignore[dict-item]
            routes={"ghost_r": {}},  # type: ignore[dict-item]
        ),
    )
    report = validate_scene(scene, strict=True)
    assert not report.ok
    found = set(codes(report.issues))
    assert {
        "TOPO_DANGLING_REF",
        "TOPO_DEGREE",
        "STATE_UNKNOWN_ELEMENT",
        "STATE_PATH_NOT_ALLOWED",
        "STATE_MISSING",
        "ROUTE_START_NOT_SIGNAL",
        "GEOM_OUT_OF_BOUNDS",
    } <= found
    assert "SCHEMA_INVALID" not in found


# ----------------------------------------------------------------------------- build + generate


def test_to_networkx_and_endpoint_ports(station_dkw_scene: SceneAnnotation):
    graph = to_networkx(station_dkw_scene.topology)
    assert set(graph.nodes) == set(station_dkw_scene.node_map())
    assert graph.has_edge("n_w", "sw1", key="e1")
    assert graph.nodes["dkw1"]["kind"] is NodeKind.DKW
    assert graph.nodes["sw1"]["label"] == "1"
    assert graph.edges["sw1", "dkw1", "e3"]["port_a"] is Port.DIVERGING
    assert graph.edges["sw1", "dkw1", "e3"]["kind"] is TrackKind.SIDING
    assert graph.number_of_edges() == len(station_dkw_scene.topology.edges)
    assert dict(graph.degree) == {
        n.id: DEGREE_BY_KIND[n.kind] for n in station_dkw_scene.topology.nodes
    }
    ports = endpoint_ports(station_dkw_scene.topology)
    assert ports["dkw1"] == [(Port.A, "e2"), (Port.B, "e3"), (Port.C, "e4"), (Port.D, "e5")]
    dangling = Topology(
        nodes=[Node(id="a", kind=NodeKind.BOUNDARY)],
        edges=[Track(id="e", a=PortRef(node="a", port=Port.A), b=PortRef(node="zz", port=Port.A))],
    )
    assert to_networkx(dangling).number_of_edges() == 0
    assert endpoint_ports(dangling) == {"a": [(Port.A, "e")], "zz": [(Port.A, "e")]}


def test_generate_topology_shape_and_determinism():
    empty = generate_topology(0, n_inner=0)
    assert [n.id for n in empty.nodes] == ["n_w", "n_e"]
    assert [(e.id, e.a.node, e.b.node) for e in empty.edges] == [("m0", "n_w", "n_e")]
    assert generate_topology(7, n_inner=5) == generate_topology(7, n_inner=5)
    assert generate_topology(7, n_inner=5) != generate_topology(8, n_inner=5)
    only_switches = generate_topology(1, n_inner=3, kinds=[NodeKind.SWITCH])
    assert [n.kind for n in only_switches.nodes if n.id.startswith("x")] == [NodeKind.SWITCH] * 3
    assert [n.label for n in only_switches.nodes if n.id.startswith("x")] == ["W0", "W1", "W2"]
    slips = generate_topology(1, n_inner=2, kinds=[NodeKind.DKW])
    assert next(n for n in slips.nodes if n.id == "x1").half_labels == ["W1a", "W1b"]
    with pytest.raises(ValueError, match="non-terminal"):
        generate_topology(0, kinds=[NodeKind.BUFFER_STOP])
    with pytest.raises(ValueError, match="non-empty"):
        generate_topology(0, kinds=[])
    with pytest.raises(ValueError, match="n_inner"):
        generate_topology(0, n_inner=-1)
    assert DEFAULT_KINDS == (
        NodeKind.JOINT,
        NodeKind.SWITCH,
        NodeKind.DKW,
        NodeKind.EKW,
        NodeKind.CROSSING,
    )


@pytest.mark.parametrize("kind", [k for k in NodeKind if k not in TERMINAL_KINDS])
def test_generate_scene_is_strict_valid_for_each_kind(kind: NodeKind):
    scene = generate_scene(3, n_inner=2, kinds=[kind])
    assert scene.scene_id == "synthetic-3-2"
    assert scene.provenance.tool == "rail_vision_bench.graph.generate"
    assert [s.id for s in scene.signals] == ["sig_w", "sig_e"]
    assert scene.routes[0].end == "n_e"
    assert validate_document(scene, strict=True).ok
    assert validate_document(scene.model_dump(mode="json"), strict=True).ok
