"""Property tests: generated scenes are valid, targeted mutations produce targeted codes."""

from __future__ import annotations

import random

from hypothesis import assume, given, strategies as st

from rail_vision_bench.graph.build import to_networkx
from rail_vision_bench.graph.rules import DEGREE_BY_KIND
from rail_vision_bench.graph.validator import validate_document, validate_scene
from rail_vision_bench.schema.issues import IssueCode
from rail_vision_bench.schema.models import SceneAnnotation, Topology
from tests.strategies import (
    scenes,
    topologies,
    with_bad_active_path,
    with_dropped_edge,
    with_extra_edge,
)


@given(scenes())
def test_generated_scenes_are_strict_valid(scene: SceneAnnotation):
    report = validate_document(scene, strict=True)
    assert report.ok, report.issues
    assert report.issues == []
    assert validate_document(scene.model_dump(mode="json"), strict=True).ok


@given(scenes(max_inner=6), st.randoms(use_true_random=False))
def test_extra_edge_breaks_degree_of_target(scene: SceneAnnotation, rng: random.Random):
    mutated, node_id = with_extra_edge(scene, rng)
    report = validate_scene(mutated)
    assert not report.ok
    degree_errors = [i for i in report.errors if i.code is IssueCode.TOPO_DEGREE]
    assert [i.element_id for i in degree_errors] == [node_id]


@given(scenes(max_inner=6), st.randoms(use_true_random=False))
def test_dropped_edge_breaks_degree_of_both_ends(scene: SceneAnnotation, rng: random.Random):
    mutated, a, b = with_dropped_edge(scene, rng)
    report = validate_scene(mutated, strict=True)
    assert not report.ok
    assert {i.element_id for i in report.errors if i.code is IssueCode.TOPO_DEGREE} == {a, b}
    assert {i.code for i in report.issues} == {IssueCode.TOPO_DEGREE}


@given(scenes())
def test_bad_active_path_is_reported(scene: SceneAnnotation):
    mutation = with_bad_active_path(scene)
    assume(mutation is not None)
    assert mutation is not None
    mutated, node_id = mutation
    report = validate_scene(mutated, strict=True)
    assert not report.ok
    assert [(i.code, i.element_id) for i in report.errors] == [
        (IssueCode.STATE_PATH_NOT_ALLOWED, node_id)
    ]


@given(scenes())
def test_validation_is_deterministic(scene: SceneAnnotation):
    first = validate_scene(scene, strict=True)
    second = validate_scene(scene, strict=True)
    assert first.issues == second.issues
    assert first == second


@given(topologies())
def test_networkx_graph_matches_topology(topology: Topology):
    graph = to_networkx(topology)
    assert graph.number_of_edges() == len(topology.edges)
    assert graph.number_of_nodes() == len(topology.nodes)
    for node in topology.nodes:
        assert graph.degree(node.id) == DEGREE_BY_KIND[node.kind]
