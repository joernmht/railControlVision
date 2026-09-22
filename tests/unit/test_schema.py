"""Structural behaviour of the v0 models and the JSON Schema export."""

from __future__ import annotations

import copy
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from rail_vision_bench import SCHEMA_VERSION
from rail_vision_bench.schema.export import (
    SCHEMA_DIALECT,
    SCHEMA_ID,
    check_json_schema,
    dump_json_schema,
    to_json_schema,
    write_json_schema,
)
from rail_vision_bench.schema.issues import (
    STRICT_PROMOTED,
    WARNING_CODES,
    IssueCode,
    ValidationIssue,
    ValidationReport,
)
from rail_vision_bench.schema.models import (
    EdgeAttachment,
    Node,
    NodeKind,
    Port,
    Route,
    SceneAnnotation,
    Signal,
    SwitchPosition,
)
from rail_vision_bench.schema.validate import load_schema, schema_issues
from tests.conftest import REPO_ROOT

SCHEMA_FILE = REPO_ROOT / "schema" / "v0.json"


@pytest.mark.parametrize("doc_fixture", ["minimal_doc", "station_dkw_doc"])
def test_round_trip_is_idempotent(doc_fixture: str, request: pytest.FixtureRequest):
    doc = request.getfixturevalue(doc_fixture)
    model = SceneAnnotation.model_validate(doc)
    dumped = model.model_dump(mode="json")
    assert SceneAnnotation.model_validate(dumped) == model
    assert SceneAnnotation.model_validate_json(model.model_dump_json()) == model


def test_extra_keys_are_rejected(minimal_doc: dict[str, Any]):
    with pytest.raises(ValidationError, match="extra"):
        SceneAnnotation.model_validate({**minimal_doc, "extra": 1})
    nested = copy.deepcopy(minimal_doc)
    nested["topology"]["nodes"][0]["colour"] = "red"
    with pytest.raises(ValidationError, match="colour"):
        SceneAnnotation.model_validate(nested)


def test_enum_pattern_and_range_constraints():
    with pytest.raises(ValidationError):
        Node(id="n1", kind="teleporter")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="pattern"):
        Node(id="-leading-dash", kind=NodeKind.JOINT)
    with pytest.raises(ValidationError, match="pattern"):
        Node(id="x" * 65, kind=NodeKind.JOINT)
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        Node(id="n1", kind=NodeKind.JOINT, confidence=1.5)
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        EdgeAttachment(edge="e1", offset=2.0, direction="a_to_b")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="schema_version"):
        SceneAnnotation.model_validate({"scene_id": "s"})
    with pytest.raises(ValidationError):
        Route(id="r", start="s", end="e", path=[])


def test_route_switch_positions_accept_positions_and_port_paths():
    route = Route(
        id="r",
        start="s",
        end="e",
        path=["e1"],
        switch_positions={"sw": "straight", "dkw": ["A", "D"]},  # type: ignore[dict-item]
    )
    assert route.switch_positions == {"sw": SwitchPosition.STRAIGHT, "dkw": (Port.A, Port.D)}
    with pytest.raises(ValidationError):
        Route(id="r", start="s", end="e", path=["e1"], switch_positions={"sw": ["A"]})  # type: ignore[dict-item]


def test_map_helpers_last_wins(minimal_scene: SceneAnnotation):
    assert set(minimal_scene.node_map()) == {"n_w", "n_j", "n_e"}
    assert set(minimal_scene.edge_map()) == {"e1", "e2"}
    assert set(minimal_scene.signal_map()) == {"sig_1"}
    assert minimal_scene.derailer_map() == {}
    assert minimal_scene.route_map() == {}
    duplicate = minimal_scene.model_copy(deep=True)
    duplicate.signals.append(
        Signal(
            id="sig_1",
            kind="shunt",  # type: ignore[arg-type]
            at=EdgeAttachment(edge="e2", direction="b_to_a"),  # type: ignore[arg-type]
        )
    )
    assert duplicate.signal_map()["sig_1"].at.edge == "e2"


def test_schema_version_constant_matches_literal():
    annotation = SceneAnnotation.model_fields["schema_version"].annotation
    assert get_args(annotation) == (SCHEMA_VERSION,)
    assert to_json_schema()["properties"]["schema_version"]["const"] == SCHEMA_VERSION


def test_json_schema_shape():
    schema = to_json_schema()
    assert schema["$schema"] == SCHEMA_DIALECT
    assert schema["$id"] == SCHEMA_ID
    assert schema["title"] == "SceneAnnotation"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "schema_version",
        "scene_id",
        "source",
        "provenance",
        "topology",
        "state",
    }
    assert schema["properties"]["meta"] == {
        "additionalProperties": True,
        "description": "Free-form notes (station name, difficulty tags, ...).",
        "title": "Meta",
        "type": "object",
    }
    defs = schema["$defs"]
    expected_models = {
        "Geometry",
        "PortRef",
        "Node",
        "Track",
        "EdgeAttachment",
        "Signal",
        "Derailer",
        "Route",
        "Topology",
        "SwitchState",
        "SignalState",
        "TrackState",
        "DerailerState",
        "RouteState",
        "State",
        "Source",
        "Provenance",
    }
    expected_enums = {
        "NodeKind",
        "Port",
        "TrackKind",
        "SignalKind",
        "SignalAspect",
        "SwitchPosition",
        "Occupancy",
        "DerailerPosition",
        "RouteStatus",
        "Direction",
        "SourceKind",
        "ProvenanceKind",
    }
    assert set(defs) == expected_models | expected_enums
    for name in expected_models:
        assert defs[name]["type"] == "object", name
        assert defs[name]["additionalProperties"] is False, name
    for name in expected_enums:
        assert defs[name]["type"] == "string", name
        assert isinstance(defs[name]["enum"], list), name
        assert defs[name]["enum"], name
    # Id-keyed dicts constrain their keys with the id pattern.
    pattern = next(iter(defs["State"]["properties"]["switches"]["patternProperties"]))
    assert pattern == r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$"


def test_dump_is_stable_and_newline_terminated():
    first = dump_json_schema()
    assert first == dump_json_schema()
    assert first.endswith("}\n")
    assert not first.endswith("\n\n")


def test_committed_schema_is_up_to_date():
    assert check_json_schema(SCHEMA_FILE), "run `bench schema export` to refresh schema/v0.json"
    assert load_schema(SCHEMA_FILE) == to_json_schema()


def test_write_and_check_round_trip(tmp_path):
    target = tmp_path / "nested" / "v0.json"
    write_json_schema(target)
    assert check_json_schema(target)
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert not check_json_schema(target)
    not_an_object = tmp_path / "list.json"
    not_an_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_schema(not_an_object)


@pytest.mark.parametrize("doc_fixture", ["minimal_doc", "station_dkw_doc"])
def test_examples_pass_jsonschema(doc_fixture: str, request: pytest.FixtureRequest):
    doc = request.getfixturevalue(doc_fixture)
    assert schema_issues(doc) == []
    assert schema_issues(doc, load_schema(SCHEMA_FILE)) == []


def test_jsonschema_rejections(minimal_doc: dict[str, Any], station_dkw_doc: dict[str, Any]):
    issues = schema_issues({**minimal_doc, "extra": 1})
    assert issues
    assert all(i.code is IssueCode.SCHEMA_INVALID and i.severity == "error" for i in issues)
    assert "extra" in issues[0].message

    bad_port = copy.deepcopy(station_dkw_doc)
    bad_port["routes"][0]["switch_positions"]["dkw1"] = ["A"]
    paths = [i.path or "" for i in schema_issues(bad_port)]
    assert paths
    assert all(p.startswith("routes/0/switch_positions/dkw1") for p in paths)

    bad_version = {**minimal_doc, "schema_version": "v1"}
    assert [i.path for i in schema_issues(bad_version)] == ["schema_version"]

    missing = copy.deepcopy(minimal_doc)
    del missing["topology"]
    assert [i.path for i in schema_issues(missing)] == [""]


def test_issue_codes_are_closed_and_named():
    assert len(IssueCode) == 18
    assert all(code.value == code.name for code in IssueCode)
    assert STRICT_PROMOTED < WARNING_CODES
    assert IssueCode.NODE_HALF_LABELS_UNEXPECTED in WARNING_CODES
    assert IssueCode.ID_DUPLICATE not in WARNING_CODES


def test_validation_report_properties():
    warning = ValidationIssue(code=IssueCode.STATE_MISSING, severity="warning", message="w")
    error = ValidationIssue(code=IssueCode.TOPO_DEGREE, severity="error", message="e")
    assert ValidationReport.from_issues([]) == ValidationReport(ok=True)
    assert ValidationReport.from_issues([warning]).ok
    report = ValidationReport.from_issues([warning, error])
    assert not report.ok
    assert report.errors == [error]
    assert report.warnings == [warning]
    with pytest.raises(ValidationError):
        ValidationIssue(code="MADE_UP", severity="error", message="x")  # type: ignore[arg-type]
