"""Every example is strict-valid; every fixture yields exactly the documented issue code."""

from __future__ import annotations

from pathlib import Path

import pytest

from rail_vision_bench.graph.validator import validate_document
from rail_vision_bench.schema.issues import IssueCode
from rail_vision_bench.schema.validate import schema_issues
from tests.conftest import EXAMPLES_DIR, FIXTURES_DIR, load_json

EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("*.json"))

# Expected error code per invalid fixture (from the schema v0 specification).
INVALID_EXPECTED: dict[str, IssueCode] = {
    "duplicate_id": IssueCode.ID_DUPLICATE,
    "dangling_ref": IssueCode.TOPO_DANGLING_REF,
    "port_unknown": IssueCode.TOPO_PORT_UNKNOWN,
    "port_dup": IssueCode.TOPO_PORT_DUP,
    "degree_violation": IssueCode.TOPO_DEGREE,
    "self_loop": IssueCode.TOPO_SELF_LOOP,
    "state_wrong_field": IssueCode.STATE_WRONG_FIELD_FOR_KIND,
    "dkw_bad_path": IssueCode.STATE_PATH_NOT_ALLOWED,
    "dkw_mixed_family": IssueCode.STATE_PATH_NOT_ALLOWED,
    "route_discontiguous": IssueCode.ROUTE_DISCONTIGUOUS,
    "route_invalid_traversal": IssueCode.ROUTE_INVALID_TRAVERSAL,
    "route_switch_mismatch": IssueCode.ROUTE_SWITCH_MISMATCH,
    "route_start_not_signal": IssueCode.ROUTE_START_NOT_SIGNAL,
    "state_unknown_element": IssueCode.STATE_UNKNOWN_ELEMENT,
}
# Expected warning (default mode) / error (strict mode) per strict fixture.
STRICT_EXPECTED: dict[str, IssueCode] = {
    "state_missing": IssueCode.STATE_MISSING,
    "geom_out_of_bounds": IssueCode.GEOM_OUT_OF_BOUNDS,
}


def _names(directory: Path) -> list[str]:
    return sorted(path.stem for path in directory.glob("*.json"))


def test_example_and_fixture_inventory_matches_spec():
    assert [path.name for path in EXAMPLE_FILES] == ["minimal.json", "station_dkw.json"]
    assert _names(FIXTURES_DIR / "invalid") == sorted(INVALID_EXPECTED)
    assert _names(FIXTURES_DIR / "strict") == sorted(STRICT_EXPECTED)


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.stem)
def test_examples_are_strict_valid(path: Path):
    report = validate_document(load_json(path), strict=True)
    assert report.ok, [issue.model_dump() for issue in report.issues]
    assert report.issues == []


@pytest.mark.parametrize(("name", "code"), sorted(INVALID_EXPECTED.items()))
def test_invalid_fixture_yields_expected_code(name: str, code: IssueCode):
    doc = load_json(FIXTURES_DIR / "invalid" / f"{name}.json")
    assert schema_issues(doc) == [], "invalid fixtures must still be schema-valid"
    report = validate_document(doc)
    assert not report.ok
    assert code in {issue.code for issue in report.errors}
    assert IssueCode.SCHEMA_INVALID not in {issue.code for issue in report.issues}


@pytest.mark.parametrize(("name", "code"), sorted(STRICT_EXPECTED.items()))
def test_strict_fixture_flips_with_strict_flag(name: str, code: IssueCode):
    doc = load_json(FIXTURES_DIR / "strict" / f"{name}.json")
    assert schema_issues(doc) == []
    lenient = validate_document(doc)
    assert lenient.ok
    assert [issue.code for issue in lenient.warnings] == [code]
    assert lenient.errors == []
    strict = validate_document(doc, strict=True)
    assert not strict.ok
    assert [issue.code for issue in strict.errors] == [code]
    assert strict.warnings == []
