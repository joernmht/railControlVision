"""Matching, per-scene metrics, scoring and run aggregation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from rail_vision_bench.config import Mode
from rail_vision_bench.eval import (
    METRIC_NAMES,
    aggregate_run,
    combine_results,
    load_metrics,
    match_all,
    match_elements,
    score_prediction,
)
from rail_vision_bench.eval.matching import Match, geometry_distance, label_distance
from rail_vision_bench.eval.metrics import (
    calibration,
    detection_prf1,
    label_cer_wer,
    latency_summary,
    route_prf1,
    state_accuracy,
    topology_agreement,
)
from rail_vision_bench.eval.records import MetricResult, PredictionRecord
from rail_vision_bench.graph.generate import generate_scene
from rail_vision_bench.runner import RUN_LAYOUT
from rail_vision_bench.schema.models import (
    Geometry,
    Occupancy,
    SceneAnnotation,
    SignalAspect,
    Source,
    SourceKind,
)

PER_SCENE = [name for name in METRIC_NAMES if not name.startswith(("latency", "cost"))]


def by_name(rows: list[MetricResult]) -> dict[str, MetricResult]:
    return {row.name: row for row in rows}


def all_matches(gt: SceneAnnotation, pred: SceneAnnotation) -> list[Match]:
    return [m for matches in match_all(gt, pred).values() for m in matches]


def rename_ids(scene: SceneAnnotation, prefix: str = "p_") -> SceneAnnotation:
    """Rename every element id (and every reference to it) consistently."""
    doc = scene.model_dump(mode="json")

    def r(element_id: str) -> str:
        return prefix + element_id

    for element in (
        doc["topology"]["nodes"]
        + doc["topology"]["edges"]
        + doc["signals"]
        + doc["derailers"]
        + doc["routes"]
    ):
        element["id"] = r(element["id"])
    for edge in doc["topology"]["edges"]:
        edge["a"]["node"] = r(edge["a"]["node"])
        edge["b"]["node"] = r(edge["b"]["node"])
    for element in doc["signals"] + doc["derailers"]:
        element["at"]["edge"] = r(element["at"]["edge"])
    for route in doc["routes"]:
        route["start"] = r(route["start"])
        route["end"] = r(route["end"])
        route["path"] = [r(edge) for edge in route["path"]]
        route["switch_positions"] = {r(k): v for k, v in route["switch_positions"].items()}
    for table in ("switches", "signals", "tracks", "derailers", "routes"):
        doc["state"][table] = {r(k): v for k, v in doc["state"][table].items()}
    return SceneAnnotation.model_validate(doc)


def edit(scene: SceneAnnotation) -> dict[str, Any]:
    return scene.model_dump(mode="json")


@pytest.fixture(params=["minimal", "station_dkw", "gen0", "gen1", "gen7"])
def scene(
    request: pytest.FixtureRequest,
    minimal_scene: SceneAnnotation,
    station_dkw_scene: SceneAnnotation,
) -> SceneAnnotation:
    if request.param == "minimal":
        return minimal_scene
    if request.param == "station_dkw":
        return station_dkw_scene
    return generate_scene(int(request.param[3:]), n_inner=5)


# --- matching ---------------------------------------------------------------------------


def test_perfect_matching_is_identity(scene: SceneAnnotation):
    for family, matches in match_all(scene, scene).items():
        assert all(m.gt_id == m.pred_id and m.cost == 0 for m in matches), family
        assert all(m.family == family for m in matches)
    assert len(match_elements(scene, scene, family="nodes")) == len(scene.topology.nodes)
    assert len(match_elements(scene, scene, family="routes")) == len(scene.routes)


def test_matching_is_independent_of_ids(station_dkw_scene):
    pred = rename_ids(station_dkw_scene)
    for matches in match_all(station_dkw_scene, pred).values():
        assert all(m.pred_id == "p_" + m.gt_id for m in matches)
    rows = by_name(score_prediction(station_dkw_scene, pred))
    for name in ("detection_prf1", "state_accuracy", "route_prf1", "topology_agreement"):
        assert rows[name].value == 1.0, name
    assert rows["label_cer"].value == 0.0


def test_kind_mismatch_and_threshold(station_dkw_scene):
    doc = edit(station_dkw_scene)
    doc["topology"]["nodes"][1]["kind"] = "joint"  # sw1: label and geometry still agree
    pred = SceneAnnotation.model_validate(doc)
    match = next(
        m for m in match_elements(station_dkw_scene, pred, family="nodes") if m.gt_id == "sw1"
    )
    assert match.cost == pytest.approx(0.3)
    strict = match_elements(station_dkw_scene, pred, family="nodes", threshold=0.2)
    assert "sw1" not in {m.gt_id for m in strict}


def test_geometry_decides_between_unlabelled_nodes(station_dkw_scene):
    # Swap the ids of the two buffer stops: geometry must map them back.
    doc = edit(station_dkw_scene)
    nodes = doc["topology"]["nodes"]
    bs1 = next(n for n in nodes if n["id"] == "n_bs1")
    bs2 = next(n for n in nodes if n["id"] == "n_bs2")
    bs1["geometry"], bs2["geometry"] = bs2["geometry"], bs1["geometry"]
    pred = SceneAnnotation.model_validate(doc)
    pairs = {m.gt_id: m.pred_id for m in match_elements(station_dkw_scene, pred, family="nodes")}
    assert pairs["n_bs1"] == "n_bs2"
    assert pairs["n_bs2"] == "n_bs1"


def test_match_elements_rejects_bad_arguments(minimal_scene):
    with pytest.raises(ValueError, match="unknown family"):
        match_elements(minimal_scene, minimal_scene, family="tracks")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        match_elements(minimal_scene, minimal_scene, family="nodes", w_kind=-1)


def test_label_and_geometry_distance():
    assert label_distance(None, "  ") is None
    assert label_distance("A", None) == 1.0
    assert label_distance("A", "a") == 1.0
    assert label_distance("N1", "N1") == 0.0
    assert label_distance("12", "13") == pytest.approx(0.5)
    src = Source(kind=SourceKind.SYNTHETIC, image="x.png", width=100, height=100)
    big = Source(kind=SourceKind.SYNTHETIC, image="x.png", width=200, height=200)
    box = Geometry(bbox=(0, 0, 10, 10))
    assert geometry_distance(box, src, box, src) == 0.0
    # Same normalised box in a twice as large image.
    assert geometry_distance(box, src, Geometry(bbox=(0, 0, 20, 20)), big) == 0.0
    half = Geometry(bbox=(5, 0, 15, 10))
    assert geometry_distance(box, src, half, src) == pytest.approx(1 - 50 / 150)
    points = geometry_distance(Geometry(point=(0, 0)), src, Geometry(point=(30, 40)), src)
    assert points == pytest.approx(0.5 / math.sqrt(2))
    flat = Geometry(polyline=[(0, 50), (100, 50)])  # zero-area box: centre distance
    assert geometry_distance(flat, src, flat, src) == 0.0
    assert geometry_distance(None, src, box, src) is None


# --- perfect, perturbed and empty predictions -------------------------------------------


def test_perfect_prediction(scene: SceneAnnotation):
    rows = by_name(score_prediction(scene, scene))
    assert list(rows) == PER_SCENE
    assert rows["detection_prf1"].value == 1.0
    assert rows["detection_prf1"].extra["fp"] == rows["detection_prf1"].extra["fn"] == 0
    assert rows["label_cer"].value == 0.0
    assert rows["label_wer"].value == 0.0
    assert rows["state_accuracy"].value == 1.0
    assert rows["state_accuracy"].n > 0
    assert rows["route_prf1"].value == 1.0
    assert rows["topology_agreement"].value == 1.0
    assert rows["topology_agreement"].extra["degree_agreement"] == 1.0


def test_perfect_prediction_counts(station_dkw_scene):
    rows = by_name(score_prediction(station_dkw_scene, station_dkw_scene))
    families = rows["detection_prf1"].extra["families"]
    assert {f: s["tp"] for f, s in families.items()} == {
        "nodes": 8,
        "edges": 8,
        "signals": 5,
        "derailers": 1,
    }
    # 3 switches + 5 aspects + 8 occupancies + 8 route_set + 1 derailer + 2 routes.
    assert rows["state_accuracy"].extra["total"] == 27
    assert rows["topology_agreement"].extra["intersection"] == 8


def test_missing_derailer(station_dkw_scene):
    doc = edit(station_dkw_scene)
    doc["derailers"] = []
    doc["state"]["derailers"] = {}
    rows = by_name(score_prediction(station_dkw_scene, SceneAnnotation.model_validate(doc)))
    det = rows["detection_prf1"]
    assert (det.extra["tp"], det.extra["fp"], det.extra["fn"]) == (21, 0, 1)
    assert det.value == pytest.approx(42 / 43)
    assert det.extra["families"]["derailers"]["recall"] == 0.0
    assert det.extra["families"]["nodes"]["f1"] == 1.0
    assert rows["state_accuracy"].extra["total"] == 26  # the derailer is not scored
    assert rows["state_accuracy"].value == 1.0


def test_wrong_states(station_dkw_scene):
    doc = edit(station_dkw_scene)
    doc["state"]["signals"]["sig_A"]["aspect"] = SignalAspect.STOP.value
    doc["state"]["tracks"]["e8"]["occupancy"] = Occupancy.UNKNOWN.value  # unknown is wrong
    doc["state"]["switches"]["dkw1"]["active_paths"] = [["A", "D"], ["B", "C"]]
    del doc["state"]["routes"]["rt_A_1"]  # missing entry is wrong
    rows = by_name(score_prediction(station_dkw_scene, SceneAnnotation.model_validate(doc)))
    state = rows["state_accuracy"]
    assert state.value == pytest.approx(23 / 27)
    assert state.extra["fields"]["aspect"] == {"correct": 4, "total": 5}
    assert state.extra["fields"]["slip_paths"] == {"correct": 0, "total": 1}
    assert state.extra["fields"]["route_status"] == {"correct": 1, "total": 2}


def test_slip_paths_are_unordered(station_dkw_scene):
    doc = edit(station_dkw_scene)
    doc["state"]["switches"]["dkw1"]["active_paths"] = [["D", "B"], ["C", "A"]]
    pred = SceneAnnotation.model_validate(doc)
    assert (
        state_accuracy(station_dkw_scene, pred, all_matches(station_dkw_scene, pred))[0].value
        == 1.0
    )


def test_label_errors(station_dkw_scene):
    doc = edit(station_dkw_scene)
    doc["topology"]["nodes"][1]["label"] = "7"  # sw1 "1" -> "7": still matched on geometry
    doc["signals"][3]["label"] = "N 1"  # sig_N1 "N1" -> "N 1"
    rows = by_name(score_prediction(station_dkw_scene, SceneAnnotation.model_validate(doc)))
    labels = [
        label
        for label in [n.label for n in station_dkw_scene.topology.nodes]
        + [s.label for s in station_dkw_scene.signals]
        + [d.label for d in station_dkw_scene.derailers]
        + [r.label for r in station_dkw_scene.routes]
        if label
    ]
    assert rows["label_cer"].n == len(labels)
    total_chars = sum(len(label) for label in labels)
    assert rows["label_cer"].value == pytest.approx(2 / total_chars)  # 1 sub + 1 insertion
    total_words = sum(len(label.split()) for label in labels)
    # "1" -> "7" is one substitution, "N1" -> "N 1" one substitution plus one insertion.
    assert rows["label_wer"].extra == {"errors": 3, "ref_len": total_words}


def test_label_cer_wer_direct():
    rows = by_name(label_cer_wer([("abc", "abd"), ("12", ""), ("", "ignored")]))
    assert rows["label_cer"].n == 2
    assert rows["label_cer"].value == pytest.approx(3 / 5)
    assert rows["label_wer"].value == pytest.approx(1.0)
    empty = by_name(label_cer_wer([]))
    assert empty["label_cer"].value == 0.0
    assert empty["label_cer"].n == 0


def test_route_changes(station_dkw_scene):
    doc = edit(station_dkw_scene)
    doc["routes"][0]["path"] = ["e1", "e3", "e5"]  # truncated path, wrong end
    pred = SceneAnnotation.model_validate(doc)
    row = route_prf1(station_dkw_scene, pred, all_matches(station_dkw_scene, pred))[0]
    assert (row.extra["tp"], row.extra["fp"], row.extra["fn"]) == (1, 1, 1)
    assert row.value == pytest.approx(0.5)
    doc["routes"] = []
    empty = SceneAnnotation.model_validate(doc)
    row = route_prf1(station_dkw_scene, empty, all_matches(station_dkw_scene, empty))[0]
    assert row.extra["precision"] == 0.0
    assert row.value == 0.0


def test_topology_changes():
    gt = generate_scene(3, n_inner=4)
    n_edges = len(gt.topology.edges)
    doc = edit(gt)
    doc["topology"]["edges"] = doc["topology"]["edges"][:-1]
    row = topology_agreement(gt, SceneAnnotation.model_validate(doc))[0]
    assert row.value == pytest.approx((n_edges - 1) / n_edges)
    assert row.extra["degree_agreement"] < 1.0
    # A wrong port breaks port-exact agreement of that edge only.
    doc = edit(gt)
    first = doc["topology"]["edges"][0]
    first["a"]["port"] = "B" if first["a"]["port"] == "A" else "A"
    row = topology_agreement(gt, SceneAnnotation.model_validate(doc))[0]
    assert row.extra["intersection"] == n_edges - 1
    assert row.value == pytest.approx((n_edges - 1) / (n_edges + 1))


def test_empty_prediction(station_dkw_scene):
    rows = by_name(score_prediction(station_dkw_scene, None))
    det = rows["detection_prf1"]
    assert det.value == 0.0
    assert det.extra["fn"] == 22
    assert det.extra["precision"] == 0.0
    assert rows["route_prf1"].value == 0.0
    assert rows["topology_agreement"].value == 0.0
    assert rows["state_accuracy"].n == 0
    assert rows["state_accuracy"].value == 0.0
    assert rows["label_cer"].n == 0
    assert rows["calibration_brier"].n == 0
    assert all(not math.isnan(row.value) for row in rows.values())


def test_empty_ground_truth_and_prediction(minimal_scene):
    doc = edit(minimal_scene)
    doc["topology"] = {"nodes": [], "edges": []}
    doc["signals"] = []
    doc["state"] = {}
    empty = SceneAnnotation.model_validate(doc)
    rows = by_name(score_prediction(empty, empty))
    assert rows["detection_prf1"].value == 1.0
    assert rows["detection_prf1"].n == 0
    assert rows["topology_agreement"].value == 1.0
    assert rows["route_prf1"].value == 1.0


def test_metric_functions_accept_hand_built_matches(minimal_scene):
    matches = [Match(gt_id=n.id, pred_id=n.id, cost=0) for n in minimal_scene.topology.nodes]
    row = detection_prf1(minimal_scene, minimal_scene, matches)[0]
    assert row.extra["families"]["nodes"]["tp"] == 3
    assert row.extra["families"]["edges"]["fn"] == 2  # edges were not matched


def test_score_prediction_subset_and_unknown(minimal_scene):
    rows = score_prediction(minimal_scene, minimal_scene, ["state_accuracy", "latency_p50_ms"])
    assert [row.name for row in rows] == ["state_accuracy"]
    with pytest.raises(ValueError, match="unknown metric"):
        score_prediction(minimal_scene, minimal_scene, ["bogus"])


# --- calibration and latency ------------------------------------------------------------


def test_calibration_known_values():
    rows = by_name(calibration([0.9, 0.8, 0.3], [True, True, False]))
    assert rows["calibration_brier"].value == pytest.approx((0.01 + 0.04 + 0.09) / 3)
    assert rows["calibration_ece"].value == pytest.approx((0.1 + 0.2 + 0.3) / 3)
    rows = by_name(calibration([1.0, 1.0, 0.0, 0.0], [True, True, False, False]))
    assert rows["calibration_brier"].value == 0.0
    assert rows["calibration_ece"].value == 0.0
    # Two items in one bin: |mean acc - mean conf| = |0.5 - 0.55|.
    rows = by_name(calibration([0.5, 0.6], [True, False], n_bins=2))
    assert rows["calibration_ece"].value == pytest.approx(0.05)
    assert rows["calibration_ece"].extra["bins"][1]["count"] == 2


def test_calibration_edge_cases():
    rows = by_name(calibration([], []))
    assert rows["calibration_brier"].n == 0
    assert rows["calibration_brier"].value == 0.0
    with pytest.raises(ValueError, match="outcomes"):
        calibration([0.5], [])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calibration([1.5], [True])


def test_calibration_uses_prediction_confidences(station_dkw_scene):
    doc = edit(station_dkw_scene)
    for node in doc["topology"]["nodes"]:
        node["confidence"] = 0.75
    doc["topology"]["nodes"].append({"id": "ghost", "kind": "joint", "confidence": 0.75})
    rows = by_name(score_prediction(station_dkw_scene, SceneAnnotation.model_validate(doc)))
    # 9 node confidences (8 right, the ghost wrong) + the sw1 state confidence 1.0 (right).
    assert rows["calibration_brier"].n == 10
    expected = (8 * 0.25**2 + 0.75**2 + 0.0) / 10
    assert rows["calibration_brier"].value == pytest.approx(expected)


def record(scene_id: str, latency: float, **kwargs: Any) -> PredictionRecord:
    return PredictionRecord(
        scene_id=scene_id,
        run_id="run-1",
        model=kwargs.pop("model", "m1"),
        mode=kwargs.pop("mode", Mode.SINGLE_SHOT),
        latency_ms=latency,
        **kwargs,
    )


def test_latency_summary():
    records = [record(f"s{i}", ms, cost_usd=0.01) for i, ms in enumerate([10, 20, 30, 40, 100])]
    records.append(record("s5", 50))
    rows = by_name(latency_summary(records))
    assert rows["latency_p50_ms"].value == pytest.approx(35.0)
    assert rows["latency_p95_ms"].value == pytest.approx(87.5)
    assert rows["latency_p50_ms"].extra["max"] == 100
    assert rows["cost_usd"].value == pytest.approx(0.05)
    assert rows["cost_usd"].extra["n_with_cost"] == 5
    assert rows["cost_usd"].n == 6
    empty = by_name(latency_summary([]))
    assert empty["latency_p95_ms"].value == 0.0
    assert empty["latency_p95_ms"].n == 0


# --- aggregation ------------------------------------------------------------------------


def test_combine_results_matches_pooled_counts(station_dkw_scene, minimal_scene):
    perfect = by_name(score_prediction(station_dkw_scene, station_dkw_scene))
    missing = by_name(score_prediction(minimal_scene, None))
    combined = combine_results([perfect["detection_prf1"], missing["detection_prf1"]])
    assert combined.extra["tp"] == 22
    assert combined.extra["fn"] == 6
    assert combined.value == pytest.approx(44 / 50)
    assert combined.extra["families"]["signals"]["recall"] == pytest.approx(5 / 6)
    state = combine_results([perfect["state_accuracy"], missing["state_accuracy"]])
    assert state.value == 1.0  # the empty scene has n = 0 and contributes nothing
    topo = combine_results([perfect["topology_agreement"], missing["topology_agreement"]])
    assert topo.value == pytest.approx(8 / 10)
    with pytest.raises(ValueError, match="exactly one"):
        combine_results([perfect["label_cer"], perfect["label_wer"]])
    with pytest.raises(ValueError, match="exactly one"):
        combine_results([])


def test_combine_calibration_equals_pooled():
    a = by_name(calibration([0.9, 0.2], [True, True]))
    b = by_name(calibration([0.6, 0.4, 0.8], [False, True, True]))
    pooled = by_name(calibration([0.9, 0.2, 0.6, 0.4, 0.8], [True, True, False, True, True]))
    for name in ("calibration_brier", "calibration_ece"):
        combined = combine_results([a[name], b[name]])
        assert combined.value == pytest.approx(pooled[name].value)
        assert combined.n == 5


def test_aggregate_run_round_trip(tmp_path: Path, station_dkw_scene, minimal_scene):
    gen = generate_scene(1)
    truth = {s.scene_id: s for s in (station_dkw_scene, minimal_scene, gen)}
    records = [
        # m1: a failed first attempt at station_dkw, then a perfect retry.
        record("station_dkw", 100, attempt=1, parse_error="bad json", cost_usd=0.5),
        record("station_dkw", 300, attempt=2, annotation=station_dkw_scene, cost_usd=0.5),
        record("minimal", 200, annotation=minimal_scene, cost_usd=0.25),
        record("not_in_gt", 400),
        # m2 (agentic): nothing parsed for minimal, the generated scene perfect.
        record("minimal", 50, model="m2", mode=Mode.AGENTIC),
        record(gen.scene_id, 70, model="m2", mode=Mode.AGENTIC, annotation=gen),
    ]
    (tmp_path / RUN_LAYOUT["predictions"]).write_text(
        "\n".join(r.model_dump_json() for r in records) + "\n\n", encoding="utf-8"
    )
    path = aggregate_run(tmp_path, ground_truth=truth)
    assert path == tmp_path / RUN_LAYOUT["metrics"]
    frame = load_metrics(path)
    assert {"model", "mode", "name", "value", "n", "extra"} <= set(frame.columns)
    run = frame[frame["scope"] == "run"].set_index(["model", "mode", "name"])
    assert set(run.loc[("m1", "single_shot")].index) == set(METRIC_NAMES)
    assert run.loc[("m1", "single_shot", "detection_prf1"), "value"] == 1.0
    assert run.loc[("m1", "single_shot", "state_accuracy"), "value"] == 1.0
    assert run.loc[("m1", "single_shot", "cost_usd"), "value"] == pytest.approx(1.25)
    assert run.loc[("m1", "single_shot", "latency_p50_ms"), "value"] == pytest.approx(250.0)
    assert run.loc[("m1", "single_shot", "latency_p50_ms"), "n"] == 4
    m2 = run.loc[("m2", "agentic", "detection_prf1")]
    gen_elements = len(gen.topology.nodes) + len(gen.topology.edges) + len(gen.signals)
    assert m2["extra"]["tp"] == gen_elements
    assert m2["extra"]["fn"] == 6  # the minimal scene: 3 nodes, 2 edges, 1 signal
    assert m2["value"] == pytest.approx(2 * gen_elements / (2 * gen_elements + 6))
    scenes = frame[frame["scope"] == "scene"]
    assert set(scenes["scene_id"]) == {"station_dkw", "minimal", gen.scene_id}
    assert scenes["scene_id"].notna().all()
    assert frame[frame["scope"] == "run"]["scene_id"].isna().all()
    raw = load_metrics(path, parse_extra=False)
    assert isinstance(json.loads(raw["extra"].iloc[0]), dict)


def test_aggregate_run_without_ground_truth_and_subset(tmp_path: Path):
    (tmp_path / RUN_LAYOUT["predictions"]).write_text(
        record("s", 10).model_dump_json() + "\n", encoding="utf-8"
    )
    frame = load_metrics(aggregate_run(tmp_path))
    assert sorted(frame["name"]) == ["cost_usd", "latency_p50_ms", "latency_p95_ms"]
    frame = load_metrics(aggregate_run(tmp_path, metrics=["latency_p95_ms"]))
    assert list(frame["name"]) == ["latency_p95_ms"]
    with pytest.raises(ValueError, match="unknown metric"):
        aggregate_run(tmp_path, metrics=["bogus"])
