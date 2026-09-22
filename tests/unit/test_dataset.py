"""Manifest round trip, the committed example manifest, partitions and difficulty tiers."""

from __future__ import annotations

from pathlib import Path

import pytest

from rail_vision_bench.dataset.manifest import ManifestRow, read_manifest, write_manifest
from rail_vision_bench.dataset.splits import (
    DifficultyThresholds,
    assign_partition,
    difficulty_tier,
)
from rail_vision_bench.ingest.quality import QualityMetrics
from rail_vision_bench.schema.models import SourceKind
from tests.conftest import REPO_ROOT

EXAMPLE_MANIFEST = REPO_ROOT / "data" / "manifest.example.jsonl"


def _row(scene_id: str, **overrides: object) -> ManifestRow:
    fields: dict[str, object] = {
        "scene_id": scene_id,
        "split": "synthetic_clean",
        "partition": assign_partition(scene_id),
        "source_kind": SourceKind.SYNTHETIC,
        "image": f"data/synthetic/{scene_id}.png",
        "gt": f"data/gt/{scene_id}.json",
        "width": 400,
        "height": 100,
    }
    fields.update(overrides)
    return ManifestRow.model_validate(fields)


def test_manifest_round_trip(tmp_path: Path):
    rows = [
        _row("s1", difficulty="easy", license="CC-BY-4.0", quality={"blur_var": 500.0}),
        _row("s2"),
    ]
    path = tmp_path / "nested" / "manifest.jsonl"
    write_manifest(rows, path)
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 2
    assert read_manifest(path) == rows


def test_read_manifest_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    write_manifest([_row("s1")], path)
    path.write_bytes(b"\n" + path.read_bytes() + b"\n\n")
    assert [row.scene_id for row in read_manifest(path)] == ["s1"]


def test_manifest_row_rejects_extra_and_bad_values():
    with pytest.raises(ValueError, match="extra"):
        _row("s1", bogus=1)
    with pytest.raises(ValueError, match="partition"):
        _row("s1", partition="train")
    with pytest.raises(ValueError, match="width"):
        _row("s1", width=0)


def test_example_manifest_resolves_and_partitions_match():
    rows = read_manifest(EXAMPLE_MANIFEST)
    assert [row.scene_id for row in rows] == ["minimal", "station_dkw"]
    for row in rows:
        assert (REPO_ROOT / row.gt).is_file(), row.gt
        assert row.partition == assign_partition(row.scene_id)
        assert row.split == "synthetic_clean"
        assert row.source_kind is SourceKind.SYNTHETIC


def test_example_manifest_sizes_match_ground_truth():
    from tests.conftest import load_json

    for row in read_manifest(EXAMPLE_MANIFEST):
        source = load_json(REPO_ROOT / row.gt)["source"]
        assert (row.width, row.height) == (source["width"], source["height"])
        assert row.image == source["image"]


def test_assign_partition_is_deterministic_and_roughly_20_percent():
    ids = [f"scene_{i:05d}" for i in range(2000)]
    first = [assign_partition(i) for i in ids]
    assert first == [assign_partition(i) for i in ids]
    share = first.count("test") / len(first)
    assert 0.12 <= share <= 0.28
    assert all(assign_partition(i, test_fraction=0.0) == "dev" for i in ids[:50])
    assert all(assign_partition(i, test_fraction=1.0) == "test" for i in ids[:50])


@pytest.mark.parametrize(
    ("blur_var", "glare_fraction", "expected"),
    [
        (500.0, 0.0, "easy"),
        (300.0, 0.05, "easy"),
        (299.9, 0.0, "medium"),
        (500.0, 0.06, "medium"),
        (99.9, 0.0, "hard"),
        (500.0, 0.16, "hard"),
        (50.0, 0.5, "hard"),
    ],
)
def test_difficulty_tier_defaults(blur_var: float, glare_fraction: float, expected: str):
    quality = QualityMetrics(blur_var=blur_var, glare_fraction=glare_fraction)
    assert difficulty_tier(quality) == expected


def test_difficulty_tier_custom_thresholds():
    quality = QualityMetrics(blur_var=200.0, glare_fraction=0.0)
    lenient = DifficultyThresholds(blur_var_medium=150.0, blur_var_hard=50.0)
    assert difficulty_tier(quality, lenient) == "easy"
    harsh = DifficultyThresholds(blur_var_hard=250.0)
    assert difficulty_tier(quality, harsh) == "hard"


def test_quality_metrics_ranges():
    with pytest.raises(ValueError, match="glare_fraction"):
        QualityMetrics(blur_var=1.0, glare_fraction=1.5)
    with pytest.raises(ValueError, match="blur_var"):
        QualityMetrics(blur_var=-1.0, glare_fraction=0.0)
