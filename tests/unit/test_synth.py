"""Synthetic data: layout, rendering, augmentation and dataset generation."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from rail_vision_bench.dataset.manifest import read_manifest
from rail_vision_bench.dataset.splits import assign_partition, difficulty_tier
from rail_vision_bench.graph.generate import generate_scene
from rail_vision_bench.graph.validator import validate_document
from rail_vision_bench.ingest.quality import QualityMetrics
from rail_vision_bench.schema.models import (
    Geometry,
    NodeKind,
    Occupancy,
    SceneAnnotation,
    SignalAspect,
    SignalState,
    SwitchPosition,
    SwitchState,
    TrackState,
)
from rail_vision_bench.synth.augment import PRESETS, augment_image, build_augmentation
from rail_vision_bench.synth.generate import (
    SPLIT_AUGMENTED,
    SPLIT_CLEAN,
    check_layout,
    generate_dataset,
)
from rail_vision_bench.synth.render import (
    STYLES,
    has_complete_geometry,
    layout_scene,
    render_png,
    render_svg,
    svg_to_png,
)
from tests.conftest import EXAMPLES_DIR, load_json

TILE_FACE = (0xBF, 0xC3, 0xB8)


def _decode(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(png)).convert("RGB")).astype(int)


def _laid_out(seed: int = 3, n_inner: int = 4) -> SceneAnnotation:
    return layout_scene(generate_scene(seed, n_inner=n_inner), tile_px=32)


def _pixel(image: np.ndarray, point: tuple[float, float]) -> tuple[int, int, int]:
    x = min(int(point[0]), image.shape[1] - 1)
    y = min(int(point[1]), image.shape[0] - 1)
    r, g, b = image[y, x]
    return int(r), int(g), int(b)


# --------------------------------------------------------------------------- layout


@pytest.mark.parametrize("seed", range(8))
def test_layout_is_strict_valid_and_in_bounds(seed: int):
    doc = layout_scene(generate_scene(seed, n_inner=1 + seed % 5), tile_px=32)
    assert has_complete_geometry(doc)
    report = validate_document(doc.model_dump(mode="json"), strict=True)
    assert report.ok, report.issues
    check_layout(doc)
    tile = doc.meta["render"]["tile_px"]
    ox, oy = doc.meta["render"]["grid_origin"]
    for node in doc.topology.nodes:
        assert node.geometry is not None
        assert node.geometry.point is not None
        x, y = node.geometry.point
        # Nodes sit at tile centres of the recorded grid.
        assert ((x - ox) / tile - 0.5) == pytest.approx(round((x - ox) / tile - 0.5))
        assert ((y - oy) / tile - 0.5) == pytest.approx(round((y - oy) / tile - 0.5))
    glyphs = [signal.geometry for signal in doc.signals]
    glyphs += [derailer.geometry for derailer in doc.derailers]
    for geometry in glyphs:
        assert geometry is not None
        assert geometry.bbox is not None
        assert geometry.point is not None


def test_layout_sets_source_size_or_fits_into_it():
    scene = generate_scene(1, n_inner=3)
    grown = layout_scene(scene, tile_px=40)
    assert grown.source.width % 40 == 0
    assert grown.source.height % 40 == 0
    fitted = layout_scene(scene)
    assert (fitted.source.width, fitted.source.height) == (800, 400)
    xs = [
        node.geometry.point[0]
        for node in fitted.topology.nodes
        if node.geometry is not None and node.geometry.point is not None
    ]
    assert len(xs) == len(fitted.topology.nodes)
    assert 0 <= min(xs) < max(xs) <= 800


def test_layout_rejects_bad_arguments():
    scene = generate_scene(0)
    with pytest.raises(ValueError, match="tile_px"):
        layout_scene(scene, tile_px=0)
    with pytest.raises(ValueError, match="margin"):
        layout_scene(scene, margin_tiles=0)


def test_check_layout_detects_overlapping_tiles():
    doc = _laid_out()
    nodes = list(doc.topology.nodes)
    nodes[1] = nodes[1].model_copy(update={"geometry": nodes[0].geometry})
    broken = doc.model_copy(update={"topology": doc.topology.model_copy(update={"nodes": nodes})})
    with pytest.raises(ValueError, match="overlap"):
        check_layout(broken)


# --------------------------------------------------------------------------- rendering


def test_render_svg_has_source_size():
    doc = _laid_out()
    svg = render_svg(doc)
    assert svg.startswith("<svg")
    assert f'width="{doc.source.width}"' in svg
    assert f'height="{doc.source.height}"' in svg


def test_render_rejects_unknown_style():
    with pytest.raises(ValueError, match="style"):
        render_svg(_laid_out(), style="watercolour")


@pytest.mark.parametrize("style", STYLES)
def test_rendered_geometry_matches_ground_truth(style: str):
    doc = _laid_out()
    image = _decode(render_png(doc, style=style))
    assert image.shape[:2] == (doc.source.height, doc.source.width)
    background = np.array(TILE_FACE if style == "stelltisch" else (0, 0, 0))
    for edge in doc.topology.edges:
        assert edge.geometry is not None
        assert edge.geometry.polyline is not None
        a, b = edge.geometry.polyline[0], edge.geometry.polyline[1]
        # A third of the way along the first segment, clear of switch gaps at the ends.
        probe = (a[0] + (b[0] - a[0]) / 3, a[1] + (b[1] - a[1]) / 3)
        assert np.abs(np.array(_pixel(image, probe)) - background).sum() > 60, edge.id
    colours = {SignalAspect.PROCEED: (0x1F, 0xC2, 0x3A), SignalAspect.STOP: (0xE3, 0x20, 0x2A)}
    for signal in doc.signals:
        aspect = doc.state.signals[signal.id].aspect
        assert signal.geometry is not None
        assert signal.geometry.point is not None
        expected = np.array(colours[aspect])
        assert np.abs(np.array(_pixel(image, signal.geometry.point)) - expected).sum() < 30


def test_render_shows_occupancy_and_switch_position():
    doc = layout_scene(generate_scene(0, n_inner=1, kinds=[NodeKind.SWITCH]), tile_px=32)
    base = _decode(render_png(doc))

    def red_pixels(image: np.ndarray) -> int:
        return int(((image[..., 0] > 200) & (image[..., 1] < 80) & (image[..., 2] < 80)).sum())

    occupied_tracks = {
        edge_id: TrackState(occupancy=Occupancy.OCCUPIED, route_set=False)
        for edge_id in doc.state.tracks
    }
    occupied = doc.model_copy(
        update={"state": doc.state.model_copy(update={"tracks": occupied_tracks})}
    )
    assert red_pixels(_decode(render_png(occupied))) > red_pixels(base) + 200

    diverging = doc.model_copy(
        update={
            "state": doc.state.model_copy(
                update={"switches": {"x0": SwitchState(position=SwitchPosition.DIVERGING, raw="-")}}
            )
        }
    )
    assert not np.array_equal(_decode(render_png(diverging)), base)


def test_render_without_geometry_uses_automatic_layout():
    scene = generate_scene(2, n_inner=3)
    assert not has_complete_geometry(scene)
    image = _decode(render_png(scene))
    assert image.shape[:2] == (400, 800)


@pytest.mark.parametrize("name", ["minimal.json", "station_dkw.json"])
def test_render_committed_examples(name: str):
    doc = SceneAnnotation.model_validate(load_json(EXAMPLES_DIR / name))
    assert has_complete_geometry(doc)
    image = _decode(render_png(doc, style="estw"))
    assert image.shape[:2] == (doc.source.height, doc.source.width)


def test_render_draws_unknown_and_dark_states():
    doc = _laid_out()
    signals = {
        signal.id: SignalState(aspect=SignalAspect.DARK if i else SignalAspect.UNKNOWN)
        for i, signal in enumerate(doc.signals)
    }
    changed = doc.model_copy(update={"state": doc.state.model_copy(update={"signals": signals})})
    assert render_svg(changed) != render_svg(doc)


def test_svg_to_png_writes_scaled_file(tmp_path: Path):
    doc = _laid_out()
    out = svg_to_png(render_svg(doc), tmp_path / "sub" / "panel.png", scale=0.5)
    with Image.open(out) as image:
        assert image.size == (doc.source.width // 2, doc.source.height // 2)
    with pytest.raises(ValueError, match="scale"):
        svg_to_png("<svg/>", tmp_path / "x.png", scale=0)


# --------------------------------------------------------------------------- augmentation


def test_build_augmentation_rejects_unknown_preset():
    with pytest.raises(ValueError, match="preset"):
        build_augmentation("instagram")


@pytest.mark.parametrize("preset", PRESETS)
def test_augmentation_is_seeded_and_keeps_shape(preset: str):
    image = _decode(render_png(_laid_out())).astype(np.uint8)
    points = [(10.0, 10.0), (100.0, 50.0)]
    first = augment_image(build_augmentation(preset, seed=7), image, points)
    second = augment_image(build_augmentation(preset, seed=7), image, points)
    assert first[0].shape == image.shape
    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1]
    assert len(first[1]) == 2


def test_augmentation_moves_points_with_the_image():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[90:110, 190:210] = 255
    moved_image, moved = augment_image(build_augmentation("phone", seed=3), image, [(200, 100)])
    grey = moved_image.astype(float).mean(axis=2)
    ys, xs = np.nonzero(grey >= np.percentile(grey, 99.5))
    centroid = (float(xs.mean()), float(ys.mean()))
    assert abs(centroid[0] - moved[0][0]) < 6
    assert abs(centroid[1] - moved[0][1]) < 6


# --------------------------------------------------------------------------- dataset


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_generate_dataset_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    manifest = generate_dataset(Path("out"), n=3, seed=5)
    assert manifest == Path("out") / "manifest.jsonl"
    rows = read_manifest(manifest)
    assert len(rows) == 3
    assert len({row.scene_id for row in rows}) == 3
    for row in rows:
        assert row.split == SPLIT_CLEAN
        assert row.partition == assign_partition(row.scene_id)
        assert row.source_kind == "synthetic"
        assert row.license is None
        assert row.quality is not None
        assert row.difficulty == difficulty_tier(QualityMetrics.model_validate(row.quality))
        assert row.image == f"out/images/{row.scene_id}.png"
        assert row.gt == f"out/gt/{row.scene_id}.json"
        document: dict[str, Any] = json.loads(Path(row.gt).read_text(encoding="utf-8"))
        assert validate_document(document, strict=True).ok
        assert document["scene_id"] == row.scene_id
        assert document["source"]["image"] == row.image
        assert (document["source"]["width"], document["source"]["height"]) == (
            row.width,
            row.height,
        )
        assert document["provenance"]["kind"] == "ground_truth"
        with Image.open(row.image) as image:
            assert image.size == (row.width, row.height)


def test_generate_dataset_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    generate_dataset(Path("a"), n=2, seed=11, augment="default")
    generate_dataset(Path("b"), n=2, seed=11, augment="default")
    first, second = _snapshot(tmp_path / "a"), _snapshot(tmp_path / "b")
    assert first.keys() == second.keys()
    for name, content in first.items():
        if name.endswith(".png"):
            assert content == second[name], name
        else:
            assert content.replace(b'"a/', b'"b/') == second[name], name
    rows = read_manifest(tmp_path / "a" / "manifest.jsonl")
    assert {row.split for row in rows} == {SPLIT_AUGMENTED}
    assert all(row.scene_id.endswith("-default") for row in rows)
    for row in rows:
        document = json.loads(Path(row.gt).read_text(encoding="utf-8"))
        assert validate_document(document, strict=True).ok


def test_generate_dataset_varies_with_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    generate_dataset(Path("a"), n=1, seed=1)
    generate_dataset(Path("b"), n=1, seed=2)
    assert _snapshot(tmp_path / "a") != _snapshot(tmp_path / "b")


def test_generate_dataset_outside_cwd_uses_absolute_paths(tmp_path: Path):
    manifest = generate_dataset(tmp_path / "ds", n=1, seed=0, style="estw")
    (row,) = read_manifest(manifest)
    assert Path(row.image).is_absolute()
    assert Path(row.image).exists()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"n": 0}, "n must"), ({"augment": "sepia"}, "preset"), ({"style": "oil"}, "style")],
)
def test_generate_dataset_rejects_bad_arguments(tmp_path: Path, kwargs: dict[str, Any], match: str):
    with pytest.raises(ValueError, match=match):
        generate_dataset(tmp_path, **kwargs)


def test_geometry_warp_keeps_bounds(tmp_path: Path):
    manifest = generate_dataset(tmp_path / "ds", n=2, seed=4, augment="phone")
    for row in read_manifest(manifest):
        doc = SceneAnnotation.model_validate_json(Path(row.gt).read_text(encoding="utf-8"))
        geometries: list[Geometry | None] = [
            *(node.geometry for node in doc.topology.nodes),
            *(edge.geometry for edge in doc.topology.edges),
        ]
        for geometry in geometries:
            assert geometry is not None
            if geometry.bbox is not None:
                x0, y0, x1, y1 = geometry.bbox
                assert 0 <= x0 <= x1 <= row.width
                assert 0 <= y0 <= y1 <= row.height
