"""Synthetic dataset generation: rendered panels with exact ground truth and a manifest.

Every scene comes from :func:`rail_vision_bench.graph.generate.generate_scene`
(valid by construction), gets trackside extras and a varied but consistent
state, is laid out on the tile grid (which writes the pixel geometry into the
ground truth), rendered, optionally augmented with the ground-truth points
warped alongside the image, measured for quality and written out. All
randomness derives from the ``seed`` argument, so the same call produces the
same files byte for byte.
"""

from __future__ import annotations

import random
from io import BytesIO
from pathlib import Path
from typing import Any, Final

from rail_vision_bench.schema.models import (
    Derailer,
    DerailerPosition,
    DerailerState,
    Direction,
    EdgeAttachment,
    Geometry,
    NodeKind,
    Occupancy,
    Port,
    ProvenanceKind,
    RouteState,
    RouteStatus,
    SceneAnnotation,
    Signal,
    SignalAspect,
    SignalKind,
    SignalState,
    SwitchPosition,
    SwitchState,
    TrackKind,
    TrackState,
)
from rail_vision_bench.synth.render import DEFAULT_TILE_PX

SPLIT_CLEAN: Final = "synthetic_clean"
"""Split name of clean renders (``data/README.md``)."""
SPLIT_AUGMENTED: Final = "synthetic_aug"
"""Split name of augmented renders (``data/README.md``)."""
GENERATOR_TOOL: Final = "rail_vision_bench.synth.generate"
"""``provenance.tool`` of the generated ground truth."""

_OCCUPIED_P: Final = 0.15
_ROUTE_SET_P: Final = 0.6
_SHUNT_SIGNAL_P: Final = 0.5
_DERAILER_P: Final = 0.4
_SLIP_STRAIGHT: Final[list[tuple[Port, Port]]] = [(Port.A, Port.C), (Port.B, Port.D)]
_SLIP_TURNOUT: Final[dict[NodeKind, list[tuple[Port, Port]]]] = {
    NodeKind.DKW: [(Port.A, Port.D), (Port.B, Port.C)],
    NodeKind.EKW: [(Port.A, Port.D)],
}


def _slip_raw(half_labels: list[str], sign: str) -> str:
    halves = half_labels if len(half_labels) == 2 else ["a", "b"]
    return f"{halves[0]}:{sign} {halves[1]}:{sign}"


def _decorate(doc: SceneAnnotation, rng: random.Random) -> SceneAnnotation:
    """Add labels, shunt signals and derailers on sidings, and draw a consistent state.

    The generated route ``rt_w`` is either set (its switches straight, its
    tracks lit, ``sig_w`` at proceed) or not set (switches in random positions,
    no track lit, ``sig_w`` at stop); occupancy is drawn per track
    independently of the route.
    """
    signals = [
        signal.model_copy(update={"label": {"sig_w": "A", "sig_e": "F"}.get(signal.id)})
        for signal in doc.signals
    ]
    derailers: list[Derailer] = []
    signal_states: dict[str, SignalState] = {}
    derailer_states: dict[str, DerailerState] = {}
    sidings = [edge for edge in doc.topology.edges if edge.kind == TrackKind.SIDING]
    for number, edge in enumerate(sidings, start=1):
        if rng.random() < _SHUNT_SIGNAL_P:
            signal_id = f"sh_{edge.id}"
            signals.append(
                Signal(
                    id=signal_id,
                    kind=SignalKind.SHUNT,
                    label=f"Ls{number}",
                    system="Sh",
                    at=EdgeAttachment(edge=edge.id, offset=0.75, direction=Direction.B_TO_A),
                )
            )
            proceed = rng.random() < 0.3
            signal_states[signal_id] = SignalState(
                aspect=SignalAspect.SHUNT_PROCEED if proceed else SignalAspect.STOP,
                raw="Sh1" if proceed else "Sh0",
            )
        if rng.random() < _DERAILER_P:
            derailer_id = f"gs_{edge.id}"
            derailers.append(
                Derailer(
                    id=derailer_id,
                    label=f"Gs{number}",
                    at=EdgeAttachment(edge=edge.id, offset=0.4, direction=Direction.B_TO_A),
                )
            )
            applied = rng.random() < 0.7
            derailer_states[derailer_id] = DerailerState(
                position=DerailerPosition.APPLIED if applied else DerailerPosition.REMOVED
            )

    route_set = rng.random() < _ROUTE_SET_P
    switches: dict[str, SwitchState] = {}
    for node in doc.topology.nodes:
        if node.kind == NodeKind.SWITCH:
            straight = route_set or rng.random() < 0.5
            switches[node.id] = SwitchState(
                position=SwitchPosition.STRAIGHT if straight else SwitchPosition.DIVERGING,
                raw="+" if straight else "-",
            )
        elif node.kind in _SLIP_TURNOUT:
            straight = route_set or rng.random() < 0.5
            switches[node.id] = SwitchState(
                active_paths=list(_SLIP_STRAIGHT if straight else _SLIP_TURNOUT[node.kind]),
                raw=_slip_raw(node.half_labels, "+" if straight else "-"),
            )
    route_edges = {edge_id for route in doc.routes for edge_id in route.path}
    tracks = {
        edge.id: TrackState(
            occupancy=Occupancy.OCCUPIED if rng.random() < _OCCUPIED_P else Occupancy.FREE,
            route_set=route_set and edge.id in route_edges,
        )
        for edge in doc.topology.edges
    }
    signal_states["sig_w"] = (
        SignalState(aspect=SignalAspect.PROCEED, raw="Hp1")
        if route_set
        else SignalState(aspect=SignalAspect.STOP, raw="Hp0")
    )
    signal_states["sig_e"] = SignalState(aspect=SignalAspect.STOP, raw="Hp0")
    routes = {
        route.id: RouteState(status=RouteStatus.SET if route_set else RouteStatus.NOT_SET)
        for route in doc.routes
    }
    state = doc.state.model_copy(
        update={
            "switches": switches,
            "signals": signal_states,
            "tracks": tracks,
            "derailers": derailer_states,
            "routes": routes,
        }
    )
    return doc.model_copy(update={"signals": signals, "derailers": derailers, "state": state})


def check_layout(doc: SceneAnnotation) -> None:
    """Check with ``shapely`` that node tiles do not overlap and unrelated tracks do not cross.

    Args:
        doc: A laid-out scene (see :func:`rail_vision_bench.synth.render.layout_scene`).

    Raises:
        ValueError: If two node tiles overlap, or two tracks that share no node
            touch or cross.
    """
    from shapely.geometry import LineString, box  # type: ignore[import-untyped]

    tiles = [
        (node.id, box(*node.geometry.bbox))
        for node in doc.topology.nodes
        if node.geometry is not None and node.geometry.bbox is not None
    ]
    for i, (first_id, first) in enumerate(tiles):
        for second_id, second in tiles[i + 1 :]:
            if first.intersection(second).area > 1e-6:
                msg = f"tiles of {first_id} and {second_id} overlap"
                raise ValueError(msg)
    lines = [
        (edge, LineString(edge.geometry.polyline))
        for edge in doc.topology.edges
        if edge.geometry is not None
        and edge.geometry.polyline is not None
        and len(edge.geometry.polyline) >= 2
    ]
    for i, (first_edge, first_line) in enumerate(lines):
        first_nodes = {first_edge.a.node, first_edge.b.node}
        for second_edge, second_line in lines[i + 1 :]:
            if first_nodes & {second_edge.a.node, second_edge.b.node}:
                continue
            if first_line.intersects(second_line):
                msg = f"tracks {first_edge.id} and {second_edge.id} cross"
                raise ValueError(msg)


def _geometry_points(doc: SceneAnnotation) -> list[tuple[float, float]]:
    """Every coordinate of the document, in the order :func:`_warp_geometry` consumes them."""
    points: list[tuple[float, float]] = []
    for geometry in _geometries(doc):
        if geometry is None:
            continue
        if geometry.point is not None:
            points.append(geometry.point)
        if geometry.polyline is not None:
            points.extend(geometry.polyline)
        if geometry.bbox is not None:
            x0, y0, x1, y1 = geometry.bbox
            points.extend([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    return points


def _geometries(doc: SceneAnnotation) -> list[Geometry | None]:
    return [
        *(node.geometry for node in doc.topology.nodes),
        *(edge.geometry for edge in doc.topology.edges),
        *(signal.geometry for signal in doc.signals),
        *(derailer.geometry for derailer in doc.derailers),
    ]


def _warp_geometry(doc: SceneAnnotation, moved: list[tuple[float, float]]) -> SceneAnnotation:
    """Replace every coordinate by its warped counterpart, clamped to the image."""
    width, height = doc.source.width, doc.source.height
    it = iter(moved)

    def clamp(point: tuple[float, float]) -> tuple[float, float]:
        return (min(max(point[0], 0.0), float(width)), min(max(point[1], 0.0), float(height)))

    def warp(geometry: Geometry | None) -> Geometry | None:
        if geometry is None:
            return None
        point = clamp(next(it)) if geometry.point is not None else None
        polyline = (
            [clamp(next(it)) for _ in geometry.polyline] if geometry.polyline is not None else None
        )
        bbox = None
        if geometry.bbox is not None:
            corners = [clamp(next(it)) for _ in range(4)]
            xs = [x for x, _ in corners]
            ys = [y for _, y in corners]
            bbox = (min(xs), min(ys), max(xs), max(ys))
        return Geometry(point=point, polyline=polyline, bbox=bbox)

    nodes = [
        node.model_copy(update={"geometry": warp(node.geometry)}) for node in doc.topology.nodes
    ]
    edges = [
        edge.model_copy(update={"geometry": warp(edge.geometry)}) for edge in doc.topology.edges
    ]
    signals = [
        signal.model_copy(update={"geometry": warp(signal.geometry)}) for signal in doc.signals
    ]
    derailers = [
        derailer.model_copy(update={"geometry": warp(derailer.geometry)})
        for derailer in doc.derailers
    ]
    return doc.model_copy(
        update={
            "topology": doc.topology.model_copy(update={"nodes": nodes, "edges": edges}),
            "signals": signals,
            "derailers": derailers,
        }
    )


def _repo_path(path: Path) -> str:
    """Path relative to the working directory (the repository root) when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def generate_dataset(
    out: Path,
    *,
    n: int = 10,
    seed: int = 0,
    augment: str | None = None,
    style: str = "stelltisch",
    tile_px: float = DEFAULT_TILE_PX,
) -> Path:
    """Generate ``n`` synthetic scenes with images, ground truth and a manifest.

    For scene ``i`` a generator seeded from ``(seed, i)`` draws the number of
    inner nodes (1-5) and the seed of
    :func:`rail_vision_bench.graph.generate.generate_scene`; the scene gets
    shunt signals and derailers on some sidings and a varied, consistent state
    (route set or not, switch positions, occupancy, signal aspects, derailer
    positions). :func:`rail_vision_bench.synth.render.layout_scene` writes the
    pixel geometry and the image size, :func:`check_layout` checks the tile
    layout with ``shapely``, and the scene is rendered with
    :func:`rail_vision_bench.synth.render.render_svg` and ``cairosvg``. With
    ``augment`` the image goes through
    :func:`rail_vision_bench.synth.augment.build_augmentation` and every
    ground-truth coordinate through the same warp. Quality is measured on the
    final image. Written files::

        out/images/<scene_id>.png
        out/gt/<scene_id>.json      strict-valid SceneAnnotation
        out/manifest.jsonl          one ManifestRow per scene

    Rows use split ``synthetic_clean`` or ``synthetic_aug``, the hashed
    partition, the difficulty tier of the measured quality, ``license: null``
    (project-generated data) and paths relative to the working directory when
    ``out`` lies below it (absolute otherwise); ``source.image`` of each ground
    truth equals the row's ``image``. Output is byte-identical for equal
    arguments.

    Args:
        out: Output directory of the dataset (created when missing).
        n: Number of scenes to generate.
        seed: Seed of the scene and augmentation randomness.
        augment: Augmentation preset name, or ``None`` for clean renders.
        style: Render style, ``stelltisch`` or ``estw``.
        tile_px: Tile pitch of the layout in pixels.

    Returns:
        The path of the written ``manifest.jsonl``.

    Raises:
        ValueError: If ``n`` is not positive, or ``augment`` or ``style`` is unknown.
        RuntimeError: If a generated ground truth fails strict validation (a bug).
    """
    import numpy as np
    import orjson
    from PIL import Image as PILImage

    from rail_vision_bench.dataset.manifest import ManifestRow, write_manifest
    from rail_vision_bench.dataset.splits import assign_partition, difficulty_tier
    from rail_vision_bench.graph.generate import generate_scene
    from rail_vision_bench.graph.validator import validate_document
    from rail_vision_bench.ingest.quality import measure
    from rail_vision_bench.synth.augment import augment_image, build_augmentation
    from rail_vision_bench.synth.render import STYLES, layout_scene, render_png

    if n < 1:
        msg = f"n must be >= 1, got {n}"
        raise ValueError(msg)
    if style not in STYLES:
        msg = f"unknown style {style!r}; expected one of {', '.join(STYLES)}"
        raise ValueError(msg)
    if augment is not None:
        build_augmentation(augment)  # fail fast on an unknown preset
    split = SPLIT_CLEAN if augment is None else SPLIT_AUGMENTED
    image_dir = out / "images"
    gt_dir = out / "gt"
    image_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    rows: list[ManifestRow] = []
    for index in range(n):
        rng = random.Random(f"rail-vision-bench:{seed}:{index}")
        n_inner = rng.randint(1, 5)
        scene_seed = rng.randrange(2**31)
        scene_id = f"synth-{seed}-{index:05d}" + ("" if augment is None else f"-{augment}")
        image_path = image_dir / f"{scene_id}.png"
        gt_path = gt_dir / f"{scene_id}.json"
        image_rel = _repo_path(image_path)

        doc = _decorate(generate_scene(scene_seed, n_inner=n_inner), rng)
        meta: dict[str, Any] = {
            "generator": {
                "seed": seed,
                "index": index,
                "scene_seed": scene_seed,
                "n_inner": n_inner,
                "style": style,
                "augment": augment,
            }
        }
        doc = doc.model_copy(
            update={
                "scene_id": scene_id,
                "source": doc.source.model_copy(update={"image": image_rel}),
                "provenance": doc.provenance.model_copy(
                    update={"kind": ProvenanceKind.GROUND_TRUTH, "tool": GENERATOR_TOOL}
                ),
                "meta": meta,
            }
        )
        doc = layout_scene(doc, tile_px=tile_px)
        check_layout(doc)
        png = render_png(doc, style=style)
        if augment is None:
            image_path.write_bytes(png)
            with PILImage.open(BytesIO(png)) as rendered:
                image = rendered.convert("RGB")
        else:
            with PILImage.open(BytesIO(png)) as rendered:
                array = np.asarray(rendered.convert("RGB"), dtype=np.uint8)
            pipeline = build_augmentation(augment, seed=rng.randrange(2**31))
            augmented, moved = augment_image(pipeline, array, _geometry_points(doc))
            doc = _warp_geometry(doc, moved)
            image = PILImage.fromarray(augmented)
            image.save(image_path, format="PNG")
        quality = measure(image)
        difficulty = difficulty_tier(quality)
        doc = doc.model_copy(update={"meta": {**doc.meta, "difficulty": difficulty}})
        document = doc.model_dump(mode="json", exclude_none=True)
        report = validate_document(document, strict=True)
        if not report.ok:
            msg = f"generated scene {scene_id} is not strict-valid: {report.errors}"
            raise RuntimeError(msg)
        gt_path.write_bytes(orjson.dumps(document, option=orjson.OPT_INDENT_2) + b"\n")
        rows.append(
            ManifestRow(
                scene_id=scene_id,
                split=split,
                partition=assign_partition(scene_id),
                difficulty=difficulty,
                source_kind=doc.source.kind,
                image=image_rel,
                gt=_repo_path(gt_path),
                width=doc.source.width,
                height=doc.source.height,
                license=None,
                quality=quality.model_dump(),
            )
        )
    manifest = out / "manifest.jsonl"
    write_manifest(rows, manifest)
    return manifest
