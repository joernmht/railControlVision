"""Rendering of scene annotations as panel-style images (stubs)."""

from __future__ import annotations

from pathlib import Path

from rail_vision_bench.schema.models import SceneAnnotation


def render_svg(doc: SceneAnnotation, *, style: str = "stelltisch") -> str:
    """Draw a scene as an SVG document.

    Intended implementation: ``svgwrite`` drawing of the topology (track
    polylines, switch symbols, signal glyphs with their aspect colours,
    occupancy and route illumination) in the tile style of a German mosaic
    panel (``stelltisch``) or an ESTW screen (``estw``), using the geometry
    stored in the document.

    Args:
        doc: The scene to render; its geometry must be complete.
        style: Visual style preset.

    Returns:
        The SVG document as a string.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.synth.render.render_svg is not implemented in the skeleton: "
        "draw the topology and state as a panel-style SVG with svgwrite"
    )


def svg_to_png(svg: str, out: Path, *, scale: float = 1.0) -> Path:
    """Rasterise an SVG document to PNG.

    Intended implementation: ``cairosvg.svg2png`` with ``scale`` as the
    output scale factor, written to ``out``.

    Args:
        svg: The SVG document.
        out: Destination PNG path.
        scale: Rasterisation scale factor.

    Returns:
        The written PNG path.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.synth.render.svg_to_png is not implemented in the skeleton: "
        "rasterise the SVG to a PNG file with cairosvg"
    )
