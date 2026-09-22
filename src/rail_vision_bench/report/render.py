"""Leaderboard rendering: template loading is real, rendering is a stub."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Final, Literal

TEMPLATE_NAME: Final = "leaderboard.html.j2"


def load_template() -> str:
    """Read the leaderboard template shipped inside the package.

    Returns:
        The Jinja2 template source.
    """
    resource = files("rail_vision_bench").joinpath("report", "templates", TEMPLATE_NAME)
    return resource.read_text(encoding="utf-8")


def render_leaderboard(
    run_dirs: Sequence[Path], out: Path, *, fmt: Literal["html", "md"] = "html"
) -> Path:
    """Render a leaderboard over evaluated runs.

    Intended implementation: load ``summary.json`` and ``metrics.parquet`` of
    every run, build one row per (model, mode, split) with the template's
    columns, render :data:`TEMPLATE_NAME` with ``jinja2`` (``html``) or a
    ``tabulate`` table (``md``), and embed per-metric bar charts drawn with
    ``matplotlib`` as inline images.

    Args:
        run_dirs: Evaluated run directories.
        out: Destination file.
        fmt: Output format.

    Returns:
        The written file.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.report.render.render_leaderboard is not implemented in the skeleton: "
        "render the leaderboard of the given runs with jinja2/tabulate/matplotlib"
    )
