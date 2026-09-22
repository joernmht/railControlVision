"""Screen capture of ESTW/CTC workstations (stub)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image


def capture_screen(monitor: int = 1, region: tuple[int, int, int, int] | None = None) -> Image:
    """Grab one screenshot of a monitor or a region of it.

    Intended implementation: ``mss.mss().grab`` of the monitor (or the given
    ``(left, top, width, height)`` region) converted to an RGB PIL image.
    Tests exercising it carry the ``requires_display`` marker.

    Args:
        monitor: One-based monitor index as reported by ``mss``.
        region: Optional ``(left, top, width, height)`` sub-rectangle in pixels.

    Returns:
        The captured image.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.ingest.screen.capture_screen is not implemented in the skeleton: "
        "grab a monitor or region with mss and return it as a PIL image"
    )
