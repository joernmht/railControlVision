"""Screen capture of ESTW/CTC workstations with ``mss``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image


def capture_screen(monitor: int = 1, region: tuple[int, int, int, int] | None = None) -> Image:
    """Grab one screenshot of a monitor or a region of it.

    Uses ``mss.mss().grab`` and converts the BGRA buffer to an RGB PIL image.
    ``region`` is relative to the top-left corner of the chosen monitor; ``mss``
    itself works in virtual-screen coordinates, so the monitor offset is added.
    Tests exercising a real display carry the ``requires_display`` marker.

    Args:
        monitor: One-based monitor index as reported by ``mss`` (``0`` is the
            virtual screen spanning all monitors).
        region: Optional ``(left, top, width, height)`` sub-rectangle in pixels.

    Returns:
        The captured image.

    Raises:
        ValueError: If ``monitor`` does not exist or ``region`` is empty or does
            not lie inside the monitor.
        mss.exception.ScreenShotError: If no display is available.
    """
    import mss
    from PIL import Image as PILImage

    with mss.mss() as sct:
        monitors = sct.monitors
        if not 0 <= monitor < len(monitors):
            msg = f"monitor {monitor} does not exist; mss reports {len(monitors) - 1} monitor(s)"
            raise ValueError(msg)
        mon = monitors[monitor]
        area = {
            "left": int(mon["left"]),
            "top": int(mon["top"]),
            "width": int(mon["width"]),
            "height": int(mon["height"]),
        }
        if region is not None:
            left, top, width, height = region
            if width < 1 or height < 1:
                msg = f"region must have a positive size, got {region}"
                raise ValueError(msg)
            if left < 0 or top < 0 or left + width > area["width"] or top + height > area["height"]:
                size = f"{area['width']}x{area['height']}"
                msg = f"region {region} is outside monitor {monitor} ({size})"
                raise ValueError(msg)
            area = {
                "left": area["left"] + left,
                "top": area["top"] + top,
                "width": width,
                "height": height,
            }
        shot = sct.grab(area)
        return PILImage.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
