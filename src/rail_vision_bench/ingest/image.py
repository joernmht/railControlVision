"""Still-image loading (stub)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from PIL.Image import Image


def load_image(path: Path) -> Image:
    """Load a photo or screenshot as an RGB PIL image.

    Intended implementation: register the HEIF/HEIC opener from ``pillow_heif``
    (phone photos of panels), open the file with Pillow, apply
    ``PIL.ImageOps.exif_transpose`` so the orientation tag is baked in, and
    convert to RGB.

    Args:
        path: The image file (PNG, JPEG, WebP or HEIC).

    Returns:
        The decoded image.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.ingest.image.load_image is not implemented in the skeleton: "
        "open a still image with Pillow (HEIF registered) and bake in the EXIF orientation"
    )
