"""Still-image loading: PNG, JPEG, WebP and HEIC with the EXIF orientation applied."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from PIL.Image import Image


def load_image(path: Path) -> Image:
    """Load a photo or screenshot as an RGB PIL image.

    The HEIF/HEIC opener of ``pillow_heif`` is registered first (phone photos of
    panels), the file is decoded completely with Pillow, the EXIF orientation
    tag is baked in with :func:`PIL.ImageOps.exif_transpose` and the result is
    converted to RGB (alpha is dropped, palettes and grey levels are expanded).

    Args:
        path: The image file (PNG, JPEG, WebP or HEIC).

    Returns:
        The decoded, upright RGB image; it does not keep the file open.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        PIL.UnidentifiedImageError: If the file is not a decodable image.
    """
    from PIL import Image as PILImage, ImageOps
    from pillow_heif.as_plugin import register_heif_opener

    register_heif_opener()
    with PILImage.open(path) as opened:
        upright = ImageOps.exif_transpose(opened)
        return upright.convert("RGB")
