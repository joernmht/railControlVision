"""Image-quality measurements that feed the difficulty tiers of the dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Final

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from PIL.Image import Image

GLARE_THRESHOLD: Final[float] = 250.0 / 255.0
"""Grey level (in ``[0, 1]``) at or above which a pixel counts as saturated glare."""


class QualityMetrics(BaseModel):
    """Objective quality of one source image."""

    model_config = ConfigDict(extra="forbid")

    blur_var: Annotated[float, Field(ge=0)] = Field(
        description="Variance of the Laplacian; lower means blurrier."
    )
    glare_fraction: Annotated[float, Field(ge=0, le=1)] = Field(
        description="Fraction of pixels that are saturated (specular glare on glass panels)."
    )


def measure(image: Image) -> QualityMetrics:
    """Compute the quality metrics of an image.

    The image is converted to grey with ``skimage.color.rgb2gray``. ``blur_var``
    is the variance of the 3x3 Laplacian (``cv2.Laplacian``; the same kernel as
    ``skimage.filters.laplace``, which ships without type information) of that
    grey image on the 0-255 scale, the scale of the usual "variance of the
    Laplacian" measure that the ``DifficultyThresholds`` defaults assume;
    ``glare_fraction`` is the share of pixels whose grey level is at least
    :data:`GLARE_THRESHOLD`.

    Args:
        image: The source image (any Pillow mode; converted to RGB first).

    Returns:
        The measured metrics.
    """
    import cv2
    import numpy as np
    from skimage.color import rgb2gray

    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    grey = rgb2gray(rgb)
    blur_var = float(np.var(cv2.Laplacian(grey * 255.0, cv2.CV_64F, ksize=1)))
    glare_fraction = float(np.mean(grey >= GLARE_THRESHOLD)) if grey.size else 0.0
    return QualityMetrics(blur_var=max(0.0, blur_var), glare_fraction=min(1.0, glare_fraction))
