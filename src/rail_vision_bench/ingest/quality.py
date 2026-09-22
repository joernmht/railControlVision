"""Image-quality measurements that feed the difficulty tiers of the dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from PIL.Image import Image


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

    Intended implementation: convert to grey with ``skimage.color.rgb2gray``,
    take the variance of ``skimage.filters.laplace`` for ``blur_var`` and the
    fraction of pixels above a near-white threshold for ``glare_fraction``.

    Args:
        image: The source image.

    Returns:
        The measured metrics.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.ingest.quality.measure is not implemented in the skeleton: "
        "measure blur (Laplacian variance) and glare fraction with scikit-image"
    )
