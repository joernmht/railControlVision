"""Photo-realistic degradation of clean renders with ``albumentations``.

Geometric transforms move pixels, so the pipelines carry keypoints: every
coordinate of the ground truth goes through the same warp as the image (see
:func:`augment_image`), which keeps augmented ground truth exact.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

PRESETS: Final[tuple[str, ...]] = ("default", "phone", "cctv")
"""Names accepted by :func:`build_augmentation`."""


def _transforms(preset: str) -> list[Any]:
    import albumentations as alb  # type: ignore[import-untyped]
    import cv2

    if preset == "default":
        return [
            alb.Perspective(
                scale=(0.01, 0.04), keep_size=True, border_mode=cv2.BORDER_REPLICATE, p=0.7
            ),
            alb.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.7),
            alb.RandomSunFlare(
                flare_roi=(0.0, 0.0, 1.0, 1.0),
                src_radius=60,
                num_flare_circles_range=(2, 5),
                p=0.3,
            ),
            alb.CoarseDropout(
                num_holes_range=(1, 2),
                hole_height_range=(0.05, 0.12),
                hole_width_range=(0.02, 0.06),
                fill="random_uniform",
                p=0.3,
            ),
            alb.MotionBlur(blur_limit=(3, 5), p=0.3),
            alb.ImageCompression(compression_type="jpeg", quality_range=(60, 95), p=0.7),
        ]
    if preset == "phone":
        return [
            alb.Perspective(
                scale=(0.03, 0.08), keep_size=True, border_mode=cv2.BORDER_REPLICATE, p=1.0
            ),
            alb.Affine(rotate=(-4, 4), scale=(0.95, 1.02), border_mode=cv2.BORDER_REPLICATE, p=0.8),
            alb.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.9),
            alb.RandomSunFlare(
                flare_roi=(0.0, 0.0, 1.0, 1.0),
                src_radius=120,
                num_flare_circles_range=(3, 7),
                p=0.6,
            ),
            alb.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.08, 0.2),
                hole_width_range=(0.03, 0.1),
                fill="random_uniform",
                p=0.4,
            ),
            alb.MotionBlur(blur_limit=(3, 9), p=0.5),
            alb.ImageCompression(compression_type="jpeg", quality_range=(45, 85), p=1.0),
        ]
    if preset == "cctv":
        return [
            alb.Perspective(
                scale=(0.02, 0.05), keep_size=True, border_mode=cv2.BORDER_REPLICATE, p=0.8
            ),
            alb.Downscale(scale_range=(0.4, 0.7), p=1.0),
            alb.GaussNoise(std_range=(0.03, 0.08), p=0.9),
            alb.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.9),
            alb.MotionBlur(blur_limit=(3, 7), p=0.5),
            alb.ImageCompression(compression_type="jpeg", quality_range=(25, 60), p=1.0),
        ]
    msg = f"unknown augmentation preset {preset!r}; expected one of {', '.join(PRESETS)}"
    raise ValueError(msg)


def build_augmentation(preset: str, *, seed: int | None = None) -> Any:
    """Build the augmentation pipeline for a preset.

    An ``albumentations.Compose`` of perspective warp, lighting change,
    specular glare (sun flare), partial occlusion (coarse dropout), motion blur
    and JPEG compression, with strengths per preset: ``default`` (mild),
    ``phone`` (hand-held photo of a panel: stronger perspective, rotation,
    glare, occlusion) and ``cctv`` (control-room camera: downscaling, sensor
    noise, heavy compression). The pipeline carries ``xy`` keypoints that are
    never dropped, so ground-truth coordinates can follow the warp; the return
    type stays ``Any`` because albumentations ships no type information.

    Args:
        preset: Name of the augmentation preset, one of :data:`PRESETS`.
        seed: Seed of the pipeline's random generator; the same seed and input
            give the same output.

    Returns:
        A callable pipeline, ``pipeline(image=array, keypoints=points)``
        returning a dict with ``image`` and ``keypoints``.

    Raises:
        ValueError: If ``preset`` is unknown.
    """
    import albumentations as alb

    return alb.Compose(
        _transforms(preset),
        keypoint_params=alb.KeypointParams(format="xy", remove_invisible=False),
        seed=seed,
    )


def augment_image(
    pipeline: Any, image: NDArray[np.uint8], points: Sequence[tuple[float, float]]
) -> tuple[NDArray[np.uint8], list[tuple[float, float]]]:
    """Run a pipeline on an image and the ground-truth points drawn in it.

    Args:
        pipeline: A pipeline from :func:`build_augmentation`.
        image: ``H x W x 3`` uint8 RGB array.
        points: Pixel coordinates to transform along with the image.

    Returns:
        The augmented image (same size) and the transformed points, in input order.

    Raises:
        ValueError: If the pipeline changed the image size.
    """
    import numpy as np

    keypoints = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    result = pipeline(image=image, keypoints=keypoints)
    out = np.ascontiguousarray(result["image"], dtype=np.uint8)
    if out.shape != image.shape:
        msg = f"augmentation changed the image shape from {image.shape} to {out.shape}"
        raise ValueError(msg)
    moved = np.asarray(result["keypoints"], dtype=np.float64).reshape(-1, 2)
    return out, [(float(x), float(y)) for x, y in moved[:, :2].tolist()]
