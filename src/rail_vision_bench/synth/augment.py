"""Photo-realistic degradation of clean renders (stub)."""

from __future__ import annotations

from typing import Any


def build_augmentation(preset: str) -> Any:
    """Build the augmentation pipeline for a preset.

    Intended implementation: an ``albumentations.Compose`` of perspective
    warp, specular glare, partial occlusion, JPEG compression and motion blur
    with strengths chosen per preset (for example ``default``, ``phone``,
    ``cctv``); the return type stays ``Any`` because albumentations ships no
    type information.

    Args:
        preset: Name of the augmentation preset.

    Returns:
        A callable pipeline taking and returning an image array.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.synth.augment.build_augmentation is not implemented in the skeleton: "
        "compose the albumentations pipeline of the given preset"
    )
