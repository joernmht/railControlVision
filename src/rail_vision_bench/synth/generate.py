"""Synthetic dataset generation (stub)."""

from __future__ import annotations

from pathlib import Path


def generate_dataset(out: Path, *, n: int = 10, seed: int = 0, augment: str | None = None) -> Path:
    """Generate ``n`` synthetic scenes with images, ground truth and a manifest.

    Intended implementation: for every scene call
    :func:`rail_vision_bench.graph.generate.generate_scene` (valid by
    construction, seeded), render it with
    :func:`rail_vision_bench.synth.render.render_svg` and
    :func:`rail_vision_bench.synth.render.svg_to_png`, check the tile layout
    for overlapping tiles and seams with ``shapely``, optionally run the
    ``albumentations`` pipeline from
    :func:`rail_vision_bench.synth.augment.build_augmentation`, then write the
    PNG, the ground-truth JSON and one ``ManifestRow`` per scene through
    :func:`rail_vision_bench.dataset.manifest.write_manifest`.

    Args:
        out: Output directory of the dataset.
        n: Number of scenes to generate.
        seed: Seed of the scene and augmentation randomness.
        augment: Augmentation preset name, or ``None`` for clean renders.

    Returns:
        The path of the written ``manifest.jsonl``.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.synth.generate.generate_dataset is not implemented in the skeleton: "
        "generate, render, augment and write n synthetic scenes plus their manifest"
    )
