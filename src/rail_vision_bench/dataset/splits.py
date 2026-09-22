"""Deterministic dev/test partitioning and difficulty tiers."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict

from rail_vision_bench.ingest.quality import QualityMetrics

Partition = Literal["dev", "test"]
Difficulty = Literal["easy", "medium", "hard"]


def assign_partition(scene_id: str, *, test_fraction: float = 0.2) -> Partition:
    """Assign a scene to ``dev`` or ``test`` from a hash of its id.

    Hashing (rather than random sampling) keeps the assignment stable when
    scenes are added or removed and reproducible across machines.

    Args:
        scene_id: The scene identifier.
        test_fraction: Expected share of ``test`` scenes.

    Returns:
        The partition.
    """
    digest = hashlib.sha1(scene_id.encode(), usedforsecurity=False).hexdigest()
    bucket = int(digest[:8], 16) % 1000
    return "test" if bucket < round(test_fraction * 1000) else "dev"


class DifficultyThresholds(BaseModel):
    """Quality cut-offs between the difficulty tiers."""

    model_config = ConfigDict(extra="forbid")

    blur_var_medium: float = 300.0
    blur_var_hard: float = 100.0
    glare_medium: float = 0.05
    glare_hard: float = 0.15


def difficulty_tier(
    quality: QualityMetrics, thresholds: DifficultyThresholds | None = None
) -> Difficulty:
    """Map quality metrics to a difficulty tier.

    Args:
        quality: The measured metrics of the image.
        thresholds: Cut-offs; the defaults when omitted.

    Returns:
        ``hard`` when the image is very blurry or very glary, ``medium`` when
        moderately so, else ``easy``.
    """
    t = thresholds if thresholds is not None else DifficultyThresholds()
    if quality.blur_var < t.blur_var_hard or quality.glare_fraction > t.glare_hard:
        return "hard"
    if quality.blur_var < t.blur_var_medium or quality.glare_fraction > t.glare_medium:
        return "medium"
    return "easy"
