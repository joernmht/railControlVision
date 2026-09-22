"""Dataset plane: the manifest contract, deterministic splits and hub publishing.

Named ``dataset`` (not ``data``) so it is not confused with the DVC-managed
``data/`` directory or with the Hugging Face ``datasets`` package.
"""

from __future__ import annotations

from rail_vision_bench.dataset.manifest import ManifestRow, read_manifest, write_manifest
from rail_vision_bench.dataset.splits import assign_partition, difficulty_tier

__all__ = ["ManifestRow", "assign_partition", "difficulty_tier", "read_manifest", "write_manifest"]
