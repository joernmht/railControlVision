"""Publishing and fetching dataset splits on the Hugging Face Hub (stubs)."""

from __future__ import annotations

from pathlib import Path


def push_split(split: str, repo_id: str, *, manifest: Path) -> str:
    """Upload the images and ground truth listed in a manifest to a Hub dataset repo.

    Intended implementation: build a ``datasets.Dataset`` from the manifest
    rows (image column plus the ground-truth JSON as a string column) and
    ``push_to_hub`` under the ``split`` name, authenticated with
    ``settings.hf_token`` through ``huggingface_hub``.

    Args:
        split: Split name on the Hub.
        repo_id: Target dataset repository, ``owner/name``.
        manifest: The local ``manifest.jsonl`` describing the split.

    Returns:
        The commit URL on the Hub.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.dataset.hub.push_split is not implemented in the skeleton: "
        "push a manifest's images and ground truth to a Hub dataset split"
    )


def pull_split(split: str, repo_id: str, *, dest: Path) -> Path:
    """Download one split of a Hub dataset repo and rebuild its local manifest.

    Intended implementation: ``huggingface_hub.snapshot_download`` of the
    split, materialise images and ground-truth JSON under ``dest`` and write
    ``dest / "manifest.jsonl"`` through ``dataset.manifest.write_manifest``.

    Args:
        split: Split name on the Hub.
        repo_id: Source dataset repository, ``owner/name``.
        dest: Local directory to populate.

    Returns:
        The path of the rebuilt manifest.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.dataset.hub.pull_split is not implemented in the skeleton: "
        "download a Hub dataset split and rebuild its local manifest"
    )
