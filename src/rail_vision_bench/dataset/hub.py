"""Publishing and fetching released dataset splits on the Hugging Face Hub.

DVC stays the system of record (``data/README.md``); the Hub is a distribution
channel for released splits only. A split lives in its own folder of a Hub
*dataset* repository::

    <split>/manifest.jsonl        ManifestRow per scene, paths relative to the repo root
    <split>/images/<scene_id>.<ext>
    <split>/gt/<scene_id>.json    strict-valid SceneAnnotation

so several splits share one repository and :func:`pull_split` downloads exactly
one of them. ``huggingface_hub`` is imported lazily; the token is
``settings.hf_token`` (``HF_TOKEN``), falling back to the library's own login
when unset.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Final

from rail_vision_bench.dataset.manifest import ManifestRow, read_manifest, write_manifest
from rail_vision_bench.graph.validator import validate_document
from rail_vision_bench.schema.models import SceneAnnotation, SourceKind
from rail_vision_bench.settings import Settings, get_settings

SPLIT_NAME_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
"""A split name is one safe path segment (it becomes a folder in the Hub repo)."""

MANIFEST_NAME: Final = "manifest.jsonl"


class HubReleaseError(ValueError):
    """A manifest does not meet the release rules of :func:`check_release`."""


def _check_split_name(split: str) -> None:
    if not SPLIT_NAME_RE.fullmatch(split):
        raise ValueError(f"invalid split name {split!r}: must match {SPLIT_NAME_RE.pattern}")


def _token(settings: Settings | None) -> str | None:
    resolved = settings if settings is not None else get_settings()
    return resolved.hf_token.get_secret_value() if resolved.hf_token is not None else None


def check_release(rows: Sequence[ManifestRow], split: str, *, root: Path) -> list[str]:
    """List every reason why ``rows`` cannot be released as ``split``.

    The rules: the manifest is not empty, every row belongs to ``split``,
    scene ids are unique, every non-synthetic row fills ``license`` (licensing
    policy rule 3), the image and ground-truth files exist, and every ground
    truth parses, carries the row's ``scene_id``, ``width`` and ``height`` and
    is strict-valid.

    Args:
        rows: The manifest rows.
        split: The split being released.
        root: Directory the manifest paths are relative to (the repository root).

    Returns:
        Human-readable problems; empty when the split may be released.
    """
    if not rows:
        return ["the manifest has no rows"]
    problems: list[str] = []
    seen: set[str] = set()
    for row in rows:
        where = f"scene {row.scene_id!r}"
        if row.scene_id in seen:
            problems.append(f"{where}: duplicate scene_id")
        seen.add(row.scene_id)
        if row.split != split:
            problems.append(f"{where}: split is {row.split!r}, not {split!r}")
        if row.license is None and row.source_kind is not SourceKind.SYNTHETIC:
            problems.append(f"{where}: license is null but the source is not synthetic")
        if not (root / row.image).is_file():
            problems.append(f"{where}: image {row.image} not found")
        gt_path = root / row.gt
        if not gt_path.is_file():
            problems.append(f"{where}: ground truth {row.gt} not found")
            continue
        try:
            doc = SceneAnnotation.model_validate_json(gt_path.read_bytes())
        except ValueError as exc:
            problems.append(f"{where}: ground truth does not parse: {exc}")
            continue
        if doc.scene_id != row.scene_id:
            problems.append(f"{where}: ground truth scene_id is {doc.scene_id!r}")
        if (doc.source.width, doc.source.height) != (row.width, row.height):
            problems.append(f"{where}: ground truth size differs from the manifest")
        report = validate_document(doc, strict=True)
        if not report.ok:
            codes = ", ".join(sorted({issue.code for issue in report.errors}))
            problems.append(f"{where}: ground truth is not strict-valid ({codes})")
    return problems


def push_split(
    split: str,
    repo_id: str,
    *,
    manifest: Path,
    root: Path = Path(),
    settings: Settings | None = None,
    private: bool = True,
) -> str:
    """Upload the images and ground truth listed in a manifest to a Hub dataset repo.

    The manifest is checked with :func:`check_release` first; nothing is
    uploaded when a rule fails. The files are staged in a temporary directory
    under the layout of this module's docstring (images and ground truth
    renamed to ``<scene_id>`` plus their suffix, manifest paths rewritten to
    the repo layout), the dataset repository is created when missing
    (``private`` by default) and the staged folder is uploaded to
    ``<split>/`` in one commit with ``HfApi.upload_folder``. Files of an
    earlier upload of the same split that are no longer staged are deleted in
    that commit.

    Args:
        split: Split name on the Hub; also the folder name in the repository.
        repo_id: Target dataset repository, ``owner/name``.
        manifest: The local ``manifest.jsonl`` describing the split.
        root: Directory the manifest paths are relative to (the repository root).
        settings: Runtime settings for ``hf_token``; ``get_settings()`` when omitted.
        private: Visibility of the repository if this call creates it.

    Returns:
        The commit URL on the Hub.

    Raises:
        ValueError: When ``split`` is not a valid folder name.
        HubReleaseError: When the manifest breaks a release rule.
    """
    from huggingface_hub import HfApi

    _check_split_name(split)
    rows = read_manifest(manifest)
    problems = check_release(rows, split, root=root)
    if problems:
        raise HubReleaseError(
            f"split {split!r} cannot be released:\n" + "\n".join(f"- {p}" for p in problems)
        )
    token = _token(settings)
    with tempfile.TemporaryDirectory(prefix="rvb-hub-") as tmp:
        staging = Path(tmp)
        staged: list[ManifestRow] = []
        for row in rows:
            image = PurePosixPath(split, "images", f"{row.scene_id}{Path(row.image).suffix}")
            gt = PurePosixPath(split, "gt", f"{row.scene_id}.json")
            for source, target in ((row.image, image), (row.gt, gt)):
                dest = staging / target.relative_to(split)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / source, dest)
            staged.append(row.model_copy(update={"image": str(image), "gt": str(gt)}))
        write_manifest(staged, staging / MANIFEST_NAME)
        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=staging,
            path_in_repo=split,
            commit_message=f"Release split {split} ({len(staged)} scenes)",
            delete_patterns="*",
        )
    return str(commit.commit_url)


def pull_split(
    split: str,
    repo_id: str,
    *,
    dest: Path,
    revision: str | None = None,
    settings: Settings | None = None,
) -> Path:
    """Download one split of a Hub dataset repo and rebuild its local manifest.

    ``huggingface_hub.snapshot_download`` fetches only ``<split>/**`` into
    ``dest`` (so the files land in ``dest / split``); the downloaded manifest's
    ``image`` and ``gt`` paths are re-anchored at ``dest`` and written to
    ``dest / "manifest.jsonl"`` with ``write_manifest``. With a relative
    ``dest`` (for example ``data/hub``) the paths stay repository-relative.

    Args:
        split: Split name on the Hub.
        repo_id: Source dataset repository, ``owner/name``.
        dest: Local directory to populate.
        revision: Branch, tag or commit to download; the default branch when omitted.
        settings: Runtime settings for ``hf_token``; ``get_settings()`` when omitted.

    Returns:
        The path of the rebuilt manifest.

    Raises:
        ValueError: When ``split`` is not a valid folder name.
        FileNotFoundError: When the split, or a file its manifest names, is missing
            from the download.
    """
    from huggingface_hub import snapshot_download

    _check_split_name(split)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"{split}/*", f"{split}/**"],
        local_dir=dest,
        token=_token(settings),
    )
    remote_manifest = dest / split / MANIFEST_NAME
    if not remote_manifest.is_file():
        raise FileNotFoundError(f"split {split!r} not found in {repo_id} (no {remote_manifest})")
    rows: list[ManifestRow] = []
    for row in read_manifest(remote_manifest):
        image, gt = dest / row.image, dest / row.gt
        for path in (image, gt):
            if not path.is_file():
                raise FileNotFoundError(f"scene {row.scene_id!r}: {path} missing from download")
        rows.append(row.model_copy(update={"image": image.as_posix(), "gt": gt.as_posix()}))
    local_manifest = dest / MANIFEST_NAME
    write_manifest(rows, local_manifest)
    return local_manifest
