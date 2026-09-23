"""Hub push/pull with ``huggingface_hub`` replaced by in-process fakes (no network)."""

from __future__ import annotations

import fnmatch
import shutil
from pathlib import Path
from typing import Any, ClassVar

import huggingface_hub
import orjson
import pytest
from pydantic import SecretStr

from rail_vision_bench.dataset.hub import (
    HubReleaseError,
    check_release,
    pull_split,
    push_split,
)
from rail_vision_bench.dataset.manifest import ManifestRow, read_manifest, write_manifest
from rail_vision_bench.schema.models import SourceKind
from rail_vision_bench.settings import Settings
from tests.conftest import EXAMPLES_DIR, REPO_ROOT

EXAMPLE_MANIFEST = REPO_ROOT / "data" / "manifest.example.jsonl"
SETTINGS = Settings(_env_file=None, hf_token=SecretStr("hf_secret"))


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    """A repository-shaped tree holding the example manifest, its ground truth and fake PNGs."""
    root = tmp_path / "repo"
    rows = read_manifest(EXAMPLE_MANIFEST)
    for row in rows:
        (root / row.gt).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(EXAMPLES_DIR / Path(row.gt).name, root / row.gt)
        (root / row.image).parent.mkdir(parents=True, exist_ok=True)
        (root / row.image).write_bytes(b"\x89PNG " + row.scene_id.encode())
    write_manifest(rows, root / "manifest.jsonl")
    return root


class FakeCommit:
    commit_url = "https://huggingface.co/datasets/o/n/commit/abc123"


class FakeHfApi:
    """Records calls and snapshots the uploaded folder before the staging dir disappears."""

    instances: ClassVar[list[FakeHfApi]] = []

    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.uploaded: dict[str, bytes] = {}
        FakeHfApi.instances.append(self)

    def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        self.calls.append(("create_repo", {"repo_id": repo_id, **kwargs}))

    def upload_folder(self, **kwargs: Any) -> FakeCommit:
        self.calls.append(("upload_folder", kwargs))
        folder = Path(kwargs["folder_path"])
        self.uploaded = {
            f"{kwargs['path_in_repo']}/{p.relative_to(folder).as_posix()}": p.read_bytes()
            for p in sorted(folder.rglob("*"))
            if p.is_file()
        }
        return FakeCommit()


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> type[FakeHfApi]:
    FakeHfApi.instances = []
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    return FakeHfApi


def test_example_manifest_is_releasable(repo_root: Path):
    rows = read_manifest(repo_root / "manifest.jsonl")
    assert check_release(rows, "synthetic_clean", root=repo_root) == []


def test_push_split_stages_layout_and_uploads(repo_root: Path, fake_api: type[FakeHfApi]):
    url = push_split(
        "synthetic_clean",
        "o/n",
        manifest=repo_root / "manifest.jsonl",
        root=repo_root,
        settings=SETTINGS,
    )
    assert url == FakeCommit.commit_url
    (api,) = fake_api.instances
    assert api.token == "hf_secret"
    assert api.calls[0] == (
        "create_repo",
        {"repo_id": "o/n", "repo_type": "dataset", "private": True, "exist_ok": True},
    )
    upload = api.calls[1][1]
    assert upload["repo_id"] == "o/n"
    assert upload["repo_type"] == "dataset"
    assert upload["path_in_repo"] == "synthetic_clean"
    assert sorted(api.uploaded) == [
        "synthetic_clean/gt/minimal.json",
        "synthetic_clean/gt/station_dkw.json",
        "synthetic_clean/images/minimal.png",
        "synthetic_clean/images/station_dkw.png",
        "synthetic_clean/manifest.jsonl",
    ]
    assert api.uploaded["synthetic_clean/images/minimal.png"] == b"\x89PNG minimal"
    remote_rows = [
        ManifestRow.model_validate(orjson.loads(line))
        for line in api.uploaded["synthetic_clean/manifest.jsonl"].splitlines()
    ]
    assert [(r.image, r.gt) for r in remote_rows] == [
        ("synthetic_clean/images/minimal.png", "synthetic_clean/gt/minimal.json"),
        ("synthetic_clean/images/station_dkw.png", "synthetic_clean/gt/station_dkw.json"),
    ]


def test_push_split_refuses_unreleasable_manifest(repo_root: Path, fake_api: type[FakeHfApi]):
    rows = read_manifest(repo_root / "manifest.jsonl")
    bad = [
        rows[0].model_copy(update={"source_kind": SourceKind.PANEL_PHOTO}),
        rows[1].model_copy(update={"split": "other", "image": "missing.png"}),
        rows[1],
    ]
    write_manifest(bad, repo_root / "bad.jsonl")
    with pytest.raises(HubReleaseError) as info:
        push_split("synthetic_clean", "o/n", manifest=repo_root / "bad.jsonl", root=repo_root)
    message = str(info.value)
    assert "license is null" in message
    assert "split is 'other'" in message
    assert "missing.png not found" in message
    assert "duplicate scene_id" in message
    assert fake_api.instances == []


def test_check_release_rejects_mismatched_ground_truth(repo_root: Path):
    rows = read_manifest(repo_root / "manifest.jsonl")
    wrong = [rows[0].model_copy(update={"gt": rows[1].gt, "width": 1})]
    problems = check_release(wrong, "synthetic_clean", root=repo_root)
    assert any("scene_id is 'station_dkw'" in p for p in problems)
    assert any("size differs" in p for p in problems)
    (repo_root / "broken.json").write_text("{}", encoding="utf-8")
    broken = [rows[0].model_copy(update={"gt": "broken.json"})]
    assert any("does not parse" in p for p in check_release(broken, "s", root=repo_root))
    assert check_release([], "s", root=repo_root) == ["the manifest has no rows"]


@pytest.mark.parametrize("split", ["", "../x", "a/b", ".hidden"])
def test_invalid_split_names(split: str, tmp_path: Path):
    with pytest.raises(ValueError, match="invalid split name"):
        push_split(split, "o/n", manifest=tmp_path / "m.jsonl")
    with pytest.raises(ValueError, match="invalid split name"):
        pull_split(split, "o/n", dest=tmp_path)


def test_push_then_pull_round_trip(
    repo_root: Path, tmp_path: Path, fake_api: type[FakeHfApi], monkeypatch: pytest.MonkeyPatch
):
    push_split("synthetic_clean", "o/n", manifest=repo_root / "manifest.jsonl", root=repo_root)
    remote = dict(fake_api.instances[0].uploaded)
    remote["other_split/manifest.jsonl"] = b""
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        local = Path(kwargs["local_dir"])
        for name, content in remote.items():
            if any(fnmatch.fnmatch(name, pattern) for pattern in kwargs["allow_patterns"]):
                (local / name).parent.mkdir(parents=True, exist_ok=True)
                (local / name).write_bytes(content)
        return str(local)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    dest = tmp_path / "pulled"
    manifest = pull_split("synthetic_clean", "o/n", dest=dest, revision="v1", settings=SETTINGS)
    assert manifest == dest / "manifest.jsonl"
    assert calls[0]["repo_id"] == "o/n"
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["revision"] == "v1"
    assert calls[0]["token"] == "hf_secret"
    assert not (dest / "other_split").exists()

    original = read_manifest(repo_root / "manifest.jsonl")
    pulled = read_manifest(manifest)
    assert [r.scene_id for r in pulled] == [r.scene_id for r in original]
    for before, after in zip(original, pulled, strict=True):
        assert Path(after.image).read_bytes() == (repo_root / before.image).read_bytes()
        assert Path(after.gt).read_bytes() == (repo_root / before.gt).read_bytes()
        assert after.model_dump(exclude={"image", "gt"}) == before.model_dump(
            exclude={"image", "gt"}
        )
    assert check_release(pulled, "synthetic_clean", root=Path("/")) == []


def test_pull_split_missing_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: kwargs["local_dir"])
    with pytest.raises(FileNotFoundError, match="not found in o/n"):
        pull_split("nope", "o/n", dest=tmp_path / "d", settings=Settings(_env_file=None))
