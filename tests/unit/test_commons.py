"""Licence allow-list, source records, attribution and the Commons ingest (mocked HTTP)."""

from __future__ import annotations

import hashlib
import io
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image
from typer.testing import CliRunner

from rail_vision_bench.cli import app
from rail_vision_bench.dataset.licensing import (
    ATTRIBUTION_LICENCE_URL,
    LicenceClass,
    classify_licence,
    is_allowed,
    spdx_id,
)
from rail_vision_bench.dataset.manifest import ManifestRow
from rail_vision_bench.dataset.sources import (
    CurationStatus,
    SourceRecord,
    attribution_markdown,
    read_sources,
    write_sources,
)
from rail_vision_bench.ingest import commons
from rail_vision_bench.schema.models import SourceKind
from tests.conftest import plain


@pytest.mark.parametrize(
    ("name", "family", "spdx"),
    [
        ("CC BY 4.0", LicenceClass.CC_BY, "CC-BY-4.0"),
        ("CC BY 2.0", LicenceClass.CC_BY, "CC-BY-2.0"),
        ("CC BY 3.0 de", LicenceClass.CC_BY, "CC-BY-3.0-DE"),
        ("CC0", LicenceClass.CC0, "CC0-1.0"),
        ("CC0 1.0", LicenceClass.CC0, "CC0-1.0"),
        ("Public domain", LicenceClass.PUBLIC_DOMAIN, "LicenseRef-PublicDomain"),
        ("PD-old", LicenceClass.PUBLIC_DOMAIN, "LicenseRef-PublicDomain"),
        ("Attribution", LicenceClass.ATTRIBUTION, "LicenseRef-Commons-Attribution"),
        ("CC BY-SA 4.0", LicenceClass.CC_BY_SA, None),
        ("CC BY-SA 2.5", LicenceClass.CC_BY_SA, None),
        ("CC BY-NC 4.0", LicenceClass.OTHER, None),
        ("GFDL", LicenceClass.OTHER, None),
        (None, LicenceClass.OTHER, None),
    ],
)
def test_licence_allow_list(name: str | None, family: LicenceClass, spdx: str | None):
    assert classify_licence(name) is family
    assert spdx_id(name) == spdx
    assert is_allowed(name) is (spdx is not None)


def _record(source_id: str, **overrides: Any) -> SourceRecord:
    fields: dict[str, Any] = {
        "source_id": source_id,
        "image": f"data/raw/commons/images/{source_id}.jpg",
        "sha256": "0" * 64,
        "width": 800,
        "height": 600,
        "title": "File:Panel (1).jpg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Panel_(1).jpg",
        "author": "A. Person",
        "credit": "Own work",
        "license": "CC-BY-4.0",
        "license_name": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0",
        "ingested_at": "2026-10-06T00:00:00+00:00",
    }
    fields.update(overrides)
    return SourceRecord.model_validate(fields)


def test_sources_round_trip_is_sorted(tmp_path: Path):
    path = tmp_path / "sources.jsonl"
    assert read_sources(path) == []
    write_sources([_record("b"), _record("a")], path)
    assert [record.source_id for record in read_sources(path)] == ["a", "b"]


def test_source_record_rejects_bad_hash_and_extra():
    with pytest.raises(ValueError, match="sha256"):
        _record("a", sha256="xyz")
    with pytest.raises(ValueError, match="extra"):
        _record("a", bogus=1)


def test_attribution_lists_only_accepted_and_escapes():
    records = [
        _record("a", status="accepted", author="Jo *Bold*", credit="Archiv X"),
        _record("b", status="rejected"),
        _record("c", status="accepted", license_url=None, license_name="Public domain"),
    ]
    text = attribution_markdown(records)
    assert "2 image(s)." in text
    assert "`b`" not in text
    assert r"Jo \*Bold\*" in text
    assert "[Panel (1).jpg](<https://commons.wikimedia.org/wiki/File:Panel_(1).jpg>)" in text
    assert "Credit: Archiv X." in text
    assert "Public domain." in text
    assert text.endswith("\n")


def test_manifest_row_carries_provenance():
    row = ManifestRow(
        scene_id="commons-0123456789ab",
        split="panel_photo_v1",
        partition="dev",
        source_kind=SourceKind.PANEL_PHOTO,
        image="data/raw/commons/images/commons-0123456789ab.jpg",
        gt="data/gt/commons-0123456789ab.json",
        width=800,
        height=600,
        license="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0",
        author="A. Person",
        source_url="https://commons.wikimedia.org/wiki/File:X.jpg",
    )
    assert row.author == "A. Person"


def _jpeg(width: int = 64, height: int = 48, color: tuple[int, int, int] = (40, 40, 40)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="JPEG")
    return buffer.getvalue()


_IMAGE = _jpeg()


def _page(title: str, licence: str, **info: Any) -> dict[str, Any]:
    image_info: dict[str, Any] = {
        "mime": "image/jpeg",
        "width": 3000,
        "height": 2000,
        "thumbwidth": 2000,
        "thumbheight": 1333,
        "url": f"https://upload.example/{title}.jpg",
        "thumburl": f"https://upload.example/thumb/{title}.jpg",
        "descriptionurl": f"https://commons.wikimedia.org/wiki/{title}",
        "extmetadata": {
            "LicenseShortName": {"value": licence},
            "Artist": {"value": '<a href="//x">Kim <b>Lee</b></a>'},
            "Credit": {"value": "Own work"},
        },
    }
    image_info.update(info)
    return {"title": title, "imageinfo": [image_info]}


def _transport(pages: list[dict[str, Any]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "commons.wikimedia.org":
            return httpx.Response(
                200, json={"query": {"pages": {str(i): p for i, p in enumerate(pages)}}}
            )
        if "missing" in request.url.path:
            return httpx.Response(404)
        if "notanimage" in request.url.path:
            return httpx.Response(200, content=b"<html>error page</html>")
        return httpx.Response(200, content=_IMAGE)

    return httpx.MockTransport(handler)


def test_ingest_filters_licences_and_records_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    pages = [
        _page("File:Good.jpg", "CC BY 4.0"),
        _page("File:ShareAlike.jpg", "CC BY-SA 4.0"),
        _page("File:Tiny.jpg", "CC0", width=100, height=100),
        _page("File:Anim.gif", "CC0", mime="image/gif"),
        _page("File:SameBytes.jpg", "Public domain"),
        _page("File:Gone.jpg", "CC0", thumburl="https://upload.example/missing.jpg"),
    ]
    dest = Path("data/raw/commons")
    with httpx.Client(transport=_transport(pages)) as client:
        counts = commons.ingest(dest, terms=["panel"], client=client, delay_s=0)
    assert counts == commons.IngestCounts(
        seen=6, new=1, duplicate=1, licence_rejected=1, unsuitable=2, failed=1
    )
    (record,) = read_sources(dest / "sources.jsonl")
    sha = hashlib.sha256(_IMAGE).hexdigest()
    assert record.source_id == f"commons-{sha[:12]}"
    assert record.status is CurationStatus.CANDIDATE
    assert record.title == "File:Good.jpg"
    assert record.author == "Kim Lee"
    assert record.license == "CC-BY-4.0"
    assert (record.width, record.height) == (64, 48)  # decoded, not the reported thumb size
    assert Path(record.image).read_bytes() == _IMAGE
    assert record.image == f"data/raw/commons/images/commons-{sha[:12]}.jpg"

    # a second run sees the same titles and keeps the file as it is
    with httpx.Client(transport=_transport(pages[:1])) as client:
        again = commons.ingest(dest, terms=["panel"], client=client, delay_s=0)
    assert (again.new, again.duplicate) == (0, 1)


def _make_db(tmp_path: Path, rows: list[tuple[str, str, bytes]]) -> Path:
    db_path = tmp_path / "pv" / "panelvision.db"
    (db_path.parent / "images").mkdir(parents=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE images (id INTEGER PRIMARY KEY, commons_title TEXT, source_url TEXT,
           file_url TEXT, author TEXT, license TEXT, license_url TEXT, credit TEXT,
           description TEXT, search_term TEXT, width INTEGER, height INTEGER, sha256 TEXT,
           path TEXT, status TEXT, ingested_at TEXT)"""
    )
    for i, (licence, status, body) in enumerate(rows):
        rel = f"images/{i}.JPG"
        (db_path.parent / rel).write_bytes(body)
        conn.execute(
            "INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                i,
                f"File:{i}.jpg",
                f"https://commons.wikimedia.org/wiki/File:{i}.jpg",
                None,
                "Kim",
                licence,
                None,
                "Own work",
                None,
                "panel",
                800,
                600,
                hashlib.sha256(body).hexdigest(),
                rel,
                status,
                "2026-07-02T00:00:00+00:00",
            ),
        )
    conn.commit()
    conn.close()
    return db_path


def test_import_panelvision_db_takes_accepted_allow_listed_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db_path = _make_db(
        tmp_path,
        [
            ("CC BY 4.0", "accepted", b"one"),
            ("CC BY-SA 4.0", "accepted", b"two"),
            ("Public domain", "rejected", b"three"),
            ("Attribution", "accepted", b"four"),
        ],
    )
    monkeypatch.chdir(tmp_path)
    records = commons.import_panelvision_db(db_path, Path("out"))
    assert sorted(r.license_name for r in records) == ["Attribution", "CC BY 4.0"]
    attribution = next(r for r in records if r.license_name == "Attribution")
    assert attribution.license_url == ATTRIBUTION_LICENCE_URL
    assert all(r.status is CurationStatus.ACCEPTED for r in records)
    assert all(Path(r.image).suffix == ".jpg" and Path(r.image).is_file() for r in records)
    assert (db_path.parent / "images" / "0.JPG").is_file()  # copied, not moved
    assert read_sources(Path("out/sources.jsonl")) == records


def test_import_panelvision_db_rejects_hash_mismatch(tmp_path: Path):
    db_path = _make_db(tmp_path, [("CC0", "accepted", b"one")])
    (db_path.parent / "images" / "0.JPG").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match"):
        commons.import_panelvision_db(db_path, tmp_path / "out")


def test_cli_commons_status_and_attribution(tmp_path: Path, cli: CliRunner):
    dest = tmp_path / "commons"
    write_sources([_record("a")], dest / "sources.jsonl")

    result = cli.invoke(app, ["commons", "status", "a", "accepted", "--dest", str(dest)])
    assert result.exit_code == 0, result.output
    assert read_sources(dest / "sources.jsonl")[0].status is CurationStatus.ACCEPTED

    assert (
        cli.invoke(app, ["commons", "status", "zz", "accepted", "--dest", str(dest)]).exit_code == 1
    )
    assert cli.invoke(app, ["commons", "status", "a", "maybe", "--dest", str(dest)]).exit_code == 2

    out = tmp_path / "ATTRIBUTION.md"
    result = cli.invoke(app, ["commons", "attribution", "--dest", str(dest), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "`a`" in out.read_text(encoding="utf-8")

    result = cli.invoke(app, ["commons", "attribution", "--dest", str(dest)])
    assert "1 image(s)." in plain(result.output)


def test_cli_commons_import_db(tmp_path: Path, cli: CliRunner):
    db_path = _make_db(tmp_path, [("CC0", "accepted", b"one")])
    dest = tmp_path / "out"
    result = cli.invoke(app, ["commons", "import-db", str(db_path), "--dest", str(dest)])
    assert result.exit_code == 0, result.output
    assert "imported 1 image(s)" in plain(result.output)


def test_ingest_skips_titles_known_elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    write_sources([_record("a", title="File:Good.jpg")], Path("raw/sources.jsonl"))
    with httpx.Client(transport=_transport([_page("File:Good.jpg", "CC0")])) as client:
        counts = commons.ingest(
            Path("staging"), terms=["panel"], client=client, delay_s=0, known=[Path("raw")]
        )
    assert (counts.new, counts.duplicate) == (0, 1)


def test_promote_copies_accepted_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    staging = Path("staging")
    for name in ("a", "b"):
        (staging / "images").mkdir(parents=True, exist_ok=True)
        (staging / "images" / f"{name}.jpg").write_bytes(name.encode())
    write_sources(
        [
            _record("a", status="accepted", image="staging/images/a.jpg"),
            _record("b", image="staging/images/b.jpg"),
        ],
        staging / "sources.jsonl",
    )
    (promoted,) = commons.promote(staging, Path("raw"))
    assert promoted.image == "raw/images/a.jpg"
    assert Path("raw/images/a.jpg").read_bytes() == b"a"
    assert not Path("raw/images/b.jpg").exists()
    assert read_sources(Path("raw/sources.jsonl")) == [promoted]
    assert Path("staging/images/a.jpg").is_file()


def test_cli_commons_promote(tmp_path: Path, cli: CliRunner):
    staging, dest = tmp_path / "staging", tmp_path / "raw"
    write_sources([], staging / "sources.jsonl")
    result = cli.invoke(app, ["commons", "promote", "--staging", str(staging), "--dest", str(dest)])
    assert result.exit_code == 0, result.output
    assert "promoted 0 image(s)" in plain(result.output)


def test_ingest_extension_comes_from_the_bytes_not_the_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # regression: Commons thumb URLs can end in utm query parameters, which once became the
    # file "extension" (commons-....orgutmcampaignimageinfoutmcontentthumbnailunscaled)
    monkeypatch.chdir(tmp_path)
    utm = (
        "https://upload.example/thumb/a.jpg/2000px-a.jpg?utm_source=commons.wikimedia.org"
        "&utm_campaign=imageinfo&utm_content=thumbnail_unscaled"
    )
    pages = [
        _page("File:Utm.jpg", "CC0", thumburl=utm),
        _page("File:Html.jpg", "CC0", thumburl="https://upload.example/notanimage.jpg"),
    ]
    with httpx.Client(transport=_transport(pages)) as client:
        counts = commons.ingest(Path("s"), terms=["panel"], client=client, delay_s=0)
    assert (counts.new, counts.unsuitable) == (1, 1)
    (record,) = read_sources(Path("s/sources.jsonl"))
    assert Path(record.image).suffix == ".jpg"


def test_sniff_image():
    assert commons.sniff_image(_IMAGE) == ("jpg", 64, 48)
    buffer = io.BytesIO()
    Image.new("RGB", (5, 7)).save(buffer, format="PNG")
    assert commons.sniff_image(buffer.getvalue()) == ("png", 5, 7)
    assert commons.sniff_image(b"not an image") is None


def _gray(image: Image.Image, xy: tuple[int, int]) -> int:
    value = image.getpixel(xy)
    assert isinstance(value, float | int)
    return int(value)


def _staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    staging = Path("staging")
    (staging / "images").mkdir(parents=True)
    (staging / "images" / "a.jpg").write_bytes(_jpeg(200, 100, (250, 250, 250)))
    write_sources([_record("a", image="staging/images/a.jpg")], staging / "sources.jsonl")
    return staging


def test_blur_changes_only_the_boxes_and_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    staging = _staged(tmp_path, monkeypatch)
    image_path = staging / "images" / "a.jpg"
    with Image.open(image_path) as image:
        drawn = image.convert("RGB")
    for x in range(20, 60, 4):  # a high-contrast pattern inside the box
        for y in range(20, 60):
            drawn.putpixel((x, y), (0, 0, 0))
    drawn.save(image_path, format="JPEG", quality=95)

    record = commons.blur(staging, "a", [(10, 10, 70, 70), (-5, 90, 20, 200)])
    assert record.modification == commons.BLUR_MODIFICATION
    assert record.blur_regions == [(10, 10, 70, 70), (0, 90, 20, 100)]  # clipped
    assert record.sha256 == hashlib.sha256(image_path.read_bytes()).hexdigest()
    with Image.open(image_path) as image:
        assert image.format == "JPEG"
        gray = image.convert("L")
        stripe = [_gray(gray, (x, 40)) for x in range(20, 60)]
        assert max(stripe) - min(stripe) < 60  # the stripes are smeared out
        assert _gray(gray, (150, 50)) > 240  # outside every box: untouched

    with pytest.raises(ValueError, match="empty"):
        commons.blur(staging, "a", [(500, 500, 600, 600)])
    with pytest.raises(KeyError):
        commons.blur(staging, "zz", [(0, 0, 1, 1)])

    text = attribution_markdown([record.model_copy(update={"status": CurationStatus.ACCEPTED})])
    assert "Modified (faces blurred)." in text


def test_status_note_group_and_promote_withdrawal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    staging = _staged(tmp_path, monkeypatch)
    commons.set_status(staging, "a", CurationStatus.ACCEPTED, note="clear panel")
    commons.set_group(staging, "a", "nl-hilversum")
    (promoted,) = commons.promote(staging, Path("raw"))
    assert (promoted.curation_note, promoted.panel_group) == ("clear panel", "nl-hilversum")
    assert Path(promoted.image).is_file()

    record = commons.set_status(staging, "a", CurationStatus.REJECTED, note="near-duplicate")
    assert record.curation_note == "near-duplicate"
    assert commons.set_status(staging, "a", CurationStatus.REJECTED).curation_note == (
        "near-duplicate"
    )
    assert commons.promote(staging, Path("raw")) == []
    assert read_sources(Path("raw/sources.jsonl")) == []
    assert not Path(promoted.image).exists()
    with pytest.raises(ValueError, match="panel_group"):
        commons.set_group(staging, "a", "bad group!")


def test_cli_commons_group_blur_and_note(tmp_path: Path, cli: CliRunner):
    staging = tmp_path / "staging"
    (staging / "images").mkdir(parents=True)
    image = staging / "images" / "a.jpg"
    image.write_bytes(_jpeg(100, 100))
    write_sources([_record("a", image=str(image))], staging / "sources.jsonl")
    d = ["--dest", str(staging)]

    result = cli.invoke(app, ["commons", "status", "a", "rejected", "--note", "dup", *d])
    assert result.exit_code == 0, result.output
    assert read_sources(staging / "sources.jsonl")[0].curation_note == "dup"

    assert cli.invoke(app, ["commons", "group", "a", "nl-x", *d]).exit_code == 0
    assert cli.invoke(app, ["commons", "group", "a", "bad id!", *d]).exit_code == 2
    assert cli.invoke(app, ["commons", "group", "zz", "nl-x", *d]).exit_code == 1

    result = cli.invoke(app, ["commons", "blur", "a", "--box", "0,0,10,10", *d])
    assert result.exit_code == 0, result.output
    assert read_sources(staging / "sources.jsonl")[0].blur_regions == [(0, 0, 10, 10)]
    assert cli.invoke(app, ["commons", "blur", "a", "--box", "1,2,3", *d]).exit_code == 2
    assert cli.invoke(app, ["commons", "blur", "a", "--box", "a,b,c,d", *d]).exit_code == 2
    assert cli.invoke(app, ["commons", "blur", "a", "--box", "500,500,600,600", *d]).exit_code == 1
