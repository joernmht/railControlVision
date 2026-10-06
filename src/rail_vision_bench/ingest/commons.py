"""Collect railway-control panel photographs from Wikimedia Commons, with provenance.

Commons is the one large web source where every file carries machine-readable licensing and
attribution. For each hit the licence is checked against the allow-list
(:mod:`rail_vision_bench.dataset.licensing`) *before* download, so share-alike and
non-commercial files are never fetched.

Downloads land in a staging directory (default ``data/incoming/commons``, gitignored and not
tracked by DVC) as ``images/<source_id>.<ext>`` plus a :class:`SourceRecord` with status
``candidate`` in ``sources.jsonl``. A human accepts or rejects each one
(``bench commons status``); :func:`promote` then copies the accepted ones into the DVC tree
(default ``data/raw/commons``). Staging keeps unreviewed images, including ones showing people
that would first need blurring under the data policy, out of the data plane.

Ported from the raiLPoperator ``panelvision`` prototype; :func:`import_panelvision_db`
migrates that prototype's accepted images.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import sqlite3
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import httpx

from rail_vision_bench import __version__
from rail_vision_bench.dataset.licensing import (
    ATTRIBUTION_LICENCE_URL,
    LicenceClass,
    classify_licence,
    spdx_id,
)
from rail_vision_bench.dataset.sources import (
    CurationStatus,
    SourceRecord,
    read_sources,
    write_sources,
)

API_URL: Final = "https://commons.wikimedia.org/w/api.php"
USER_AGENT: Final = (
    f"rail-vision-bench/{__version__} "
    "(research benchmark; https://github.com/joernmht/railControlVision)"
)

SEARCH_TERMS: Final[tuple[str, ...]] = (
    # German: relay and push-button interlockings and their panels
    "Gleisbildstellwerk",
    "Drucktastenstellwerk",
    "Spurplanstellwerk",
    "Stelltisch Stellwerk",
    "Stellwerk Bedienpult",
    "Meldetafel Stellwerk",
    # English
    "signal box control panel",
    "railway signalling control panel",
    "NX panel signalling",
    "entrance exit panel signalling",
    "CTC panel railroad",
    "railway control panel mimic",
    "train describer panel",
    # Dutch, Polish, Czech
    "NX-tableau seinhuis",
    "bedieningstableau seinhuis",
    "pulpit nastawczy",
    "nastawnia przekaźnikowa",
    "reléové zabezpečovací zařízení",
    "stavědlo ovládací pult",
)
"""Multilingual search terms for track-diagram panels and control desks."""

MIN_PIXELS: Final = 500 * 400
"""Files below this area are thumbnails or icons and are skipped."""

THUMB_WIDTH: Final = 2000
"""Commons renders a thumbnail at most this wide instead of the (often huge) original."""

_EXT_METADATA: Final = "LicenseShortName|LicenseUrl|Artist|Credit|ImageDescription"
_TAG_RE: Final = re.compile(r"<[^>]+>")
_UNSAFE_RE: Final = re.compile(r"[^a-z0-9]+")


@dataclass
class IngestCounts:
    """What one ingest run did with the search hits."""

    seen: int = 0
    new: int = 0
    duplicate: int = 0
    licence_rejected: int = 0
    unsuitable: int = 0
    failed: int = 0


def utcnow() -> str:
    """Return the current UTC time as an ISO 8601 string with seconds."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def source_id_for(sha256: str) -> str:
    """Return the stable source id of an image from its content hash."""
    return f"commons-{sha256[:12]}"


def _plain(value: Any) -> str | None:
    """Strip the HTML that Commons ``extmetadata`` values carry; empty becomes ``None``."""
    if value is None:
        return None
    text = " ".join(_TAG_RE.sub("", str(value)).split())
    return text.replace("&amp;", "&").replace("&quot;", '"').replace("&#039;", "'") or None


def _meta(extmeta: Mapping[str, Any], key: str) -> str | None:
    """Return one ``extmetadata`` field as plain text."""
    entry = extmeta.get(key)
    return _plain(entry.get("value")) if isinstance(entry, Mapping) else None


def licence_url_for(name: str | None, stated: str | None) -> str | None:
    """Return the licence URL to record: the stated one, else a known one for the family."""
    if stated:
        return stated
    if classify_licence(name) is LicenceClass.ATTRIBUTION:
        return ATTRIBUTION_LICENCE_URL
    return None


def search_files(client: httpx.Client, term: str, limit: int) -> list[dict[str, Any]]:
    """Search the Commons File namespace and return the pages that have image info.

    Args:
        client: An HTTP client (its User-Agent is sent to Commons).
        term: The full-text search query.
        limit: Maximum number of hits (Commons caps it at 50 for anonymous clients).

    Returns:
        The ``query.pages`` values that carry an ``imageinfo`` list, in title order.
    """
    response = client.get(
        API_URL,
        params={
            "format": "json",
            "action": "query",
            "generator": "search",
            "gsrsearch": term,
            "gsrnamespace": 6,
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": THUMB_WIDTH,
            "iiextmetadatafilter": _EXT_METADATA,
        },
        timeout=30,
    )
    response.raise_for_status()
    pages: dict[str, dict[str, Any]] = response.json().get("query", {}).get("pages", {})
    return sorted(
        (page for page in pages.values() if page.get("imageinfo")),
        key=lambda page: str(page.get("title", "")),
    )


def ingest(
    dest: Path,
    *,
    terms: Iterable[str] = SEARCH_TERMS,
    per_term: int = 25,
    client: httpx.Client | None = None,
    delay_s: float = 0.3,
    known: Iterable[Path] = (),
) -> IngestCounts:
    """Search Commons and download new, licence-compatible candidate images.

    Args:
        dest: The source directory (``images/`` and ``sources.jsonl`` live under it).
            Image paths in the records are relative to the current working directory,
            which is the repository root in normal use.
        terms: The search queries.
        per_term: Hits requested per query.
        client: HTTP client to use; one with the project User-Agent is created if omitted.
        delay_s: Pause after each download, to be polite to Commons.
        known: Further source directories (e.g. the promoted set) whose titles and hashes
            count as already seen.

    Returns:
        Counters of what happened to the hits.
    """
    own_client = client is None
    http = client or httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True)
    sources_path = dest / "sources.jsonl"
    records = {record.source_id: record for record in read_sources(sources_path)}
    seen = {r.source_id: r for path in known for r in read_sources(path / "sources.jsonl")}
    titles = {record.title for record in [*records.values(), *seen.values()]}
    counts = IngestCounts()
    images_dir = dest / "images"
    try:
        for term in terms:
            for page in search_files(http, term, per_term):
                counts.seen += 1
                record = _fetch(http, page, term, images_dir, titles, {**seen, **records}, counts)
                if record is None:
                    continue
                records[record.source_id] = record
                titles.add(record.title)
                write_sources(records.values(), sources_path)
                counts.new += 1
                time.sleep(delay_s)
    finally:
        if own_client:
            http.close()
    return counts


def _fetch(
    http: httpx.Client,
    page: Mapping[str, Any],
    term: str,
    images_dir: Path,
    titles: set[str],
    records: Mapping[str, SourceRecord],
    counts: IngestCounts,
) -> SourceRecord | None:
    """Filter one search hit and download it; ``None`` (with a counter bumped) if skipped."""
    info: Mapping[str, Any] = page["imageinfo"][0]
    title = str(page["title"])
    mime = str(info.get("mime", ""))
    if title in titles:
        counts.duplicate += 1
        return None
    extmeta: Mapping[str, Any] = info.get("extmetadata", {})
    licence_name = _meta(extmeta, "LicenseShortName")
    licence = spdx_id(licence_name)
    if licence is None or licence_name is None:
        counts.licence_rejected += 1
        return None
    if (
        not mime.startswith("image/")
        or mime == "image/gif"
        or int(info.get("width", 0)) * int(info.get("height", 0)) < MIN_PIXELS
    ):
        counts.unsuitable += 1
        return None

    file_url = str(info.get("thumburl") or info["url"])
    response = http.get(file_url, timeout=60)
    if response.status_code != httpx.codes.OK:
        counts.failed += 1
        return None
    body = response.content
    sha = hashlib.sha256(body).hexdigest()
    source_id = source_id_for(sha)
    if source_id in records:
        counts.duplicate += 1  # the same bytes under another title
        return None

    suffix = file_url.rsplit(".", 1)[-1].lower()
    images_dir.mkdir(parents=True, exist_ok=True)
    local = images_dir / f"{source_id}.{_UNSAFE_RE.sub('', suffix) or 'jpg'}"
    local.write_bytes(body)
    return SourceRecord(
        source_id=source_id,
        image=local.as_posix(),
        sha256=sha,
        width=int(info.get("thumbwidth") or info["width"]),
        height=int(info.get("thumbheight") or info["height"]),
        title=title,
        source_url=str(info.get("descriptionurl", f"https://commons.wikimedia.org/wiki/{title}")),
        file_url=file_url,
        author=_meta(extmeta, "Artist"),
        credit=_meta(extmeta, "Credit"),
        license=licence,
        license_name=licence_name,
        license_url=licence_url_for(licence_name, _meta(extmeta, "LicenseUrl")),
        description=_meta(extmeta, "ImageDescription"),
        search_term=term,
        ingested_at=utcnow(),
    )


def import_panelvision_db(
    db_path: Path, dest: Path, *, statuses: Iterable[str] = ("accepted",)
) -> list[SourceRecord]:
    """Import images from the raiLPoperator ``panelvision`` SQLite prototype.

    Only rows whose licence is on the allow-list and whose curation status is in
    ``statuses`` are taken. Image files are copied (never moved), renamed to their source id
    and re-hashed; a file whose hash differs from the database is an error. Records already
    in ``<dest>/sources.jsonl`` are replaced by the imported version.

    Args:
        db_path: The prototype's ``panelvision.db``; image paths in it are relative to its
            directory.
        dest: The source directory to import into.
        statuses: Curation statuses to import.

    Returns:
        The imported records, sorted by ``source_id``.

    Raises:
        ValueError: If a copied file's sha256 does not match the database.
    """
    wanted = set(statuses)
    sources_path = dest / "sources.jsonl"
    records = {record.source_id: record for record in read_sources(sources_path)}
    imported: list[SourceRecord] = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM images ORDER BY id").fetchall()
    for row in rows:
        licence = spdx_id(row["license"])
        if licence is None or row["status"] not in wanted:
            continue
        original = db_path.parent / row["path"]
        sha = hashlib.sha256(original.read_bytes()).hexdigest()
        if sha != row["sha256"]:
            msg = f"{original}: sha256 {sha} does not match the database ({row['sha256']})"
            raise ValueError(msg)
        source_id = source_id_for(sha)
        local = dest / "images" / f"{source_id}{original.suffix.lower()}"
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, local)
        record = SourceRecord(
            source_id=source_id,
            status=CurationStatus(row["status"]),
            image=local.as_posix(),
            sha256=sha,
            width=row["width"],
            height=row["height"],
            title=row["commons_title"],
            source_url=row["source_url"],
            file_url=row["file_url"],
            author=_plain(row["author"]),
            credit=_plain(row["credit"]),
            license=licence,
            license_name=row["license"],
            license_url=licence_url_for(row["license"], row["license_url"]),
            description=_plain(row["description"]),
            search_term=row["search_term"],
            ingested_at=row["ingested_at"],
        )
        records[source_id] = record
        imported.append(record)
    write_sources(records.values(), sources_path)
    return sorted(imported, key=lambda record: record.source_id)


def set_status(dest: Path, source_id: str, status: CurationStatus) -> SourceRecord:
    """Set the curation status of one record in ``<dest>/sources.jsonl``.

    Args:
        dest: The source directory.
        source_id: The record to change.
        status: The new status.

    Returns:
        The updated record.

    Raises:
        KeyError: If no record has that id.
    """
    sources_path = dest / "sources.jsonl"
    records = {record.source_id: record for record in read_sources(sources_path)}
    if source_id not in records:
        raise KeyError(source_id)
    updated = records[source_id].model_copy(update={"status": status})
    records[source_id] = updated
    write_sources(records.values(), sources_path)
    return updated


def promote(src: Path, dest: Path) -> list[SourceRecord]:
    """Copy the accepted records of a staging directory into the data-plane directory.

    The image is copied to ``<dest>/images/`` and the record's ``image`` path rewritten;
    records already in ``<dest>/sources.jsonl`` are replaced. The staging copy stays, so a
    later ingest still knows the title.

    Args:
        src: The staging source directory.
        dest: The data-plane source directory.

    Returns:
        The promoted records, sorted by ``source_id``.
    """
    records = {record.source_id: record for record in read_sources(dest / "sources.jsonl")}
    promoted: list[SourceRecord] = []
    for record in read_sources(src / "sources.jsonl"):
        if record.status is not CurationStatus.ACCEPTED:
            continue
        local = dest / "images" / Path(record.image).name
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(record.image, local)
        moved = record.model_copy(update={"image": local.as_posix()})
        records[record.source_id] = moved
        promoted.append(moved)
    write_sources(records.values(), dest / "sources.jsonl")
    return promoted
