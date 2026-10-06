"""The image-licence allow-list for third-party imagery.

The benchmark cuts tiles and crops out of every image, and a crop is an adaptation. Under
CC BY-SA an adaptation has to be shared alike, which would bind the derived data to BY-SA, so
the main set admits only licences whose adaptations carry no share-alike or non-commercial
condition: CC BY (any version or port), CC0, public domain, and the Wikimedia Commons
``Attribution`` licence (any use, attribution only). Everything else is rejected before
download.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Final


class LicenceClass(StrEnum):
    """The family a licence belongs to, as far as the allow-list cares."""

    CC_BY = "cc_by"
    CC0 = "cc0"
    PUBLIC_DOMAIN = "public_domain"
    ATTRIBUTION = "attribution"
    CC_BY_SA = "cc_by_sa"
    OTHER = "other"


ALLOWED_LICENCES: Final[frozenset[LicenceClass]] = frozenset(
    {LicenceClass.CC_BY, LicenceClass.CC0, LicenceClass.PUBLIC_DOMAIN, LicenceClass.ATTRIBUTION}
)
"""The licence families admitted to the main set."""

ATTRIBUTION_LICENCE_URL: Final = "https://commons.wikimedia.org/wiki/Template:Attribution"
"""Terms of the Commons ``Attribution`` licence, which has no Creative Commons URL."""

_CC_BY_RE: Final = re.compile(r"^CC[ -]BY[ -](\d\.\d)(?:[ -]([a-z]{2,3}))?$", re.IGNORECASE)
_CC_BY_SA_RE: Final = re.compile(r"^CC[ -]BY[ -]SA\b", re.IGNORECASE)
_CC0_RE: Final = re.compile(r"^CC0(?:[ -]1\.0)?$", re.IGNORECASE)
_PD_RE: Final = re.compile(r"^(?:public domain|pd(?:[ -].*)?)$", re.IGNORECASE)


def classify_licence(name: str | None) -> LicenceClass:
    """Map a licence short name (as Wikimedia Commons reports it) to its family.

    Args:
        name: The short name, e.g. ``CC BY 4.0``, ``CC BY-SA 3.0 de``, ``CC0``,
            ``Public domain`` or ``Attribution``; ``None`` when the source has none.

    Returns:
        The family; an unknown or missing name is ``OTHER``.
    """
    if name is None:
        return LicenceClass.OTHER
    text = name.strip()
    if _CC_BY_RE.match(text):
        return LicenceClass.CC_BY
    if _CC_BY_SA_RE.match(text):
        return LicenceClass.CC_BY_SA
    if _CC0_RE.match(text):
        return LicenceClass.CC0
    if _PD_RE.match(text):
        return LicenceClass.PUBLIC_DOMAIN
    if text.lower() == "attribution":
        return LicenceClass.ATTRIBUTION
    return LicenceClass.OTHER


def is_allowed(name: str | None) -> bool:
    """Return whether a licence short name is on the allow-list."""
    return classify_licence(name) in ALLOWED_LICENCES


def spdx_id(name: str | None) -> str | None:
    """Return the SPDX identifier for an allowed licence short name.

    Ported CC BY licences keep their jurisdiction (``CC BY 3.0 de`` → ``CC-BY-3.0-DE``).
    Public domain and the Commons ``Attribution`` licence have no SPDX id and get a
    ``LicenseRef-`` identifier.

    Args:
        name: The licence short name.

    Returns:
        The identifier, or ``None`` when the licence is not on the allow-list.
    """
    family = classify_licence(name)
    if family is LicenceClass.CC0:
        return "CC0-1.0"
    if family is LicenceClass.PUBLIC_DOMAIN:
        return "LicenseRef-PublicDomain"
    if family is LicenceClass.ATTRIBUTION:
        return "LicenseRef-Commons-Attribution"
    if family is LicenceClass.CC_BY and name is not None:
        match = _CC_BY_RE.match(name.strip())
        if match is not None:
            version, port = match.groups()
            return f"CC-BY-{version}" + (f"-{port.upper()}" if port else "")
    return None
