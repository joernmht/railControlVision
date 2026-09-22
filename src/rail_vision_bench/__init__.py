"""rail-vision-bench: benchmark generative AI on railway-control imagery.

Only the package version and the schema version live here so that importing
the package (and therefore ``bench --help``) stays fast.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Final

try:
    __version__: str = version("rail-vision-bench")
except PackageNotFoundError:  # running from a source tree without an installed distribution
    __version__ = "0.0.0"

SCHEMA_VERSION: Final[str] = "v0"

__all__ = ["SCHEMA_VERSION", "__version__"]
