"""Versioned prompts shipped as package data and loaded with ``importlib.resources``."""

from __future__ import annotations

from importlib.resources import files
from typing import Final

PROMPT_NAMES: Final[tuple[str, ...]] = (
    "single_shot",
    "agentic/planner",
    "agentic/reader",
    "agentic/interpreter",
    "agentic/geometer",
    "agentic/builder",
    "agentic/critic",
)
"""Every prompt in the package, as ``<name>`` of ``prompts/<name>.md``."""


def load_prompt(name: str) -> str:
    """Return the text of a packaged prompt.

    Args:
        name: One of :data:`PROMPT_NAMES`.

    Returns:
        The Markdown prompt.

    Raises:
        ValueError: If ``name`` is not a known prompt.
    """
    if name not in PROMPT_NAMES:
        msg = f"unknown prompt {name!r}; known: {list(PROMPT_NAMES)}"
        raise ValueError(msg)
    return files("rail_vision_bench").joinpath("prompts", f"{name}.md").read_text(encoding="utf-8")
