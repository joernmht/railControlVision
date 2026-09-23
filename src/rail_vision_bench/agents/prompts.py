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


def split_prompt(text: str) -> tuple[str, str]:
    """Split a prompt with ``## System`` and ``## User`` sections into its two turns.

    Everything under ``## System`` up to ``## User`` becomes the system prompt,
    everything after ``## User`` the user turn. A prompt without these headings
    (the agentic node specifications) is returned whole as the system prompt
    with an empty user turn.

    Args:
        text: A packaged or user-supplied prompt.

    Returns:
        ``(system, user)``, both stripped.
    """
    system_marker, user_marker = "\n## System\n", "\n## User\n"
    padded = f"\n{text}"
    system_at = padded.find(system_marker)
    user_at = padded.find(user_marker)
    if system_at == -1 or user_at == -1 or user_at < system_at:
        return text.strip(), ""
    system = padded[system_at + len(system_marker) : user_at]
    user = padded[user_at + len(user_marker) :]
    return system.strip(), user.strip()


def fill_prompt(template: str, **values: object) -> str:
    """Substitute ``{name}`` placeholders without touching other braces.

    ``str.format`` cannot be used because the prompts contain literal JSON
    examples; only the placeholders named in ``values`` are replaced.

    Args:
        template: The prompt text.
        **values: Placeholder values, converted with ``str``.

    Returns:
        The filled prompt.
    """
    for name, value in values.items():
        template = template.replace(f"{{{name}}}", str(value))
    return template
