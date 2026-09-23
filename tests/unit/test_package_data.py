"""Package data shipped inside the wheel: the prompts and the leaderboard template."""

from __future__ import annotations

from importlib.resources import files

import pytest

from rail_vision_bench.agents.prompts import PROMPT_NAMES, load_prompt
from rail_vision_bench.agents.state import NODE_NAMES
from rail_vision_bench.report.render import TEMPLATE_NAME, load_template

AGENTIC_SECTIONS = ("## Reads", "## Writes", "## Instructions", "## Stop condition")
SINGLE_SHOT_PLACEHOLDERS = ("{scene_id}", "{source_kind}", "{image}", "{width}", "{height}")
LEADERBOARD_COLUMNS = ("model", "mode", "split")


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_prompt_loads_non_empty(name: str):
    assert load_prompt(name).strip()


def test_prompt_names_match_the_shipped_files():
    """Every packaged .md is listed and every listed name is packaged (no orphans either way)."""
    root = files("rail_vision_bench").joinpath("prompts")
    shipped: set[str] = set()
    for entry in root.iterdir():
        if entry.is_dir():
            shipped.update(
                f"{entry.name}/{child.name.removesuffix('.md')}"
                for child in entry.iterdir()
                if child.name.endswith(".md")
            )
        elif entry.name.endswith(".md"):
            shipped.add(entry.name.removesuffix(".md"))
    assert shipped == set(PROMPT_NAMES)


def test_agentic_prompts_cover_every_node():
    agentic = {name for name in PROMPT_NAMES if name.startswith("agentic/")}
    assert agentic == {f"agentic/{node}" for node in NODE_NAMES}


@pytest.mark.parametrize("node", NODE_NAMES)
def test_agentic_prompt_states_its_contract(node: str):
    text = load_prompt(f"agentic/{node}")
    for section in AGENTIC_SECTIONS:
        assert section in text, f"{node}.md lacks the {section!r} section"


def test_single_shot_prompt_contract():
    text = load_prompt("single_shot")
    assert "## System" in text
    assert "## User" in text
    assert "schema_version" in text
    user_turn = text.split("## User", 1)[1]
    for placeholder in SINGLE_SHOT_PLACEHOLDERS:
        assert placeholder in user_turn


def test_unknown_prompt_is_a_value_error():
    with pytest.raises(ValueError, match="nope"):
        load_prompt("nope")


def test_template_loads_with_its_markers():
    text = load_template()
    assert TEMPLATE_NAME.endswith(".j2")
    assert text.lstrip().startswith("<!DOCTYPE html>")
    for marker in ("{{ title }}", "{{ generated_at }}", "{% for row in rows %}", "{% endfor %}"):
        assert marker in text
    for column in LEADERBOARD_COLUMNS:
        assert f"row.{column}" in text, f"template does not render column {column!r}"
