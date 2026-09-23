"""JSON recovery from model answers."""

from __future__ import annotations

import pytest

from rail_vision_bench.providers.parsing import extract_json


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '  {"a": 1}\n',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Here is the document:\n{"a": 1}\nHope this helps.',
        'Sure. ```json\n{"a": 1}``` done',
    ],
)
def test_extracts_the_object(text: str):
    assert extract_json(text) == {"a": 1}


def test_braces_inside_strings_do_not_end_the_object():
    assert extract_json('prefix {"label": "}{", "n": {"x": "\\"}"}} suffix') == {
        "label": "}{",
        "n": {"x": '"}'},
    }


def test_skips_a_non_json_brace_span_and_finds_a_later_object():
    assert extract_json('use {braces} like {"a": 2}') == {"a": 2}


def test_a_truncated_document_does_not_yield_a_nested_fragment():
    assert extract_json('{"scene_id": "s", "topology": {"nodes": []}, "signals": [') is None


@pytest.mark.parametrize("text", ["", "no json here", "[1, 2]", '{"a": ', "42"])
def test_returns_none_without_an_object(text: str):
    assert extract_json(text) is None
