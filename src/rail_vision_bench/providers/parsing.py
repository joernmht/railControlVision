"""Recover the JSON object a model was asked to produce from its raw text.

Models asked for "only JSON" still wrap it in Markdown fences or a sentence of
prose now and then; every provider and every agent node goes through
:func:`extract_json` so they all tolerate the same deviations.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Iterator
from typing import Any

_FENCE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)


def _object_spans(text: str) -> Iterator[str]:
    """Yield the top-level balanced ``{...}`` spans of ``text`` in order.

    Braces inside JSON strings are ignored. Scanning resumes after each span, so
    a nested object is never offered on its own, and an unclosed ``{`` ends the
    scan: everything after it belongs to a truncated object. Linear in ``len(text)``.
    """
    start = text.find("{")
    while start != -1:
        end = -1
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end == -1:
            return
        yield text[start : end + 1]
        start = text.find("{", end + 1)


def extract_json(text: str) -> dict[str, Any] | None:
    """Parse the JSON object contained in a model answer.

    Tries, in order: the whole text, the content of each fenced code block, and
    each top-level balanced ``{...}`` span in order. Only a JSON *object* counts.

    Args:
        text: The raw model output.

    Returns:
        The parsed object, or None when no JSON object can be recovered.
    """
    fenced = (block.strip() for block in _FENCE.findall(text))
    for candidate in itertools.chain([text.strip()], fenced, _object_spans(text)):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None
