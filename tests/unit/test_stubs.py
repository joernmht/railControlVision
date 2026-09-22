"""Every stub in the package raises NotImplementedError and carries a docstring.

Stubs are discovered mechanically instead of being listed by hand: every module
under ``rail_vision_bench`` is parsed with ``ast`` and a function or method
whose body is exactly a docstring followed by one ``raise NotImplementedError(...)``
counts as a stub. A stub added later is therefore covered automatically, and a
``raise NotImplementedError`` hidden in a partially implemented function is
caught because it does not sit inside a stub-shaped body.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import inspect
import sys
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Union, get_args, get_origin

import pytest
from pydantic import BaseModel, ValidationError

import rail_vision_bench
from rail_vision_bench.config import RunConfig
from rail_vision_bench.graph.generate import generate_scene
from rail_vision_bench.providers.base import (
    ImageInput,
    VisionProvider,
    VisionRequest,
    VisionResponse,
)
from rail_vision_bench.schema.models import SceneAnnotation
from rail_vision_bench.settings import Settings

PACKAGE = rail_vision_bench.__name__
PACKAGE_DIR = Path(rail_vision_bench.__file__).resolve().parent
STUB_PREFIX = "is not implemented in the skeleton: "

# Placeholder for parameters whose type cannot be built here (PIL images are
# only imported under TYPE_CHECKING); stubs raise before touching any argument.
_ANY: Any = None


class FakeProvider:
    """Provider whose ``complete`` echoes the prompt; stands in for provider-taking stubs."""

    name: ClassVar[str] = "fake"

    async def complete(self, request: VisionRequest) -> VisionResponse:
        return VisionResponse(text=request.prompt)


@dataclass(frozen=True)
class StubRef:
    """A stub located by module, qualified name and the line of its ``raise``."""

    module: str
    qualname: str
    raise_lineno: int

    @property
    def dotted(self) -> str:
        return f"{self.module}.{self.qualname}"

    @property
    def label(self) -> str:
        return self.dotted.removeprefix(f"{PACKAGE}.")


def _names_not_implemented(node: ast.Raise) -> bool:
    exc = node.exc
    name = exc.func if isinstance(exc, ast.Call) else exc
    return isinstance(name, ast.Name) and name.id == "NotImplementedError"


def _is_stub(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """A stub is exactly a docstring plus one ``raise NotImplementedError(<message>)``."""
    if len(fn.body) != 2 or ast.get_docstring(fn) is None:
        return False
    raise_stmt = fn.body[1]
    return (
        isinstance(raise_stmt, ast.Raise)
        and isinstance(raise_stmt.exc, ast.Call)
        and _names_not_implemented(raise_stmt)
    )


def _module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE_DIR.parent).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _discover() -> tuple[list[StubRef], set[tuple[str, int]]]:
    """Return every stub and the location of every NotImplementedError raise."""
    stubs: list[StubRef] = []
    raises: set[tuple[str, int]] = set()
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        module = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        raises.update(
            (module, node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.Raise) and _names_not_implemented(node)
        )
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _is_stub(node):
                stubs.append(StubRef(module, node.name, node.body[1].lineno))
            elif isinstance(node, ast.ClassDef):
                stubs.extend(
                    StubRef(module, f"{node.name}.{member.name}", member.body[1].lineno)
                    for member in node.body
                    if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef)
                    and _is_stub(member)
                )
    return stubs, raises


STUBS, NOT_IMPLEMENTED_RAISES = _discover()

# Domain types that need a valid instance rather than a bare constructor call.
_EXACT: dict[Any, Callable[[], Any]] = {
    Settings: lambda: Settings(_env_file=None),  # never read a developer's .env
    SceneAnnotation: lambda: generate_scene(0),
    RunConfig: lambda: RunConfig(name="x", task=Path("t.yaml"), models=["m"]),
    ImageInput: lambda: ImageInput(data=b""),
    VisionRequest: lambda: VisionRequest(model="m", prompt="p"),
    VisionProvider: FakeProvider,
}
_SCALARS: dict[Any, Any] = {
    bool: False,
    int: 0,
    float: 0.0,
    str: "x",
    bytes: b"",
    Path: Path("x"),
}


def _placeholder(annotation: Any) -> Any:
    """Build a minimal value of the annotated type, or ``_ANY`` when there is no rule."""
    if annotation in _EXACT:
        return _EXACT[annotation]()
    if annotation in _SCALARS:
        return _SCALARS[annotation]
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _placeholder(args[0])
    if origin is Literal:
        return args[0]
    if origin in (Union, types.UnionType):
        return None if type(None) in args else _placeholder(args[0])
    if origin is tuple:
        return tuple(_placeholder(arg) for arg in args if arg is not Ellipsis)
    if origin in (list, Sequence, Iterable):
        return []
    if origin in (dict, Mapping):
        return {}
    if isinstance(annotation, type):
        if issubclass(annotation, dict):  # includes TypedDict classes such as AgentState
            return {}
        if issubclass(annotation, BaseModel):
            try:
                return annotation()
            except ValidationError:
                return _ANY
    return _ANY


def _annotation(param: inspect.Parameter, scope: dict[str, Any]) -> Any:
    """Resolve a parameter's annotation; names imported under TYPE_CHECKING stay unresolved."""
    if not isinstance(param.annotation, str):
        return param.annotation
    try:
        return eval(param.annotation, scope)  # resolves the PEP 563 string annotation
    except Exception:  # any unresolvable name means "no typed placeholder"
        return _ANY


def _placeholder_arguments(fn: Any) -> tuple[list[Any], dict[str, Any]]:
    """Positional and keyword placeholders for every parameter without a default."""
    scope = getattr(fn, "__globals__", None) or vars(sys.modules[fn.__module__])
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for param in inspect.signature(fn).parameters.values():
        if param.default is not param.empty or param.kind in (
            param.VAR_POSITIONAL,
            param.VAR_KEYWORD,
        ):
            continue
        value = _placeholder(_annotation(param, scope))
        if param.kind is param.KEYWORD_ONLY:
            kwargs[param.name] = value
        else:
            args.append(value)
    return args, kwargs


def _call(fn: Any, args: list[Any], kwargs: dict[str, Any]) -> Any:
    """Invoke a stub; async stubs are driven to completion so their raise surfaces."""
    if inspect.iscoroutinefunction(fn):
        return asyncio.run(fn(*args, **kwargs))
    return fn(*args, **kwargs)


def _resolve(stub: StubRef) -> Any:
    """Import the stub; methods are bound to an instance built from the class signature."""
    module = importlib.import_module(stub.module)
    owner_name, _, method_name = stub.qualname.rpartition(".")
    if not owner_name:
        return getattr(module, stub.qualname)
    owner = getattr(module, owner_name)
    args, kwargs = _placeholder_arguments(owner)
    return getattr(owner(*args, **kwargs), method_name)


def test_discovery_is_non_empty_and_covers_known_stubs():
    labels = {stub.label for stub in STUBS}
    assert labels >= {"runner.run_benchmark", "providers.claude.AnthropicProvider.complete"}
    assert len(labels) == len(STUBS)


def test_every_not_implemented_raise_belongs_to_a_stub():
    """No partially implemented function hides a NotImplementedError; stubs are all-or-nothing."""
    assert {(stub.module, stub.raise_lineno) for stub in STUBS} == NOT_IMPLEMENTED_RAISES


@pytest.mark.parametrize("stub", STUBS, ids=[stub.label for stub in STUBS])
def test_stub_raises_not_implemented(stub: StubRef):
    fn = _resolve(stub)
    assert (fn.__doc__ or "").strip(), f"{stub.dotted} has no docstring"
    args, kwargs = _placeholder_arguments(fn)
    with pytest.raises(NotImplementedError) as exc:
        _call(fn, args, kwargs)
    assert str(exc.value).startswith(f"{stub.dotted} {STUB_PREFIX}")
    assert str(exc.value) != f"{stub.dotted} {STUB_PREFIX}", "the stub message names no intent"
