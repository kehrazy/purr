"""
Internal, immutable data structures for fully-resolved IR definitions.

These objects are the final product of the DSL introspection and are consumed
by code generators.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from . import builders as _builders
from . import domain as _domain
from . import traits as _traits
from . import types as _types


@dataclass(frozen=True, kw_only=True)
class OperandDef:
    name: str
    ir_type: type[_types.IRType]
    is_variadic: bool


@dataclass(frozen=True, kw_only=True)
class ResultDef:
    name: str
    ir_type: type[_types.IRType]
    is_variadic: bool


@dataclass(frozen=True, kw_only=True)
class AttributeDef:
    name: str
    ir_type: type[_types.IRAttribute]
    has_default: bool
    default_value: object | None


@dataclass(frozen=True, kw_only=True)
class OpDefinition:
    """A complete, immutable definition of an operation."""

    dialect: _domain.Dialect
    py_op_name: str
    docstring: str
    operands: Sequence[OperandDef]
    results: Sequence[ResultDef]
    attributes: Sequence[AttributeDef]
    traits: frozenset[type[_traits.Trait]]
    verifier: Callable[..., None] | None
    cpp_builders: Sequence[_builders.CppBuilderDef]
    cpp_extra_decls: Sequence[_builders.CppExtraDef]
    assembly_format: Sequence[_builders.AsmDirective] | None

    @property
    def mlir_name(self) -> str:
        """The MLIR-style name for the op (e.g., 'meow.paws_add')."""
        return f"{self.dialect.name}.{self.py_op_name}"
