"""
Internal, temporary builder objects created by the public DSL.

These objects hold the state of a definition before it is finalized into
an immutable `OpDefinition`. They are not meant to be used directly.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from . import types as _types


@dataclass(frozen=True)
class OperandBuilder:
    ir_type: type[_types.IRType]
    doc: str | None
    is_variadic: bool


@dataclass(frozen=True)
class ResultBuilder:
    ir_type: type[_types.IRType]
    doc: str | None
    is_variadic: bool


@dataclass(frozen=True)
class AttributeBuilder:
    ir_type: type[_types.IRAttribute]
    doc: str | None
    has_default: bool
    default_value: Any | None


@dataclass(frozen=True)
class CppBuilderArg:
    name: str
    cpp_type: str


@dataclass(frozen=True)
class CppBuilderDef:
    arguments: Sequence[CppBuilderArg]
    body: str


@dataclass(frozen=True)
class CppExtraDef:
    body: str


class AsmDirective:
    """Base class for all assembly format directives."""

    pass


@dataclass(frozen=True)
class AsmKeyword(AsmDirective):
    value: str


@dataclass(frozen=True)
class AsmOperand(AsmDirective):
    name: str


@dataclass(frozen=True)
class AsmAttribute(AsmDirective):
    name: str


@dataclass(frozen=True)
class AsmResultType(AsmDirective):
    index: int


@dataclass(frozen=True)
class AsmAttrDict(AsmDirective):
    pass


@dataclass(frozen=True)
class AsmVariadicList(AsmDirective):
    name: str
    delimiter: str


PrintCondition = Literal["any", "all"] | Sequence[str]


@dataclass(frozen=True)
class AsmGroup(AsmDirective):
    content: Sequence[AsmDirective]
    print_if: PrintCondition
