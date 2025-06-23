"""Internal implementation of the main user-facing classes and DSL builders."""

from __future__ import annotations

import inspect
import re
from collections.abc import Sequence
from enum import Enum as PythonEnum
from types import MappingProxyType

from . import builders as _b
from . import defs as _d
from . import exceptions as _e
from . import traits as _t
from . import types as _ty


class OpBuilderNamespace:
    """The `op` builder for defining core operation components."""

    def operand(
        self,
        ir_type: type[_ty.IRType],
        doc: str | None = None,
        *,
        variadic: bool = False,
    ) -> _b.OperandBuilder:
        """Defines an operand."""
        if not issubclass(ir_type, _ty.IRType):
            raise TypeError("Must be a purr.types class.")
        return _b.OperandBuilder(ir_type=ir_type, doc=doc, is_variadic=variadic)

    def result(
        self,
        ir_type: type[_ty.IRType],
        doc: str | None = None,
        *,
        variadic: bool = False,
    ) -> _b.ResultBuilder:
        """Defines a result."""
        if not issubclass(ir_type, _ty.IRType):
            raise TypeError("Must be a purr.types class.")
        return _b.ResultBuilder(ir_type=ir_type, doc=doc, is_variadic=variadic)

    def attribute(
        self,
        ir_type: type[_ty.IRAttribute],
        *,
        default: object = inspect.Parameter.empty,
        doc: str | None = None,
    ) -> _b.AttributeBuilder:
        """Defines an attribute."""
        if not issubclass(ir_type, _ty.IRAttribute):
            raise TypeError("Must be a purr.types class.")
        return _b.AttributeBuilder(
            ir_type=ir_type,
            doc=doc,
            has_default=(default is not inspect.Parameter.empty),
            default_value=default,
        )


class CppBuilderNamespace:
    """The `cpp` builder for defining C++-specific helpers."""

    def builder(self, args: Sequence[tuple[str, str]], body: str) -> _b.CppBuilderDef:
        """Defines a custom C++ builder method."""
        b_args = [_b.CppBuilderArg(name=n, cpp_type=t) for n, t in args]
        return _b.CppBuilderDef(arguments=tuple(b_args), body=body)

    def extra_decl(self, body: str) -> _b.CppExtraDef:
        """Injects an extra C++ declaration into the op class body."""
        return _b.CppExtraDef(body=body)


class AsmBuilderNamespace:
    """The `asm` builder for defining a declarative assembly format."""

    def keyword(self, value: str) -> _b.AsmKeyword:
        """Represents a literal keyword."""
        return _b.AsmKeyword(value=value)

    def operand(self, name: str) -> _b.AsmOperand:
        """Represents an operand, referenced by its Python attribute name."""
        return _b.AsmOperand(name=name)

    def attribute(self, name: str) -> _b.AsmAttribute:
        """Represents an attribute, referenced by its Python attribute name."""
        return _b.AsmAttribute(name=name)

    def result_type(self, index: int = 0) -> _b.AsmResultType:
        """Represents the type of a result."""
        return _b.AsmResultType(index=index)

    def attr_dict(self) -> _b.AsmAttrDict:
        """Represents the optional attribute dictionary."""
        return _b.AsmAttrDict()

    def list(self, name: str, *, delimiter: str = ",") -> _b.AsmVariadicList:
        """Represents a delimited list of a variadic operand or attribute."""
        return _b.AsmVariadicList(name=name, delimiter=delimiter)

    def group(
        self, content: Sequence[_b.AsmDirective], *, print_if: _b.PrintCondition = "any"
    ) -> _b.AsmGroup:
        """Defines a syntactic group that is printed conditionally."""
        return _b.AsmGroup(content=tuple(content), print_if=print_if)


op_builder_instance = OpBuilderNamespace()
cpp_builder_instance = CppBuilderNamespace()
asm_builder_instance = AsmBuilderNamespace()


def _to_snake_case(name: str) -> str:
    """Converts CamelCase to snake_case for op names."""
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


class Dialect:
    """A container for a set of related operations."""

    def __init__(self, name: str, cpp_namespace: str, description: str | None = None):
        self.name = name
        self.cpp_namespace = cpp_namespace
        self.description = description
        self._ops: dict[str, _d.OpDefinition] = {}
        self._enums: dict[str, type[PythonEnum]] = {}

    @property
    def ops(self) -> MappingProxyType[str, _d.OpDefinition]:
        """A read-only view of the operations defined in this dialect."""
        return MappingProxyType(self._ops)

    @property
    def enums(self) -> MappingProxyType[str, type[PythonEnum]]:
        """A read-only view of the enums defined in this dialect."""
        return MappingProxyType(self._enums)

    def _register_op(self, op_def: _d.OpDefinition) -> None:
        if op_def.py_op_name in self._ops:
            raise _e.DefinitionError(
                f"Op '{op_def.py_op_name}' is already defined in dialect '{self.name}'."
            )
        self._ops[op_def.py_op_name] = op_def


class Op:
    """Base class for all class-based operation definitions."""

    dialect: Dialect
    traits: set[type[_t.Trait]] = set()
    assembly_format: Sequence[_b.AsmDirective] | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Introspects the subclass to build and register an OpDefinition."""
        super().__init_subclass__(**kwargs)

        # Don't register the base class or abstract intermediates.
        if cls.__name__ == "Op" or not hasattr(cls, "dialect"):
            return

        py_op_name = _to_snake_case(cls.__name__)
        docstring = inspect.getdoc(cls) or ""

        operands: list[_d.OperandDef] = []
        results: list[_d.ResultDef] = []
        attributes: list[_d.AttributeDef] = []
        cpp_builders: list[_b.CppBuilderDef] = []
        cpp_extras: list[_b.CppExtraDef] = []

        # walk class attributes for builder objects
        for name, value in inspect.getmembers(cls):
            if isinstance(value, _b.OperandBuilder):
                operands.append(
                    _d.OperandDef(
                        name=name, ir_type=value.ir_type, is_variadic=value.is_variadic
                    )
                )
            elif isinstance(value, _b.ResultBuilder):
                results.append(
                    _d.ResultDef(
                        name=name, ir_type=value.ir_type, is_variadic=value.is_variadic
                    )
                )
            elif isinstance(value, _b.AttributeBuilder):
                attributes.append(
                    _d.AttributeDef(
                        name=name,
                        ir_type=value.ir_type,
                        has_default=value.has_default,
                        default_value=value.default_value,
                    )
                )
                # ..if this is an enum attribute, auto-register its type.
                if hasattr(value.ir_type, "EnumType") and issubclass(
                    value.ir_type.EnumType, PythonEnum
                ):
                    enum_cls = value.ir_type.EnumType
                    if enum_cls.__name__ not in cls.dialect._enums:
                        cls.dialect._enums[enum_cls.__name__] = enum_cls

            elif isinstance(value, _b.CppBuilderDef):
                cpp_builders.append(value)
            elif isinstance(value, _b.CppExtraDef):
                cpp_extras.append(value)

        variadic_operands = [p.name for p in operands if p.is_variadic]
        if len(variadic_operands) > 1:
            raise _e.DefinitionError(
                f"Op '{py_op_name}' has more than one variadic operand: "
                f"{variadic_operands}"
            )

        # Find the verifier method, if it was overridden
        verifier = cls.verify if "verify" in cls.__dict__ else None

        # Build the final, immutable, OpDefinition
        op_def = _d.OpDefinition(
            dialect=cls.dialect,
            py_op_name=py_op_name,
            docstring=docstring,
            operands=tuple(operands),
            results=tuple(results),
            attributes=tuple(attributes),
            traits=frozenset(cls.traits),
            verifier=verifier,
            cpp_builders=tuple(cpp_builders),
            cpp_extra_decls=tuple(cpp_extras),
            assembly_format=cls.assembly_format,
        )
        cls.dialect._register_op(op_def)

    def verify(self) -> None:
        """Optional method to define the operation's verifier logic."""
        pass
