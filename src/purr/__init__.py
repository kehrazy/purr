"""
Purr: A cute, Pythonic, and extensible DSL for defining MLIR-like IR.

This package provides a modern, discoverable API for defining compiler
dialects and operations directly in Python, avoiding the pitfalls of
TableGen.

Example:
    from purr import Dialect, Op, op, types

    my_dialect = Dialect("my_dialect", "::mlir::my_dialect")

    class AddOp(Op):
        dialect = my_dialect
        lhs = op.operand(types.Tensor)
        rhs = op.operand(types.Tensor)
        output = op.result(types.Tensor)
"""

__version__ = "0.0.0-dev"

from ._internal import traits, types
from ._internal.domain import Dialect, Op
from ._internal.domain import asm_builder_instance as asm
from ._internal.domain import cpp_builder_instance as cpp
from ._internal.domain import op_builder_instance as op
from ._internal.exceptions import DefinitionError, PurrError

__all__ = [
    "Dialect",
    "Op",
    "op",
    "cpp",
    "asm",
    "types",
    "traits",
    "PurrError",
    "DefinitionError",
]
