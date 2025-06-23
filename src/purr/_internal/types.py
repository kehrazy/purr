"""Internal marker classes for Purr's type system."""

from enum import Enum as PythonEnum


class IRType:
    """Base class for all IR data types (e.g., Tensor, F32)."""

    pass


class IRAttribute:
    """Base class for all IR attributes (e.g., I64Attr, StringAttr)."""

    pass


_EnumAttrCache: dict[type[PythonEnum], type[IRAttribute]] = {}


def EnumAttr(enum_type: type[PythonEnum]) -> type[IRAttribute]:
    """
    A factory for creating MLIR-style enum attribute types from Python enums.

    This function dynamically creates and caches a new `IRAttribute` subclass
    for each unique Python enum type provided. This allows for type-safe
    and Pythonic definition of enum attributes in the Purr DSL.

    Args:
        enum_type: The Python `enum.Enum` class to wrap.

    Returns:
        A new class that inherits from `IRAttribute`, representing the specific
        enum attribute.
    """
    if not issubclass(enum_type, PythonEnum):
        raise TypeError("EnumAttr must be initialized with a Python Enum subclass.")

    # TODO(kehrazy): cache enum attributes
    # TODO(kehrazy): enums can be bitfields - so
    #                we need a custom bitfield decorator/builder
    if enum_type in _EnumAttrCache:
        return _EnumAttrCache[enum_type]

    attr_class_name = f"{enum_type.__name__}Attr"
    NewEnumAttr = type(
        attr_class_name,
        (IRAttribute,),
        {"EnumType": enum_type, "__doc__": f"An attribute for `{enum_type.__name__}`."},
    )

    _EnumAttrCache[enum_type] = NewEnumAttr
    return NewEnumAttr


class Tensor(IRType):
    """Represents a tensor of any rank or element type."""

    pass


class F32(IRType):
    """Represents a 32-bit float scalar type."""

    pass


class F64(IRType):
    """Represents a 64-bit float scalar type."""

    pass


class BoolAttr(IRAttribute):
    """Represents a boolean attribute."""

    pass


class F64Attr(IRAttribute):
    """Represents a 64-bit float attribute."""

    pass


class I64Attr(IRAttribute):
    """Represents a 64-bit integer attribute."""

    pass


class StringAttr(IRAttribute):
    """Represents a string attribute."""

    pass


class TypeAttr(IRAttribute):
    """Represents an attribute that holds a type."""

    pass


class SymbolRefAttr(IRAttribute):
    """Represents a symbolic reference to another operation (e.g., a function)."""

    pass
