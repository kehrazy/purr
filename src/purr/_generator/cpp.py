"""
Emits C++ header (.h.inc) and source (.cpp.inc) files for Purr dialects.

This module translates the Python-based Purr IR definitions into the C++
class declarations and definitions that form the primary API for interacting
with operations in MLIR.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum as PythonEnum
from typing import TYPE_CHECKING

from .._internal import builders as _b
from .._internal import defs as _d
from .._internal import types as _ty
from .emitter import CodeEmitter
from .tablegen import PURR_TO_MLIR_TRAIT_MAP, to_camel_case

PURR_TO_CPP_ATTR_MAP: dict[type[_ty.IRAttribute], str] = {
    _ty.BoolAttr: "::mlir::BoolAttr",
    _ty.F64Attr: "::mlir::FloatAttr",
    _ty.I64Attr: "::mlir::IntegerAttr",
    _ty.StringAttr: "::mlir::StringAttr",
    _ty.TypeAttr: "::mlir::TypeAttr",
    _ty.SymbolRefAttr: "::mlir::SymbolRefAttr",
}

if TYPE_CHECKING:
    from .._internal.domain import Dialect


class CppEmitter:
    """Emits C++ header and source files for a given Purr dialect."""

    def __init__(self, dialect: Dialect):
        self.dialect = dialect
        self.ops_h = CodeEmitter()
        self.ops_cpp = CodeEmitter()
        self.enums_h = CodeEmitter()
        self.enums_cpp = CodeEmitter()
        self.attrs_h = CodeEmitter()
        self.attrs_cpp = CodeEmitter()
        self.dialect_h = CodeEmitter()
        self.dialect_cpp = CodeEmitter()

    def emit(self) -> dict[str, str]:
        """
        Generates all C++ files for the dialect.

        This is the main entry point that orchestrates the generation of
        ops, enums, attributes, and the dialect class itself.
        """
        self._emit_enums_h()
        self._emit_enums_cpp()
        self._emit_attrs_h()
        self._emit_attrs_cpp()
        self._emit_ops_h()
        self._emit_ops_cpp()
        self._emit_dialect_h()
        self._emit_dialect_cpp()

        return {
            f"{self.dialect.name}Enums.h.inc": self.enums_h.get(),
            f"{self.dialect.name}Enums.cpp.inc": self.enums_cpp.get(),
            f"{self.dialect.name}Attrs.h.inc": self.attrs_h.get(),
            f"{self.dialect.name}Attrs.cpp.inc": self.attrs_cpp.get(),
            f"{self.dialect.name}Ops.h.inc": self.ops_h.get(),
            f"{self.dialect.name}Ops.cpp.inc": self.ops_cpp.get(),
            f"{self.dialect.name}Dialect.h.inc": self.dialect_h.get(),
            f"{self.dialect.name}Dialect.cpp.inc": self.dialect_cpp.get(),
        }

    def _get_operand_type(self, op: _d.OperandDef) -> str:
        """
        Returns the C++ type for an operand.

        TODO(kehrazy): add support for more specific types based on constraints.
        """
        return "::mlir::Value"

    def _get_result_type(self, res: _d.ResultDef) -> str:
        """
        Returns the C++ type for a result.

        Results can be single values or ranges.
        """
        if res.is_variadic:
            return "::mlir::ResultRange"
        return "::mlir::Value"

    def _get_attribute_type(self, attr: _d.AttributeDef) -> str:
        """
        Returns the C++ type for an attribute.
        """
        if hasattr(attr.ir_type, "EnumType"):
            return (
                f"::{self.dialect.cpp_namespace}::{attr.ir_type.EnumType.__name__}Attr"
            )
        return PURR_TO_CPP_ATTR_MAP.get(attr.ir_type, "::mlir::Attribute")

    def _emit_ops_h(self) -> None:
        """Generates the C++ Ops header (.h.inc) file content."""
        emitter = self.ops_h
        dialect_name_pascal = to_camel_case(self.dialect.name)
        guard = f"PURR_GEN_{self.dialect.name.upper()}OPS_H_INC_"

        with emitter.macro_guard(guard):
            emitter.line('#include "mlir/IR/OpDefinition.h"')
            emitter.line('#include "mlir/IR/Builders.h"')
            emitter.line('#include "mlir/IR/OpImplementation.h"')
            emitter.blank_line()
            if self.dialect.enums:
                emitter.line(f'#include "{dialect_name_pascal}Enums.h.inc"')
                emitter.line(f'#include "{dialect_name_pascal}Attrs.h.inc"')
            emitter.blank_line()
            emitter.blank_line()
            with emitter.namespace(self.dialect.cpp_namespace):
                with emitter.macro_guard("GET_OP_CLASSES"):
                    for op_def in self.dialect.ops.values():
                        self._emit_op_adaptor_decl(emitter, op_def)
                        self._emit_op_class_decl(emitter, op_def)

    def _emit_ops_cpp(self) -> None:
        """Generates the C++ Ops source (.cpp.inc) file content."""
        if not self.dialect.ops:
            return
        emitter = self.ops_cpp

        with emitter.namespace(self.dialect.cpp_namespace):
            for op_def in self.dialect.ops.values():
                self._emit_op_adaptor_def(emitter, op_def)
                self._emit_op_class_def(emitter, op_def)

    def _emit_enums_h(self) -> None:
        """Generates the C++ Enums header (.h.inc) file content."""
        if not self.dialect.enums:
            return

        emitter = self.enums_h
        guard = f"PURR_GEN_{self.dialect.name.upper()}ENUMS_H_INC_"
        with emitter.macro_guard(guard):
            emitter.line(f"namespace {self.dialect.cpp_namespace} {{")
            for name, enum_cls in self.dialect.enums.items():
                self._emit_single_enum_h(emitter, name, enum_cls)
            emitter.line(f"}} // namespace {self.dialect.cpp_namespace}")

    def _emit_single_enum_h(
        self, emitter: CodeEmitter, name: str, enum_cls: type[PythonEnum]
    ) -> None:
        emitter.blank_line()

        # TODO(kehrazy): enums may not be uint32_t
        emitter.line(f"enum class {name} : uint32_t {{")
        with emitter.indent():
            for i, member in enumerate(enum_cls):
                emitter.line(f"{member.name} = {i},")
        emitter.line("};")
        emitter.blank_line()
        emitter.line(f"::std::optional<{name}> symbolize{name}(uint32_t);")
        emitter.line(f"::llvm::StringRef stringify{name}({name});")
        emitter.line(f"::std::optional<{name}> symbolize{name}(::llvm::StringRef);")

    def _emit_enums_cpp(self) -> None:
        """Generates the C++ Enums source (.cpp.inc) file content."""
        if not self.dialect.enums:
            return

        emitter = self.enums_cpp
        with emitter.namespace(self.dialect.cpp_namespace):
            for name, enum_cls in self.dialect.enums.items():
                self._emit_single_enum_cpp(emitter, name, enum_cls)

    def _emit_single_enum_cpp(
        self, emitter: CodeEmitter, name: str, enum_cls: type[PythonEnum]
    ) -> None:
        """
        Generates the C++ Enums source (.cpp.inc) file content for a single enum.
        """
        emitter.blank_line()
        emitter.line(f"::llvm::StringRef stringify{name}({name} val) {{")
        with emitter.block("switch (val) {", "}"):
            for member in enum_cls:
                emitter.line(f'case {name}::{member.name}: return "{member.value}";')
        emitter.line('return "";')
        emitter.line("}")

        emitter.blank_line()
        emitter.line(
            f"::std::optional<{name}> symbolize{name}(::llvm::StringRef str) {{"
        )
        with emitter.block(
            f"return ::llvm::StringSwitch<::std::optional<{name}>>(str)",
            ".Default(::std::nullopt);",
        ):
            for member in enum_cls:
                emitter.line(f'.Case("{member.value}", {name}::{member.name})')
        emitter.line("}")

        emitter.blank_line()
        emitter.line(f"::std::optional<{name}> symbolize{name}(uint32_t value) {{")
        with emitter.block("switch (value) {", "}"):
            for i, member in enumerate(enum_cls):
                emitter.line(f"case {i}: return {name}::{member.name};")
            emitter.line("default: return ::std::nullopt;")
        emitter.line("}")

    def _emit_attrs_h(self) -> None:
        """Generates the C++ Attrs header (.h.inc) file content."""
        dialect_name_pascal = to_camel_case(self.dialect.name)
        if not self.dialect.enums:
            return

        emitter = self.attrs_h
        guard = f"PURR_GEN_{self.dialect.name.upper()}ATTRS_H_INC_"
        with emitter.macro_guard(guard):
            emitter.line('#include "mlir/IR/Attributes.h"')
            emitter.line(f'#include "{dialect_name_pascal}Enums.h.inc"')
            emitter.blank_line()
            with emitter.namespace(self.dialect.cpp_namespace):
                with emitter.macro_guard("GET_ATTRDEF_CLASSES"):
                    for enum_name in self.dialect.enums:
                        self._emit_single_attr_h(emitter, enum_name)

    def _emit_single_attr_h(self, emitter: CodeEmitter, enum_name: str) -> None:
        attr_name = f"{enum_name}Attr"
        storage_name = f"{enum_name}AttrStorage"

        emitter.blank_line()
        emitter.line(f"class {attr_name};")
        with emitter.namespace("detail"):
            emitter.line(f"struct {storage_name};")

        emitter.line(
            f"class {attr_name} : public ::mlir::Attribute::AttrBase<{attr_name}, ::mlir::Attribute, detail::{storage_name}> {{"
        )
        emitter.line("public:")
        with emitter.indent():
            emitter.line("using Base::Base;")
            emitter.line(
                f"static {attr_name} get(::mlir::MLIRContext *context, {enum_name} value);"
            )
            emitter.line(
                "static ::mlir::Attribute parse(::mlir::AsmParser &parser, ::mlir::Type type);"
            )
            emitter.line("void print(::mlir::AsmPrinter &printer) const;")
            emitter.line(f"{enum_name} getValue() const;")
        emitter.line("};")

    def _emit_attrs_cpp(self) -> None:
        """Generates the C++ Attrs source (.cpp.inc) file content."""
        if not self.dialect.enums:
            return

        emitter = self.attrs_cpp

        with emitter.namespace("mlir"):
            with emitter.namespace(self.dialect.cpp_namespace):
                self._emit_generated_attr_parser_printer(emitter)

                with emitter.macro_guard("GET_ATTRDEF_CLASSES"):
                    for enum_name in self.dialect.enums:
                        self._emit_single_attr_cpp(emitter, enum_name)

                emitter.blank_line()
                with emitter.macro_guard("GET_ATTRDEF_LIST"):
                    for enum_name in self.dialect.enums:
                        emitter.line(
                            f"::mlir::{self.dialect.cpp_namespace}::{enum_name}Attr"
                        )

    def _emit_single_attr_cpp(self, emitter: CodeEmitter, enum_name: str) -> None:
        attr_name = f"{enum_name}Attr"
        storage_name = f"detail::{enum_name}AttrStorage"

        with emitter.namespace("detail"):
            emitter.line(f"using KeyTy = {enum_name};")
            emitter.line(f"{enum_name}AttrStorage(KeyTy value) : value(value) {{}}")
            emitter.line(
                "bool operator==(const KeyTy &key) const { return key == value; }"
            )
            emitter.line(
                f"static {enum_name}AttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy key) {{"
            )
            emitter.line(
                f"    return new (allocator.allocate<{enum_name}AttrStorage>()) {enum_name}AttrStorage(key);"
            )
            emitter.line("}")
            emitter.line("KeyTy value;")

        # ::get
        emitter.line(
            f"{attr_name} {attr_name}::get(::mlir::MLIRContext *context, {enum_name} value) {{"
        )
        emitter.line(f"    return ::mlir::Attribute::get(context, value);")
        emitter.line("}")

        # Parse
        emitter.line(
            f"::mlir::Attribute {attr_name}::parse(::mlir::AsmParser &parser, ::mlir::Type type) {{"
        )
        emitter.line("    // Implementation left as an exercise for the reader :)")
        emitter.line(
            f"    return get(parser.getContext(), {enum_name}::{next(iter(self.dialect.enums[enum_name]))});"
        )
        emitter.line("}")

        # Print
        emitter.line(f"void {attr_name}::print(::mlir::AsmPrinter &printer) const {{")
        emitter.line('    printer << "<" << stringify(getValue()) << ">";')
        emitter.line("}")

        # GetValue
        emitter.line(f"{enum_name} {attr_name}::getValue() const {{")
        emitter.line(f"    return getImpl()->value;")
        emitter.line("}")

    def _emit_generated_attr_parser_printer(self, emitter: CodeEmitter) -> None:
        if not self.dialect.enums:
            return

        # Parser
        emitter.line(
            "static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {"
        )
        with emitter.block(
            "  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)",
            "    .Default([&](llvm::StringRef keyword, llvm::SMLoc) { *mnemonic = keyword; return std::nullopt; });",
        ):
            for enum_name in self.dialect.enums:
                attr_name = f"{enum_name}Attr"
                emitter.line(
                    f".Case({attr_name}::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {{"
                )
                emitter.line(f"  value = {attr_name}::parse(parser, type);")
                emitter.line(f"  return ::mlir::success(!!value);")
                emitter.line("})")
        emitter.line("}")
        emitter.blank_line()

        # Printer
        emitter.line(
            "static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {"
        )
        with emitter.block(
            "  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)",
            "    .Default([](auto) { return ::mlir::failure(); });",
        ):
            for enum_name in self.dialect.enums:
                attr_name = f"{enum_name}Attr"
                emitter.line(f".Case<{attr_name}>([&](auto t) {{")
                emitter.line(f"  printer << {attr_name}::getMnemonic();")
                emitter.line(f"  t.print(printer);")
                emitter.line(f"  return ::mlir::success();")
                emitter.line("})")
        emitter.line("}")
        emitter.blank_line()

    def _emit_op_adaptor_decl(
        self, emitter: CodeEmitter, op_def: _d.OpDefinition
    ) -> None:
        op_name_pascal = to_camel_case(op_def.py_op_name)
        adaptor_name = f"{op_name_pascal}Adaptor"

        with emitter.block(
            f"class {adaptor_name} : public ::mlir::OpAdaptor<{op_name_pascal}> {{",
            "};",
        ):
            emitter.line("public:")
            with emitter.indent():
                emitter.line(f"using OpAdaptor<{op_name_pascal}>::OpAdaptor;")
                emitter.blank_line()

                for operand in op_def.operands:
                    emitter.line(f"::mlir::Value get{to_camel_case(operand.name)}();")
                for attr in op_def.attributes:
                    emitter.line(
                        f"{self._get_attribute_type(attr)} get{to_camel_case(attr.name)}();"
                    )
        emitter.blank_line()

    def _emit_op_adaptor_def(
        self, emitter: CodeEmitter, op_def: _d.OpDefinition
    ) -> None:
        op_name_pascal = to_camel_case(op_def.py_op_name)
        adaptor_name = f"{op_name_pascal}Adaptor"

        for i, operand in enumerate(op_def.operands):
            with emitter.block(
                f"::mlir::Value {adaptor_name}::get{to_camel_case(operand.name)}() {{",
                "}",
            ):
                emitter.line(f"return getOperands()[{i}];")
        emitter.blank_line()

        for attr in op_def.attributes:
            with emitter.block(
                f"{self._get_attribute_type(attr)} {adaptor_name}::get{to_camel_case(attr.name)}() {{",
                "}",
            ):
                emitter.line(f"return getOp()->get{to_camel_case(attr.name)}Attr();")
        emitter.blank_line()

    def _emit_op_class_decl(
        self, emitter: CodeEmitter, op_def: _d.OpDefinition
    ) -> None:
        op_name_pascal = to_camel_case(op_def.py_op_name)

        trait_list = ", ".join(
            f"::mlir::{PURR_TO_MLIR_TRAIT_MAP[t]}"
            for t in op_def.traits
            if t in PURR_TO_MLIR_TRAIT_MAP
        )

        emitter.line(
            f"class {op_name_pascal} : public ::mlir::Op<{op_name_pascal}{', ' + trait_list if trait_list else ''}> {{"
        )
        emitter.line("public:")
        with emitter.indent():
            emitter.line("using Op::Op;")
            emitter.line(f"using Adaptor = {op_name_pascal}Adaptor;")
            emitter.blank_line()

            emitter.line("static ::llvm::StringRef getOperationName();")

            for operand in op_def.operands:
                emitter.line(f"{self._get_operand_type(operand)} get_{operand.name}();")
            for result in op_def.results:
                emitter.line(f"{self._get_result_type(result)} get_{result.name}();")
            for attr in op_def.attributes:
                emitter.line(
                    f"{self._get_attribute_type(attr)} get{to_camel_case(attr.name)}Attr();"
                )

            emitter.blank_line()

            self._emit_op_builder_decls(emitter, op_def)

            if op_def.verifier:
                emitter.line("::mlir::LogicalResult verify();")

            if op_def.assembly_format:
                emitter.line(
                    f"static void print(::mlir::OpAsmPrinter &p, {op_name_pascal} op);"
                )
                emitter.line(
                    "static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, "
                    "::mlir::OperationState &result);"
                )

        emitter.line("};")
        emitter.blank_line()

    def _emit_op_builder_decls(
        self, emitter: CodeEmitter, op_def: _d.OpDefinition
    ) -> None:
        emitter.line(
            "static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,"
        )
        with emitter.indent():
            for op in op_def.operands:
                emitter.line(f"{self._get_operand_type(op)} {op.name},")
            for attr in op_def.attributes:
                emitter.line(f"{self._get_attribute_type(attr)} {attr.name},")

        emitter.line(");")
        emitter.blank_line()

        for cpp_builder in op_def.cpp_builders:
            emitter.write(
                f"static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state"
            )
            if cpp_builder.arguments:
                emitter.write(", ")
                args_str = ", ".join(
                    f"{arg.cpp_type} {arg.name}" for arg in cpp_builder.arguments
                )
                emitter.write(args_str)
            emitter.line(");")
        emitter.blank_line()

    def _emit_op_class_def(self, emitter: CodeEmitter, op_def: _d.OpDefinition) -> None:
        op_name_pascal = to_camel_case(op_def.py_op_name)

        emitter.line(f"::llvm::StringRef {op_name_pascal}::getOperationName() {{")
        with emitter.indent():
            emitter.line(f'return "{op_def.mlir_name}";')
        emitter.line("}")
        emitter.blank_line()

        for i, operand in enumerate(op_def.operands):
            emitter.line(
                f"{self._get_operand_type(operand)} {op_name_pascal}::get_{operand.name}() {{ return getOperand({i}); }}"
            )
        for result in op_def.results:
            emitter.line(
                f"{self._get_result_type(result)} {op_name_pascal}::get_{result.name}() {{ return getResult({i}); }}"
            )
        for attr in op_def.attributes:
            emitter.line(
                f"{self._get_attribute_type(attr)} {op_name_pascal}::get{to_camel_case(attr.name)}Attr() {{"
            )
            with emitter.indent():
                emitter.line(
                    f'return getAttr("{attr.name}").cast<{self._get_attribute_type(attr)}>();'
                )
            emitter.line("}")
        emitter.blank_line()

        self._emit_op_builder_defs(emitter, op_def)

        if op_def.verifier:
            emitter.line(f"::mlir::LogicalResult {op_name_pascal}::verify() {{")
            with emitter.indent():
                emitter.line("// TODO: Implement user-defined verifier logic.")
                emitter.line("return ::mlir::success();")
            emitter.line("}")
            emitter.blank_line()

        if op_def.assembly_format:
            self._emit_op_printer_def(emitter, op_def)
            self._emit_op_parser_def(emitter, op_def)

    def _get_asm_members(self, directives: Sequence[_b.AsmDirective]) -> list[str]:
        """Recursively finds all operand/attribute names in a format string."""
        members = []
        for d in directives:
            if isinstance(d, (_b.AsmOperand, _b.AsmAttribute)):
                members.append(d.name)
            elif isinstance(d, _b.AsmGroup):
                members.extend(self._get_asm_members(d.content))
        return members

    def _build_print_condition(
        self, op_var: str, op_def: _d.OpDefinition, condition: _b.PrintCondition
    ) -> str:
        """Builds a C++ 'if' condition string for a print_if directive."""
        names_to_check: list[str] = []
        if isinstance(condition, str) and condition in ("any", "all"):
            # TODO(kehrazy): this is ambiguous, implement group-based checks
            names_to_check = [op.name for op in op_def.operands] + [
                attr.name for attr in op_def.attributes
            ]
        elif isinstance(condition, Sequence):
            names_to_check = list(condition)

        if not names_to_check:
            return "true"

        checks = []
        for name in names_to_check:
            # Find the definition to know if it's an operand or attribute
            if any(op.name == name for op in op_def.operands):
                checks.append(f"{op_var}.get_{name}()")  # mlir::Value is truthy
            elif any(attr.name == name for attr in op_def.attributes):
                checks.append(
                    f"{op_var}.get{to_camel_case(name)}Attr()"
                )  # mlir::Attribute is truthy

        op = " || " if isinstance(condition, str) and condition == "any" else " && "
        return op.join(checks)

    def _emit_op_printer_def(
        self, emitter: CodeEmitter, op_def: _d.OpDefinition
    ) -> None:
        if not op_def.assembly_format:
            return
        op_name_pascal = to_camel_case(op_def.py_op_name)
        with emitter.curly_block(
            f"void {op_name_pascal}::print(::mlir::OpAsmPrinter &p, {op_name_pascal} op)"
        ):
            elided_attrs = self._get_elided_attributes(op_def.assembly_format)
            elided_attrs_str = ", ".join(f'"{a}"' for a in elided_attrs)
            emitter.line(
                f"p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{{{elided_attrs_str}}});"
            )
            self._emit_asm_printer_body(emitter, "op", op_def, op_def.assembly_format)
            emitter.line(
                'p << " : " << op.getOperation()->getOperandTypes() << " -> " << op.getOperation()->getResultTypes();'
            )

    def _get_elided_attributes(
        self, directives: Sequence[_b.AsmDirective]
    ) -> list[str]:
        elided = []
        for directive in directives:
            if isinstance(directive, _b.AsmAttribute):
                elided.append(directive.name)
            elif isinstance(directive, _b.AsmGroup):
                elided.extend(self._get_elided_attributes(directive.content))
        return elided

    def _emit_asm_printer_body(
        self,
        emitter: CodeEmitter,
        op_var: str,
        op_def: _d.OpDefinition,
        directives: Sequence[_b.AsmDirective],
    ) -> None:
        for i, directive in enumerate(directives):
            emitter.line('p << " ";')
            if isinstance(directive, _b.AsmKeyword):
                emitter.line(f'p << "{directive.value}";')
            elif isinstance(directive, _b.AsmOperand):
                emitter.line(f"p.printOperand({op_var}.get_{directive.name}());")
            elif isinstance(directive, _b.AsmAttribute):
                emitter.line(
                    f"p.printAttribute({op_var}.get{to_camel_case(directive.name)}Attr());"
                )
            elif isinstance(directive, _b.AsmGroup):
                condition = self._build_print_condition(
                    op_var, op_def, directive.print_if
                )
                with emitter.curly_block(f"if ({condition})"):
                    self._emit_asm_printer_body(
                        emitter, op_var, op_def, directive.content
                    )

    def _emit_op_parser_def(
        self, emitter: CodeEmitter, op_def: _d.OpDefinition
    ) -> None:
        if not op_def.assembly_format:
            return
        op_name_pascal = to_camel_case(op_def.py_op_name)
        with emitter.curly_block(
            f"::mlir::ParseResult {op_name_pascal}::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)"
        ):
            self._emit_asm_parser_body(emitter, op_def, op_def.assembly_format)

            emitter.blank_line()
            emitter.line("::mlir::FunctionType op_type;")
            emitter.line("if (parser.parseOptionalAttrDict(result.attributes) ||")
            emitter.line("    parser.parseColonType(op_type))")
            emitter.line("  return ::mlir::failure();")
            emitter.blank_line()

            emitter.line("result.addTypes(op_type.getResults());")
            op_var_list = ", ".join([op.name for op in op_def.operands])
            emitter.line(
                f"if (parser.resolveOperands({{{op_var_list}}}, op_type.getInputs(), parser.getCurrentLocation(), result.operands))"
            )
            emitter.line("  return ::mlir::failure();")
            emitter.blank_line()
            emitter.line("return ::mlir::success();")

    def _emit_asm_parser_body(
        self,
        emitter: CodeEmitter,
        op_def: _d.OpDefinition,
        directives: Sequence[_b.AsmDirective],
    ) -> None:
        for op in op_def.operands:
            emitter.line(f"::mlir::OpAsmParser::UnresolvedOperand {op.name};")

        self._emit_asm_parser_recursive(emitter, op_def, directives)

    def _emit_asm_parser_recursive(
        self,
        emitter: CodeEmitter,
        op_def: _d.OpDefinition,
        directives: Sequence[_b.AsmDirective],
    ) -> None:
        for directive in directives:
            if isinstance(directive, _b.AsmKeyword):
                emitter.line(
                    f'if (parser.parseKeyword("{directive.value}")) return ::mlir::failure();'
                )
            elif isinstance(directive, _b.AsmOperand):
                emitter.line(
                    f"if (parser.parseOperand({directive.name})) return ::mlir::failure();"
                )
            elif isinstance(directive, _b.AsmAttribute):
                emitter.line(
                    f"{self._get_attribute_type_for_parser(op_def, directive.name)} parsed_{directive.name};"
                )
                emitter.line(
                    f'if (parser.parseAttribute(parsed_{directive.name}, "{directive.name}", result.attributes)) return ::mlir::failure();'
                )
            elif isinstance(directive, _b.AsmGroup):
                if not directive.content or not isinstance(
                    directive.content[0], _b.AsmKeyword
                ):
                    # TODO(kehrazy): raise an error
                    continue

                keyword = directive.content[0].value
                with emitter.curly_block(
                    f'if (succeeded(parser.parseOptionalKeyword("{keyword}")))'
                ):
                    self._emit_asm_parser_recursive(
                        emitter, op_def, directive.content[1:]
                    )

    def _get_attribute_type_for_parser(
        self, op_def: _d.OpDefinition, attr_name: str
    ) -> str:
        for attr_def in op_def.attributes:
            if attr_def.name == attr_name:
                return self._get_attribute_type(attr_def)
        return "::mlir::Attribute"

    def _emit_op_builder_defs(
        self, emitter: CodeEmitter, op_def: _d.OpDefinition
    ) -> None:
        op_name_pascal = to_camel_case(op_def.py_op_name)

        emitter.line(
            f"void {op_name_pascal}::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,"
        )
        with emitter.indent():
            for op in op_def.operands:
                emitter.line(f"{self._get_operand_type(op)} {op.name},")
            for attr in op_def.attributes:
                emitter.line(f"{self._get_attribute_type(attr)} {attr.name},")

        emitter.line(") {")
        with emitter.indent():
            if op_def.operands:
                op_names = ", ".join(op.name for op in op_def.operands)
                emitter.line(f"state.addOperands({{{op_names}}});")

            if op_def.attributes:
                emitter.line("::mlir::NamedAttrList attributes;")
                for attr in op_def.attributes:
                    emitter.line(
                        f'attributes.append(builder.getStringAttr("{attr.name}"), {attr.name});'
                    )
                emitter.line("state.addAttributes(attributes);")

        emitter.line("}")
        emitter.blank_line()

        for cpp_builder in op_def.cpp_builders:
            emitter.write(
                f"void {op_name_pascal}::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state"
            )
            if cpp_builder.arguments:
                emitter.write(", ")
                args_str = ", ".join(
                    f"{arg.cpp_type} {arg.name}" for arg in cpp_builder.arguments
                )
                emitter.write(args_str)
            emitter.line(") {")
            with emitter.indent():
                emitter.line(cpp_builder.body)
            emitter.line("}")
            emitter.blank_line()

        if op_def.assembly_format:
            self._emit_op_printer_def(emitter, op_def)
            self._emit_op_parser_def(emitter, op_def)

    def _emit_dialect_h(self) -> None:
        """Generates the C++ Dialect header (.h.inc) file content."""
        emitter = self.dialect_h
        dialect_pascal = to_camel_case(self.dialect.name) + "Dialect"

        emitter.line('#include "mlir/IR/Dialect.h"')
        emitter.blank_line()

        emitter.line(f"// Dialect for {self.dialect.name}")
        emitter.line(f"class {dialect_pascal} : public ::mlir::Dialect {{")
        emitter.line("public:")
        with emitter.indent():
            emitter.line(f"explicit {dialect_pascal}(::mlir::MLIRContext *context);")
            emitter.line(
                f'static ::llvm::StringRef getDialectNamespace() {{ return "{self.dialect.name}"; }}'
            )
            emitter.blank_line()
            emitter.line("void initialize();")
            if self.dialect.enums:
                emitter.line(
                    "::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser, ::mlir::Type type) const override;"
                )
                emitter.line(
                    "void printAttribute(::mlir::Attribute attr, ::mlir::DialectAsmPrinter &printer) const override;"
                )

        emitter.line("};")

    def _emit_dialect_cpp(self) -> None:
        """Generates the C++ Dialect source (.cpp.inc) file content."""
        emitter = self.dialect_cpp
        dialect_pascal = to_camel_case(self.dialect.name) + "Dialect"

        emitter.line(f'#include "{self.dialect.name}Dialect.h"')
        if self.dialect.ops:
            emitter.line(f'#include "{self.dialect.name}Ops.h"')
        if self.dialect.enums:
            emitter.line(f'#include "{self.dialect.name}Attrs.h"')
        emitter.line('#include "mlir/IR/OpImplementation.h"')
        emitter.blank_line()

        # Bring in all op definitions
        if self.dialect.ops:
            emitter.line(f'#include "{self.dialect.name}Ops.cpp.inc"')
            emitter.blank_line()

        emitter.line(f"void {dialect_pascal}::initialize() {{")
        with emitter.indent():
            if self.dialect.enums:
                emitter.line("addAttributes<")
                with emitter.indent():
                    emitter.line("#define GET_ATTRDEF_LIST")
                    emitter.line(f'#include "{self.dialect.name}Attrs.cpp.inc"')
                emitter.line(">();")

            if self.dialect.ops:
                emitter.line("addOperations<")
                with emitter.indent():
                    for op_def in self.dialect.ops.values():
                        op_name_pascal = to_camel_case(op_def.py_op_name)
                        emitter.line(f"{self.dialect.cpp_namespace}::{op_name_pascal},")
                if emitter._line_buffer and self.dialect.ops:
                    emitter._line_buffer[-1] = emitter._line_buffer[-1].rstrip(",")
                emitter.line(">();")

        emitter.line("}")
        emitter.blank_line()

        # Attribute Parsers/Printers
        if self.dialect.enums:
            emitter.line("::mlir::Attribute")
            with emitter.block(
                f"{dialect_pascal}::parseAttribute(::mlir::DialectAsmParser &parser, ::mlir::Type type) const {{",
                "}",
            ):
                emitter.line("::llvm::SMLoc typeLoc = parser.getCurrentLocation();")
                emitter.line("::llvm::StringRef attrTag;")
                emitter.line("::mlir::Attribute attr;")
                emitter.line(
                    "auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);"
                )
                emitter.line("if (parseResult.has_value())")
                emitter.line("    return attr;")
                emitter.line(
                    'parser.emitError(typeLoc) << "unknown attribute `" << attrTag << "` in dialect `" << getNamespace() << "`";'
                )
                emitter.line("return {};")
            emitter.blank_line()

            emitter.line("void")
            with emitter.block(
                f"{dialect_pascal}::printAttribute(::mlir::Attribute attr, ::mlir::DialectAsmPrinter &printer) const {{",
                "}",
            ):
                emitter.line(
                    "if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))"
                )
                emitter.line("    return;")
            emitter.blank_line()


def emit_dialect_cpp(dialect: Dialect) -> dict[str, str]:
    """
    Top-level function to generate C++ files for a Purr dialect.

    Returns:
        A dictionary mapping generated filenames to their content.
    """
    emitter = CppEmitter(dialect)
    return emitter.emit()
