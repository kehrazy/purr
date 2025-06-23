"""
A general-purpose, indentation-aware code emission helper.
"""

from __future__ import annotations

import io
from contextlib import contextmanager
from typing import Generator
from collections.abc import Iterator


class CodeEmitter:
    """
    A helper class for emitting code with support for indentation and vertical
    alignment points.

    This class is designed to make generating structured, readable code easier.
    It handles tracking indentation levels and allows for defining alignment
    points to vertically align parts of the code, like variable names or
    assignment operators, across multiple lines.
    """

    def __init__(self, indent_str: str = "  "):
        self._buffer = io.StringIO()
        self._indent_level = 0
        self._indent_str = indent_str
        self._line_buffer: list[str] = []
        self._align_points: dict[str, int] = {}
        self._just_newlined = True

    def get(self) -> str:
        """Returns the entire emitted string."""
        self._flush_line()
        return self._buffer.getvalue()

    def _write_indent(self) -> None:
        if self._just_newlined:
            self._buffer.write(self._indent_str * self._indent_level)
            self._just_newlined = False

    def write(self, *parts: str) -> None:
        """Writes one or more strings to the current line."""
        self._write_indent()
        for part in parts:
            self._line_buffer.append(part)

    def _flush_line(self) -> None:
        if self._line_buffer:
            self._buffer.write("".join(self._line_buffer))
            self._line_buffer = []

    def nl(self, count: int = 1) -> None:
        """Writes one or more newlines."""
        for _ in range(count):
            self._flush_line()
            self._buffer.write("\n")
            self._just_newlined = True

    @contextmanager
    def indent(self, levels: int = 1) -> Generator[None, None, None]:
        """A context manager for temporarily increasing the indent level."""
        self._indent_level += levels
        yield
        self._indent_level -= levels

    def unindent(self, levels: int = 1) -> None:
        """Decreases the indent level."""
        self._indent_level -= levels

    @contextmanager
    def block(
        self,
        prefix: str = "",
        suffix: str = "",
        indent: bool = True,
        trailing_nl: bool = False,
    ) -> Generator[None, None, None]:
        """
        A context manager for a generic, indented code block.

        Example:
            with emitter.block("if (x) {", "}"):
                emitter.emit_line("return 1;")
        """
        self.line(prefix)
        if indent:
            with self.indent():
                yield
        else:
            yield
        self.line(suffix)
        if trailing_nl:
            self.nl()

    def line(self, *parts: str) -> None:
        """Writes a single line, followed by a newline."""
        self.write(*parts)
        self.nl()

    def blank_line(self, count: int = 1) -> None:
        """Writes one or more blank lines."""
        self.nl(count)

    def define_align(self, name: str) -> None:
        """
        Defines a vertical alignment point at the current cursor position.

        Args:
            name: A unique name for the alignment point.
        """
        self._write_indent()
        self._align_points[name] = len("".join(self._line_buffer))

    def align(self, name: str) -> None:
        """
        Aligns the cursor to a previously defined alignment point.

        If the current position is already past the alignment point, this does
        nothing.

        Args:
            name: The name of the alignment point to align to.
        """
        if name not in self._align_points:
            # TODO(kehrazy): raise an error
            return
        target_col = self._align_points[name]
        current_col = len("".join(self._line_buffer))
        if target_col > current_col:
            self.write(" " * (target_col - current_col))

    @contextmanager
    def macro_guard(self, guard_name: str) -> Generator[None, None, None]:
        """
        A context manager to wrap code in C-style macro guards.

        Example:
            with emitter.macro_guard("MY_HEADER_H_"):
                emitter.line("#define MY_AWESOME_MACRO")
        """
        self.line(f"#ifndef {guard_name}")
        self.line(f"#define {guard_name}")
        self.nl()
        yield
        self.nl()
        self.line(f"#endif  // {guard_name}")

    @contextmanager
    def curly_block(self, prefix: str = "") -> Iterator[None]:
        """
        A context manager for a C++-style curly brace block.

        Handles single-line empty blocks (`{}`) vs multi-line non-empty
        blocks.

        Example:
            with emitter.curly_block("if (x)"):
                ...
        """
        self.line(prefix, " {")
        self.indent()

        original_buffer = self._buffer
        self._buffer = io.StringIO()

        yield

        block_content = self._buffer.getvalue()
        self._buffer = original_buffer

        self.unindent()

        if not block_content:
            last_line = self._line_buffer.pop().rstrip()
            self._line_buffer.append(last_line + "}")
        else:
            # Block had content, so write it out and close the brace.
            self._buffer.write(block_content)
            self.line("}")

    @contextmanager
    def namespace(self, name: str) -> Generator[None, None, None]:
        """A context manager for C++ namespaces."""
        self.line(f"namespace {name} {{")
        self.blank_line()
        yield
        self.blank_line()
        self.line(f"}} // namespace {name}")

    def statement(self, line: str) -> Iterator[None]:
        with self.curly_block(line):
            yield
