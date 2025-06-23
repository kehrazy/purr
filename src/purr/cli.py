import ast
import importlib.util
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import rich
import rich.progress
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from purr._generator.cpp import emit_dialect_cpp
from purr._generator.tablegen import emit_dialect_tablegen
from purr._internal.domain import Dialect

ATTR_TYPE_MAP = {
    "BoolAttr": "bool",
    "F32Attr": "float",
    "F64Attr": "float",
    "I32Attr": "int",
    "I64Attr": "int",
    "StringAttr": "str",
    "SymbolRefAttr": "str",
    "TypeAttr": "object",
}


@dataclass
class FieldInfo:
    name: str
    field_type: str


@dataclass
class MethodInfo:
    name: str
    signature: str


@dataclass
class StubClassInfo:
    name: str
    docstring: str
    fields: list[FieldInfo] = field(default_factory=list)
    methods: list[MethodInfo] = field(default_factory=list)


class PurrVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stub_classes: list[StubClassInfo] = []
        self._current_class_info: StubClassInfo | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        is_purr_op = any(
            (isinstance(base, ast.Name) and base.id == "Op")
            or (isinstance(base, ast.Attribute) and base.attr == "Op")
            for base in node.bases
        )
        if not is_purr_op:
            self.generic_visit(node)
            return
        self._current_class_info = StubClassInfo(
            name=node.name, docstring=ast.get_docstring(node) or ""
        )
        for child in node.body:
            self.visit(child)
        if self._current_class_info:
            self.stub_classes.append(self._current_class_info)
        self._current_class_info = None

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._current_class_info is None:
            return
        if not (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "op"
        ):
            return

        field_name = node.targets[0].id  # type: ignore
        dsl_call_name = node.value.func.attr
        field_info = None
        if dsl_call_name == "operand":
            field_info = FieldInfo(name=field_name, field_type="OperandStub")
        elif dsl_call_name == "result":
            field_info = FieldInfo(name=field_name, field_type="ResultStub")
        elif dsl_call_name == "attribute":
            attr_type_node = node.value.args[0]
            py_type = "object"
            if isinstance(attr_type_node, ast.Attribute) and hasattr(
                attr_type_node, "attr"
            ):
                py_type = ATTR_TYPE_MAP.get(attr_type_node.attr, "object")
            field_info = FieldInfo(
                name=field_name, field_type=f"AttributeStub[{py_type}]"
            )
        if field_info:
            self._current_class_info.fields.append(field_info)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._current_class_info is None or node.name.startswith("_"):
            return
        args = [arg.arg for arg in node.args.args]
        signature = f"def {node.name}({', '.join(args)})"
        if node.returns:
            return_type_str = "object"
            if isinstance(node.returns, ast.Name):
                return_type_str = node.returns.id
            elif isinstance(node.returns, ast.Attribute):
                return_type_str = f"{node.returns.value.id}.{node.returns.attr}"  # type: ignore
            signature += f" -> {return_type_str}"
        signature += ": ..."
        self._current_class_info.methods.append(
            MethodInfo(name=node.name, signature=signature)
        )


@lru_cache
def find_executable(name: str) -> str | None:
    """Tries to find an executable in the PATH."""
    return shutil.which(name)


def executable_exists(name: str) -> bool:
    """Checks if an executable exists in the PATH."""
    return find_executable(name) is not None


def format_with_ruff(source: str) -> str:
    """Format a Python file with ruff."""
    args = ["ruff", "format", "--check", "--diff", "--stdin-filename", "stdin.py"]
    process = subprocess.run(args, input=source, capture_output=True, text=True)
    if process.returncode != 0:
        return source
    return process.stdout


def format_python_file(source: str) -> str:
    """
    Tries to format a Python file with `ruff` if it is available.
    Otherwise, returns the source unmodified.
    """
    if executable_exists("ruff"):
        return format_with_ruff(source)
    return source


def format_stub_file_content(
    stub_infos: list[StubClassInfo],
    progress: Progress,
    task: rich.progress.TaskID,
) -> str:
    """Format a stub file content."""
    progress.update(task, description="Generating the stub file...")
    lines = [
        "# AUTO-GENERATED by purr. DO NOT EDIT.",
        "# flake8: noqa",
        "# pylint: skip-file",
        "",
        "from purr import Dialect",
        "from purr._internal.stubs import OpStub, OperandStub, ResultStub, AttributeStub",
        "from purr._internal import types",
        "",
    ]
    for info in stub_infos:
        lines.append(f"class {info.name}(OpStub):")
        if info.docstring:
            lines.append(f'    """{info.docstring}"""')
        if not info.fields and not info.methods:
            lines.append("    ...")
            lines.append("")
            continue
        for inner_field in info.fields:
            lines.append(f"    {inner_field.name}: {inner_field.field_type}")
        if info.fields and info.methods:
            lines.append("")
        for method in info.methods:
            lines.append(f"    {method.signature}")
        lines.append("")
    contents = "\n".join(lines)
    progress.update(task, description="Formatting the stub file...")
    formatted = format_python_file(contents)
    progress.update(task, description="Stub file formatted.")
    return formatted


app = typer.Typer(
    name="purr",
    help="🐱 A cute, Pythonic, and extensible DSL for defining MLIR dialects and ops.",
    add_completion=True,
)
console = Console()


@app.command()
def stub(
    files: list[Path] = typer.Argument(  # noqa: B008
        ...,
        help="One or more Python files containing purr dialect definitions.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output."
    ),
) -> None:
    """
    Generate .pyi stub files for purr dialect definitions.

    This enhances your IDE's autocompletion and type-checking capabilities
    by providing static type information for your dynamically defined Ops.
    """

    total_files = len(files)
    ops_found_counts: list[int] = [0] * total_files

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing files...", total=total_files)
        for i, filepath in enumerate(files):
            progress.update(
                task, description=f"Processing [cyan]{filepath.name}[/cyan]..."
            )
            if verbose:
                console.print(f"\n[dim]Inspecting: {filepath}[/dim]")

            try:
                source = filepath.read_text()
                tree = ast.parse(source)
            except SyntaxError as e:
                console.print(
                    f"[bold red]Error:[/] Could not parse [cyan]{filepath.name}[/]: {e}"
                )
                progress.advance(task)
                continue
            except Exception as e:
                console.print(
                    f"[bold red]Error:[/] An unexpected error occurred while "
                    f"processing [cyan]{filepath.name}[/]: {e}"
                )
                progress.advance(task)
                continue

            visitor = PurrVisitor()
            visitor.visit(tree)

            ops_found_counts[i] = len(visitor.stub_classes)

            if not visitor.stub_classes:
                if verbose:
                    console.print(
                        "  [dim]-> No `purr.Op` classes found. Skipping.[/dim]"
                    )
                progress.advance(task)
                continue

            stub_content_task = progress.add_task(
                f"Generating stub for [cyan]{filepath.name}[/cyan]...", total=1
            )
            stub_content = format_stub_file_content(
                visitor.stub_classes, progress, stub_content_task
            )
            stub_filepath = filepath.with_suffix(".pyi")
            stub_filepath.write_text(stub_content)
            progress.update(stub_content_task, completed=1)

            progress.update(task, advance=1)

    for i, filepath in enumerate(files):
        if ops_found_counts[i] > 0:
            console.print(
                f"  [green]-> Found {ops_found_counts[i]} Op(s). "
                f"Stub written to [cyan]{filepath.name}[/cyan]"
            )


gen_app = typer.Typer(
    name="generate",
    help="Generate various outputs from purr dialect definitions.",
    add_completion=True,
    pretty_exceptions_enable=False,
    short_help="Generate MLIR dialect files from DSL dialect definitions.",
)
app.add_typer(gen_app, name="generate")


@gen_app.command("tablegen")
def generate_tablegen(
    file: Path = typer.Argument(  # noqa: B008
        ...,
        help="The Python file containing purr dialect definitions.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    dialect_name: str = typer.Option(
        ...,
        "--dialect",
        "-d",
        help="The variable name of the Dialect object to generate.",
    ),
    output: Path = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        help="The output path for the TableGen file. Prints to stdout by default.",
        file_okay=True,
        dir_okay=False,
        writable=True,
    ),
) -> None:
    """
    Generate a TableGen (.td) file from a purr dialect definition.
    """

    try:
        spec = importlib.util.spec_from_file_location(file.stem, file)
        if spec is None or spec.loader is None:
            console.print(
                "[bold red]Error:[/] "
                f"Could not create module spec for [cyan]{file.name}[/]."
            )
            raise typer.Exit(1)

        module = importlib.util.module_from_spec(spec)
        sys.modules[file.stem] = module
        spec.loader.exec_module(module)

        dialect_obj = getattr(module, dialect_name, None)

        if dialect_obj is None or not isinstance(dialect_obj, Dialect):
            console.print(
                "[bold red]Error:[/] Dialect object [bold cyan]"
                f"'{dialect_name}'[/bold cyan] not found in [cyan]{file.name}[/]."
            )
            raise typer.Exit(1)

        console.print(
            f"Found dialect [bold cyan]'{dialect_obj.name}'[/bold cyan]. "
            "Generating TableGen..."
        )

        tablegen_str = emit_dialect_tablegen(dialect_obj)

        if output:
            output.write_text(tablegen_str)
            console.print(
                "[bold green]Success![/] "
                f"TableGen file written to [cyan]{output}[/cyan]."
            )
        else:
            console.print("\n" + tablegen_str)

    except (ModuleNotFoundError, AttributeError) as e:
        console.print(
            "[bold red]Error:[/] "
            f"Failed to import or find dialect in [cyan]{file.name}[/]: {e}"
        )
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[bold red]Error:[/] An unexpected error occurred: {e}")
        raise typer.Exit(1) from e


@gen_app.command("cpp")
def generate_cpp(
    file: Path = typer.Argument(  # noqa: B008
        ...,
        help="The Python file containing purr dialect definitions.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    dialect_name: str = typer.Option(
        ...,
        "--dialect",
        "-d",
        help="The variable name of the Dialect object to generate.",
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        ...,
        "--output-dir",
        "-o",
        help="The output directory for the generated C++ files.",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
) -> None:
    """
    Generate C++ (.h.inc, .cpp.inc) files from a purr dialect definition.
    """

    try:
        spec = importlib.util.spec_from_file_location(file.stem, file)
        if spec is None or spec.loader is None:
            console.print(
                "[bold red]Error:[/] "
                f"Could not create module spec for [cyan]{file.name}[/]."
            )
            raise typer.Exit(1)

        module = importlib.util.module_from_spec(spec)
        sys.modules[file.stem] = module
        spec.loader.exec_module(module)

        dialect_obj = getattr(module, dialect_name, None)

        if dialect_obj is None or not isinstance(dialect_obj, Dialect):
            console.print(
                "[bold red]Error:[/] Dialect object [bold cyan]"
                f"'{dialect_name}'[/bold cyan] not found in [cyan]{file.name}[/]."
            )
            raise typer.Exit(1)

        console.print(
            f"Found dialect [bold cyan]'{dialect_obj.name}'[/bold cyan]. "
            "Generating C++..."
        )

        generated_files = emit_dialect_cpp(dialect_obj)

        output_dir.mkdir(parents=True, exist_ok=True)

        console.print(
            f"[bold green]✓ Success![/] C++ files written to [cyan]{output_dir}[/cyan]:"
        )
        for filename, content in generated_files.items():
            if content:
                path = output_dir / filename
                path.write_text(content)
                console.print(f"  - [green]Generated[/green] [cyan]{filename}[/cyan]")
            else:
                console.print(
                    f"  - [yellow]Skipped[/yellow]   [dim]{filename}[/dim] (no content)"
                )

    except (ModuleNotFoundError, AttributeError) as e:
        console.print(
            "[bold red]Error:[/] "
            f"Failed to import or find dialect in [cyan]{file.name}[/]: {e}"
        )
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[bold red]Error:[/] An unexpected error occurred: {e}")
        raise typer.Exit(1) from e


@app.command()
def version() -> None:
    """
    Show the Purr CLI version.
    """
    from purr import __version__

    console.print(f"Purr CLI Version: [bold cyan]{__version__}[/bold cyan]")


if __name__ == "__main__":
    app()  # Typer handles the rest
