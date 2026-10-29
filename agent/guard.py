"""Check model-written code before it runs and block risky operations.

This protects against MISTAKES of the model (deleting files, writing data
files past export_data, saving charts past save_chart, shell or network
calls), not against deliberately malicious code. The check reads the code
with `ast`, so it sees what is called, not text that merely looks similar.
It can still be bypassed, e.g. with getattr(), importlib, or names built
from strings at run time. Real isolation comes from running the kernel in a
Docker container (stage 6).
"""

import ast
import re

# Modules whose import alone is blocked: shell commands and network access.
BLOCKED_MODULES = {
    "subprocess": "runs shell commands",
    "socket": "opens network connections",
    "requests": "accesses the network",
    "urllib": "accesses the network",
    "http": "accesses the network",
    "httpx": "accesses the network",
    "ftplib": "accesses the network",
    "smtplib": "accesses the network",
}

# Fully qualified functions that are blocked, after resolving import aliases.
BLOCKED_CALLS = {
    "os.system": "runs shell commands",
    "os.popen": "runs shell commands",
    "os.remove": "deletes files",
    "os.unlink": "deletes files",
    "os.rmdir": "deletes folders",
    "os.removedirs": "deletes folders",
    "shutil.rmtree": "deletes folders",
    "shutil.move": "moves files",
    "eval": "runs code built from a string",
    "exec": "runs code built from a string",
    "compile": "builds code from a string",
    "__import__": "imports modules dynamically",
    "numpy.save": "writes a data file",
    "numpy.savez": "writes a data file",
    "numpy.savetxt": "writes a data file",
}

# pandas methods that write a file when given a path or buffer. Without one,
# to_csv()/to_json() only return a string, which is harmless.
WRITE_METHODS = {
    "to_csv", "to_excel", "to_parquet", "to_json", "to_pickle", "to_feather",
    "to_hdf", "to_sql", "to_stata", "to_orc", "to_xml", "to_html", "to_latex",
}
# Methods that always write a file.
ALWAYS_WRITE_METHODS = {"write_text": "writes a file", "write_bytes": "writes a file"}

WRITE_MODE = re.compile(r"[wax+]")
LOOKS_LIKE_MODE = re.compile(r"^[rwxabt+]+$")

HINT_EXPORT = "To give the user a data file, use the export_data tool (it asks the user first)."
HINT_CHART = "To save a chart, use the save_chart tool; the figure is captured automatically."
HINT_OTHER = "This sandbox does not allow it. Do the analysis with pandas, numpy and scipy on `df`."
MAX_REPORTED = 3


def check_code(code: str) -> str | None:
    """Return an error message if the code uses a blocked operation, else None.

    Code with a syntax error passes: running it shows the real SyntaxError.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    problems = _find_problems(tree)
    if not problems:
        return None
    lines = [f"- line {line}: {what}" for line, what, _ in problems[:MAX_REPORTED]]
    if len(problems) > MAX_REPORTED:
        lines.append(f"- ... and {len(problems) - MAX_REPORTED} more")
    hints = list(dict.fromkeys(hint for _, _, hint in problems))
    return "Blocked before running, nothing was executed:\n" + "\n".join(lines) + "\n" + " ".join(hints)


def _find_problems(tree: ast.Module) -> list[tuple[int, str, str]]:
    aliases = _import_aliases(tree)
    problems: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                root = name.split(".")[0]
                if root in BLOCKED_MODULES:
                    problems.append((node.lineno, f"import {name} ({BLOCKED_MODULES[root]})", HINT_OTHER))
        elif isinstance(node, ast.Call):
            problem = _check_call(node, aliases)
            if problem:
                problems.append((node.lineno, *problem))
    return sorted(problems)


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    """Local name -> full name, e.g. {'np': 'numpy', 'rm': 'os.remove'}."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    # The kernel imports numpy as np at startup.
    aliases.setdefault("np", "numpy")
    return aliases


def _dotted_name(node: ast.expr, aliases: dict[str, str]) -> str | None:
    """'os.system' for os.system(...), with import aliases resolved."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(aliases.get(node.id, node.id))
    return ".".join(reversed(parts))


def _check_call(node: ast.Call, aliases: dict[str, str]) -> tuple[str, str] | None:
    name = _dotted_name(node.func, aliases)
    if name in BLOCKED_CALLS:
        what = BLOCKED_CALLS[name]
        return f"{name}() {what}", HINT_EXPORT if what == "writes a data file" else HINT_OTHER
    if name and name.split(".")[0] in BLOCKED_MODULES:
        return f"{name}() {BLOCKED_MODULES[name.split('.')[0]]}", HINT_OTHER

    if name == "open" or name == "io.open":
        mode = _mode_argument(node, position=1)
        if mode is None or WRITE_MODE.search(mode):
            return f"open(..., mode={mode!r}) writes a file", HINT_EXPORT
        return None

    if not isinstance(node.func, ast.Attribute):
        return None
    method = node.func.attr
    if method == "savefig":
        return "savefig() writes a chart file", HINT_CHART
    if method in ALWAYS_WRITE_METHODS:
        return f".{method}() {ALWAYS_WRITE_METHODS[method]}", HINT_EXPORT
    if method in WRITE_METHODS and (node.args or _has_target_keyword(node)):
        return f".{method}(...) writes a data file", HINT_EXPORT
    if method == "open":
        # Path(...).open("w"): the mode is the first argument, if it looks like one.
        mode = _mode_argument(node, position=0)
        if mode is not None and LOOKS_LIKE_MODE.match(mode) and WRITE_MODE.search(mode):
            return f".open(mode={mode!r}) writes a file", HINT_EXPORT
    return None


def _mode_argument(node: ast.Call, position: int) -> str | None:
    """The file mode of an open() call: 'r' if absent, None if not a literal string."""
    for keyword in node.keywords:
        if keyword.arg == "mode":
            return keyword.value.value if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str) else None
    if len(node.args) > position:
        arg = node.args[position]
        return arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None
    return "r"


def _has_target_keyword(node: ast.Call) -> bool:
    targets = {"path_or_buf", "path", "excel_writer", "buf", "name", "con", "fname"}
    return any(keyword.arg in targets for keyword in node.keywords)
