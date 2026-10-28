"""Tool definitions for Claude and the code that executes them."""

from collections.abc import Callable
from typing import Any

from agent.config import MAX_OUTPUT_CHARS, OUTPUT_HEAD_CHARS, OUTPUT_TAIL_CHARS
from agent.sandbox import Sandbox

GET_SCHEMA_TOOL: dict[str, Any] = {
    "name": "get_schema",
    "description": (
        "Return a compact overview of a pandas DataFrame in the kernel: number of rows "
        "and columns, full duplicate rows, memory use, and for each column its dtype, "
        "missing values, unique count, sample values, min/max/mean for numbers, "
        "min/max for dates, and a 'looks like date' flag for text columns that hold dates. "
        "Use this FIRST to explore the structure of the data instead of writing your own "
        "exploration code. Call it again on derived DataFrames (for example a cleaned "
        "copy named `data`) to check the result of your transformations."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "variable": {
                "type": "string",
                "description": "Name of the DataFrame variable in the kernel. Defaults to 'df'.",
            },
        },
    },
}

RUN_CODE_TOOL: dict[str, Any] = {
    "name": "run_code",
    "description": (
        "Execute Python code in a persistent Jupyter kernel and return its output. "
        "The dataset is already loaded into a pandas DataFrame named `df`. "
        "Variables, imports and functions persist between calls, so build the "
        "analysis step by step instead of repeating earlier work. "
        "To see a value, print() it or leave the expression as the last line of "
        "the code. Matplotlib charts are captured automatically; you will be told "
        "how many charts were generated, but you cannot see them. "
        "Errors are returned with the exception type and traceback. "
        "Available libraries: pandas, numpy, matplotlib, scipy. Installing other "
        "packages is not allowed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute.",
            },
        },
        "required": ["code"],
    },
}

TOOLS: list[dict[str, Any]] = [GET_SCHEMA_TOOL, RUN_CODE_TOOL]

MAX_SCHEMA_COLUMNS = 50
DATE_SAMPLE_SIZE = 200

# Runs inside the kernel, where the data lives. Errors are returned as text
# starting with "ERROR:" so the tool can report them cleanly.
SCHEMA_CODE = f'''
def _agent_schema(name):
    import re
    import warnings
    import pandas as pd

    obj = globals().get(name)
    if obj is None:
        return f"ERROR: variable '{{name}}' does not exist in the kernel."
    if not isinstance(obj, pd.DataFrame):
        return f"ERROR: '{{name}}' is a {{type(obj).__name__}}, not a pandas DataFrame."

    frame = obj
    rows, cols = frame.shape
    memory_mb = frame.memory_usage(deep=True).sum() / 1024 ** 2
    lines = [
        f"DataFrame '{{name}}': {{rows}} rows x {{cols}} columns, "
        f"{{int(frame.duplicated().sum())}} full duplicate rows, {{memory_mb:.1f}} MB"
    ]

    date_like = re.compile(r"\\d{{1,4}}[-/.]\\d{{1,2}}")
    for col in frame.columns[:{MAX_SCHEMA_COLUMNS}]:
        s = frame[col]
        missing = int(s.isna().sum())
        pct = missing / rows * 100 if rows else 0.0
        non_null = s.dropna()
        examples = [str(v)[:40] for v in non_null.unique()[:3]]
        line = (
            f"- {{col}}: {{s.dtype}}, missing {{missing}} ({{pct:.1f}}%), "
            f"unique {{s.nunique()}}, examples {{examples}}"
        )
        if pd.api.types.is_datetime64_any_dtype(s):
            line += f", min {{s.min()}}, max {{s.max()}}"
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            line += f", min {{s.min():.6g}}, max {{s.max():.6g}}, mean {{s.mean():.6g}}"
        elif pd.api.types.is_string_dtype(s) or s.dtype == object:
            # Check only a small sample so this stays fast on large data.
            sample = non_null.head({DATE_SAMPLE_SIZE}).astype(str)
            if len(sample) and sample.str.contains(date_like).mean() >= 0.9:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                if parsed.notna().mean() >= 0.9:
                    line += ", looks like date"
        lines.append(line)

    hidden = cols - {MAX_SCHEMA_COLUMNS}
    if hidden > 0:
        lines.append(f"... {{hidden}} more columns not shown")
    return "\\n".join(lines)
'''


def truncate_output(text: str) -> str:
    """Keep the head and tail of a long output so it fits in MAX_OUTPUT_CHARS."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    omitted = len(text) - OUTPUT_HEAD_CHARS - OUTPUT_TAIL_CHARS
    marker = (
        f"\n... [truncated {omitted} chars] ... "
        "Tip: print less — use .head(), aggregates or value_counts().\n"
    )
    return text[:OUTPUT_HEAD_CHARS] + marker + text[-OUTPUT_TAIL_CHARS:]


def run_code(tool_input: dict[str, Any], sandbox: Sandbox) -> tuple[str, bool]:
    """Execute model-written code and turn the result into text."""
    code = tool_input.get("code")
    if not isinstance(code, str) or not code.strip():
        return "Missing required input: code (a non-empty string).", True

    result = sandbox.execute(code)

    parts: list[str] = []
    if result.stdout:
        parts.append(result.stdout.rstrip())
    if result.result is not None:
        parts.append(result.result)
    if result.images:
        parts.append(f"[{len(result.images)} chart(s) generated]")
    if result.error:
        parts.append(result.error)

    text = "\n".join(parts) if parts else "[code executed, no output]"
    return text, not result.success


def get_schema(tool_input: dict[str, Any], sandbox: Sandbox) -> tuple[str, bool]:
    """Describe a DataFrame that lives in the kernel."""
    variable = tool_input.get("variable", "df")
    # Only plain names: the value is inserted into code that runs in the kernel.
    if not isinstance(variable, str) or not variable.isidentifier():
        return f"Invalid variable name: {variable!r}. Use a plain Python name like 'df'.", True

    code = f"{SCHEMA_CODE}\nprint(_agent_schema({variable!r}))\ndel _agent_schema"
    result = sandbox.execute(code)
    if not result.success:
        return f"get_schema failed:\n{result.error}", True

    text = result.stdout.strip()
    return text, text.startswith("ERROR:")


TOOL_HANDLERS: dict[str, Callable[[dict[str, Any], Sandbox], tuple[str, bool]]] = {
    GET_SCHEMA_TOOL["name"]: get_schema,
    RUN_CODE_TOOL["name"]: run_code,
}


def execute_tool(name: str, tool_input: dict[str, Any], sandbox: Sandbox) -> tuple[str, bool]:
    """Run a tool requested by the model.

    Returns the text to send back to the model and whether it is an error.
    """
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        available = ", ".join(TOOL_HANDLERS)
        return f"Unknown tool: {name}. Available tools: {available}.", True

    text, is_error = handler(tool_input, sandbox)
    return truncate_output(text), is_error
