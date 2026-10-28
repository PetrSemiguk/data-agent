"""Tool definitions for Claude and the code that executes them."""

import base64
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.config import (
    MAX_OUTPUT_CHARS,
    MAX_USER_QUESTIONS,
    OUTPUT_HEAD_CHARS,
    OUTPUT_TAIL_CHARS,
)
from agent.sandbox import Sandbox

# Shows a question (with optional choices) to a human and returns the raw answer.
AskUserFn = Callable[[str, list[str] | None], str]


@dataclass
class ToolContext:
    """Per-run state that tools need: the kernel, the run folder and the human."""

    sandbox: Sandbox
    output_dir: Path | None = None
    ask_user_fn: AskUserFn | None = None
    questions_asked: int = 0

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
        "pandas as pd, numpy as np and matplotlib.pyplot as plt are already imported. "
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

SAVE_CHART_TOOL: dict[str, Any] = {
    "name": "save_chart",
    "description": (
        "Run matplotlib code that builds ONE final chart and save it as a PNG file for "
        "the user. Use this only for final charts that belong in the answer; for quick "
        "checks while exploring, use run_code. Make a separate call for each chart. "
        "pandas as pd, numpy as np and matplotlib.pyplot as plt are already imported. "
        "Always label the axes and give the chart a title. Do not call plt.show() or "
        "plt.savefig(): the figure is captured and saved automatically. "
        "Returns the saved file path and size."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Matplotlib code that builds one chart. `df` and earlier variables are available.",
            },
            "filename": {
                "type": "string",
                "description": "Short file name without extension, e.g. 'revenue_by_category'.",
            },
            "title": {
                "type": "string",
                "description": "Optional chart title; replaces the title set in the code.",
            },
        },
        "required": ["code", "filename"],
    },
}

ASK_USER_TOOL: dict[str, Any] = {
    "name": "ask_user",
    "description": (
        "Ask the user a clarifying question. Ask ONLY when the question is ambiguous, "
        "AND different interpretations give substantially different numbers, AND the "
        "data cannot resolve it. Never ask about anything you can check with code. "
        "Ask one question at a time. If you can make a reasonable assumption, make it "
        "and state it in the final answer instead of asking. The number of questions "
        "per run is limited."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "One short, specific question.",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 4,
                "description": "Optional 2-4 answer choices the user can pick from.",
            },
        },
        "required": ["question"],
    },
}

TOOLS: list[dict[str, Any]] = [GET_SCHEMA_TOOL, RUN_CODE_TOOL, SAVE_CHART_TOOL, ASK_USER_TOOL]

USER_UNAVAILABLE = (
    "User is not available. Proceed with the most reasonable assumption "
    "and state it explicitly in the final answer."
)
QUESTION_LIMIT_REACHED = "Question limit reached. Proceed with your best assumption and state it."
NO_ANSWER = (
    "The user gave no answer and left the decision to you. Proceed with the most "
    "reasonable assumption and state it explicitly in the final answer."
)

MAX_FILENAME_CHARS = 60

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


def safe_filename(name: str) -> str:
    """Reduce a model-supplied name to [a-z0-9_-] so it cannot escape the run folder."""
    cleaned = re.sub(r"[^a-z0-9_-]+", "_", name.lower()).strip("_-")
    return cleaned[:MAX_FILENAME_CHARS] or "chart"


def unique_path(folder: Path, stem: str) -> Path:
    """Return folder/stem.png, adding _2, _3, ... instead of overwriting a file."""
    path = folder / f"{stem}.png"
    counter = 2
    while path.exists():
        path = folder / f"{stem}_{counter}.png"
        counter += 1
    return path


def run_code(tool_input: dict[str, Any], ctx: ToolContext) -> tuple[str, bool]:
    """Execute model-written code and turn the result into text."""
    code = tool_input.get("code")
    if not isinstance(code, str) or not code.strip():
        return "Missing required input: code (a non-empty string).", True

    result = ctx.sandbox.execute(code)

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


def get_schema(tool_input: dict[str, Any], ctx: ToolContext) -> tuple[str, bool]:
    """Describe a DataFrame that lives in the kernel."""
    variable = tool_input.get("variable", "df")
    # Only plain names: the value is inserted into code that runs in the kernel.
    if not isinstance(variable, str) or not variable.isidentifier():
        return f"Invalid variable name: {variable!r}. Use a plain Python name like 'df'.", True

    code = f"{SCHEMA_CODE}\nprint(_agent_schema({variable!r}))\ndel _agent_schema"
    result = ctx.sandbox.execute(code)
    if not result.success:
        return f"get_schema failed:\n{result.error}", True

    text = result.stdout.strip()
    return text, text.startswith("ERROR:")


def save_chart(tool_input: dict[str, Any], ctx: ToolContext) -> tuple[str, bool]:
    """Run chart code and save every captured PNG into the run folder."""
    output_dir = ctx.output_dir
    if output_dir is None:
        return "save_chart is not available: no output folder for this run.", True
    code = tool_input.get("code")
    if not isinstance(code, str) or not code.strip():
        return "Missing required input: code (a non-empty string).", True

    title = tool_input.get("title")
    if isinstance(title, str) and title.strip():
        # Only touch a figure the code left open, never create an empty one.
        code += (
            "\nimport matplotlib.pyplot as _plt\n"
            f"if _plt.get_fignums():\n    _plt.gca().set_title({title!r})\n"
        )

    result = ctx.sandbox.execute(code)
    if not result.success:
        return result.error or "Chart code failed.", True
    if not result.images:
        return (
            "Error: no figure was produced. Build a matplotlib chart in the code "
            "and do not call plt.close() before the end.",
            True,
        )

    stem = safe_filename(str(tool_input.get("filename", "")))
    lines: list[str] = []
    for index, image in enumerate(result.images, start=1):
        name = stem if len(result.images) == 1 else f"{stem}_{index}"
        path = unique_path(output_dir, name)
        path.write_bytes(base64.b64decode(image))
        lines.append(f"Saved chart: {path.as_posix()} ({path.stat().st_size / 1024:.1f} KB)")
    if result.stdout.strip():
        lines.append(result.stdout.rstrip())
    return "\n".join(lines), False


def ask_user(tool_input: dict[str, Any], ctx: ToolContext) -> tuple[str, bool]:
    """Pass a clarifying question to the human, within the per-run limit."""
    question = tool_input.get("question")
    if not isinstance(question, str) or not question.strip():
        return "Missing required input: question (a non-empty string).", True
    options = tool_input.get("options")
    if options is not None and (
        not isinstance(options, list)
        or not 2 <= len(options) <= 4
        or not all(isinstance(option, str) for option in options)
    ):
        return "Invalid input: options must be a list of 2-4 strings.", True

    if ctx.ask_user_fn is None:
        return USER_UNAVAILABLE, False
    if ctx.questions_asked >= MAX_USER_QUESTIONS:
        return QUESTION_LIMIT_REACHED, False

    ctx.questions_asked += 1
    answer = ctx.ask_user_fn(question, options).strip()
    if not answer:
        return NO_ANSWER, False
    # Turn "2" into the text of option 2 so the model sees the actual choice.
    if options and answer.isdigit() and 1 <= int(answer) <= len(options):
        answer = options[int(answer) - 1]
    return f"User answered: {answer}", False


ToolHandler = Callable[[dict[str, Any], ToolContext], tuple[str, bool]]

TOOL_HANDLERS: dict[str, ToolHandler] = {
    GET_SCHEMA_TOOL["name"]: get_schema,
    RUN_CODE_TOOL["name"]: run_code,
    SAVE_CHART_TOOL["name"]: save_chart,
    ASK_USER_TOOL["name"]: ask_user,
}


def execute_tool(name: str, tool_input: dict[str, Any], ctx: ToolContext) -> tuple[str, bool]:
    """Run a tool requested by the model.

    Returns the text to send back to the model and whether it is an error.
    """
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        available = ", ".join(TOOL_HANDLERS)
        return f"Unknown tool: {name}. Available tools: {available}.", True

    text, is_error = handler(tool_input, ctx)
    return truncate_output(text), is_error
