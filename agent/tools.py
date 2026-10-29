"""Tool definitions for Claude and the code that executes them."""

import base64
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.config import (
    MAX_APPROVALS,
    MEMORY_MAX_NOTES,
    MEMORY_NOTE_MAX_CHARS,
    MAX_OUTPUT_CHARS,
    MAX_USER_QUESTIONS,
    OUTPUT_HEAD_CHARS,
    OUTPUT_TAIL_CHARS,
)
from agent.guard import check_code
from agent.memory import NOTE_KINDS
from agent.sandbox import Sandbox

# Shows a question (with optional choices) to a human and returns the raw answer.
AskUserFn = Callable[[str, list[str] | None], str]
# Shows an approval request to a human and returns True if they approve.
ApproveFn = Callable[["ApprovalRequest"], bool]


@dataclass
class ToolContext:
    """Per-run state that tools need: the kernel, the run folder and the human."""

    sandbox: Sandbox
    output_dir: Path | None = None
    ask_user_fn: AskUserFn | None = None
    questions_asked: int = 0
    approvals_requested: int = 0
    # Artifacts of the latest tool call, for displaying them; reset on every call.
    last_images: list[str] = field(default_factory=list)
    last_chart_paths: list[str] = field(default_factory=list)
    # The agent's current analysis plan, replaced as a whole by update_plan.
    plan: list[dict[str, str]] | None = None
    # Separate kernel namespace for run_code and get_schema (the verifier's).
    namespace: str | None = None
    # Ids of the dataset's notes in memory; None when memory is off for this run.
    memory_note_ids: set[int] | None = None
    # Notes from remember calls, saved to memory only when the run succeeds.
    pending_notes: list[dict[str, Any]] = field(default_factory=list)

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
        "Code that writes files, deletes files, runs shell commands or uses the network "
        "is blocked before it runs; use save_chart for charts and export_data for data files. "
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

CLAIM_STATUSES = ("supported", "contradicted", "unverifiable")

REPORT_VERIFICATION_TOOL: dict[str, Any] = {
    "name": "report_verification",
    "description": (
        "Submit the result of your verification. You MUST finish your work by calling "
        "this tool exactly once, with every checkable claim of the answer. Use it after "
        "you have recomputed the claims with code; do not write a text answer instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": "The claim quoted verbatim from the answer (a short fragment).",
                        },
                        "status": {"type": "string", "enum": list(CLAIM_STATUSES)},
                        "evidence": {
                            "type": "string",
                            "description": "What your code showed, with the numbers it printed.",
                        },
                        "correction": {
                            "type": "string",
                            "description": "Only for 'contradicted': what the correct statement is.",
                        },
                    },
                    "required": ["claim", "status", "evidence"],
                },
            },
        },
        "required": ["claims"],
    },
}

PLAN_STATUSES = ("pending", "in_progress", "done", "skipped")
MAX_PLAN_ITEMS = 8

UPDATE_PLAN_TOOL: dict[str, Any] = {
    "name": "update_plan",
    "description": (
        "Record your analysis plan and its progress. Required when the question needs "
        "several stages of analysis; skip it for a simple question that 1-2 code calls "
        "answer. After get_schema, create a plan of 3-6 items. Update statuses in the same response as the work on an item: mark it "
        "in_progress with its first tool call and done with its last one (for example "
        "together with save_chart). Never spend a separate step only on update_plan; the "
        "plan must already be closed when you give the final answer. Always send the "
        "WHOLE plan, not only the changed items. Mark an item 'skipped' only with the "
        "reason in 'note'. This tool runs no code; it only stores the plan."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_PLAN_ITEMS,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Short stable id, e.g. '1'."},
                        "text": {"type": "string", "description": "What this step does."},
                        "status": {"type": "string", "enum": list(PLAN_STATUSES)},
                        "note": {
                            "type": "string",
                            "description": "Optional; required when status is 'skipped'.",
                        },
                    },
                    "required": ["id", "text", "status"],
                },
            },
        },
        "required": ["items"],
    },
}

REMEMBER_TOOL: dict[str, Any] = {
    "name": "remember",
    "description": (
        "Save a note to this dataset's memory, for future conversations with this data. "
        "Remember only lasting facts about the data (e.g. duplicates, cancellations, "
        "date range, data quality) and definitions or preferences the user stated "
        "explicitly (e.g. 'revenue means net revenue'). Do not remember answers to "
        "specific questions. Do not duplicate a note that is already in memory; refine "
        "it with `replaces`. One fact per note, at most 160 characters; split longer "
        "notes into separate calls. A data_fact needs evidence: numbers your code "
        "printed. A data_fact is saved only if the fact-checker confirms it, and notes "
        "are saved only if this run finishes successfully. When the user asks you to "
        "remember something, call this tool."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": list(NOTE_KINDS)},
            "text": {"type": "string", "description": "One fact in one short sentence (max 160 characters)."},
            "evidence": {
                "type": "string",
                "description": "Required for data_fact: what your code printed, with numbers.",
            },
            "replaces": {
                "type": "integer",
                "description": "Id of the memory note this one corrects or refines.",
            },
        },
        "required": ["kind", "text"],
    },
}

EXPORT_DATA_TOOL: dict[str, Any] = {
    "name": "export_data",
    "description": (
        "Save a DataFrame from the kernel as a CSV or Excel file for the user to download. "
        "Use it ONLY when the user explicitly asks to export, save or download data. "
        "The user must approve every export first; if they deny it, do not retry. "
        "Prepare the DataFrame with run_code first, then pass its variable name."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "variable": {
                "type": "string",
                "description": "Name of the DataFrame variable in the kernel, e.g. 'negative_rows'.",
            },
            "filename": {
                "type": "string",
                "description": "Short file name without extension, e.g. 'negative_quantity_rows'.",
            },
            "format": {"type": "string", "enum": ["csv", "xlsx"]},
            "reason": {
                "type": "string",
                "description": "One sentence for the user: what the file contains and why it is exported.",
            },
        },
        "required": ["variable", "filename", "format", "reason"],
    },
}

TOOLS: list[dict[str, Any]] = [
    GET_SCHEMA_TOOL,
    UPDATE_PLAN_TOOL,
    RUN_CODE_TOOL,
    SAVE_CHART_TOOL,
    EXPORT_DATA_TOOL,
    ASK_USER_TOOL,
]

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


def unique_path(folder: Path, stem: str, suffix: str = ".png") -> Path:
    """Return folder/stem.suffix, adding _2, _3, ... instead of overwriting a file."""
    path = folder / f"{stem}{suffix}"
    counter = 2
    while path.exists():
        path = folder / f"{stem}_{counter}{suffix}"
        counter += 1
    return path


def run_code(tool_input: dict[str, Any], ctx: ToolContext) -> tuple[str, bool]:
    """Execute model-written code and turn the result into text."""
    code = tool_input.get("code")
    if not isinstance(code, str) or not code.strip():
        return "Missing required input: code (a non-empty string).", True
    blocked = check_code(code)
    if blocked:
        return blocked, True

    result = ctx.sandbox.execute(code, namespace=ctx.namespace)

    parts: list[str] = []
    if result.stdout:
        parts.append(result.stdout.rstrip())
    if result.result is not None:
        parts.append(result.result)
    if result.images:
        parts.append(f"[{len(result.images)} chart(s) generated]")
        ctx.last_images = result.images
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
    result = ctx.sandbox.execute(code, namespace=ctx.namespace)
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
    blocked = check_code(code)
    if blocked:
        return blocked, True

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
        ctx.last_chart_paths.append(path.as_posix())
        lines.append(f"Saved chart: {path.as_posix()} ({path.stat().st_size / 1024:.1f} KB)")
    if result.stdout.strip():
        lines.append(result.stdout.rstrip())
    return "\n".join(lines), False


def check_ask_user(tool_input: dict[str, Any], ctx: ToolContext) -> tuple[str, bool] | None:
    """Reply for an ask_user call that must not reach the human, or None if it may.

    Covers invalid input and the per-run question limit. Used both here and by the
    agent loop when it pauses the run to wait for an answer.
    """
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
    if ctx.questions_asked >= MAX_USER_QUESTIONS:
        return QUESTION_LIMIT_REACHED, False
    return None


def resolve_answer(answer: str, options: list[str] | None) -> str:
    """Clean the human's answer; an option number becomes the option text."""
    answer = answer.strip()
    if options and answer.isdigit() and 1 <= int(answer) <= len(options):
        return options[int(answer) - 1]
    return answer


def answer_for_model(resolved: str) -> str:
    """Tool result text for an answer; empty means the user left it to the agent."""
    return f"User answered: {resolved}" if resolved else NO_ANSWER


def ask_user(tool_input: dict[str, Any], ctx: ToolContext) -> tuple[str, bool]:
    """Pass a clarifying question to the human, within the per-run limit."""
    rejected = check_ask_user(tool_input, ctx)
    if rejected is not None:
        return rejected
    if ctx.ask_user_fn is None:
        return USER_UNAVAILABLE, False

    ctx.questions_asked += 1
    options = tool_input.get("options")
    answer = ctx.ask_user_fn(tool_input["question"], options)
    return answer_for_model(resolve_answer(answer, options)), False


def validate_plan(items: Any) -> list[str]:
    """Problems with a plan sent to update_plan; an empty list means it is valid."""
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_PLAN_ITEMS:
        return [f"items must be a list of 1-{MAX_PLAN_ITEMS} plan items."]
    problems: list[str] = []
    seen: set[str] = set()
    for position, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            problems.append(f"item {position} must be an object.")
            continue
        item_id, text, status = item.get("id"), item.get("text"), item.get("status")
        if not isinstance(item_id, str) or not item_id.strip():
            problems.append(f"item {position} needs a non-empty 'id'.")
        elif item_id in seen:
            problems.append(f"duplicate id '{item_id}'.")
        else:
            seen.add(item_id)
        if not isinstance(text, str) or not text.strip():
            problems.append(f"item {position} needs a non-empty 'text'.")
        if status not in PLAN_STATUSES:
            problems.append(f"item {position} has invalid status {status!r}; use one of {', '.join(PLAN_STATUSES)}.")
        note = item.get("note")
        if status == "skipped" and not (isinstance(note, str) and note.strip()):
            problems.append(f"item {position} is 'skipped' without a reason in 'note'.")
    return problems


def plan_summary(plan: list[dict[str, str]]) -> str:
    """Short progress line such as '2/5 done, 1 skipped'."""
    done = sum(item["status"] == "done" for item in plan)
    skipped = sum(item["status"] == "skipped" for item in plan)
    return f"{done}/{len(plan)} done" + (f", {skipped} skipped" if skipped else "")


def update_plan(tool_input: dict[str, Any], ctx: ToolContext) -> tuple[str, bool]:
    """Replace the stored plan with the one the model sent."""
    items = tool_input.get("items")
    problems = validate_plan(items)
    if problems:
        return "Plan not updated: " + " ".join(problems) + " Send the whole plan again.", True
    ctx.plan = [
        {key: str(item[key]).strip() for key in ("id", "text", "status", "note") if item.get(key)}
        for item in items
    ]
    return f"Plan updated: {plan_summary(ctx.plan)}.", False


def validate_claims(claims: Any) -> list[str]:
    """Problems with the claims sent to report_verification; empty means valid."""
    if not isinstance(claims, list) or not claims:
        return ["claims must be a non-empty list."]
    problems: list[str] = []
    for position, item in enumerate(claims, start=1):
        if not isinstance(item, dict):
            problems.append(f"claim {position} must be an object.")
            continue
        for key in ("claim", "evidence"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                problems.append(f"claim {position} needs a non-empty '{key}'.")
        status = item.get("status")
        if status not in CLAIM_STATUSES:
            problems.append(f"claim {position} has invalid status {status!r}; use one of {', '.join(CLAIM_STATUSES)}.")
        correction = item.get("correction")
        if status == "contradicted" and not (isinstance(correction, str) and correction.strip()):
            problems.append(f"claim {position} is 'contradicted' without a 'correction'.")
    return problems


def report_verification(tool_input: dict[str, Any], ctx: ToolContext) -> tuple[str, bool]:
    """Check the verifier's report; the loop ends the run once it is accepted."""
    problems = validate_claims(tool_input.get("claims"))
    if problems:
        return "Report not accepted: " + " ".join(problems) + " Send the whole report again.", True
    return f"Report accepted: {len(tool_input['claims'])} claims.", False


EXPORT_FORMATS = ("csv", "xlsx")
EXCEL_MAX_ROWS = 1_048_575  # one row of the sheet is the header
MAX_SHOWN_COLUMNS = 8
SIZE_SAMPLE_ROWS = 1000
APPROVAL_LIMIT_REACHED = "Approval limit reached for this run. Do not request more exports."
EXPORT_DENIED = "User denied the export. Do not retry; continue without it and mention it in the answer."
EXPORT_NO_APPROVER = (
    "No user is available to approve the export, so it was denied automatically. "
    "Do not retry; continue without it and mention it in the answer."
)

# Runs in the kernel: shape, columns and an estimated CSV size of a DataFrame.
EXPORT_INFO_CODE = f'''
def _agent_export_info(name):
    import json
    import pandas as pd

    obj = globals().get(name)
    if obj is None:
        return json.dumps({{"error": f"variable '{{name}}' does not exist in the kernel."}})
    if not isinstance(obj, pd.DataFrame):
        return json.dumps({{"error": f"'{{name}}' is a {{type(obj).__name__}}, not a pandas DataFrame."}})
    sample = obj.head({SIZE_SAMPLE_ROWS})
    per_row = len(sample.to_csv(index=False).encode()) / max(len(sample), 1)
    return json.dumps({{"rows": len(obj), "columns": [str(c) for c in obj.columns], "bytes": per_row * len(obj)}})
'''


@dataclass(frozen=True)
class ApprovalRequest:
    """An action the model wants to take that a human must approve first."""

    tool: str
    # One line, e.g. "Export `neg` (10,624 rows) to neg.xlsx".
    action: str
    # Everything the human should see before deciding, in display order.
    details: dict[str, str]
    variable: str
    file_format: str
    path: str


def format_size(size: float) -> str:
    if size < 1024:
        return f"{size:.0f} B"
    return f"{size / 1024 / 1024:.1f} MB" if size >= 1024 * 1024 else f"{size / 1024:.0f} KB"


def prepare_export(tool_input: dict[str, Any], ctx: ToolContext) -> ApprovalRequest | tuple[str, bool]:
    """Validate an export_data call and describe it for approval; nothing is written yet.

    Returns the request, or (message for the model, is_error) if it cannot go ahead.
    """
    if ctx.output_dir is None:
        return "export_data is not available: no output folder for this run.", True
    variable = tool_input.get("variable")
    if not isinstance(variable, str) or not variable.isidentifier():
        return f"Invalid variable name: {variable!r}. Use a plain Python name of a DataFrame.", True
    file_format = tool_input.get("format")
    if file_format not in EXPORT_FORMATS:
        return f"Invalid format {file_format!r}; use one of {', '.join(EXPORT_FORMATS)}.", True
    reason = tool_input.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return "Missing required input: reason (one sentence for the user).", True
    if ctx.approvals_requested >= MAX_APPROVALS:
        return APPROVAL_LIMIT_REACHED, False

    result = ctx.sandbox.execute(f"{EXPORT_INFO_CODE}\nprint(_agent_export_info({variable!r}))\ndel _agent_export_info")
    if not result.success:
        return f"export_data failed:\n{result.error}", True
    info = json.loads(result.stdout.strip())
    if "error" in info:
        return f"Cannot export: {info['error']}", True
    rows, columns = info["rows"], info["columns"]
    if file_format == "xlsx" and rows > EXCEL_MAX_ROWS:
        return f"Cannot export {rows:,} rows to xlsx (limit {EXCEL_MAX_ROWS:,}); use csv.", True

    path = unique_path(ctx.output_dir, safe_filename(str(tool_input.get("filename", ""))), f".{file_format}")
    shown = ", ".join(columns[:MAX_SHOWN_COLUMNS])
    if len(columns) > MAX_SHOWN_COLUMNS:
        shown += f", ... ({len(columns) - MAX_SHOWN_COLUMNS} more)"
    return ApprovalRequest(
        tool=EXPORT_DATA_TOOL["name"],
        action=f"Export `{variable}` ({rows:,} rows) to {path.name}",
        details={
            "Variable": variable,
            "Size": f"{rows:,} rows x {len(columns)} columns",
            "Columns": shown,
            "Format": file_format,
            "File": path.as_posix(),
            "Approx. size": format_size(info["bytes"]),
            "Reason": reason.strip(),
        },
        variable=variable,
        file_format=file_format,
        path=path.as_posix(),
    )


def run_export(request: ApprovalRequest, ctx: ToolContext) -> tuple[str, bool, str | None]:
    """Write the approved file; returns (message, is_error, file path)."""
    path = Path(request.path)
    if path.exists():
        # Never overwrite: a file with this name appeared after the request.
        # Number from the base name again, so "x_2" becomes "x_3", not "x_2_2".
        path = unique_path(path.parent, re.sub(r"_\d+$", "", path.stem), path.suffix)
    writer = "to_csv" if request.file_format == "csv" else "to_excel"
    # Our own code, not the model's: it is not checked by the guard.
    result = ctx.sandbox.execute(
        f"globals()[{request.variable!r}].{writer}({path.as_posix()!r}, index=False)", timeout=120
    )
    if not result.success or not path.exists():
        return f"Export failed:\n{result.error or 'no file was written.'}", True, None
    size = format_size(path.stat().st_size)
    return f"Exported to {path.as_posix()} ({size}). Tell the user the file name.", False, path.as_posix()


def resolve_approval(
    request: ApprovalRequest, approved: bool, ctx: ToolContext, automatic: bool = False
) -> tuple[str, bool, str | None]:
    """Carry out the human's decision: (message for the model, is_error, file path)."""
    if not approved:
        return (EXPORT_NO_APPROVER if automatic else EXPORT_DENIED), False, None
    return run_export(request, ctx)



def remember(tool_input: dict[str, Any], ctx: ToolContext) -> tuple[str, bool]:
    """Validate a note and keep it for saving at the end of a successful run."""
    if ctx.memory_note_ids is None:
        return "Memory is off for this run; nothing was remembered.", True
    kind, text = tool_input.get("kind"), tool_input.get("text")
    evidence, replaces = tool_input.get("evidence"), tool_input.get("replaces")
    if kind not in NOTE_KINDS:
        return f"Invalid kind {kind!r}; use one of {', '.join(NOTE_KINDS)}.", True
    if not isinstance(text, str) or not text.strip():
        return "Missing required input: text (one short sentence).", True
    if len(text) > MEMORY_NOTE_MAX_CHARS:
        return (
            f"The note is too long ({len(text)} chars, limit {MEMORY_NOTE_MAX_CHARS}): "
            "split into separate notes, one fact each.",
            True,
        )
    has_evidence = isinstance(evidence, str) and bool(evidence.strip())
    if kind == "data_fact" and not has_evidence:
        return "A data_fact needs evidence: the numbers your code printed that show it.", True
    if replaces is not None and replaces not in ctx.memory_note_ids:
        known = ", ".join(f"#{i}" for i in sorted(ctx.memory_note_ids)) or "none"
        return f"There is no memory note #{replaces} for this dataset (notes: {known}).", True
    # A second note for the same target replaces the first one of this run. The list
    # is changed in place: the run's state holds the same list.
    ctx.pending_notes[:] = [n for n in ctx.pending_notes if replaces is None or n["replaces"] != replaces]
    new_notes = sum(n["replaces"] is None for n in ctx.pending_notes)
    if replaces is None and len(ctx.memory_note_ids) + new_notes >= MEMORY_MAX_NOTES:
        return (
            f"Memory is full ({MEMORY_MAX_NOTES} notes for this dataset). "
            "Refine an existing note with `replaces` instead.",
            True,
        )
    ctx.pending_notes.append(
        {
            "kind": kind,
            "text": text.strip(),
            "evidence": evidence.strip() if has_evidence else None,
            "replaces": replaces,
            # For a data_fact: the fact-checker's verdict, filled in by the loop.
            "check": None,
        }
    )
    target = f" (replaces #{replaces})" if replaces is not None else ""
    return f"Noted{target}. It is saved to memory only if this run finishes successfully.", False


ToolHandler = Callable[[dict[str, Any], ToolContext], tuple[str, bool]]

TOOL_HANDLERS: dict[str, ToolHandler] = {
    GET_SCHEMA_TOOL["name"]: get_schema,
    UPDATE_PLAN_TOOL["name"]: update_plan,
    RUN_CODE_TOOL["name"]: run_code,
    SAVE_CHART_TOOL["name"]: save_chart,
    ASK_USER_TOOL["name"]: ask_user,
    REPORT_VERIFICATION_TOOL["name"]: report_verification,
    REMEMBER_TOOL["name"]: remember,
}


def execute_tool(name: str, tool_input: dict[str, Any], ctx: ToolContext) -> tuple[str, bool]:
    """Run a tool requested by the model.

    Returns the text to send back to the model and whether it is an error.
    """
    ctx.last_images = []
    ctx.last_chart_paths = []
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        available = ", ".join(TOOL_HANDLERS)
        return f"Unknown tool: {name}. Available tools: {available}.", True

    text, is_error = handler(tool_input, ctx)
    return truncate_output(text), is_error
