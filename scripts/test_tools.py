"""Checks for agent.tools (no API calls)."""

import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd

# Make the project root importable when running this file as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.config import (  # noqa: E402
    MAX_OUTPUT_CHARS,
    MAX_USER_QUESTIONS,
    MEMORY_MAX_NOTES,
    MEMORY_NOTE_MAX_CHARS,
    OUTPUT_HEAD_CHARS,
    OUTPUT_TAIL_CHARS,
)
from agent.sandbox import Sandbox  # noqa: E402
from agent.tools import (  # noqa: E402
    EXPORT_DENIED,
    QUESTION_LIMIT_REACHED,
    USER_UNAVAILABLE,
    ApprovalRequest,
    ToolContext,
    execute_tool,
    prepare_export,
    resolve_approval,
    run_export,
    truncate_output,
)

DATA_DIR = PROJECT_ROOT / "data"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
CHART_CODE = 'df.groupby("Category")["Sales"].sum().plot.bar(title="Revenue by category")'

passed = 0
failed = 0


def report(name: str, ok: bool, details: str) -> None:
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"       {details}")


def neg_count(sb: Sandbox) -> int:
    """Number of rows of `neg` in the kernel, to compare with what was exported."""
    return int(sb.execute("len(neg)").result or 0)


def main() -> int:
    # a) Short text is returned unchanged
    short = "x" * MAX_OUTPUT_CHARS
    report("a) short text unchanged", truncate_output(short) == short,
           f"input {len(short)} chars")

    # Long text with recognizable start and end
    long_text = "START" + "a" * 50_000 + "b" * 50_000 + "END"
    out = truncate_output(long_text)

    # b) Long text is cut to about MAX_OUTPUT_CHARS
    report("b) long text truncated", len(out) <= MAX_OUTPUT_CHARS,
           f"input {len(long_text)} chars -> output {len(out)} chars (limit {MAX_OUTPUT_CHARS})")

    # c) Marker reports the exact number of removed chars
    omitted = len(long_text) - OUTPUT_HEAD_CHARS - OUTPUT_TAIL_CHARS
    marker = f"[truncated {omitted} chars]"
    report("c) marker present", marker in out and "Tip: print less" in out,
           f"looking for '{marker}'")

    # d) Head and tail are preserved
    ok = (out.startswith(long_text[:OUTPUT_HEAD_CHARS])
          and out.endswith(long_text[-OUTPUT_TAIL_CHARS:]))
    report("d) head and tail kept", ok,
           f"starts with {out[:5]!r}, ends with {out[-3:]!r}")

    # ask_user never touches the kernel, so these tests need no sandbox
    question = {"question": "Gross or net revenue?", "options": ["Gross", "Net"]}

    # o) The human's free-text answer reaches the model
    ctx = ToolContext(sandbox=None, ask_user_fn=lambda q, opts: "net of returns")  # type: ignore[arg-type]
    text, is_error = execute_tool("ask_user", question, ctx)
    report("o) answer reaches model", not is_error and text == "User answered: net of returns", text)

    # p) An option number becomes the option text
    ctx = ToolContext(sandbox=None, ask_user_fn=lambda q, opts: "2")  # type: ignore[arg-type]
    text, is_error = execute_tool("ask_user", question, ctx)
    report("p) number -> option text", text == "User answered: Net", text)

    # q) Without a human the model is told to assume
    ctx = ToolContext(sandbox=None)  # type: ignore[arg-type]
    text, is_error = execute_tool("ask_user", question, ctx)
    report("q) no handler -> not available", not is_error and text == USER_UNAVAILABLE, text)

    # r) Questions beyond the limit are not shown to the human
    calls: list[str] = []
    ctx = ToolContext(sandbox=None, ask_user_fn=lambda q, opts: calls.append(q) or "ok")  # type: ignore[arg-type]
    answers = [execute_tool("ask_user", question, ctx)[0] for _ in range(MAX_USER_QUESTIONS + 1)]
    report("r) limit reached", answers[-1] == QUESTION_LIMIT_REACHED and len(calls) == MAX_USER_QUESTIONS,
           f"handler called {len(calls)} times; last reply: {answers[-1]}")

    # remember only validates and keeps notes for the end of the run: no sandbox needed
    def remember_call(
        note_ids: frozenset[int] | set[int] | None = frozenset(), pending: int = 0, **tool_input: Any
    ) -> tuple[str, bool, ToolContext]:
        """Call remember with the dataset's note ids; note_ids=None means memory is off."""
        ids = None if note_ids is None else set(note_ids)
        ctx = ToolContext(sandbox=None, memory_note_ids=ids)  # type: ignore[arg-type]
        ctx.pending_notes = [{"kind": "preference", "text": "x", "evidence": None, "replaces": None}] * pending
        text, is_error = execute_tool("remember", tool_input, ctx)
        return text, is_error, ctx

    fact = {"kind": "data_fact", "text": "34,335 rows are full duplicates.", "evidence": "duplicated().sum() = 34335"}
    bad_notes = {
        "data_fact without evidence": remember_call(kind="data_fact", text="Many duplicates."),
        "unknown kind": remember_call(kind="opinion", text="x"),
        "empty text": remember_call(kind="preference", text=" "),
        "too long": remember_call(kind="preference", text="x" * (MEMORY_NOTE_MAX_CHARS + 1)),
        "unknown replaces id": remember_call(note_ids={1, 2}, replaces=7, **fact),
        "memory full": remember_call(note_ids=set(range(MEMORY_MAX_NOTES - 1)), pending=1, **fact),
        "memory off": remember_call(note_ids=None, **fact),
    }
    report("cc) remember validation",
           all(r[1] for r in bad_notes.values()) and "split into separate notes" in bad_notes["too long"][0],
           "; ".join(f"{k}: {r[0][:60]}" for k, r in bad_notes.items()))

    text, is_error, ctx = remember_call(**fact)
    definition = remember_call(note_ids={4}, kind="definition", text="Revenue means net revenue.", replaces=4)
    full_but_replaces = remember_call(note_ids=set(range(MEMORY_MAX_NOTES)), replaces=3, **fact)
    report("dd) remember keeps valid notes until the end of the run",
           not is_error and ctx.pending_notes == [{**fact, "replaces": None, "check": None}]
           and not definition[1] and definition[2].pending_notes[0]["replaces"] == 4
           and not full_but_replaces[1],
           f"{text} | replaces: {definition[0]} | full memory, replaces=3: {full_but_replaces[0][:40]}")

    # update_plan only validates and stores the plan, so no sandbox is needed
    def plan_call(items: Any) -> tuple[str, bool, ToolContext]:
        ctx = ToolContext(sandbox=None)  # type: ignore[arg-type]
        text, is_error = execute_tool("update_plan", {"items": items}, ctx)
        return text, is_error, ctx

    good = [
        {"id": "1", "text": "Explore schema", "status": "done"},
        {"id": "2", "text": "Revenue by category", "status": "in_progress"},
        {"id": "3", "text": "Seasonality test", "status": "skipped", "note": "only 4 years"},
    ]
    text, is_error, ctx = plan_call(good)
    report("s) valid plan stored", not is_error and ctx.plan is not None and len(ctx.plan) == 3,
           text.splitlines()[0])

    bad_cases = {
        "t) duplicate ids": ([good[0], {**good[1], "id": "1"}], "duplicate id"),
        "u) invalid status": ([{**good[0], "status": "finished"}], "invalid status"),
        "v) skipped without note": ([{"id": "1", "text": "x", "status": "skipped"}], "without a reason"),
        "w) empty plan": ([], "1-8"),
    }
    for name, (items, expected) in bad_cases.items():
        text, is_error, ctx = plan_call(items)
        report(name, is_error and expected in text and ctx.plan is None, text)

    # get_schema runs inside a real kernel
    with Sandbox() as sb:
        sb.load_dataframe(DATA_DIR / "Titanic-Dataset.csv")
        text, is_error = execute_tool("get_schema", {}, ToolContext(sb))
        header = text.splitlines()[0] if text else ""

        # e) Shape is correct
        report("e) Titanic shape", not is_error and "891 rows x 12 columns" in header, header)

        # f) Missing ages are reported
        age = next((line for line in text.splitlines() if line.startswith("- Age:")), "")
        report("f) Age missing 177", "missing 177" in age, age[:90])

        # g) Unknown variable is an error
        text, is_error = execute_tool("get_schema", {"variable": "nope"}, ToolContext(sb))
        report("g) unknown variable", is_error, text)

        # h) Text dates are detected
        sb.load_dataframe(DATA_DIR / "Sample - Superstore.csv")
        text, _ = execute_tool("get_schema", {}, ToolContext(sb))
        order_date = next((line for line in text.splitlines() if line.startswith("- Order Date:")), "")
        report("h) Order Date looks like date", "looks like date" in order_date, order_date)

        # save_chart writes into a throwaway run folder
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)

            # j) Simple chart is saved as a real PNG
            text, is_error = execute_tool(
                "save_chart", {"code": CHART_CODE, "filename": "revenue"}, ToolContext(sb, run_dir))
            first = run_dir / "revenue.png"
            data = first.read_bytes() if first.exists() else b""
            width, height = struct.unpack(">II", data[16:24]) if len(data) > 24 else (0, 0)
            report("j) chart saved as PNG", not is_error and data.startswith(PNG_SIGNATURE),
                   f"{first.name}, {len(data) / 1024:.1f} KB, {width}x{height} px")

            # k) Dangerous name stays inside the run folder
            text, is_error = execute_tool(
                "save_chart", {"code": CHART_CODE, "filename": "../../evil name!"}, ToolContext(sb, run_dir))
            files = sorted(p.name for p in run_dir.iterdir())
            outside = list(run_dir.parent.glob("*evil*"))
            report("k) unsafe filename sanitized",
                   not is_error and (run_dir / "evil_name.png").exists() and not outside,
                   f"files in run folder: {files}")

            # l) Same name again gets a suffix, first file is untouched
            before = first.read_bytes()
            text, is_error = execute_tool(
                "save_chart", {"code": CHART_CODE, "filename": "revenue"}, ToolContext(sb, run_dir))
            report("l) no overwrite",
                   not is_error and (run_dir / "revenue_2.png").exists()
                   and first.read_bytes() == before,
                   f"second call -> {first.with_name('revenue_2.png').name}")

            # m) Code without a figure is an error
            text, is_error = execute_tool(
                "save_chart", {"code": "x = 1", "filename": "empty"}, ToolContext(sb, run_dir))
            report("m) no figure -> error", is_error and "no figure" in text, text.splitlines()[0])

            # n) Failing code is an error
            text, is_error = execute_tool(
                "save_chart", {"code": "df['Nope'].plot()", "filename": "broken"}, ToolContext(sb, run_dir))
            report("n) failing code -> error", is_error and "KeyError" in text, text.splitlines()[0])

        # export_data: prepared for approval, then written into the run folder
        sb.execute("neg = df[df['Profit'] < 0][['Order ID', 'Sales', 'Profit']]\nnot_a_frame = 5")
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            ctx = ToolContext(sb, run_dir)

            def export(**overrides: Any) -> Any:
                tool_input = {"variable": "neg", "filename": "losses", "format": "csv",
                              "reason": "Orders with a loss.", **overrides}
                return prepare_export(tool_input, ctx)

            # x) Wrong inputs are errors and nothing is written
            bad = {
                "missing variable": export(variable="nope"),
                "not a DataFrame": export(variable="not_a_frame"),
                "invalid name": export(variable="neg; import os"),
                "bad format": export(format="parquet"),
                "no reason": export(reason=" "),
            }
            ok = all(isinstance(r, tuple) and r[1] for r in bad.values()) and not any(run_dir.iterdir())
            report("x) export_data validation", ok,
                   "; ".join(f"{k}: {r[0].splitlines()[0] if isinstance(r, tuple) else 'NOT REJECTED'}"
                             for k, r in bad.items()))

            # y) The request describes the export; the file name is made safe
            request = export(filename="../../evil name!")
            ok = (isinstance(request, ApprovalRequest) and Path(request.path).parent == run_dir
                  and Path(request.path).name == "evil_name.csv"
                  and request.details["Size"].startswith(f"{neg_count(sb):,} rows x 3 columns")
                  and request.details["Reason"] == "Orders with a loss.")
            report("y) request details + safe name", ok,
                   f"{request.action if isinstance(request, ApprovalRequest) else request}; "
                   f"details={request.details if isinstance(request, ApprovalRequest) else '-'}")

            # z) csv and xlsx are written and read back with the same shape
            shapes = {}
            for file_format in ("csv", "xlsx"):
                request = export(format=file_format)
                assert isinstance(request, ApprovalRequest)
                text, is_error, path = run_export(request, ctx)
                read = pd.read_csv if file_format == "csv" else pd.read_excel
                shapes[file_format] = read(path).shape if path else None
            expected = (neg_count(sb), 3)
            report("z) csv and xlsx written", shapes == {"csv": expected, "xlsx": expected},
                   f"read back: {shapes}, expected {expected}")

            # aa) No overwrite: a taken name gets a suffix, also if it appears after the request
            first = run_dir / "losses.csv"
            before = first.read_bytes()
            request = export()
            assert isinstance(request, ApprovalRequest)
            (run_dir / "losses_2.csv").write_text("someone else's file")
            _, _, path = run_export(request, ctx)
            report("aa) export never overwrites",
                   Path(request.path).name == "losses_2.csv" and path is not None and Path(path).name == "losses_3.csv"
                   and first.read_bytes() == before
                   and (run_dir / "losses_2.csv").read_text() == "someone else's file",
                   f"planned {Path(request.path).name}, written {Path(path).name if path else None}")

            # bb) Denied: nothing is written, the model is told not to retry
            files_before = sorted(run_dir.iterdir())
            text, is_error, path = resolve_approval(request, False, ctx)
            report("bb) denied -> no file", text == EXPORT_DENIED and path is None
                   and sorted(run_dir.iterdir()) == files_before, text)

        # i) Fast enough on 1M rows
        sb.load_dataframe(DATA_DIR / "online_retail_II.csv")
        start = time.perf_counter()
        text, is_error = execute_tool("get_schema", {}, ToolContext(sb))
        elapsed = time.perf_counter() - start
        report("i) Online Retail under 10s", not is_error and elapsed < 10,
               f"{elapsed:.1f}s, {len(text)} chars")
        print(f"\n--- get_schema output for Online Retail II ---\n{text}\n")

    print(f"\nTotal: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
