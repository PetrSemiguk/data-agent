"""Checks for agent.tools (no API calls)."""

import struct
import sys
import tempfile
import time
from pathlib import Path

# Make the project root importable when running this file as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.config import (  # noqa: E402
    MAX_OUTPUT_CHARS,
    MAX_USER_QUESTIONS,
    OUTPUT_HEAD_CHARS,
    OUTPUT_TAIL_CHARS,
)
from agent.sandbox import Sandbox  # noqa: E402
from agent.tools import (  # noqa: E402
    QUESTION_LIMIT_REACHED,
    USER_UNAVAILABLE,
    ToolContext,
    execute_tool,
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
