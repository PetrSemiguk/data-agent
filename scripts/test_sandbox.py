"""Manual checks for agent.sandbox using the Superstore dataset."""

import os
import sys
from pathlib import Path

# Make the project root importable when running this file as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.sandbox import ExecutionResult, Sandbox  # noqa: E402

DATASET = PROJECT_ROOT / "data" / "Sample - Superstore.csv"

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


def first_line(text: str | None) -> str:
    return text.splitlines()[0] if text else ""


def timeout_path(error: str | None) -> str:
    # The sandbox states in the timeout message how the timeout was handled.
    text = error or ""
    if "kernel was restarted" in text:
        return "restart"
    if "was interrupted" in text:
        return "interrupt"
    return "unknown"


def main() -> int:
    # Fake secrets in the app process; the kernel must not inherit them
    os.environ["ANTHROPIC_API_KEY"] = "fake-key-for-test"
    os.environ["SOME_SERVICE_TOKEN"] = "fake-token-for-test"

    with Sandbox() as sb:
        # n) Secrets are not passed to the kernel, also after a restart
        code = 'import os\n[os.environ.get(n) for n in ("ANTHROPIC_API_KEY", "SOME_SERVICE_TOKEN")]'
        before = sb.execute(code).result
        sb.restart()
        after = sb.execute(code).result
        report("n) kernel env has no secrets",
               before == after == "[None, None]" and os.environ.get("ANTHROPIC_API_KEY") is not None,
               f"in the kernel: before restart={before}, after restart={after}; in the app process: set")

        # 0) Common libraries are imported at startup (before load_dataframe)
        r = sb.execute("pd.__name__, np.__name__, plt.__name__")
        report("0) pd, np, plt preimported",
               r.success and r.result == "('pandas', 'numpy', 'matplotlib.pyplot')",
               f"result: {r.result or first_line(r.error)}")

        # a) DataFrame is loaded
        load = sb.load_dataframe(DATASET)
        r = sb.execute("df.shape")
        report("a) df is loaded", r.success and r.result == "(9994, 21)",
               f"{load.stdout.strip()}, df.shape = {r.result}")

        # b) State is kept between calls
        sb.execute("x = 5")
        r = sb.execute("x + 1")
        report("b) state persists", r.success and r.result == "6",
               f"x + 1 = {r.result}")

        # c) Runtime error
        r = sb.execute("1 / 0")
        report("c) ZeroDivisionError", not r.success and "ZeroDivisionError" in (r.error or ""),
               f"success={r.success}, error: {first_line(r.error)}")

        # d) Undefined variable
        r = sb.execute("undefined_variable")
        report("d) NameError", not r.success and "NameError" in (r.error or ""),
               f"success={r.success}, error: {first_line(r.error)}")

        # e) Timeout, then the kernel and df must still work
        r = sb.execute("import time\ntime.sleep(10)", timeout=3)
        after: ExecutionResult = sb.execute("df.shape")
        report("e) timeout + kernel alive",
               not r.success and "TimeoutError" in (r.error or "")
               and after.success and after.result == "(9994, 21)",
               f"handled via: {timeout_path(r.error)}; then df.shape = {after.result}")

        # f) Chart is returned as an image
        r = sb.execute('df["Sales"].plot.hist();')
        sizes = ", ".join(str(len(img)) for img in r.images)
        report("f) chart image", r.success and len(r.images) >= 1,
               f"images: {len(r.images)}, base64 length: {sizes or '-'}")

        # g) Pure Python loop must be stopped by interrupt, keeping variables
        sb.execute("y = 42")
        r = sb.execute("while True:\n    pass", timeout=3)
        path = timeout_path(r.error)
        after = sb.execute("y")
        report("g) interrupt keeps variables",
               not r.success and path == "interrupt"
               and after.success and after.result == "42",
               f"handled via: {path}; then y = {after.result}")

        # Separate namespace (used by the verifier): its code must not touch the kernel's variables
        ns = "test_ns"
        sb.execute('data = "agent"')
        sb.execute("data = 1", namespace=ns)
        after = sb.execute("data")
        report("h) namespace keeps agent variable", after.result == "'agent'",
               f"agent's data after namespace code 'data = 1': {after.result}")

        r = sb.execute('df.drop(columns=["Sales"], inplace=True)\ndf.shape', namespace=ns)
        after = sb.execute('df.shape, "Sales" in df.columns')
        report("i) namespace inplace drop leaves df",
               r.success and r.result == "(9994, 20)" and after.result == "((9994, 21), True)",
               f"namespace df.shape = {r.result}; agent's (df.shape, has Sales) = {after.result}")

        r = sb.execute("x", namespace=ns)
        report("j) agent variables hidden", not r.success and "NameError" in (r.error or ""),
               f"agent's x in namespace: {first_line(r.error)}")

        r = sb.execute("y = 2\ny * 21", namespace=ns)
        report("k) last expression shown", r.success and r.result == "42", f"result: {r.result}")

        # m) The kernel works in the project root, also after a restart: relative paths in
        #    model code and the guard's checks mean the same folder
        before = sb.execute("import os\nos.getcwd()").result
        sb.restart()
        after = sb.execute("import os\nos.getcwd()").result
        sb.load_dataframe(DATASET)
        root = repr(str(PROJECT_ROOT))
        report("m) kernel cwd is the project root", before == root and after == root, f"before={before}, after restart={after}")

        sb.drop_namespace(ns)
        r = sb.execute("y", namespace=ns)
        report("l) dropped namespace is gone", not r.success and "NameError" in (r.error or ""),
               f"namespace variable y after drop: {first_line(r.error) or r.result}")

    print(f"\nTotal: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
