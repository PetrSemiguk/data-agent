"""Checks for the public mode: visitor keys, masking, sessions, limits, demo data, UI (no real API calls)."""

import io
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd
from streamlit.testing.v1 import AppTest

# Make the project root importable when running this file as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent import sessions  # noqa: E402
from agent.events import Event, ToolResult  # noqa: E402
from agent.guard import check_code  # noqa: E402
from agent.loop import create_client, run_agent  # noqa: E402
from agent.mode import LOCAL, MODE_VARIABLE, PUBLIC, get_mode, load_server_environment  # noqa: E402
from agent.redact import MASK, SecretFilter, mask_secrets  # noqa: E402
from agent.sandbox import Sandbox  # noqa: E402
from agent.sessions import KernelPool, ServerBusy, cleanup_sessions, session_dirs, upload_error  # noqa: E402
from make_demo_data import DEMO_FILE, make_demo_data, write_demo_data  # noqa: E402
from test_loop import FakeClient, api_error, make_response, text, tool_use  # noqa: E402

FAKE_KEY = "sk-ant-api03-FAKEvisitorKEY_for_tests-0123456789abcdef"
SERVER_KEY = "sk-ant-api03-FAKEserverKEY_for_tests-0123456789abcdef"

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


def files_containing(folders: list[Path], needle: str) -> list[Path]:
    """Every file under the folders whose bytes contain the text."""
    found = []
    for folder in folders:
        for path in folder.rglob("*"):
            if path.is_file() and needle.encode() in path.read_bytes():
                found.append(path)
    return found


def session_folders(session_id: str) -> list[Path]:
    dirs = session_dirs(session_id)
    return [dirs.uploads, dirs.runs.outputs, dirs.runs.logs]


def remove_session(session_id: str) -> None:
    for folder in session_folders(session_id):
        shutil.rmtree(folder, ignore_errors=True)


def test_modes() -> None:
    saved = os.environ.pop(MODE_VARIABLE, None)
    try:
        default = get_mode()
        os.environ[MODE_VARIABLE] = "public"
        public = get_mode()
        os.environ[MODE_VARIABLE] = "cloud"
        try:
            get_mode()
            invalid = "no error"
        except ValueError as exc:
            invalid = str(exc)
    finally:
        os.environ.pop(MODE_VARIABLE, None)
        if saved is not None:
            os.environ[MODE_VARIABLE] = saved
    report("a) mode: local by default, public by variable, typo rejected",
           default is LOCAL and public is PUBLIC and "must be one of" in invalid,
           f"default={default.name}, public={public.name}, invalid -> {invalid}")


def test_server_key_ignored() -> None:
    os.environ["ANTHROPIC_API_KEY"] = SERVER_KEY
    load_server_environment(PUBLIC)
    visitor_client = create_client(FAKE_KEY)
    # What a client would get if some code forgot to pass the visitor's key.
    fallback = anthropic.Anthropic().api_key
    report("b) public ignores the server key",
           "ANTHROPIC_API_KEY" not in os.environ and visitor_client.api_key == FAKE_KEY and fallback is None,
           f"server key in env after start: {'ANTHROPIC_API_KEY' in os.environ}; "
           f"client uses visitor key: {visitor_client.api_key == FAKE_KEY}; client without key gets: {fallback}")


def test_mask_text_and_logs() -> None:
    masked = mask_secrets(f"Error for key {FAKE_KEY}, retry")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("test_public.masking")
    logger.addFilter(SecretFilter())
    logger.addHandler(handler)
    logger.propagate = False
    try:
        raise RuntimeError(f"request failed with {FAKE_KEY}")
    except RuntimeError:
        logger.exception("Call with %s failed", FAKE_KEY)
    output = stream.getvalue()
    report("c) masking: text and log records (message + traceback)",
           FAKE_KEY not in masked and MASK in masked and FAKE_KEY not in output and output.count(MASK) >= 2,
           f"text: {masked!r}; log lines: {len(output.splitlines())}, key in log: {FAKE_KEY in output}")


def fake_run(sb: Sandbox, session_id: str, client: FakeClient, events: list[Event]) -> Any:
    return run_agent("Q", sb, client, verify=False, on_event=events.append,  # type: ignore[arg-type]
                     dirs=session_dirs(session_id).runs)


def test_key_never_saved(sb: Sandbox) -> None:
    session_id = sessions.new_session_id()
    events: list[Event] = []
    # The code's output contains the key (the model does not know it, so the code text
    # does not), then the API fails with the key in its error message.
    head, tail = FAKE_KEY[:7], FAKE_KEY[7:]
    client = FakeClient([
        make_response([text("Checking."), tool_use(f'print("token: " + "{head}" + "{tail}")')], "tool_use"),
        api_error(anthropic.BadRequestError, 400, f"invalid request for {FAKE_KEY}"),
    ])
    try:
        result = fake_run(sb, session_id, client, events)
        in_events = [type(e).__name__ for e in events if FAKE_KEY in repr(e)]
        in_files = files_containing(session_folders(session_id), FAKE_KEY)
        seen_by_model = FAKE_KEY in repr(client.requests)
        written = sum(1 for folder in session_folders(session_id) for p in folder.rglob("*") if p.is_file())
        report("d) key in code output and API error: not in result, events, files, model input",
               result.status == "api_error" and FAKE_KEY not in repr(result) and not in_events
               and not in_files and not seen_by_model and written > 0 and MASK in (result.error_message or ""),
               f"status={result.status}; error: {result.error_message}; events with key: {in_events}; "
               f"session files: {written}, with key: {len(in_files)}; model saw key: {seen_by_model}")
    finally:
        remove_session(session_id)


def test_session_isolation(sb: Sandbox) -> None:
    a, b = sessions.new_session_id(), sessions.new_session_id()
    shared_logs = PROJECT_ROOT / "logs" / "runs"
    before = set(shared_logs.glob("*.json")) if shared_logs.exists() else set()
    try:
        results = {}
        for session_id in (a, b):
            client = FakeClient([make_response([text("Done: 4.")], "end_turn")])
            results[session_id] = fake_run(sb, session_id, client, [])
        after = set(shared_logs.glob("*.json"))
        a_files = {p for f in session_folders(a) for p in f.rglob("*")}
        b_files = {p for f in session_folders(b) for p in f.rglob("*")}
        own = all(Path(results[s].log_path).is_relative_to(session_dirs(s).runs.logs) for s in (a, b))
        overlap = set(map(str, session_folders(a))) & set(map(str, session_folders(b)))
        report("e) two sessions: own folders, no shared files",
               own and a_files and b_files and not (a_files & b_files) and not overlap and after == before,
               f"A files={len(a_files)}, B files={len(b_files)}, common={len(a_files & b_files)}; "
               f"logs in each session's folder: {own}; new files in shared logs/runs: {len(after - before)}")
    finally:
        remove_session(a)
        remove_session(b)


def test_guard_reads_per_session() -> None:
    a, b = session_dirs(sessions.new_session_id()), session_dirs(sessions.new_session_id())
    readable = a.runs.readable

    def blocked(path: Path | str) -> bool:
        return check_code(f"pd.read_csv({str(path)!r})", readable) is not None

    cases = {
        "own upload": blocked(a.uploads / "sales.csv"),
        "demo": blocked(DEMO_FILE),
        "other session": blocked(b.uploads / "sales.csv"),
        "uploads root": blocked(sessions.UPLOADS_ROOT / "x.csv"),
        "data/": blocked(PROJECT_ROOT / "data" / "Titanic-Dataset.csv"),
        "relative data/": blocked("data/Titanic-Dataset.csv"),
        "escape with ..": blocked(a.uploads / ".." / b.uploads.name / "sales.csv"),
    }
    local = check_code("pd.read_csv('data/Titanic-Dataset.csv')") is None and session_dirs(None).runs.readable is None
    expected = {"own upload": False, "demo": False, "other session": True, "uploads root": True,
                "data/": True, "relative data/": True, "escape with ..": True}
    report("q) guard in public: reads only the own session folder and demo/; local unchanged",
           cases == expected and local,
           f"blocked: {cases}; local mode reads data/: {local}")


def test_session_cannot_read_other(sb: Sandbox) -> None:
    a, b = sessions.new_session_id(), sessions.new_session_id()
    secret = "B-ONLY-CONTENT-42"
    other_file = session_dirs(b).uploads / "private.csv"
    other_file.parent.mkdir(parents=True, exist_ok=True)
    other_file.write_text(f"col\n{secret}\n", encoding="utf-8")
    client = FakeClient([
        make_response([tool_use(f"print(pd.read_csv({str(other_file)!r}))")], "tool_use"),
        make_response([text("Could not read it.")], "end_turn"),
    ])
    events: list[Event] = []
    try:
        fake_run(sb, a, client, events)
        output = next((e.output for e in events if isinstance(e, ToolResult)), "")
        report("r) session A's code cannot read session B's upload (guard blocks it)",
               output.startswith("Blocked before running") and "reads outside the dataset folder" in output
               and secret not in repr(client.requests),
               f"tool result: {output.splitlines()[1] if output else '-'}; B's content reached the model: "
               f"{secret in repr(client.requests)}")
    finally:
        remove_session(a)
        remove_session(b)


def test_cleanup() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        now = time.time()
        old_time = now - 3 * 3600
        names = {
            "old": sessions.new_session_id(),
            "fresh": sessions.new_session_id(),
            "old_active": sessions.new_session_id(),
        }
        for name, session_id in names.items():
            folder = root / session_id
            (folder / "run").mkdir(parents=True)
            (folder / "run" / "chart.png").write_bytes(b"x")
            if name.startswith("old"):
                for path in [folder, folder / "run", folder / "run" / "chart.png"]:
                    os.utime(path, (old_time, old_time))
        # A local run folder is old too, but it is not a session folder.
        local = root / "2024-01-01_10-00-00"
        local.mkdir()
        os.utime(local, (old_time, old_time))
        deleted = cleanup_sessions(2, keep=[names["old_active"]], roots=[root], now=now)
        left = sorted(p.name for p in root.iterdir())
        report("f) TTL cleanup: only old, inactive session folders",
               deleted == [root / names["old"]] and len(left) == 3 and local.exists(),
               f"deleted: {[p.name[:16] for p in deleted]}; left: {[n[:16] for n in left]}")


def test_upload_limit() -> None:
    mb = 1024 * 1024
    small, big, unlimited = upload_error(49 * mb, 50), upload_error(51 * mb, 50), upload_error(500 * mb, None)
    report("g) upload limit",
           small is None and big is not None and "limit is 50 MB" in big and unlimited is None,
           f"49 MB -> {small}; 51 MB -> {big!r}; no limit -> {unlimited}")


class FakeSandbox:
    """Counts starts and shutdowns instead of running a kernel."""

    started = 0
    stopped = 0

    def start(self) -> None:
        FakeSandbox.started += 1

    def shutdown(self) -> None:
        FakeSandbox.stopped += 1


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_session_limit() -> None:
    clock = Clock()
    pool = KernelPool(max_active=3, idle_seconds=15 * 60, factory=FakeSandbox, clock=clock)  # type: ignore[arg-type]
    for name in ("s1", "s2", "s3"):
        with pool.using(name):
            clock.now += 60
    try:
        with pool.using("s4"):
            busy = "no error"
    except ServerBusy as exc:
        busy = str(exc)
    with pool.using("s2") as (_, fresh_again):
        pass
    report("h) session limit: 4th session gets 'server busy'",
           "busy" in busy and pool.active_sessions() == {"s1", "s2", "s3"} and not fresh_again,
           f"4th: {busy!r}; active: {sorted(pool.active_sessions())}; known session reuses its kernel: {not fresh_again}")


def test_idle_close() -> None:
    clock = Clock()
    FakeSandbox.stopped = 0
    pool = KernelPool(max_active=3, idle_seconds=15 * 60, factory=FakeSandbox, clock=clock)  # type: ignore[arg-type]
    with pool.using("idle"):
        pass
    with pool.using("working"):
        # A long question: the kernel is in use while the clock passes the idle limit.
        clock.now += 20 * 60
        closed_while_busy = pool.close_idle()
    clock.now += 60
    closed = pool.close_idle()
    with pool.using("new"):
        pass
    with pool.using("idle") as (_, fresh):
        pass
    report("i) idle kernel closed (not one in use), started again on next use",
           closed_while_busy == ["idle"] and closed == [] and fresh and FakeSandbox.stopped >= 1,
           f"closed after 20 min: {closed_while_busy}; kernel in use kept; slot reused by 'new'; "
           f"'idle' gets a fresh kernel: {fresh}")


def test_real_kernel_restart() -> None:
    clock = Clock()
    pool = KernelPool(max_active=1, idle_seconds=60, clock=clock)
    try:
        with pool.using("s") as (sandbox, first):
            sandbox.execute("x = 1")
        clock.now += 120
        with pool.using("s") as (sandbox, again):
            after = sandbox.execute("x")
        report("j) real kernel: closed when idle, a new one starts on the next question",
               first and again and not after.success and "NameError" in (after.error or ""),
               f"first start: {first}; started again: {again}; old variable: {(after.error or '').splitlines()[0]}")
    finally:
        pool.close_all()


def demo_traps(data: pd.DataFrame) -> dict[str, Any]:
    dates = pd.to_datetime(data["order_date"])
    months = dates.dt.to_period("M").value_counts().sort_index()
    last = dates.max()
    products = data[data["category"] != "Service"].drop_duplicates()
    by_region = products.groupby("region")["amount"].agg(["count", "mean"])
    smallest = by_region["count"].idxmin()
    top = products["amount"].nlargest(max(1, len(products) // 100)).sum() / products["amount"].sum()
    return {
        "duplicates": int(data.duplicated().sum()),
        "last month incomplete": last.day < last.days_in_month and months.iloc[-1] < 0.5 * months.iloc[-2],
        "service rows": int((data["category"] == "Service").sum()),
        "small group tops average": smallest == by_region["mean"].idxmax() and by_region["count"].min() < 10,
        "skew (mean/median)": round(products["amount"].mean() / products["amount"].median(), 1),
        "top 1% share": round(top, 2),
    }


def test_demo_data() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        first = write_demo_data(Path(tmp) / "a.csv").read_bytes()
        second = write_demo_data(Path(tmp) / "b.csv").read_bytes()
    committed = DEMO_FILE.read_bytes()
    report("k) demo dataset is deterministic and matches demo/demo_sales.csv",
           first == second == committed and make_demo_data(1).equals(make_demo_data(1)),
           f"two runs equal: {first == second}; equal to the committed file: {first == committed}; "
           f"size {len(committed) / 1024:.0f} KB")
    traps = demo_traps(pd.read_csv(DEMO_FILE))
    ok = (traps["duplicates"] >= 20 and traps["last month incomplete"] and traps["service rows"] > 0
          and traps["small group tops average"] and traps["skew (mean/median)"] >= 2 and traps["top 1% share"] >= 0.2)
    report("l) demo dataset has every trap", ok, str(traps))


def test_key_format() -> None:
    # A malformed key is refused before any request is sent.
    error = sessions.check_api_key("not-a-key")
    report("m) malformed key refused without a request", error is not None and "sk-ant-" in error, f"{error}")


def app_test(mode: str | None) -> AppTest:
    if mode is None:
        os.environ.pop(MODE_VARIABLE, None)
    else:
        os.environ[MODE_VARIABLE] = mode
    # Set before the app runs: load_dotenv() never overrides it with the real key.
    os.environ["ANTHROPIC_API_KEY"] = SERVER_KEY
    return AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=90)


def page_text(at: AppTest) -> str:
    parts = [e.value for kind in ("markdown", "caption", "info", "error", "success", "warning") for e in getattr(at, kind)]
    return " ".join(str(p) for p in parts)


def labels(at: AppTest, kind: str) -> list[str]:
    return [e.label for e in getattr(at, kind)]


def test_app_local() -> None:
    at = app_test(None).run()
    report("n) local app: as before (memory toggle, no key field, no demo button)",
           not at.exception and "Use memory" in labels(at, "toggle") and "Anthropic API key" not in labels(at, "text_input")
           and "Use demo dataset" not in labels(at, "button") and at.chat_input[0].proto.disabled
           and at.chat_input[0].proto.placeholder == "Upload a dataset to start",
           f"exception: {bool(at.exception)}; toggles: {labels(at, 'toggle')}; buttons: {labels(at, 'button')}; "
           f"chat: {at.chat_input[0].proto.placeholder!r}")


def test_app_public_without_key() -> None:
    at = app_test("public").run()
    chat = at.chat_input[0].proto
    report("o) public app without key: opens, explains, chat disabled, server key unused",
           not at.exception and chat.disabled and "API key" in chat.placeholder
           and "Anthropic API key" in labels(at, "text_input") and "Use memory" not in labels(at, "toggle")
           and "Use demo dataset" in labels(at, "button") and "Example" in page_text(at)
           and "ANTHROPIC_API_KEY" not in os.environ and SERVER_KEY not in page_text(at),
           f"exception: {bool(at.exception)}; chat: {chat.placeholder!r} (disabled={chat.disabled}); "
           f"server key in env: {'ANTHROPIC_API_KEY' in os.environ}")


def test_app_public_with_key() -> None:
    original = sessions.check_api_key
    try:
        # A wrong key: the visitor sees the error, nothing is kept.
        sessions.check_api_key = lambda key: "The key was rejected: it is invalid, revoked or disabled."
        at = app_test("public").run()
        at.text_input(key="api_key_input").input(FAKE_KEY).run()
        rejected = "rejected" in page_text(at) and "client" not in at.session_state and FAKE_KEY not in page_text(at)

        # A good key (the check is replaced, no request is sent).
        sessions.check_api_key = lambda key: None
        at.text_input(key="api_key_input").input(FAKE_KEY).run()
        accepted = at.session_state["client"].api_key == FAKE_KEY and "Forget my key" in labels(at, "button")
        # The field is not shown once a key is accepted; its stored value must be empty.
        field_cleared = "api_key_input" not in at.session_state or at.session_state["api_key_input"] == ""
        before_data = at.chat_input[0].proto.disabled

        next(b for b in at.button if b.label == "Use demo dataset").click().run()
        chat_enabled = not at.chat_input[0].proto.disabled
        demo_loaded = "demo_sales.csv" in page_text(at)

        next(b for b in at.button if b.label == "Forget my key").click().run()
        forgotten = "client" not in at.session_state and "api_key" not in at.session_state and at.chat_input[0].proto.disabled
        report("p) public app with key: wrong key refused, good key enables chat, Forget my key disables it",
               not at.exception and rejected and accepted and field_cleared and before_data and chat_enabled
               and demo_loaded and forgotten and FAKE_KEY not in page_text(at),
               f"wrong key refused: {rejected}; accepted: {accepted}; field emptied: {field_cleared}; "
               f"demo loaded: {demo_loaded}; chat enabled: {chat_enabled}; forgotten: {forgotten}")
    finally:
        sessions.check_api_key = original


def main() -> int:
    saved_key = os.environ.get("ANTHROPIC_API_KEY")
    saved_mode = os.environ.get(MODE_VARIABLE)
    try:
        test_modes()
        test_server_key_ignored()
        test_mask_text_and_logs()
        with Sandbox() as sb:
            test_key_never_saved(sb)
            test_session_isolation(sb)
            test_session_cannot_read_other(sb)
        test_guard_reads_per_session()
        test_cleanup()
        test_upload_limit()
        test_session_limit()
        test_idle_close()
        test_real_kernel_restart()
        test_demo_data()
        test_key_format()
        test_app_local()
        test_app_public_without_key()
        test_app_public_with_key()
    finally:
        for name, value in (("ANTHROPIC_API_KEY", saved_key), (MODE_VARIABLE, saved_mode)):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    print(f"\nTotal: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
