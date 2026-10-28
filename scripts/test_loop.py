"""Checks for agent.loop.run_agent with a fake Claude client (no real API calls)."""

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2

# Make the project root importable when running this file as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.config import MAX_TOTAL_TOKENS, OUTPUTS_DIR  # noqa: E402
from agent.loop import RunResult, run_agent  # noqa: E402
from agent.sandbox import Sandbox  # noqa: E402

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


def make_response(blocks: list[Any], stop_reason: str, tokens_in: int = 100, tokens_out: int = 20) -> Any:
    """A minimal stand-in for anthropic's Message object."""
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=tokens_in, output_tokens=tokens_out),
    )


def tool_use(code: str, block_id: str = "toolu_1") -> Any:
    return SimpleNamespace(type="tool_use", id=block_id, name="run_code", input={"code": code})


def text(value: str) -> Any:
    return SimpleNamespace(type="text", text=value)


def api_error(cls: type[anthropic.APIStatusError], status: int, message: str) -> anthropic.APIStatusError:
    """Build a real SDK exception, as the client would raise it."""
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    body = {"type": "error", "error": {"type": "error", "message": message}}
    return cls(f"Error code: {status}", response=response, body=body)


class FakeClient:
    """Has client.messages.create like the SDK; replays responses or raises."""

    def __init__(self, replies: list[Any] | None = None, repeat: Any = None) -> None:
        self.replies = list(replies or [])
        self.repeat = repeat
        self.calls = 0
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        reply = self.replies.pop(0) if self.replies else self.repeat
        if isinstance(reply, Exception):
            raise reply
        return reply


def read_log(result: RunResult) -> dict[str, Any] | None:
    try:
        return json.loads(Path(result.log_path or "").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def cleanup(result: RunResult) -> None:
    """Remove the log and output folder this test run created."""
    log_path = Path(result.log_path or "")
    shutil.rmtree(PROJECT_ROOT / OUTPUTS_DIR / log_path.stem, ignore_errors=True)
    log_path.unlink(missing_ok=True)


def run(sb: Sandbox, client: FakeClient, max_steps: int = 15) -> tuple[RunResult, dict[str, Any] | None]:
    result = run_agent("Test question", sb, client, max_steps=max_steps, dataset_file="test.csv")  # type: ignore[arg-type]
    log = read_log(result)
    cleanup(result)
    return result, log


def main() -> int:
    with Sandbox() as sb:
        # a) Two steps: tool call, then a final answer
        client = FakeClient([
            make_response([text("Let me check."), tool_use("print(2 + 2)")], "tool_use"),
            make_response([text("The answer is 4.")], "end_turn"),
        ])
        result, log = run(sb, client)
        tool_output = log["steps"][0]["tool_calls"][0]["output"] if log else ""
        report("a) completed in 2 steps",
               result.status == "completed" and result.final_answer == "The answer is 4."
               and log is not None and len(log["steps"]) == 2 and log["status"] == "completed",
               f"status={result.status}, steps in log={len(log['steps']) if log else '-'}, "
               f"tool output={tool_output.splitlines()[0] if tool_output else '-'}")

        # b) Bad API key
        client = FakeClient([api_error(anthropic.AuthenticationError, 401, "invalid x-api-key")])
        result, log = run(sb, client)
        report("b) AuthenticationError -> api_error",
               result.status == "api_error" and "ANTHROPIC_API_KEY" in (result.error_message or "")
               and log is not None and log["status"] == "api_error",
               f"{result.error_message} | log saved: {log is not None}")

        # c) Unknown model
        client = FakeClient([api_error(anthropic.NotFoundError, 404, "model: claude-nonexistent")])
        result, log = run(sb, client)
        report("c) NotFoundError -> api_error",
               result.status == "api_error" and "model" in (result.error_message or ""),
               f"{result.error_message}")

        # d) Huge usage stops the run before the next call
        big = MAX_TOTAL_TOKENS // 2 + 1
        client = FakeClient(repeat=make_response([tool_use("1")], "tool_use", tokens_in=big, tokens_out=0))
        result, log = run(sb, client)
        report("d) budget_exceeded",
               result.status == "budget_exceeded" and client.calls == 2,
               f"API calls made: {client.calls}; {result.error_message}")

        # e) Model never stops calling tools
        client = FakeClient(repeat=make_response([tool_use("1")], "tool_use"))
        result, log = run(sb, client, max_steps=3)
        report("e) max_steps",
               result.status == "max_steps" and result.steps == 3 and client.calls == 3,
               f"steps={result.steps}; {result.error_message}")

    print(f"\nTotal: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
