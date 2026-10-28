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

from agent.config import MAX_TOTAL_TOKENS, MAX_USER_QUESTIONS, OUTPUTS_DIR  # noqa: E402
from agent.events import (  # noqa: E402
    ApiCallFinished,
    AssistantText,
    Event,
    RunFinished,
    RunStarted,
    StepStarted,
    StreamReset,
    TextDelta,
    ToolCall,
    ToolResult,
)
from agent.loop import RunResult, resume_agent, run_agent  # noqa: E402
from agent.tools import QUESTION_LIMIT_REACHED  # noqa: E402
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


def ask(question: str, options: list[str] | None = None, block_id: str = "toolu_ask") -> Any:
    tool_input: dict[str, Any] = {"question": question}
    if options:
        tool_input["options"] = options
    return SimpleNamespace(type="tool_use", id=block_id, name="ask_user", input=tool_input)


def last_tool_results(client: "FakeClient", call: int) -> list[dict[str, Any]]:
    """tool_result blocks the model received on API call number `call` (1-based)."""
    content = client.requests[call - 1][-1]["content"]
    return [block for block in content if isinstance(block, dict) and block.get("type") == "tool_result"]


def api_error(cls: type[anthropic.APIStatusError], status: int, message: str) -> anthropic.APIStatusError:
    """Build a real SDK exception, as the client would raise it."""
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    body = {"type": "error", "error": {"type": "error", "message": message}}
    return cls(f"Error code: {status}", response=response, body=body)


class BrokenStream:
    """A reply whose stream sends `deltas_before_break` text pieces, then fails.

    By default the connection is lost; `error` replaces that with another exception,
    e.g. an error event the API sends inside the stream.
    """

    def __init__(self, response: Any, deltas_before_break: int = 2, error: Exception | None = None) -> None:
        self.response = response
        self.deltas_before_break = deltas_before_break
        self.error = error


def stream_error_event(error_type: str, message: str) -> anthropic.APIStatusError:
    """What the SDK raises for an `error` event inside an open stream (HTTP status is 200)."""
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(200, request=request)
    body = {"type": "error", "error": {"type": error_type, "message": message}}
    return anthropic.APIStatusError(str(body), response=response, body=body)


class FakeStream:
    """Context manager + iterator of stream events, like the SDK's MessageStream."""

    def __init__(self, reply: Any) -> None:
        self.reply = reply

    def __enter__(self) -> "FakeStream":
        # Errors before the stream opens: the SDK raises them from __enter__.
        if isinstance(self.reply, Exception):
            raise self.reply
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def __iter__(self) -> Any:
        response = self.reply.response if isinstance(self.reply, BrokenStream) else self.reply
        sent = 0
        for block in response.content:
            if block.type != "text":
                continue
            # Split text into small pieces, as the API does.
            for i in range(0, len(block.text), 4):
                if isinstance(self.reply, BrokenStream) and sent == self.reply.deltas_before_break:
                    raise self.reply.error or httpx2.ReadError("connection lost")
                sent += 1
                yield SimpleNamespace(type="text", text=block.text[i:i + 4])

    def get_final_message(self) -> Any:
        return self.reply


class FakeClient:
    """Has client.messages.stream like the SDK; replays responses or raises."""

    def __init__(self, replies: list[Any] | None = None, repeat: Any = None) -> None:
        self.replies = list(replies or [])
        self.repeat = repeat
        self.calls = 0
        self.requests: list[list[Any]] = []
        self.messages = self

    def stream(self, **kwargs: Any) -> FakeStream:
        self.calls += 1
        # A copy of what the model would see on this call.
        self.requests.append(list(kwargs["messages"]))
        reply = self.replies.pop(0) if self.replies else self.repeat
        return FakeStream(reply)


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


def run(
    sb: Sandbox, client: FakeClient, max_steps: int = 15, on_event: Any = None
) -> tuple[RunResult, dict[str, Any] | None]:
    result = run_agent(
        "Test question", sb, client, max_steps=max_steps,  # type: ignore[arg-type]
        dataset_file="test.csv", on_event=on_event,
    )
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

        # f) The same two-step run reports events in order
        events: list[Event] = []
        client = FakeClient([
            make_response([text("Let me check."), tool_use("print(2 + 2)")], "tool_use"),
            make_response([text("The answer is 4.")], "end_turn"),
        ])
        result, _ = run(sb, client, on_event=events.append)
        all_events = list(events)
        # Text deltas are checked in test h; here only the step structure matters.
        events = [e for e in all_events if not isinstance(e, TextDelta)]
        expected = [RunStarted, StepStarted, ApiCallFinished, AssistantText, ToolCall, ToolResult,
                    StepStarted, ApiCallFinished, AssistantText, RunFinished]
        kinds = [type(e) for e in events]
        tool_result = next((e for e in events if isinstance(e, ToolResult)), None)
        ok = (
            kinds == expected
            and [e.step for e in events if isinstance(e, StepStarted)] == [1, 2]
            and events[3].text == "Let me check."  # type: ignore[union-attr]
            and events[4].input == {"code": "print(2 + 2)"}  # type: ignore[union-attr]
            and tool_result is not None and tool_result.output.startswith("4") and not tool_result.is_error
            and events[-1].result.status == "completed"  # type: ignore[union-attr]
        )
        report("f) events in order", ok, " -> ".join(k.__name__ for k in kinds))

        # h) Text arrives as deltas that add up to the full text
        ok = True
        details = []
        for step in (1, 2):
            joined = "".join(e.text for e in all_events if isinstance(e, TextDelta) and e.step == step)
            full = next(e.text for e in all_events if isinstance(e, AssistantText) and e.step == step)
            deltas = sum(isinstance(e, TextDelta) and e.step == step for e in all_events)
            ok = ok and joined == full and deltas > 1
            details.append(f"step {step}: {deltas} deltas -> {joined!r}")
        report("h) deltas join to full text", ok, "; ".join(details))

        # i) The stream breaks mid-way, the step is retried and the run completes
        events = []
        client = FakeClient([
            BrokenStream(make_response([text("Partial answer that breaks")], "end_turn")),
            make_response([text("Full answer.")], "end_turn"),
        ])
        result, log = run(sb, client, on_event=events.append)
        kinds = [type(e).__name__ for e in events if not isinstance(e, TextDelta)]
        resets = [e for e in events if isinstance(e, StreamReset)]
        after_reset = "".join(
            e.text for e in events[events.index(resets[0]) + 1:] if isinstance(e, TextDelta)
        ) if resets else ""
        report("i) broken stream retried",
               result.status == "completed" and result.final_answer == "Full answer."
               and len(resets) == 1 and client.calls == 2 and after_reset == "Full answer."
               and log is not None and len(log["steps"]) == 1,
               f"status={result.status}, API calls={client.calls}, events: {' -> '.join(kinds)}")

        # j) overloaded_error arrives as an event mid-stream: reset, retry, complete
        events = []
        client = FakeClient([
            BrokenStream(make_response([text("Partial answer that breaks")], "end_turn"),
                         error=stream_error_event("overloaded_error", "Overloaded")),
            make_response([text("Full answer.")], "end_turn"),
        ])
        result, _ = run(sb, client, on_event=events.append)
        kinds = [type(e).__name__ for e in events if not isinstance(e, TextDelta)]
        resets = sum(isinstance(e, StreamReset) for e in events)
        report("j) mid-stream overloaded_error retried",
               result.status == "completed" and result.final_answer == "Full answer."
               and resets == 1 and client.calls == 2,
               f"status={result.status}, API calls={client.calls}, events: {' -> '.join(kinds)}")

        # k) A permanent error event mid-stream is not retried
        events = []
        client = FakeClient([
            BrokenStream(make_response([text("Partial answer that breaks")], "end_turn"),
                         error=stream_error_event("invalid_request_error", "Bad input")),
            make_response([text("Should not be requested.")], "end_turn"),
        ])
        result, _ = run(sb, client, on_event=events.append)
        resets = sum(isinstance(e, StreamReset) for e in events)
        report("k) mid-stream permanent error -> api_error",
               result.status == "api_error" and resets == 0 and client.calls == 1,
               f"status={result.status}, API calls={client.calls}; {result.error_message}")

        # l) ask_user pauses the run; resume with an answer completes it, in one log
        client = FakeClient([
            make_response([ask("Gross or net revenue?", ["Gross", "Net"])], "tool_use"),
            make_response([text("Net revenue is 10.")], "end_turn"),
        ])
        events = []
        paused = run_agent("Revenue?", sb, client, on_event=events.append, pause_on_ask=True)  # type: ignore[arg-type]
        waiting_ok = (paused.status == "waiting_for_user" and paused.question == "Gross or net revenue?"
                      and paused.options == ["Gross", "Net"] and paused.state is not None)
        final = resume_agent(paused.state, "2", sb, client, on_event=events.append)  # type: ignore[arg-type]
        results = last_tool_results(client, 2)
        log = read_log(final)
        report("l) pause -> resume -> completed",
               waiting_ok and final.status == "completed"
               and [r["content"] for r in results] == ["User answered: Net"]
               and final.log_path == paused.log_path and final.steps == 2
               and log is not None and log["status"] == "completed" and len(log["steps"]) == 2
               and log["steps"][0]["tool_calls"][0]["output"] == "User answered: Net"
               and [e.step for e in events if isinstance(e, StepStarted)] == [1, 2],
               f"paused={paused.status} ({paused.question!r}), final={final.status}, "
               f"model got {[r['content'] for r in results]}, log steps={len(log['steps']) if log else '-'}")
        cleanup(final)

        # m) run_code and ask_user in one answer: both results go back together, in order
        client = FakeClient([
            make_response([tool_use("print(2 + 2)", "toolu_code"), ask("Which year?", block_id="toolu_ask")],
                          "tool_use"),
            make_response([text("Done.")], "end_turn"),
        ])
        paused = run_agent("Q", sb, client, pause_on_ask=True)  # type: ignore[arg-type]
        ran_before_pause = len(paused.state.pending_tool_results) == 2 if paused.state else False
        final = resume_agent(paused.state, "2011", sb, client)  # type: ignore[arg-type]
        results = last_tool_results(client, 2)
        report("m) tool results sent together",
               ran_before_pause and final.status == "completed"
               and [r["tool_use_id"] for r in results] == ["toolu_code", "toolu_ask"]
               and results[0]["content"].startswith("4") and results[1]["content"] == "User answered: 2011"
               and len(client.requests[1][-1]["content"]) == 2,
               f"ids={[r['tool_use_id'] for r in results]}, contents="
               f"{[r['content'].splitlines()[0] for r in results]}")
        cleanup(final)

        # n) The question limit counts across resumes
        client = FakeClient([
            make_response([ask("First?", block_id="a1")], "tool_use"),
            make_response([ask("Second?", block_id="a2")], "tool_use"),
            make_response([ask("Third?", block_id="a3")], "tool_use"),
            make_response([text("Final.")], "end_turn"),
        ])
        result = run_agent("Q", sb, client, pause_on_ask=True)  # type: ignore[arg-type]
        pauses = 0
        while result.status == "waiting_for_user" and result.state is not None:
            pauses += 1
            result = resume_agent(result.state, "", sb, client)  # type: ignore[arg-type]
        third = last_tool_results(client, 4)
        report("n) question limit across resumes",
               pauses == MAX_USER_QUESTIONS and result.status == "completed"
               and third[0]["content"].startswith(QUESTION_LIMIT_REACHED),
               f"pauses={pauses}, third question got: {third[0]['content'].splitlines()[0]!r}")
        cleanup(result)

        # g) A crashing event handler does not break the run
        def broken_handler(event: Event) -> None:
            raise RuntimeError("display is broken")

        client = FakeClient([
            make_response([tool_use("print(1)")], "tool_use"),
            make_response([text("Done.")], "end_turn"),
        ])
        result, _ = run(sb, client, on_event=broken_handler)
        report("g) failing handler ignored",
               result.status == "completed" and result.final_answer == "Done.",
               f"status={result.status}, answer={result.final_answer!r}")

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
