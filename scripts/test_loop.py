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

from agent.config import (  # noqa: E402
    MAX_HISTORY_TURNS,
    MAX_TOTAL_TOKENS,
    MAX_USER_QUESTIONS,
    OUTPUTS_DIR,
)
import agent.loop  # noqa: E402
from agent.events import (  # noqa: E402
    AnswerRevised,
    ApiCallFinished,
    AssistantText,
    Event,
    PlanUpdated,
    RunFinished,
    RunStarted,
    StepStarted,
    StreamReset,
    TextDelta,
    ToolCall,
    ToolResult,
    VerificationFinished,
    VerificationStarted,
    VerifierEvent,
)
from agent.loop import RunResult, build_messages, resume_agent, run_agent  # noqa: E402
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


def plan_use(statuses: list[str], block_id: str = "toolu_plan") -> Any:
    items = [{"id": str(i), "text": f"Step {i}", "status": s} for i, s in enumerate(statuses, start=1)]
    return SimpleNamespace(type="tool_use", id=block_id, name="update_plan", input={"items": items})


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
        self.systems: list[str] = []
        self.messages = self

    def stream(self, **kwargs: Any) -> FakeStream:
        self.calls += 1
        # A copy of what the model would see on this call.
        self.requests.append(list(kwargs["messages"]))
        self.systems.append(kwargs["system"])
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


def report_use(claims: list[dict[str, str]], block_id: str = "toolu_report") -> Any:
    return SimpleNamespace(type="tool_use", id=block_id, name="report_verification", input={"claims": claims})


CLAIM_OK = {"claim": "Revenue is 10", "status": "supported", "evidence": "sum = 10"}
CLAIM_BAD = {"claim": "Revenue is 12", "status": "contradicted", "evidence": "sum = 10",
             "correction": "Revenue is 10"}
LOGS = PROJECT_ROOT / "logs" / "runs"


def verified_run(sb: Sandbox, client: FakeClient, events: list[Event] | None = None,
                 verify: bool | None = True) -> tuple[RunResult, dict[str, Any] | None]:
    result = run_agent("Revenue?", sb, client, verify=verify,  # type: ignore[arg-type]
                       on_event=events.append if events is not None else None)
    log = read_log(result)
    cleanup(result)
    return result, log


def main() -> int:
    # The older tests script no verifier replies, so checking is off unless a test turns it on.
    agent.loop.VERIFY_ANSWERS = False
    verifier_logs_before = set(LOGS.glob("*_verify.json"))
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

        # o) Earlier chat turns come before the new question; the log links the chat
        client = FakeClient([make_response([text("Technology.")], "end_turn")])
        history = [("Top category?", "Technology."), ("By how much?", "By 4%.")]
        result = run_agent(
            "And the most profitable?", sb, client, history=history,  # type: ignore[arg-type]
            conversation_id="conv-test", turn=3,
        )
        sent = client.requests[0]
        last = sent[-1]["content"]
        log = read_log(result)
        ok = (
            [m["role"] for m in sent] == ["user", "assistant", "user", "assistant", "user"]
            and [m["content"] for m in sent[:4]] == ["Top category?", "Technology.", "By how much?", "By 4%."]
            and isinstance(last, list) and last[-1]["text"] == "And the most profitable?"
            and "continues the conversation" in last[0]["text"]
            and log is not None and log["conversation_id"] == "conv-test" and log["turn"] == 3
            and log["history_turns_sent"] == 2
        )
        report("o) history before the question", ok,
               f"roles={[m['role'] for m in sent]}, log conversation_id="
               f"{log['conversation_id'] if log else '-'}, turn={log['turn'] if log else '-'}")
        cleanup(result)

        # p) Only the last MAX_HISTORY_TURNS turns are sent
        client = FakeClient([make_response([text("Ok.")], "end_turn")])
        history = [(f"Q{i}", f"A{i}") for i in range(1, MAX_HISTORY_TURNS + 3)]
        result = run_agent("New?", sb, client, history=history)  # type: ignore[arg-type]
        sent = client.requests[0]
        past_questions = [m["content"] for m in sent[:-1] if m["role"] == "user"]
        expected = [q for q, _ in history[-MAX_HISTORY_TURNS:]]
        report("p) history trimmed", past_questions == expected,
               f"{len(history)} turns given, sent: {past_questions}")
        cleanup(result)

        # q) update_plan and run_code in ONE response: both run, the plan reaches the result
        events = []
        client = FakeClient([
            make_response([plan_use(["done", "done"]), tool_use("print(2 + 2)", "toolu_code")], "tool_use"),
            make_response([text("Answer.")], "end_turn"),
        ])
        result = run_agent("Q", sb, client, on_event=events.append)  # type: ignore[arg-type]
        results = last_tool_results(client, 2)
        plan_events = [e for e in events if isinstance(e, PlanUpdated)]
        log = read_log(result)
        report("q) update_plan alongside run_code",
               [r["tool_use_id"] for r in results] == ["toolu_plan", "toolu_code"]
               and results[0]["content"].startswith("Plan updated: 2/2 done")
               and results[1]["content"].startswith("4")
               and len(plan_events) == 1 and plan_events[0].step == 1
               and result.plan == plan_events[0].items and not result.plan_incomplete
               and log is not None and log["plan"] == result.plan
               and log["steps"][0]["plan_updates"] == [result.plan],
               f"results={[r['content'].splitlines()[0] for r in results]}, "
               f"PlanUpdated={len(plan_events)}, plan_incomplete={result.plan_incomplete}")
        cleanup(result)

        # r) The plan survives a pause for a question
        client = FakeClient([
            make_response([plan_use(["done", "pending"]), ask("Which year?", block_id="toolu_ask")], "tool_use"),
            make_response([plan_use(["done", "done"], "toolu_plan2")], "tool_use"),
            make_response([text("Answer.")], "end_turn"),
        ])
        paused = run_agent("Q", sb, client, pause_on_ask=True)  # type: ignore[arg-type]
        kept = paused.state.plan if paused.state else None
        final = resume_agent(paused.state, "2011", sb, client)  # type: ignore[arg-type]
        report("r) plan survives pause/resume",
               paused.status == "waiting_for_user" and kept is not None
               and [i["status"] for i in kept] == ["done", "pending"]
               and final.status == "completed" and [i["status"] for i in final.plan or []] == ["done", "done"]
               and not final.plan_incomplete,
               f"plan at pause={[i['status'] for i in kept or []]}, "
               f"final={[i['status'] for i in final.plan or []]}")
        cleanup(final)

        # s) A final answer with open plan items is flagged, not blocked
        client = FakeClient([
            make_response([plan_use(["done", "in_progress", "pending"])], "tool_use"),
            make_response([text("Answer.")], "end_turn"),
        ])
        result = run_agent("Q", sb, client)  # type: ignore[arg-type]
        log = read_log(result)
        report("s) plan_incomplete flagged",
               result.status == "completed" and result.plan_incomplete
               and log is not None and log["plan_incomplete"] and "2, 3" in log["warnings"][0],
               f"status={result.status}, plan_incomplete={result.plan_incomplete}, "
               f"warning={log['warnings'] if log else '-'}")
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

        # t) Verification ok: the answer is kept, revised=False, tokens include the check
        events = []
        client = FakeClient([
            make_response([text("Revenue is 10.")], "end_turn"),
            make_response([report_use([CLAIM_OK])], "tool_use", tokens_in=50, tokens_out=10),
        ])
        result, log = verified_run(sb, client, events)
        kinds = [type(e).__name__ for e in events if not isinstance(e, TextDelta)]
        inner = [type(e.event).__name__ for e in events if isinstance(e, VerifierEvent)]
        report("t) verification ok -> no revision",
               result.status == "completed" and result.final_answer == "Revenue is 10." and not result.revised
               and result.verification is not None and result.verification.status == "ok"
               and client.calls == 2 and client.systems[1].startswith("You are a fact-checker")
               and result.input_tokens == 150 and result.output_tokens == 30
               and "VerificationStarted" in kinds and "VerificationFinished" in kinds
               and "AnswerRevised" not in kinds and "StepStarted" in inner
               and log is not None and log["verification"]["status"] == "ok" and log["revised"] is False
               and log["totals"]["verification"]["input_tokens"] == 50,
               f"verification={result.verification.status if result.verification else None}, "
               f"tokens in={result.input_tokens}, verifier events: {' -> '.join(dict.fromkeys(inner))}")

        # u) Contradicted claim: the agent gets the remarks, answers again, no second check
        events = []
        client = FakeClient([
            make_response([text("Revenue is 12.")], "end_turn"),
            make_response([report_use([CLAIM_BAD])], "tool_use"),
            make_response([text("Corrected: revenue is 10.")], "end_turn"),
        ])
        result, log = verified_run(sb, client, events)
        remarks = client.requests[2][-1]["content"] if client.calls >= 3 else ""
        revised_events = [e for e in events if isinstance(e, AnswerRevised)]
        finished = [e for e in events if isinstance(e, VerificationFinished)]
        history_sent = build_messages("Next?", [("Revenue?", result.final_answer or "")])
        report("u) contradicted -> revised answer",
               result.status == "completed" and result.revised
               and result.draft_answer == "Revenue is 12." and result.final_answer == "Corrected: revenue is 10."
               and client.calls == 3 and client.systems[2] == client.systems[0]
               and "Revenue is 12" in remarks and "Correction: Revenue is 10" in remarks
               and len(finished) == 1 and finished[0].revision_requested and len(revised_events) == 1
               and history_sent[1]["content"] == "Corrected: revenue is 10."
               and log is not None and log["revised"] and log["draft_answer"] == "Revenue is 12."
               and log["verification"]["claims"][0]["status"] == "contradicted",
               f"calls={client.calls}, draft={result.draft_answer!r}, final={result.final_answer!r}, "
               f"history gets {history_sent[1]['content']!r}")

        # u2) The revision runs into max_steps: the draft is kept as the answer, with a warning
        client = FakeClient([
            make_response([text("Revenue is 12.")], "end_turn"),
            make_response([report_use([CLAIM_BAD])], "tool_use"),
        ], repeat=make_response([tool_use("1")], "tool_use"))
        result = run_agent("Revenue?", sb, client, max_steps=3, verify=True)  # type: ignore[arg-type]
        log = read_log(result)
        cleanup(result)
        report("u2) revision hits max_steps -> draft + revision_failed",
               result.status == "max_steps" and result.final_answer == "Revenue is 12."
               and result.revision_failed and not result.revised
               and result.verification_warning == agent.loop.REVISION_FAILED_WARNING
               and log is not None and log["final_answer"] == "Revenue is 12." and log["revision_failed"]
               and agent.loop.REVISION_FAILED_WARNING in log["warnings"],
               f"status={result.status}, answer={result.final_answer!r}, "
               f"revision_failed={result.revision_failed}, warning={result.verification_warning!r}")

        # v) The verifier never reports: answer unchanged, warning, still completed
        client = FakeClient([
            make_response([text("Revenue is 12.")], "end_turn"),
            make_response([text("I think it is fine.")], "end_turn"),
        ])
        result, log = verified_run(sb, client)
        report("v) incomplete check -> answer kept + warning",
               result.status == "completed" and result.final_answer == "Revenue is 12." and not result.revised
               and result.verification is not None and result.verification.status == "incomplete"
               and "incomplete" in (result.verification_warning or "")
               and log is not None and any("incomplete" in w for w in log["warnings"]),
               f"warning: {result.verification_warning}")

        # w) API error inside the verifier: answer unchanged, warning
        client = FakeClient([
            make_response([text("Revenue is 12.")], "end_turn"),
            api_error(anthropic.InternalServerError, 500, "server down"),
        ])
        result, log = verified_run(sb, client)
        report("w) verifier API error -> answer kept + warning",
               result.status == "completed" and result.final_answer == "Revenue is 12."
               and result.verification is not None and result.verification.status == "incomplete"
               and "HTTP 500" in (result.verification_warning or ""),
               f"warning: {result.verification_warning}")

        # x) The verifier crashes with an unexpected exception: the run still completes
        client = FakeClient([
            make_response([text("Revenue is 12.")], "end_turn"),
            RuntimeError("kernel is gone"),
        ])
        result, log = verified_run(sb, client)
        report("x) verifier crash -> answer kept + warning",
               result.status == "completed" and result.final_answer == "Revenue is 12."
               and result.verification is None and "kernel is gone" in (result.verification_warning or "")
               and log is not None and any("Verification failed" in w for w in log["warnings"]),
               f"warning: {result.verification_warning}")

        # y) An answer without digits is not checked
        client = FakeClient([make_response([text("Technology leads.")], "end_turn")])
        events = []
        result, _ = verified_run(sb, client, events)
        report("y) no digits -> no check",
               client.calls == 1 and result.verification is None
               and not any(isinstance(e, VerificationStarted) for e in events),
               f"calls={client.calls}, verification={result.verification}")

        # z) Checking switched off: verify=False (--no-verify) and VERIFY_ANSWERS=False
        client = FakeClient([make_response([text("Revenue is 12.")], "end_turn")])
        off_arg, _ = verified_run(sb, client, verify=False)
        calls_arg = client.calls
        client = FakeClient([make_response([text("Revenue is 12.")], "end_turn")])
        off_config, _ = verified_run(sb, client, verify=None)
        report("z) verification switched off",
               calls_arg == 1 and client.calls == 1
               and off_arg.verification is None and off_config.verification is None,
               f"API calls: verify=False -> {calls_arg}, VERIFY_ANSWERS=False -> {client.calls}")

    for path in set(LOGS.glob("*_verify.json")) - verifier_logs_before:
        path.unlink()
    print(f"\nTotal: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
