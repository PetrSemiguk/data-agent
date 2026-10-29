"""Checks for agent.loop.run_agent with a fake Claude client (no real API calls)."""

import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pandas as pd

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
    ApprovalRequested,
    ApprovalResolved,
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
from agent.loop import HistoryTurn, RunResult, action_record, build_messages, resume_agent, run_agent  # noqa: E402
from agent.memory import DatasetMemory, MemoryStore  # noqa: E402
from agent.tools import EXPORT_DENIED, EXPORT_NO_APPROVER, QUESTION_LIMIT_REACHED  # noqa: E402
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
        self.tools: list[list[str]] = []
        self.messages = self

    def stream(self, **kwargs: Any) -> FakeStream:
        self.calls += 1
        # A copy of what the model would see on this call.
        self.requests.append(list(kwargs["messages"]))
        self.systems.append(kwargs["system"])
        self.tools.append([tool["name"] for tool in kwargs["tools"]])
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


def export_use(file_format: str = "csv", block_id: str = "toolu_export") -> Any:
    tool_input = {"variable": "neg", "filename": "negative", "format": file_format, "reason": "Rows to review."}
    return SimpleNamespace(type="tool_use", id=block_id, name="export_data", input=tool_input)


NOTE = {"kind": "data_fact", "text": "34,335 rows are full duplicates.", "evidence": "duplicated().sum() = 34335"}


def remember_use(block_id: str = "toolu_remember", **note: str) -> Any:
    return SimpleNamespace(type="tool_use", id=block_id, name="remember", input=note or dict(NOTE))


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

        # Approvals: export_data needs the human's decision before it writes a file
        sb.execute("neg = pd.DataFrame({'Quantity': [-1, -2, -3], 'Price': [1.0, 2.0, 3.0]})")

        # ab) pause on approval -> approve -> the file is written and reported
        events = []
        client = FakeClient([
            make_response([text("Exporting."), export_use()], "tool_use"),
            make_response([text("Done, the file is ready.")], "end_turn"),
        ])
        paused = run_agent("Export", sb, client, pause_on_ask=True, on_event=events.append)  # type: ignore[arg-type]
        request = paused.approval
        waiting_ok = (paused.status == "waiting_for_user" and paused.waiting_for == "approval"
                      and request is not None and request.details["Size"] == "3 rows x 2 columns"
                      and not Path(request.path).exists())
        final = resume_agent(paused.state, True, sb, client, on_event=events.append)  # type: ignore[arg-type]
        results = last_tool_results(client, 2)
        log = read_log(final)
        entry = log["steps"][0]["tool_calls"][0] if log else {}
        resolved = [e for e in events if isinstance(e, ApprovalResolved)]
        exported = Path(final.exports[0]) if final.exports else None
        report("ab) approval: approve -> file written",
               waiting_ok and final.status == "completed" and exported is not None and exported.exists()
               and pd.read_csv(exported).shape == (3, 2) and results[0]["content"].startswith("Exported to")
               and any(isinstance(e, ApprovalRequested) for e in events)
               and len(resolved) == 1 and resolved[0].approved and resolved[0].file_path == exported.as_posix()
               and entry.get("approval", {}).get("decision") == "approved",
               f"waiting_for={paused.waiting_for}, action={request.action if request else None!r}, "
               f"model got {results[0]['content'].splitlines()[0]!r}, log decision={entry.get('approval', {}).get('decision')}")
        cleanup(final)

        # ac) pause on approval -> deny -> no file, the model is told
        client = FakeClient([
            make_response([export_use()], "tool_use"),
            make_response([text("The export was denied.")], "end_turn"),
        ])
        paused = run_agent("Export", sb, client, pause_on_ask=True)  # type: ignore[arg-type]
        planned = Path(paused.approval.path) if paused.approval else None
        final = resume_agent(paused.state, False, sb, client)  # type: ignore[arg-type]
        results = last_tool_results(client, 2)
        log = read_log(final)
        entry = log["steps"][0]["tool_calls"][0] if log else {}
        report("ac) approval: deny -> no file",
               final.status == "completed" and planned is not None and not planned.exists() and not final.exports
               and results[0]["content"] == EXPORT_DENIED and entry["approval"]["decision"] == "denied",
               f"model got {results[0]['content']!r}, files={final.exports}")
        cleanup(final)

        # ad) No human (no approve_fn, no pause): denied automatically, nothing written
        events = []
        client = FakeClient([
            make_response([export_use()], "tool_use"),
            make_response([text("I could not export.")], "end_turn"),
        ])
        result = run_agent("Export", sb, client, on_event=events.append)  # type: ignore[arg-type]
        results = last_tool_results(client, 2)
        resolved = [e for e in events if isinstance(e, ApprovalResolved)]
        log = read_log(result)
        report("ad) no human -> auto-denied",
               result.status == "completed" and not result.exports
               and results[0]["content"].startswith(EXPORT_NO_APPROVER)
               and len(resolved) == 1 and resolved[0].automatic and not resolved[0].approved
               and log is not None and log["steps"][0]["tool_calls"][0]["approval"]["decision"] == "auto-denied",
               f"model got {results[0]['content'].splitlines()[0]!r}")
        cleanup(result)

        # ae) A blocking approve_fn (console) approves without pausing
        client = FakeClient([
            make_response([export_use()], "tool_use"),
            make_response([text("Done.")], "end_turn"),
        ])
        asked: list[str] = []
        result = run_agent("Export", sb, client,  # type: ignore[arg-type]
                           approve_fn=lambda request: asked.append(request.action) or True)
        report("ae) approve_fn approves in place",
               result.status == "completed" and len(asked) == 1 and len(result.exports) == 1
               and Path(result.exports[0]).exists(),
               f"asked={asked}, exports={[Path(e).name for e in result.exports]}")
        cleanup(result)

        # af) ask_user and an approval in different steps of one run
        client = FakeClient([
            make_response([ask("CSV or Excel?", ["csv", "xlsx"])], "tool_use"),
            make_response([export_use(file_format="xlsx", block_id="toolu_export2")], "tool_use"),
            make_response([text("Exported as Excel.")], "end_turn"),
        ])
        first = run_agent("Export", sb, client, pause_on_ask=True)  # type: ignore[arg-type]
        second = resume_agent(first.state, "2", sb, client)  # type: ignore[arg-type]
        final = resume_agent(second.state, True, sb, client)  # type: ignore[arg-type]
        log = read_log(final)
        calls = [c["name"] for s in log["steps"] for c in s["tool_calls"]] if log else []
        report("af) question then approval in one run",
               first.waiting_for == "question" and second.waiting_for == "approval"
               and final.status == "completed" and len(final.exports) == 1 and final.exports[0].endswith(".xlsx")
               and last_tool_results(client, 2)[0]["content"] == "User answered: xlsx"
               and calls == ["ask_user", "export_data"] and final.log_path == first.log_path,
               f"pauses: {first.waiting_for} -> {second.waiting_for} -> {final.status}, tool calls={calls}")
        cleanup(final)

        # ah) The next question's history carries the app's record of an approved / denied export
        records = {}
        for decision in (True, False):
            client = FakeClient([
                make_response([export_use()], "tool_use"),
                make_response([text("Here is the file." if decision else "The export was denied.")], "end_turn"),
            ])
            first = run_agent("Export", sb, client, approve_fn=lambda request: decision)  # type: ignore[arg-type]
            record = action_record(first)
            client = FakeClient([make_response([text("Yes.")], "end_turn")])
            second = run_agent("Did you export a file?", sb, client,  # type: ignore[arg-type]
                               history=[HistoryTurn("Export", first.final_answer or "", record)])
            last = client.requests[0][-1]["content"]
            records[decision] = (last[1]["text"] if isinstance(last, list) and len(last) == 3 else "", first)
            cleanup(first)
            cleanup(second)
        approved_text, approved_run = records[True]
        denied_text, _ = records[False]
        report("ah) history records exports",
               approved_text.startswith("[System record")
               and "export approved by the user: negative.csv, 3 rows x 2 columns" in approved_text
               and approved_run.exports[0] in approved_text
               and "export denied by the user: negative.csv was not written" in denied_text,
               f"approved: {approved_text.splitlines()[1] if approved_text else '-'} | "
               f"denied: {denied_text.splitlines()[1] if denied_text else '-'}")

        # ai) The record of an earlier turn precedes the next question; plain pairs still work
        sent = build_messages("Q3", [HistoryTurn("Q1", "A1", "[System record: x]"), ("Q2", "A2")])
        report("ai) record goes into the next user message",
               sent[2]["content"] == [{"type": "text", "text": "[System record: x]"}, {"type": "text", "text": "Q2"}]
               and sent[1]["content"] == "A1" and [b["text"] for b in sent[-1]["content"]][1:] == ["Q3"],
               f"roles={[m['role'] for m in sent]}, message 3={sent[2]['content']}")

        # ag) Code blocked by the guard never runs; the model gets an error with a hint
        client = FakeClient([
            make_response([tool_use("df.to_csv('out.csv')\nprint('written')")], "tool_use"),
            make_response([text("Ok.")], "end_turn"),
        ])
        result = run_agent("Export", sb, client)  # type: ignore[arg-type]
        results = last_tool_results(client, 2)
        report("ag) guard blocks run_code",
               results[0]["is_error"] and "export_data" in results[0]["content"]
               and "written" not in results[0]["content"] and not Path("out.csv").exists(),
               results[0]["content"].splitlines()[1])
        cleanup(result)

        # Memory: notes from remember are saved only after a fully successful run
        memory_dir = Path(tempfile.mkdtemp())
        store = MemoryStore(memory_dir / "memory.db")
        memory = DatasetMemory(store, "fp_loop")

        # aj) A data_fact the verifier supports is saved; the next run gets it in its first message
        client = FakeClient([
            make_response([remember_use()], "tool_use"),
            make_response([text("Done.")], "end_turn"),
            make_response([report_use([{"claim": NOTE["text"], "status": "supported",
                                        "evidence": "duplicated().sum() = 34335"}])], "tool_use"),
        ])
        first = run_agent("Revenue?", sb, client, verify=True, memory=memory)  # type: ignore[arg-type]
        saved = store.list_notes("fp_loop")
        offered = client.tools[0]
        verifier_request = client.requests[2][0]["content"]
        client = FakeClient([make_response([text("Done.")], "end_turn")])
        second = run_agent("Next?", sb, client, memory=memory,  # type: ignore[arg-type]
                           history=[HistoryTurn("Revenue?", "Done.")])
        first_message = client.requests[0][0]["content"]
        log = read_log(first)
        report("aj) note saved, then loaded into the first message",
               first.remembered == [NOTE["text"]] and len(saved) == 1 and saved[0].source == "agent"
               and saved[0].evidence == NOTE["evidence"] and saved[0].run_id == Path(first.log_path or "").stem
               and "remember" in offered and second.memory_loaded == 1
               and f"<note>{NOTE['text']}</note>" in verifier_request
               and isinstance(first_message, list) and first_message[0]["text"].startswith("[Memory for this dataset")
               and f"#{saved[0].id} data_fact" in first_message[0]["text"]
               and first_message[1]["text"] == "Revenue?"
               and log is not None and log["memory"]["saved"] == [NOTE["text"]],
               f"saved={[(n.id, n.text) for n in saved]}, next run's first message starts with "
               f"{first_message[0]['text'].splitlines()[1] if isinstance(first_message, list) else first_message!r}")
        cleanup(first)
        cleanup(second)

        # ak) Without memory (--no-memory, toggle off): no block, no remember tool
        client = FakeClient([make_response([text("Done.")], "end_turn")])
        result = run_agent("Revenue?", sb, client)  # type: ignore[arg-type]
        report("ak) memory off -> nothing loaded",
               client.requests[0][0]["content"] == "Revenue?" and "remember" not in client.tools[0]
               and result.memory_loaded == 0,
               f"first message={client.requests[0][0]['content']!r}, tools={client.tools[0]}")
        cleanup(result)

        # ao) A data_fact the verifier contradicts is not saved; the answer itself is not revised
        client = FakeClient([
            make_response([remember_use()], "tool_use"),
            make_response([text("Revenue is 10.")], "end_turn"),
            make_response([report_use([CLAIM_OK, {"claim": NOTE["text"], "status": "contradicted",
                                                  "evidence": "duplicated().sum() = 30000",
                                                  "correction": "30,000 rows are full duplicates."}])], "tool_use"),
        ])
        before = len(store.list_notes("fp_loop"))
        result = run_agent("Q", sb, client, verify=True, memory=memory)  # type: ignore[arg-type]
        report("ao) contradicted fact -> not remembered",
               result.status == "completed" and not result.revised and client.calls == 3
               and len(store.list_notes("fp_loop")) == before and not result.remembered
               and result.not_remembered == [f"{NOTE['text']} (contradicted: 30,000 rows are full duplicates.)"]
               and result.verification is not None and result.verification.status == "ok"
               and result.verification.note_checks[0].status == "contradicted",
               f"not remembered: {result.not_remembered}; answer verification: "
               f"{result.verification.status if result.verification else None}")
        cleanup(result)

        # ap) A user's definition is saved without a check (no numbers, so no verifier call)
        client = FakeClient([
            make_response([remember_use(kind="definition", text="Revenue means net revenue.")], "tool_use"),
            make_response([text("I will use net revenue.")], "end_turn"),
        ])
        result = run_agent("Q", sb, client, verify=True, memory=memory)  # type: ignore[arg-type]
        saved = [n for n in store.list_notes("fp_loop") if n.text == "Revenue means net revenue."]
        report("ap) user definition saved without check",
               client.calls == 2 and result.verification is None and len(saved) == 1 and saved[0].source == "user",
               f"API calls={client.calls}, saved={[(n.kind, n.source, n.text) for n in saved]}")
        cleanup(result)

        # aq) With verification off, a data_fact is not saved (a definition still is)
        client = FakeClient([
            make_response([remember_use(), remember_use(kind="preference", text="Answer in bullets.", block_id="t2")],
                          "tool_use"),
            make_response([text("Revenue is 10.")], "end_turn"),
        ])
        result = run_agent("Q", sb, client, verify=False, memory=memory)  # type: ignore[arg-type]
        report("aq) verification off -> data_fact not saved",
               result.remembered == ["Answer in bullets."]
               and result.not_remembered == [f"{NOTE['text']} (not checked: answer verification is off)"],
               f"remembered={result.remembered}, not remembered={result.not_remembered}")
        cleanup(result)

        # al) An API error drops the run's notes
        client = FakeClient([
            make_response([remember_use()], "tool_use"),
            api_error(anthropic.AuthenticationError, 401, "invalid x-api-key"),
        ])
        before = len(store.list_notes("fp_loop"))
        result = run_agent("Q", sb, client, memory=memory)  # type: ignore[arg-type]
        log = read_log(result)
        report("al) api_error -> not saved",
               result.status == "api_error" and len(store.list_notes("fp_loop")) == before
               and not result.remembered and "status api_error" in (result.memory_not_saved or "")
               and log is not None and result.memory_not_saved in log["warnings"]
               and log["memory"]["not_saved_reason"] == result.memory_not_saved,
               f"{result.memory_not_saved}")
        cleanup(result)

        # am) A failed revision drops the run's notes
        client = FakeClient([
            make_response([remember_use()], "tool_use"),
            make_response([text("Revenue is 12.")], "end_turn"),
            make_response([report_use([CLAIM_BAD])], "tool_use"),
        ], repeat=make_response([tool_use("1")], "tool_use"))
        result = run_agent("Q", sb, client, max_steps=3, verify=True, memory=memory)  # type: ignore[arg-type]
        report("am) revision_failed -> not saved",
               result.revision_failed and len(store.list_notes("fp_loop")) == before
               and "could not fix" in (result.memory_not_saved or ""),
               f"status={result.status}, {result.memory_not_saved}")
        cleanup(result)

        # an) An answer with a contradicted claim that was not corrected drops the notes
        agent.loop.MAX_REVISIONS = 0
        client = FakeClient([
            make_response([remember_use()], "tool_use"),
            make_response([text("Revenue is 12.")], "end_turn"),
            make_response([report_use([CLAIM_BAD])], "tool_use"),
        ])
        result = run_agent("Q", sb, client, verify=True, memory=memory)  # type: ignore[arg-type]
        agent.loop.MAX_REVISIONS = 1
        report("an) uncorrected contradicted -> not saved",
               result.status == "completed" and not result.revised
               and len(store.list_notes("fp_loop")) == before
               and "not corrected" in (result.memory_not_saved or "")
               # The verifier (third call) gets neither the memory nor the remember tool.
               and client.tools[2] == ["get_schema", "run_code", "report_verification"]
               and "Memory for this dataset" not in str(client.requests[2]),
               f"status={result.status}, {result.memory_not_saved}; verifier tools={client.tools[2]}")
        cleanup(result)
        shutil.rmtree(memory_dir, ignore_errors=True)

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
