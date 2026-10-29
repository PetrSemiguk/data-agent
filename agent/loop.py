"""The agent loop: ask Claude, run the tools it requests, repeat until done."""

import copy
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import anthropic
import httpx2

from agent.config import (
    API_MAX_RETRIES,
    API_TIMEOUT_SECONDS,
    LOGS_DIR,
    MAX_HISTORY_TURNS,
    MAX_REVISIONS,
    MAX_STEPS,
    MAX_TOTAL_TOKENS,
    OUTPUTS_DIR,
    VERIFY_ANSWERS,
)
from agent.events import (
    AnswerRevised,
    ApiCallFinished,
    ApprovalRequested,
    ApprovalResolved,
    AssistantText,
    Event,
    EventHandler,
    RunFinished,
    RunStarted,
    StepStarted,
    StreamReset,
    TextDelta,
    ToolCall,
    ToolResult,
    PlanUpdated,
    UserAnswered,
    UserQuestion,
    VerificationFinished,
    VerificationStarted,
    VerifierEvent,
)
from agent.memory import DatasetMemory, MemoryFullError, memory_context
from agent.prompts import SYSTEM_PROMPT
from agent.sandbox import Sandbox
from agent.tools import (
    ASK_USER_TOOL,
    EXPORT_DATA_TOOL,
    REMEMBER_TOOL,
    TOOLS,
    UPDATE_PLAN_TOOL,
    ApprovalRequest,
    ApproveFn,
    AskUserFn,
    ToolContext,
    answer_for_model,
    check_ask_user,
    execute_tool,
    prepare_export,
    resolve_answer,
    resolve_approval,
)

if TYPE_CHECKING:
    from agent.verifier import VerificationReport

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)
HAS_DIGIT = re.compile(r"\d")
REVISION_FAILED_WARNING = (
    "Verification found errors that the agent could not fix. "
    "Showing the original draft — see the issues below."
)

# Errors raised while reading an already open stream; the SDK does not retry these.
STREAM_BROKEN_ERRORS = (httpx2.TransportError, anthropic.APIConnectionError)
# Error types the API may send as an event inside an open stream that are worth retrying.
TRANSIENT_ERROR_TYPES = {"overloaded_error", "api_error", "rate_limit_error", "timeout_error"}
STREAM_RETRY_BASE_DELAY_S = 0.5

CONTINUATION_NOTE = (
    "This question continues the conversation above about the same dataset. "
    "Variables you created while answering earlier questions may still exist in the "
    "kernel, and `df` is still the original data. If you want to reuse an earlier "
    "variable, check that it exists first with get_schema or dir()."
)

class HistoryTurn(NamedTuple):
    """One earlier turn of a chat: the question, the final answer and, if the
    run did something besides answering (exports, charts), the app's record of it."""

    question: str
    answer: str
    record: str | None = None


RunStatus = Literal[
    "running",
    "completed",
    "max_steps",
    "budget_exceeded",
    "api_error",
    "stopped",
    "waiting_for_user",
]


InteractionKind = Literal["question", "approval"]


@dataclass
class PendingInteraction:
    """A tool call the run is paused on until the human responds.

    Two kinds share one pause/resume mechanism:
    - "question" (ask_user): the human answers with text or picks an option;
    - "approval" (a risky action such as export_data): the human approves or
      denies the prepared `request`.
    """

    tool_use_id: str
    kind: InteractionKind
    # The JSON-log entry of this call, completed when the human responds.
    log_entry: dict[str, Any]
    question: str | None = None
    options: list[str] | None = None
    request: ApprovalRequest | None = None


@dataclass
class AgentState:
    """Everything needed to continue a run later: conversation, counters, log.

    Kept by the caller between `run_agent` and `resume_agent` (the web UI keeps
    it in st.session_state). It lives in memory only and is not saved to disk.
    """

    run_id: str
    # None for runs that save no files (the verifier).
    output_dir: Path | None
    log_path: Path
    log: dict[str, Any]
    model: str
    max_steps: int
    messages: list[dict[str, Any]]
    step: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    api_seconds: float = 0.0
    code_seconds: float = 0.0
    # Time the agent actually worked; waiting for the human is not counted.
    active_seconds: float = 0.0
    questions_asked: int = 0
    approvals_requested: int = 0
    # Results of the other tools from the step that paused; sent with the response.
    pending_tool_results: list[dict[str, Any]] = field(default_factory=list)
    pending: PendingInteraction | None = None
    charts: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    # Facts about approvals (exported or denied files), recorded as they happen.
    actions: list[str] = field(default_factory=list)
    plan: list[dict[str, str]] | None = None
    # What the model is told and offered; the verifier passes its own.
    system: str = SYSTEM_PROMPT
    tools: list[dict[str, Any]] = field(default_factory=lambda: TOOLS)
    # A successful call of this tool ends the run; its input is kept here.
    finish_tool: str | None = None
    finish_input: dict[str, Any] | None = None
    # Kernel namespace for the run's code; None means the kernel's globals.
    namespace: str | None = None
    # Checking the final answer. Only run_agent turns `verify` on, so the
    # verifier's own run is never verified.
    question: str = ""
    verify: bool = False
    verified: bool = False
    revisions: int = 0
    verification: "VerificationReport | None" = None
    verification_warning: str | None = None
    # The answer before the agent revised it after the verifier's remarks.
    draft_answer: str | None = None
    # Memory of the dataset (None = off), the ids of the notes shown to the model,
    # notes from remember calls (saved at the end) and the outcome of saving them.
    memory: DatasetMemory | None = None
    memory_note_ids: list[int] = field(default_factory=list)
    pending_notes: list[dict[str, Any]] = field(default_factory=list)
    remembered: list[str] = field(default_factory=list)
    not_remembered: list[str] = field(default_factory=list)
    memory_not_saved: str | None = None


@dataclass
class RunResult:
    """Outcome of one agent run; returned instead of raising on API problems.

    With status "waiting_for_user" the run is paused; `waiting_for` says on what:
    - "question": show `question` and `options`, then `resume_agent(state, answer, ...)`;
    - "approval": show `approval`, then `resume_agent(state, True or False, ...)`.
    """

    status: RunStatus
    final_answer: str | None
    error_message: str | None
    steps: int
    input_tokens: int
    output_tokens: int
    duration_s: float
    charts: list[str] = field(default_factory=list)
    log_path: str | None = None
    waiting_for: InteractionKind | None = None
    question: str | None = None
    options: list[str] | None = None
    approval: ApprovalRequest | None = None
    state: AgentState | None = None
    # Data files written by approved export_data calls.
    exports: list[str] = field(default_factory=list)
    # What the run did besides answering, as written by the app (see action_record).
    actions: list[str] = field(default_factory=list)
    plan: list[dict[str, str]] | None = None
    # True if the agent answered while plan items were still pending/in progress.
    plan_incomplete: bool = False
    # The verifier's report (None if the answer was not checked). Tokens and
    # duration above include the verification; the report has its own share.
    verification: "VerificationReport | None" = None
    draft_answer: str | None = None
    revised: bool = False
    # The agent could not finish its revision: final_answer is the unrevised draft.
    revision_failed: bool = False
    verification_warning: str | None = None
    # Memory: notes shown to the model, notes saved by this run, notes refused
    # one by one (with the reason), or why none were saved at all.
    memory_loaded: int = 0
    remembered: list[str] = field(default_factory=list)
    not_remembered: list[str] = field(default_factory=list)
    memory_not_saved: str | None = None


def create_client() -> anthropic.Anthropic:
    """Claude API client; the SDK retries rate limits, 5xx and network errors itself."""
    return anthropic.Anthropic(max_retries=API_MAX_RETRIES, timeout=API_TIMEOUT_SECONDS)


def run_agent(
    question: str,
    sandbox: Sandbox,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-5-5",
    max_steps: int = MAX_STEPS,
    ask_user_fn: AskUserFn | None = None,
    dataset_file: str | None = None,
    on_event: EventHandler | None = None,
    pause_on_ask: bool = False,
    history: list[HistoryTurn] | list[tuple[str, str]] | None = None,
    conversation_id: str | None = None,
    turn: int | None = None,
    verify: bool | None = None,
    approve_fn: ApproveFn | None = None,
    memory: DatasetMemory | None = None,
) -> RunResult:
    """Answer a question about the data in the sandbox.

    `history` holds earlier turns of the same chat as HistoryTurn (or plain
    (question, final answer) pairs); the last MAX_HISTORY_TURNS of them are sent
    before the new question, each with the app's record of its actions. Every
    question is still its own run; `conversation_id` and `turn` link the runs of
    one chat in their logs.

    How the agent's clarifying questions (ask_user) are answered:
    - `ask_user_fn` given: it is called and blocks until the human answers (console);
    - `pause_on_ask=True`: the run returns status "waiting_for_user" and is
      continued with `resume_agent` (web UI, which cannot block);
    - neither: the agent is told the user is not available and must assume.
    Risky actions (export_data) need approval the same way: `approve_fn` blocks
    until the human decides, `pause_on_ask=True` pauses the run, and without
    either the action is denied automatically.
    `on_event` receives progress events; the loop itself prints nothing.
    API errors do not raise: they end the run with status "api_error".

    With `memory`, the dataset's notes from earlier sessions are put into the
    first user message and the agent gets the remember tool; its new notes are
    saved only if the run succeeds.

    With `verify` (default VERIFY_ANSWERS), a final answer that contains numbers
    is checked by the verifier; if it contradicts claims, the agent gets the
    remarks and revises the answer within the same run and limits.
    """
    emit = safe_emitter(on_event)
    notes = memory.store.list_notes(memory.fingerprint) if memory else []
    state = new_state(
        build_messages(question, history, memory_context(notes)),
        model,
        max_steps,
        {
            "conversation_id": conversation_id,
            "turn": turn,
            "history_turns_sent": len((history or [])[-MAX_HISTORY_TURNS:]),
            "dataset_file": dataset_file,
            "question": question,
            "memory": {
                "fingerprint": memory.fingerprint,
                "loaded": [note.id for note in notes],
            } if memory else None,
        },
        question=question,
        verify=VERIFY_ANSWERS if verify is None else verify,
        tools=[*TOOLS, REMEMBER_TOOL] if memory else TOOLS,
        memory=memory,
        memory_note_ids=[note.id for note in notes],
    )
    assert state.output_dir is not None
    emit(RunStarted(run_id=state.run_id, output_dir=state.output_dir.as_posix(), log_path=state.log_path.as_posix()))
    return continue_run(state, sandbox, client, emit, ask_user_fn, pause_on_ask, approve_fn)


def new_state(
    messages: list[dict[str, Any]],
    model: str,
    max_steps: int,
    log_fields: dict[str, Any],
    log_suffix: str | None = None,
    **settings: Any,
) -> AgentState:
    """Create the run folder, the log file and the state of a new run.

    `log_fields` are extra keys for the log; `settings` are other AgentState
    fields (the verifier passes its system prompt, tools and finish tool).
    A run with `log_suffix` gets no output folder, only the log file
    logs/runs/<timestamp><log_suffix>.json.
    """
    run_dir: Path | None = None
    if log_suffix is None:
        # Every run saves its files (charts) and its log under the same run id.
        run_dir = create_run_dir()
        log_path = PROJECT_ROOT / LOGS_DIR / f"{run_dir.name}.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_path = create_unique(PROJECT_ROOT / LOGS_DIR, f"{log_suffix}.json", create_empty_file)
    run_id = log_path.stem
    state = AgentState(
        run_id=run_id,
        output_dir=run_dir,
        log_path=log_path,
        model=model,
        max_steps=max_steps,
        messages=messages,
        log={
            "run_id": run_id,
            **log_fields,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
            "model": model,
            "status": "running",
            "error_message": None,
            "final_answer": None,
            "totals": {},
            "charts": [],
            "steps": [],
        },
        **settings,
    )
    write_log(log_path, state.log)
    return state


def build_messages(
    question: str,
    history: list[HistoryTurn] | list[tuple[str, str]] | None,
    memory_text: str | None = None,
) -> list[dict[str, Any]]:
    """Start the conversation: recent chat turns as plain messages, then the question.

    The note about continuing a chat goes into the new user message rather than
    the system prompt, so the system prompt stays the same for every run.
    The record of an answer's actions goes into the NEXT user message: it comes
    from the app, so it is never put into the model's own words.
    The dataset's memory goes into the FIRST user message, so in a chat it stays
    at the start of the conversation on every turn.
    """
    recent = [HistoryTurn(*turn) for turn in (history or [])[-MAX_HISTORY_TURNS:]]
    messages: list[dict[str, Any]] = []
    previous_record: str | None = None
    for turn in recent:
        messages.append({"role": "user", "content": with_record(previous_record, turn.question)})
        messages.append({"role": "assistant", "content": turn.answer})
        previous_record = turn.record
    if not recent:
        messages.append({"role": "user", "content": question})
    else:
        content = [{"type": "text", "text": CONTINUATION_NOTE}]
        if previous_record:
            content.append({"type": "text", "text": previous_record})
        content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": content})
    if memory_text:
        first = messages[0]["content"]
        blocks = [{"type": "text", "text": first}] if isinstance(first, str) else first
        messages[0] = {"role": "user", "content": [{"type": "text", "text": memory_text}, *blocks]}
    return messages


def with_record(record: str | None, text: str) -> str | list[dict[str, str]]:
    """A user message, preceded by the record of the answer before it, if any."""
    if not record:
        return text
    return [{"type": "text", "text": record}, {"type": "text", "text": text}]


def action_record(result: RunResult) -> str | None:
    """The app's record of what a finished run did besides answering, for the chat history.

    Built from facts the loop saw (approvals, written files), not from the
    model's text, so the next run can rely on it. None if there is nothing to record.
    """
    lines = list(result.actions)
    lines += [f"chart saved: {Path(chart).name} at {chart}" for chart in result.charts]
    if not lines:
        return None
    body = "\n".join(f"- {line}" for line in lines)
    return f"[System record of the actions in your previous answer, written by the app:\n{body}]"


def describe_approval(request: ApprovalRequest, approved: bool, automatic: bool, file_path: str | None) -> str:
    """One line for the record of an export decision."""
    name = Path(request.path).name
    if approved and file_path:
        return f"export approved by the user: {Path(file_path).name}, {request.details['Size']}, written to {file_path}"
    if approved:
        return f"export approved by the user but it failed: {name} was not written"
    if automatic:
        return f"export denied automatically (no user available): {name} was not written"
    return f"export denied by the user: {name} was not written"


def resume_agent(
    state: AgentState,
    reply: str | bool,
    sandbox: Sandbox,
    client: anthropic.Anthropic,
    on_event: EventHandler | None = None,
) -> RunResult:
    """Continue a paused run with the human's response.

    A question is answered with text (empty lets the agent decide); an approval
    with True (approve) or False (deny). Uses the same run id, log, step numbers
    and limits as the original run.
    """
    pending = state.pending
    if pending is None:
        raise ValueError("This run is not waiting for the user.")
    emit = safe_emitter(on_event)

    is_error = False
    if pending.kind == "question":
        if not isinstance(reply, str):
            raise TypeError("A question is answered with text.")
        resolved = resolve_answer(reply, pending.options)
        output = answer_for_model(resolved)
        emit(UserAnswered(step=state.step, question=pending.question or "", answer=resolved))
    else:
        if not isinstance(reply, bool) or pending.request is None:
            raise TypeError("An approval is answered with True (approve) or False (deny).")
        ctx = ToolContext(sandbox=sandbox, output_dir=state.output_dir)
        start = time.perf_counter()
        output, is_error, file_path = resolve_approval(pending.request, reply, ctx)
        state.code_seconds += time.perf_counter() - start
        state.actions.append(describe_approval(pending.request, reply, False, file_path))
        pending.log_entry["approval"]["decision"] = "approved" if reply else "denied"
        emit(
            ApprovalResolved(
                step=state.step,
                request=pending.request,
                approved=reply,
                automatic=False,
                output=output,
                file_path=file_path,
            )
        )
    for tool_result in state.pending_tool_results:
        if tool_result["tool_use_id"] == pending.tool_use_id:
            tool_result["content"] = output
            tool_result["is_error"] = is_error
    pending.log_entry.update(output=output, is_error=is_error)

    # All results of that step, the response included, go back in one message.
    state.messages.append({"role": "user", "content": state.pending_tool_results})
    state.pending_tool_results = []
    state.pending = None
    write_log(state.log_path, state.log)
    return continue_run(state, sandbox, client, emit, None, pause_on_ask=True)


def approve_now(
    request: ApprovalRequest,
    approve_fn: ApproveFn | None,
    ctx: ToolContext,
    step: int,
    log_entry: dict[str, Any],
    emit: EventHandler,
    actions: list[str],
) -> tuple[str, bool]:
    """Decide an approval without pausing: ask `approve_fn`, or deny if there is no human."""
    ctx.approvals_requested += 1
    emit(ApprovalRequested(step=step, request=request))
    automatic = approve_fn is None
    approved = approve_fn(request) if approve_fn is not None else False
    output, is_error, file_path = resolve_approval(request, approved, ctx, automatic)
    actions.append(describe_approval(request, approved, automatic, file_path))
    log_entry["approval"]["decision"] = "approved" if approved else ("auto-denied" if automatic else "denied")
    emit(
        ApprovalResolved(
            step=step, request=request, approved=approved, automatic=automatic, output=output, file_path=file_path
        )
    )
    return output, is_error


def continue_run(
    state: AgentState,
    sandbox: Sandbox,
    client: anthropic.Anthropic,
    emit: EventHandler,
    ask_user_fn: AskUserFn | None,
    pause_on_ask: bool,
    approve_fn: ApproveFn | None = None,
) -> RunResult:
    """Run steps until the run finishes or pauses for the human."""
    segment_start = time.perf_counter()
    status: RunStatus = "running"
    final_answer: str | None = None
    error_message: str | None = None
    ctx = ToolContext(
        sandbox=sandbox,
        output_dir=state.output_dir,
        ask_user_fn=ask_user_fn,
        questions_asked=state.questions_asked,
        approvals_requested=state.approvals_requested,
        plan=state.plan,
        namespace=state.namespace,
        memory_note_ids=set(state.memory_note_ids) if state.memory else None,
        pending_notes=state.pending_notes,
    )
    step_logs: list[dict[str, Any]] = state.log["steps"]

    try:
        while True:
            if state.step >= state.max_steps:
                status = "max_steps"
                error_message = f"Reached the limit of {state.max_steps} steps without a final answer."
                break
            # Stop before spending more once the budget is used up.
            if state.input_tokens + state.output_tokens >= MAX_TOTAL_TOKENS:
                status = "budget_exceeded"
                error_message = (
                    f"Token budget exceeded: {state.input_tokens + state.output_tokens} tokens "
                    f"used, limit is {MAX_TOTAL_TOKENS}."
                )
                break

            state.step += 1
            step = state.step
            emit(StepStarted(step=step))
            start = time.perf_counter()
            try:
                response = stream_message(
                    client, state.model, state.system, state.tools, state.messages, step, emit
                )
            except (anthropic.AnthropicError, *STREAM_BROKEN_ERRORS) as exc:
                state.api_seconds += time.perf_counter() - start
                status = "api_error"
                error_message = describe_api_error(exc, state.model)
                break
            elapsed = time.perf_counter() - start
            state.api_seconds += elapsed
            state.input_tokens += response.usage.input_tokens
            state.output_tokens += response.usage.output_tokens
            emit(
                ApiCallFinished(
                    step=step,
                    duration_s=elapsed,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
            )

            step_log: dict[str, Any] = {
                "step": step,
                "api_duration_s": round(elapsed, 2),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "assistant_text": extract_text(response.content),
                "tool_calls": [],
            }
            step_logs.append(step_log)

            # Keep the whole assistant turn, including tool_use blocks.
            state.messages.append({"role": "assistant", "content": response.content})
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    emit(AssistantText(step=step, text=block.text.strip()))
                elif block.type == "tool_use":
                    emit(ToolCall(step=step, name=block.name, input=block.input))

            if response.stop_reason == "end_turn":
                answer = extract_text(response.content)
                if needs_verification(state, answer) and verify_final_answer(state, answer, sandbox, client, emit):
                    # The remarks are in the messages now: the agent works on.
                    write_log(state.log_path, state.log)
                    continue
                status = "completed"
                final_answer = answer
                if state.draft_answer is not None:
                    emit(AnswerRevised(step=step, draft_answer=state.draft_answer, final_answer=answer))
                break

            if response.stop_reason != "tool_use":
                status = "stopped"
                error_message = f"Agent stopped unexpectedly (stop_reason={response.stop_reason})."
                final_answer = extract_text(response.content) or None
                break

            # Run every requested tool. A question or an approval for the human is
            # held back (its slot keeps the order) and the other tools still run first.
            tool_results: list[dict[str, Any]] = []
            pending: PendingInteraction | None = None
            for block in response.content:
                if block.type != "tool_use":
                    continue
                log_entry: dict[str, Any] = {"name": block.name, "input": block.input}
                held: PendingInteraction | None = None
                start = time.perf_counter()
                outcome: tuple[str, bool] | None
                if block.name == ASK_USER_TOOL["name"] and pause_on_ask and ask_user_fn is None:
                    outcome = check_ask_user(block.input, ctx)
                    if outcome is None:
                        held = PendingInteraction(
                            tool_use_id=block.id,
                            kind="question",
                            log_entry=log_entry,
                            question=block.input["question"],
                            options=block.input.get("options"),
                        )
                elif block.name == EXPORT_DATA_TOOL["name"]:
                    prepared = prepare_export(block.input, ctx)
                    if isinstance(prepared, ApprovalRequest):
                        log_entry["approval"] = {"request": prepared.details, "decision": None}
                        if pause_on_ask and approve_fn is None:
                            held = PendingInteraction(
                                tool_use_id=block.id, kind="approval", log_entry=log_entry, request=prepared
                            )
                            outcome = None
                        else:
                            outcome = approve_now(prepared, approve_fn, ctx, step, log_entry, emit, state.actions)
                    else:
                        outcome = prepared
                else:
                    outcome = execute_tool(block.name, block.input, ctx)

                if held is not None:
                    if pending is None:
                        if held.kind == "question":
                            ctx.questions_asked += 1
                        else:
                            ctx.approvals_requested += 1
                        log_entry.update(output=None, is_error=False, duration_s=0.0)
                        step_log["tool_calls"].append(log_entry)
                        tool_results.append(
                            {"type": "tool_result", "tool_use_id": block.id, "content": None, "is_error": False}
                        )
                        pending = held
                        continue
                    outcome = ("Only one question or approval at a time. Try again after the user responds.", True)
                assert outcome is not None
                output, is_error = outcome
                elapsed = time.perf_counter() - start
                state.code_seconds += elapsed
                # Let the model see slow operations.
                output = f"{output}\n[executed in {elapsed:.1f}s]"
                emit(
                    ToolResult(
                        step=step,
                        name=block.name,
                        output=output,
                        is_error=is_error,
                        duration_s=elapsed,
                        images=list(ctx.last_images),
                        chart_paths=list(ctx.last_chart_paths),
                    )
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                        "is_error": is_error,
                    }
                )
                log_entry.update(output=output, is_error=is_error, duration_s=round(elapsed, 2))
                step_log["tool_calls"].append(log_entry)
                if block.name == UPDATE_PLAN_TOOL["name"] and not is_error and ctx.plan:
                    plan = copy.deepcopy(ctx.plan)
                    step_log.setdefault("plan_updates", []).append(plan)
                    emit(PlanUpdated(step=step, items=plan))
                if block.name == state.finish_tool and not is_error:
                    state.finish_input = block.input

            if state.finish_input is not None:
                status = "completed"
                final_answer = extract_text(response.content) or None
                break

            if pending is not None:
                state.pending_tool_results = tool_results
                state.pending = pending
                status = "waiting_for_user"
                if pending.request is not None:
                    emit(ApprovalRequested(step=step, request=pending.request))
                else:
                    emit(UserQuestion(step=step, question=pending.question or "", options=pending.options))
                break

            state.messages.append({"role": "user", "content": tool_results})
            # Save after every step so a crash still leaves a record.
            write_log(state.log_path, state.log)
    finally:
        state.questions_asked = ctx.questions_asked
        state.approvals_requested = ctx.approvals_requested
        state.pending_notes = ctx.pending_notes
        state.plan = ctx.plan
        state.active_seconds += time.perf_counter() - segment_start
        if state.output_dir is not None:
            state.charts = [chart.as_posix() for chart in sorted(state.output_dir.glob("*.png"))]
            state.exports = [
                path.as_posix() for path in sorted(state.output_dir.iterdir()) if path.suffix in (".csv", ".xlsx")
            ]
        # Not enforced for now: a final answer with open plan items is only flagged.
        open_items = [
            item["id"] for item in state.plan or [] if item["status"] in ("pending", "in_progress")
        ]
        plan_incomplete = status == "completed" and bool(open_items)
        warnings = (
            [f"Final answer given while plan items are not finished: {', '.join(open_items)}."]
            if plan_incomplete
            else []
        )
        report = state.verification
        revised = status == "completed" and state.draft_answer is not None
        # The revision ended without a new answer (step limit, budget, API error):
        # the draft is better than nothing, but it is shown with a warning.
        revision_failed = state.draft_answer is not None and status not in ("completed", "waiting_for_user")
        if revision_failed:
            final_answer = state.draft_answer
            state.verification_warning = REVISION_FAILED_WARNING
        if state.verification_warning:
            warnings.append(state.verification_warning)
        if status != "waiting_for_user":
            save_memory(state, status, revision_failed)
        if state.memory_not_saved:
            warnings.append(state.memory_not_saved)
        if state.memory is not None:
            state.log["memory"] = {
                **(state.log.get("memory") or {}),
                "remember_calls": state.pending_notes,
                "saved": state.remembered,
                "not_remembered": state.not_remembered,
                "not_saved_reason": state.memory_not_saved,
            }
        # Totals include the verifier's run; its own share is listed separately.
        verify_in = report.input_tokens if report else 0
        verify_out = report.output_tokens if report else 0
        state.log.update(
            finished_at=datetime.now().isoformat(timespec="seconds"),
            status=status,
            error_message=error_message,
            final_answer=final_answer,
            plan=state.plan,
            plan_incomplete=plan_incomplete,
            warnings=warnings,
            charts=state.charts,
            exports=state.exports,
            actions=state.actions,
            verification=asdict(report) if report else None,
            draft_answer=state.draft_answer,
            revised=revised,
            revision_failed=revision_failed,
            totals={
                "steps": len(step_logs),
                "input_tokens": state.input_tokens + verify_in,
                "output_tokens": state.output_tokens + verify_out,
                "duration_s": round(state.active_seconds, 2),
                "api_duration_s": round(state.api_seconds, 2),
                "code_duration_s": round(state.code_seconds, 2),
                "verification": {
                    "input_tokens": verify_in,
                    "output_tokens": verify_out,
                    "duration_s": report.duration_s if report else 0.0,
                },
            },
        )
        write_log(state.log_path, state.log)
        pending_now = state.pending if status == "waiting_for_user" else None
        result = RunResult(
            status=status,
            final_answer=final_answer,
            error_message=error_message,
            steps=len(step_logs),
            input_tokens=state.input_tokens + verify_in,
            output_tokens=state.output_tokens + verify_out,
            duration_s=round(state.active_seconds, 2),
            charts=state.charts,
            log_path=state.log_path.as_posix(),
            waiting_for=pending_now.kind if pending_now else None,
            question=pending_now.question if pending_now else None,
            options=pending_now.options if pending_now else None,
            approval=pending_now.request if pending_now else None,
            state=state if pending_now else None,
            exports=list(state.exports),
            actions=list(state.actions),
            memory_loaded=len(state.memory_note_ids),
            remembered=list(state.remembered),
            not_remembered=list(state.not_remembered),
            memory_not_saved=state.memory_not_saved,
            plan=copy.deepcopy(state.plan),
            plan_incomplete=plan_incomplete,
            verification=report,
            draft_answer=state.draft_answer,
            revised=revised,
            revision_failed=revision_failed,
            verification_warning=state.verification_warning,
        )
        emit(
            RunFinished(
                result=result,
                api_duration_s=state.api_seconds,
                code_duration_s=state.code_seconds,
            )
        )

    return result


def save_memory(state: AgentState, status: RunStatus, revision_failed: bool) -> None:
    """Save the run's remember notes, but only after a fully successful answer.

    Notes from a run that failed, or whose answer still has contradicted
    claims, may rest on wrong numbers, so they are dropped with a reason.
    """
    if state.memory is None or not state.pending_notes:
        return
    report = state.verification
    revised = status == "completed" and state.draft_answer is not None
    reason = None
    if revision_failed:
        reason = "the answer has errors the agent could not fix"
    elif status != "completed":
        reason = f"the run ended with status {status}"
    elif report is not None and report.status == "issues_found" and not revised:
        reason = "the verifier contradicted the answer and it was not corrected"
    if reason:
        state.memory_not_saved = f"{len(state.pending_notes)} memory notes were not saved: {reason}."
        return
    memory = state.memory
    to_save = []
    for note in state.pending_notes:
        refused = unchecked_fact_reason(state, note)
        if refused:
            state.not_remembered.append(f"{note['text']} ({refused})")
        else:
            to_save.append(note)
    for note in to_save:
        # Facts about the data come from the agent; definitions and preferences from the user.
        source = "agent" if note["kind"] == "data_fact" else "user"
        # A checked fact keeps the verifier's evidence: numbers it recomputed itself.
        evidence = note["check"]["evidence"] if note.get("check") else note["evidence"]
        fields = (memory.fingerprint, note["kind"], note["text"], evidence, source, state.run_id)
        try:
            if note["replaces"] is not None:
                saved = memory.store.replace_note(note["replaces"], *fields)
            else:
                memory.store.add_note(*fields)
                saved = True
        except MemoryFullError:
            saved = False
        if saved:
            state.remembered.append(note["text"])
        else:
            state.not_remembered.append(f"{note['text']} (memory is full or the note to replace was deleted)")


def unchecked_fact_reason(state: AgentState, note: dict[str, Any]) -> str | None:
    """Why a note may not be saved, or None if it may.

    Definitions and preferences are the user's decisions and are not checked.
    A data_fact needs a "supported" verdict: better to forget than to remember
    something wrong.
    """
    if note["kind"] != "data_fact":
        return None
    check = note.get("check")
    if check is None:
        if not state.verify:
            return "not checked: answer verification is off"
        if state.verification is None or state.verification.status == "incomplete":
            return "not checked: the verification did not complete"
        return "not checked: noted after the answer was verified"
    if check["status"] == "supported":
        return None
    detail = check.get("correction") or check["evidence"]
    return f"{check['status']}: {detail}"


def needs_verification(state: AgentState, answer: str) -> bool:
    """Check a final answer once per run, if it contains numbers or the agent
    wants to remember facts about the data."""
    return state.verify and not state.verified and (bool(HAS_DIGIT.search(answer)) or bool(facts_to_check(state)))


def facts_to_check(state: AgentState) -> list[dict[str, Any]]:
    """The run's data_fact notes; only these go to the fact-checker."""
    return [note for note in state.pending_notes if note["kind"] == "data_fact"]


def verify_final_answer(
    state: AgentState,
    answer: str,
    sandbox: Sandbox,
    client: anthropic.Anthropic,
    emit: EventHandler,
) -> bool:
    """Run the verifier on the answer; True if the agent must now revise it.

    A failed or incomplete check never stops the run: the answer stays as it
    is and a warning is kept for the result and the log.
    """
    # Imported here because agent.verifier itself imports this module.
    from agent.verifier import verify_answer

    state.verified = True
    emit(VerificationStarted(step=state.step))
    report: VerificationReport | None = None
    try:
        facts = facts_to_check(state)
        report = verify_answer(
            state.question,
            answer,
            sandbox,
            client,
            state.model,
            on_event=lambda event: emit(VerifierEvent(event=event)),
            notes=facts,
        )
        for note, check in zip(facts, report.note_checks):
            note["check"] = {"status": check.status, "evidence": check.evidence, "correction": check.correction}
    except Exception as exc:
        logger.exception("Verification crashed")
        state.verification_warning = f"Verification failed ({type(exc).__name__}: {exc}); the answer was not checked."
    else:
        state.verification = report
        if report.status == "incomplete":
            state.verification_warning = f"Verification incomplete: {report.error_message} The answer was not checked."

    revise = report is not None and report.status == "issues_found" and state.revisions < MAX_REVISIONS
    emit(
        VerificationFinished(
            step=state.step, report=report, warning=state.verification_warning, revision_requested=revise
        )
    )
    if revise and report is not None:
        state.revisions += 1
        state.draft_answer = answer
        state.messages.append({"role": "user", "content": revision_request(report)})
    return revise


def revision_request(report: VerificationReport) -> str:
    """The message that hands the verifier's remarks to the agent."""
    remarks = "\n".join(
        f"{number}. Claim: \"{claim.claim}\"\n   Evidence: {claim.evidence}\n   Correction: {claim.correction}"
        for number, claim in enumerate(
            (claim for claim in report.claims if claim.status == "contradicted"), start=1
        )
    )
    return (
        "A fact-checker recomputed the claims of your answer from the data and found "
        f"these claims wrong:\n\n{remarks}\n\n"
        "Check these points (you may run code), then give the complete corrected final "
        "answer to the original question. Keep the parts that were not contradicted. "
        "Write it as a normal final answer to the user: do not mention the fact-check, "
        "and do not add headings such as \"Corrected answer\" or \"What changed\"."
    )


def safe_emitter(on_event: EventHandler | None) -> EventHandler:
    """Wrap an event handler so that a broken display never stops the analysis."""

    def emit(event: Event) -> None:
        if on_event is None:
            return
        try:
            on_event(event)
        except Exception:
            logger.exception("Event handler failed on %s", type(event).__name__)

    return emit


def stream_message(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    step: int,
    emit: EventHandler,
) -> Any:
    """Stream one model turn, reporting text deltas, and return the final message.

    Opening the stream is retried by the SDK. A connection that breaks while the
    stream is being read is not, so this function retries the whole turn itself.
    """
    for attempt in range(API_MAX_RETRIES + 1):
        stream_open = False
        try:
            with client.messages.stream(
                model=model,
                max_tokens=4096,
                system=system,
                tools=tools,
                messages=messages,
            ) as stream:
                stream_open = True
                for event in stream:
                    # Thinking and tool-input events are not shown; the final
                    # message still contains every block unchanged.
                    if event.type == "text":
                        emit(TextDelta(step=step, text=event.text))
                return stream.get_final_message()
        except (*STREAM_BROKEN_ERRORS, anthropic.APIStatusError) as exc:
            # Before the stream opened, the SDK has already retried what it could.
            retryable = stream_open and (
                isinstance(exc, STREAM_BROKEN_ERRORS) or is_transient_api_error(exc)
            )
            if not retryable or attempt == API_MAX_RETRIES:
                raise
            logger.warning("Stream broke on step %d (%s); retrying", step, exc)
            emit(StreamReset(step=step))
            time.sleep(STREAM_RETRY_BASE_DELAY_S * 2**attempt)
    raise AssertionError("unreachable")


def create_run_dir() -> Path:
    """Create outputs/<timestamp>/, adding a suffix if that run id is taken."""
    return create_unique(PROJECT_ROOT / OUTPUTS_DIR, "", Path.mkdir)


def create_empty_file(path: Path) -> None:
    """Create an empty file; FileExistsError if it already exists."""
    path.open("x").close()


def create_unique(root: Path, ending: str, create: Callable[[Path], None]) -> Path:
    """Create root/<timestamp><ending> with `create`, adding _2, _3, ... if it exists."""
    base = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root.mkdir(parents=True, exist_ok=True)
    counter = 1
    while True:
        path = root / f"{base if counter == 1 else f'{base}_{counter}'}{ending}"
        try:
            create(path)
            return path
        except FileExistsError:
            counter += 1


def write_log(path: Path, log: dict[str, Any]) -> None:
    """Write the run log as readable JSON."""
    path.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def is_transient_api_error(exc: Exception) -> bool:
    """True for overload, rate-limit and server errors, including ones sent mid-stream.

    An error event inside an open stream arrives with HTTP status 200, so its
    kind is read from the error type in the body.
    """
    if not isinstance(exc, anthropic.APIStatusError):
        return False
    return exc.type in TRANSIENT_ERROR_TYPES or exc.status_code == 429 or exc.status_code >= 500


def describe_api_error(exc: Exception, model: str) -> str:
    """Turn an SDK or connection exception into a short, actionable message."""
    detail = api_error_detail(exc)
    retried = f"after {API_MAX_RETRIES} retries"
    if isinstance(exc, httpx2.TransportError):
        return f"Connection lost while streaming the response {retried}: {type(exc).__name__}."
    # Status 200 means the error came as an event inside an open stream.
    if isinstance(exc, anthropic.APIStatusError) and exc.status_code == 200:
        when = f" {retried}" if exc.type in TRANSIENT_ERROR_TYPES else ""
        return f"Anthropic API error during the response ({exc.type}){when}: {detail}"
    # Permanent errors: retrying cannot help.
    if isinstance(exc, anthropic.AuthenticationError):
        return "Authentication failed: check ANTHROPIC_API_KEY in .env."
    if isinstance(exc, anthropic.PermissionDeniedError):
        return f"Permission denied: this API key cannot use model '{model}'. {detail}"
    if isinstance(exc, anthropic.NotFoundError):
        return f"Not found: check the model name '{model}'. {detail}"
    if isinstance(exc, anthropic.BadRequestError):
        return f"Bad request: {detail}"
    # Temporary errors the SDK already retried.
    if isinstance(exc, anthropic.RateLimitError):
        return f"Rate limit still exceeded {retried}. Wait a minute and try again."
    if isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
        return f"Anthropic API unavailable (HTTP {exc.status_code}) {retried}. Try again later."
    if isinstance(exc, anthropic.APITimeoutError):
        return f"Request timed out {retried} (timeout {API_TIMEOUT_SECONDS}s)."
    if isinstance(exc, anthropic.APIConnectionError):
        return f"Could not connect to the Anthropic API {retried}: check your network."
    if isinstance(exc, anthropic.APIStatusError):
        return f"API error (HTTP {exc.status_code}): {detail}"
    return f"API client error: {exc}"


def api_error_detail(exc: Exception) -> str:
    """The API's own error message, without the raw JSON body."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
    return getattr(exc, "message", str(exc))


def extract_text(content: list[Any]) -> str:
    """Join all text blocks of an assistant message."""
    return "\n".join(block.text for block in content if block.type == "text").strip()
