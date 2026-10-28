"""The agent loop: ask Claude, run the tools it requests, repeat until done."""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import anthropic
import httpx2

from agent.config import (
    API_MAX_RETRIES,
    API_TIMEOUT_SECONDS,
    LOGS_DIR,
    MAX_HISTORY_TURNS,
    MAX_STEPS,
    MAX_TOTAL_TOKENS,
    OUTPUTS_DIR,
)
from agent.events import (
    ApiCallFinished,
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
    UserAnswered,
    UserQuestion,
)
from agent.prompts import SYSTEM_PROMPT
from agent.sandbox import Sandbox
from agent.tools import (
    ASK_USER_TOOL,
    TOOLS,
    AskUserFn,
    ToolContext,
    answer_for_model,
    check_ask_user,
    execute_tool,
    resolve_answer,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)

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

RunStatus = Literal[
    "running",
    "completed",
    "max_steps",
    "budget_exceeded",
    "api_error",
    "stopped",
    "waiting_for_user",
]


@dataclass
class PendingQuestion:
    """An ask_user call the run is paused on."""

    tool_use_id: str
    question: str
    options: list[str] | None
    # The JSON-log entry of this call, completed when the answer arrives.
    log_entry: dict[str, Any]


@dataclass
class AgentState:
    """Everything needed to continue a run later: conversation, counters, log.

    Kept by the caller between `run_agent` and `resume_agent` (the web UI keeps
    it in st.session_state). It lives in memory only and is not saved to disk.
    """

    run_id: str
    output_dir: Path
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
    # Results of the other tools from the step that asked; sent with the answer.
    pending_tool_results: list[dict[str, Any]] = field(default_factory=list)
    pending_question: PendingQuestion | None = None
    charts: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    """Outcome of one agent run; returned instead of raising on API problems.

    With status "waiting_for_user" the run is paused: show `question` and
    `options` to the human, then call `resume_agent(state, answer, ...)`.
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
    question: str | None = None
    options: list[str] | None = None
    state: AgentState | None = None


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
    history: list[tuple[str, str]] | None = None,
    conversation_id: str | None = None,
    turn: int | None = None,
) -> RunResult:
    """Answer a question about the data in the sandbox.

    `history` holds earlier (question, final answer) pairs of the same chat; the
    last MAX_HISTORY_TURNS of them are sent before the new question. Every
    question is still its own run; `conversation_id` and `turn` link the runs of
    one chat in their logs.

    How the agent's clarifying questions (ask_user) are answered:
    - `ask_user_fn` given: it is called and blocks until the human answers (console);
    - `pause_on_ask=True`: the run returns status "waiting_for_user" and is
      continued with `resume_agent` (web UI, which cannot block);
    - neither: the agent is told the user is not available and must assume.
    `on_event` receives progress events; the loop itself prints nothing.
    API errors do not raise: they end the run with status "api_error".
    """
    emit = safe_emitter(on_event)

    # Every run saves its files (charts) and its log under the same run id.
    run_dir = create_run_dir()
    log_path = PROJECT_ROOT / LOGS_DIR / f"{run_dir.name}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state = AgentState(
        run_id=run_dir.name,
        output_dir=run_dir,
        log_path=log_path,
        model=model,
        max_steps=max_steps,
        messages=build_messages(question, history),
        log={
            "run_id": run_dir.name,
            "conversation_id": conversation_id,
            "turn": turn,
            "history_turns_sent": len((history or [])[-MAX_HISTORY_TURNS:]),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
            "model": model,
            "dataset_file": dataset_file,
            "question": question,
            "status": "running",
            "error_message": None,
            "final_answer": None,
            "totals": {},
            "charts": [],
            "steps": [],
        },
    )
    emit(RunStarted(run_id=state.run_id, output_dir=run_dir.as_posix(), log_path=log_path.as_posix()))
    write_log(log_path, state.log)
    return continue_run(state, sandbox, client, emit, ask_user_fn, pause_on_ask)


def build_messages(question: str, history: list[tuple[str, str]] | None) -> list[dict[str, Any]]:
    """Start the conversation: recent chat turns as plain messages, then the question.

    The note about continuing a chat goes into the new user message rather than
    the system prompt, so the system prompt stays the same for every run.
    """
    recent = (history or [])[-MAX_HISTORY_TURNS:]
    messages: list[dict[str, Any]] = []
    for past_question, past_answer in recent:
        messages.append({"role": "user", "content": past_question})
        messages.append({"role": "assistant", "content": past_answer})
    if not recent:
        messages.append({"role": "user", "content": question})
        return messages
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": CONTINUATION_NOTE},
                {"type": "text", "text": question},
            ],
        }
    )
    return messages


def resume_agent(
    state: AgentState,
    answer: str,
    sandbox: Sandbox,
    client: anthropic.Anthropic,
    on_event: EventHandler | None = None,
) -> RunResult:
    """Continue a run paused on a question; an empty answer lets the agent decide.

    Uses the same run id, log, step numbers and limits as the original run.
    """
    pending = state.pending_question
    if pending is None:
        raise ValueError("This run is not waiting for an answer.")
    emit = safe_emitter(on_event)

    resolved = resolve_answer(answer, pending.options)
    output = answer_for_model(resolved)
    for tool_result in state.pending_tool_results:
        if tool_result["tool_use_id"] == pending.tool_use_id:
            tool_result["content"] = output
    pending.log_entry["output"] = output
    emit(UserAnswered(step=state.step, question=pending.question, answer=resolved))

    # All results of that step, the answer included, go back in one message.
    state.messages.append({"role": "user", "content": state.pending_tool_results})
    state.pending_tool_results = []
    state.pending_question = None
    write_log(state.log_path, state.log)
    return continue_run(state, sandbox, client, emit, None, pause_on_ask=True)


def continue_run(
    state: AgentState,
    sandbox: Sandbox,
    client: anthropic.Anthropic,
    emit: EventHandler,
    ask_user_fn: AskUserFn | None,
    pause_on_ask: bool,
) -> RunResult:
    """Run steps until the run finishes or pauses for a question."""
    segment_start = time.perf_counter()
    status: RunStatus = "running"
    final_answer: str | None = None
    error_message: str | None = None
    ctx = ToolContext(
        sandbox=sandbox,
        output_dir=state.output_dir,
        ask_user_fn=ask_user_fn,
        questions_asked=state.questions_asked,
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
                response = stream_message(client, state.model, state.messages, step, emit)
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
                status = "completed"
                final_answer = extract_text(response.content)
                break

            if response.stop_reason != "tool_use":
                status = "stopped"
                error_message = f"Agent stopped unexpectedly (stop_reason={response.stop_reason})."
                final_answer = extract_text(response.content) or None
                break

            # Run every requested tool. A question for the human is held back
            # (its slot keeps the order) and the other tools still run first.
            tool_results: list[dict[str, Any]] = []
            pending: PendingQuestion | None = None
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if block.name == ASK_USER_TOOL["name"] and pause_on_ask and ask_user_fn is None:
                    rejected = check_ask_user(block.input, ctx)
                    if rejected is None and pending is not None:
                        rejected = ("Only one question at a time. Ask again after the answer.", True)
                    if rejected is None:
                        ctx.questions_asked += 1
                        log_entry = {
                            "name": block.name,
                            "input": block.input,
                            "output": None,
                            "is_error": False,
                            "duration_s": 0.0,
                        }
                        step_log["tool_calls"].append(log_entry)
                        tool_results.append(
                            {"type": "tool_result", "tool_use_id": block.id, "content": None, "is_error": False}
                        )
                        pending = PendingQuestion(
                            tool_use_id=block.id,
                            question=block.input["question"],
                            options=block.input.get("options"),
                            log_entry=log_entry,
                        )
                        continue
                    output, is_error = rejected
                    elapsed = 0.0
                else:
                    start = time.perf_counter()
                    output, is_error = execute_tool(block.name, block.input, ctx)
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
                step_log["tool_calls"].append(
                    {
                        "name": block.name,
                        "input": block.input,
                        "output": output,
                        "is_error": is_error,
                        "duration_s": round(elapsed, 2),
                    }
                )

            if pending is not None:
                state.pending_tool_results = tool_results
                state.pending_question = pending
                status = "waiting_for_user"
                emit(UserQuestion(step=step, question=pending.question, options=pending.options))
                break

            state.messages.append({"role": "user", "content": tool_results})
            # Save after every step so a crash still leaves a record.
            write_log(state.log_path, state.log)
    finally:
        state.questions_asked = ctx.questions_asked
        state.active_seconds += time.perf_counter() - segment_start
        state.charts = [chart.as_posix() for chart in sorted(state.output_dir.glob("*.png"))]
        state.log.update(
            finished_at=datetime.now().isoformat(timespec="seconds"),
            status=status,
            error_message=error_message,
            final_answer=final_answer,
            charts=state.charts,
            totals={
                "steps": len(step_logs),
                "input_tokens": state.input_tokens,
                "output_tokens": state.output_tokens,
                "duration_s": round(state.active_seconds, 2),
                "api_duration_s": round(state.api_seconds, 2),
                "code_duration_s": round(state.code_seconds, 2),
            },
        )
        write_log(state.log_path, state.log)
        waiting = status == "waiting_for_user" and state.pending_question is not None
        result = RunResult(
            status=status,
            final_answer=final_answer,
            error_message=error_message,
            steps=len(step_logs),
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            duration_s=round(state.active_seconds, 2),
            charts=state.charts,
            log_path=state.log_path.as_posix(),
            question=state.pending_question.question if waiting else None,
            options=state.pending_question.options if waiting else None,
            state=state if waiting else None,
        )
        emit(
            RunFinished(
                result=result,
                api_duration_s=state.api_seconds,
                code_duration_s=state.code_seconds,
            )
        )

    return result


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
                system=SYSTEM_PROMPT,
                tools=TOOLS,
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
    base = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = PROJECT_ROOT / OUTPUTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    counter = 1
    while True:
        run_dir = root / (base if counter == 1 else f"{base}_{counter}")
        try:
            run_dir.mkdir()
            return run_dir
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
