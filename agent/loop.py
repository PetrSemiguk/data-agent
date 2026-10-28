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
)
from agent.prompts import SYSTEM_PROMPT
from agent.sandbox import Sandbox
from agent.tools import TOOLS, AskUserFn, ToolContext, execute_tool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)

# Errors raised while reading an already open stream; the SDK does not retry these.
STREAM_BROKEN_ERRORS = (httpx2.TransportError, anthropic.APIConnectionError)
# Error types the API may send as an event inside an open stream that are worth retrying.
TRANSIENT_ERROR_TYPES = {"overloaded_error", "api_error", "rate_limit_error", "timeout_error"}
STREAM_RETRY_BASE_DELAY_S = 0.5

RunStatus = Literal["running", "completed", "max_steps", "budget_exceeded", "api_error", "stopped"]


@dataclass
class RunResult:
    """Outcome of one agent run; returned instead of raising on API problems."""

    status: RunStatus
    final_answer: str | None
    error_message: str | None
    steps: int
    input_tokens: int
    output_tokens: int
    duration_s: float
    charts: list[str] = field(default_factory=list)
    log_path: str | None = None


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
) -> RunResult:
    """Answer a question about the data in the sandbox.

    `ask_user_fn` lets the agent ask the human a clarifying question; with None
    the agent is told the user is not available and must make assumptions.
    `on_event` receives progress events; the loop itself prints nothing.
    API errors do not raise: they end the run with status "api_error".
    """

    def emit(event: Event) -> None:
        # A broken display must never stop the analysis.
        if on_event is None:
            return
        try:
            on_event(event)
        except Exception:
            logger.exception("Event handler failed on %s", type(event).__name__)

    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    input_tokens = 0
    output_tokens = 0
    api_seconds = 0.0
    code_seconds = 0.0
    run_start = time.perf_counter()

    status: RunStatus = "running"
    final_answer: str | None = None
    error_message: str | None = None

    # Every run saves its files (charts) and its log under the same run id.
    run_dir = create_run_dir()
    run_id = run_dir.name
    log_path = PROJECT_ROOT / LOGS_DIR / f"{run_id}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(sandbox=sandbox, output_dir=run_dir, ask_user_fn=ask_user_fn)
    emit(RunStarted(run_id=run_id, output_dir=run_dir.as_posix(), log_path=log_path.as_posix()))

    step_logs: list[dict[str, Any]] = []
    log: dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "model": model,
        "dataset_file": dataset_file,
        "question": question,
        "status": status,
        "error_message": None,
        "final_answer": None,
        "totals": {},
        "charts": [],
        "steps": step_logs,
    }
    write_log(log_path, log)

    try:
        for step in range(1, max_steps + 1):
            # Stop before spending more once the budget is used up.
            if input_tokens + output_tokens >= MAX_TOTAL_TOKENS:
                status = "budget_exceeded"
                error_message = (
                    f"Token budget exceeded: {input_tokens + output_tokens} tokens used, "
                    f"limit is {MAX_TOTAL_TOKENS}."
                )
                break

            emit(StepStarted(step=step))
            start = time.perf_counter()
            try:
                response = stream_message(client, model, messages, step, emit)
            except (anthropic.AnthropicError, *STREAM_BROKEN_ERRORS) as exc:
                api_seconds += time.perf_counter() - start
                status = "api_error"
                error_message = describe_api_error(exc, model)
                break
            elapsed = time.perf_counter() - start
            api_seconds += elapsed
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens
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
            messages.append({"role": "assistant", "content": response.content})
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

            # Run every requested tool and send all results back in one message.
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                start = time.perf_counter()
                output, is_error = execute_tool(block.name, block.input, ctx)
                elapsed = time.perf_counter() - start
                code_seconds += elapsed
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
            messages.append({"role": "user", "content": tool_results})
            # Save after every step so a crash still leaves a record.
            write_log(log_path, log)
        else:
            status = "max_steps"
            error_message = f"Reached the limit of {max_steps} steps without a final answer."
    finally:
        duration_s = time.perf_counter() - run_start
        charts = [chart.as_posix() for chart in sorted(run_dir.glob("*.png"))]
        log.update(
            finished_at=datetime.now().isoformat(timespec="seconds"),
            status=status,
            error_message=error_message,
            final_answer=final_answer,
            charts=charts,
            totals={
                "steps": len(step_logs),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "duration_s": round(duration_s, 2),
                "api_duration_s": round(api_seconds, 2),
                "code_duration_s": round(code_seconds, 2),
            },
        )
        write_log(log_path, log)
        result = RunResult(
            status=status,
            final_answer=final_answer,
            error_message=error_message,
            steps=len(step_logs),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_s=round(duration_s, 2),
            charts=charts,
            log_path=log_path.as_posix(),
        )
        emit(RunFinished(result=result, api_duration_s=api_seconds, code_duration_s=code_seconds))

    return result


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
