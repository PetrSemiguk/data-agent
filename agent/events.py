"""Events the agent loop reports while it runs (for the console, the web UI, tests)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.loop import RunResult


@dataclass(frozen=True)
class RunStarted:
    run_id: str
    output_dir: str
    log_path: str


@dataclass(frozen=True)
class StepStarted:
    step: int


@dataclass(frozen=True)
class ApiCallFinished:
    step: int
    duration_s: float
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class TextDelta:
    """A piece of model text as it is generated; the full text follows as AssistantText."""

    step: int
    text: str


@dataclass(frozen=True)
class StreamReset:
    """The stream broke mid-way and the step is retried: discard its partial text."""

    step: int


@dataclass(frozen=True)
class AssistantText:
    step: int
    text: str


@dataclass(frozen=True)
class ToolCall:
    step: int
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    step: int
    name: str
    output: str
    is_error: bool
    duration_s: float
    images: list[str] = field(default_factory=list)  # base64 PNGs from run_code
    chart_paths: list[str] = field(default_factory=list)  # files saved by save_chart


@dataclass(frozen=True)
class UserQuestion:
    """The run pauses: the agent waits for the human to answer this question."""

    step: int
    question: str
    options: list[str] | None


@dataclass(frozen=True)
class UserAnswered:
    """The run resumes with the human's answer; empty means "let the agent decide"."""

    step: int
    question: str
    answer: str


@dataclass(frozen=True)
class RunFinished:
    """Sent when a run ends and also when it pauses for a question."""

    result: RunResult
    api_duration_s: float
    code_duration_s: float


Event = (
    RunStarted
    | StepStarted
    | TextDelta
    | StreamReset
    | ApiCallFinished
    | AssistantText
    | ToolCall
    | ToolResult
    | UserQuestion
    | UserAnswered
    | RunFinished
)
EventHandler = Callable[[Event], None]
