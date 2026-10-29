"""Events the agent loop reports while it runs (for the console, the web UI, tests)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.loop import RunResult
    from agent.tools import ApprovalRequest
    from agent.verifier import VerificationReport


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
class PlanUpdated:
    """The agent replaced its plan; `items` is the whole new plan."""

    step: int
    items: list[dict[str, str]]


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
class ApprovalRequested:
    """The agent wants to do a risky action (e.g. export a file) and needs the human's approval."""

    step: int
    request: ApprovalRequest


@dataclass(frozen=True)
class ApprovalResolved:
    """The human (or, with no human, the loop automatically) decided on an approval."""

    step: int
    request: ApprovalRequest
    approved: bool
    # True if no human was available and the request was denied automatically.
    automatic: bool
    output: str
    # The written file, if the action was approved and succeeded.
    file_path: str | None


@dataclass(frozen=True)
class VerificationStarted:
    """The agent gave a final answer; the verifier starts checking it."""

    step: int


@dataclass(frozen=True)
class VerifierEvent:
    """An event of the verifier's own run (its steps, tool calls, text).

    Wrapping keeps the verifier's steps apart from the agent's: they have their
    own step numbers, and a display can draw them in a separate block.
    """

    event: Event


@dataclass(frozen=True)
class VerificationFinished:
    """The check ended. `report` is None if the verifier crashed; then `warning` says why."""

    step: int
    report: VerificationReport | None
    warning: str | None
    # True if the agent now gets the remarks and revises its answer.
    revision_requested: bool


@dataclass(frozen=True)
class AnswerRevised:
    """The agent replaced its answer after the verifier's remarks."""

    step: int
    draft_answer: str
    final_answer: str


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
    | PlanUpdated
    | UserQuestion
    | UserAnswered
    | ApprovalRequested
    | ApprovalResolved
    | VerificationStarted
    | VerifierEvent
    | VerificationFinished
    | AnswerRevised
    | RunFinished
)
EventHandler = Callable[[Event], None]
