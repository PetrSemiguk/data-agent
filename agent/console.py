"""Print agent loop events to the terminal."""

import json

from agent.events import (
    AnswerRevised,
    ApiCallFinished,
    ApprovalRequested,
    ApprovalResolved,
    AssistantText,
    Event,
    PlanUpdated,
    RunFinished,
    RunStarted,
    SkillLoaded,
    StepStarted,
    StreamReset,
    TextDelta,
    ToolCall,
    ToolResult,
    UserAnswered,
    UserQuestion,
    VerificationFinished,
    VerificationStarted,
    VerifierEvent,
)

SEPARATOR = "-" * 80
PLAN_MARKS = {"pending": "☐", "in_progress": "▶", "done": "✔", "skipped": "✖"}
CLAIM_MARKS = {"supported": "✔", "contradicted": "✖", "unverifiable": "?"}


class ConsolePrinter:
    """Event handler that prints the loop's progress, streaming text word by word."""

    def __init__(self) -> None:
        # Text of the current step already shown through deltas.
        self.streamed = False
        self.line_open = False
        # Who writes the streamed text: the agent or, during a check, the verifier.
        self.speaker = "Agent"

    def __call__(self, event: Event) -> None:
        match event:
            case RunStarted(output_dir=output_dir):
                print(f"Output folder: {output_dir}")
            case StepStarted(step=step):
                self.streamed = False
                print(f"\n{'=' * 30} Step {step} {'=' * 30}")
            case TextDelta(text=text):
                if not self.streamed:
                    print(f"[{self.speaker}]")
                    self.streamed = True
                print(text, end="", flush=True)
                self.line_open = not text.endswith("\n")
            case StreamReset():
                self._end_line()
                print("[Stream interrupted, retrying this step]")
                self.streamed = False
            case ApiCallFinished():
                self._end_line()
                print(
                    f"[API call: {event.duration_s:.1f}s, tokens in={event.input_tokens} "
                    f"out={event.output_tokens}]"
                )
            case AssistantText(text=text):
                # Already printed piece by piece while it was generated.
                if not self.streamed:
                    print(f"[{self.speaker}]\n{text}")
            case ToolCall(name=name, input=tool_input):
                code = tool_input.get("code")
                if isinstance(code, str):
                    print(f"[Tool: {name}]\n{SEPARATOR}\n{code}\n{SEPARATOR}")
                else:
                    print(f"[Tool: {name}] args: {json.dumps(tool_input)}")
            case ToolResult(output=output, is_error=is_error):
                print(f"[{'Error' if is_error else 'Result'}]\n{output}")
            case PlanUpdated(items=items):
                print("[Plan]")
                for item in items:
                    note = f" ({item['note']})" if item.get("note") else ""
                    print(f"  {PLAN_MARKS.get(item['status'], '?')} {item['id']}. {item['text']}{note}")
            case UserQuestion(question=question):
                print(f"[Waiting for the user] {question}")
            case UserAnswered(answer=answer):
                print(f"[User answered] {answer or '(left to the agent)'}")
            case SkillLoaded(name=name):
                print(f"[Skill loaded] {name}")
            case ApprovalRequested(request=request):
                self._end_line()
                print(f"[Approval needed] {request.action}")
                for name, value in request.details.items():
                    print(f"  {name}: {value}")
            case ApprovalResolved(approved=approved, automatic=automatic, output=output):
                decision = "Approved" if approved else ("Denied automatically (no user)" if automatic else "Denied")
                print(f"[Approval] {decision}: {output.splitlines()[0]}")
            case VerificationStarted():
                self._end_line()
                self.speaker = "Verifier"
                print(f"\n{'#' * 28} Verification {'#' * 28}")
            case VerifierEvent(event=inner):
                # The verifier's steps are printed like the agent's, with their own header.
                if isinstance(inner, StepStarted):
                    self.streamed = False
                    print(f"\n{'-' * 26} Check step {inner.step} {'-' * 26}")
                elif not isinstance(inner, (RunStarted, RunFinished)):
                    self(inner)
            case VerificationFinished(report=report, warning=warning):
                self._end_line()
                self.speaker = "Agent"
                print(f"\n{'#' * 22} Verification result {'#' * 22}")
                if report is not None:
                    for claim in report.claims:
                        print(f"  {CLAIM_MARKS.get(claim.status, '?')} {claim.claim}")
                        if claim.status != "supported":
                            print(f"      {claim.evidence}")
                        if claim.correction:
                            print(f"      -> {claim.correction}")
                    counts = ", ".join(f"{count} {name}" for name, count in report.counts.items())
                    print(
                        f"Status: {report.status} ({counts}); tokens in={report.input_tokens} "
                        f"out={report.output_tokens}, {report.duration_s:.1f}s"
                    )
                if warning:
                    print(f"[Warning] {warning}")
                if event.revision_requested:
                    print("[Contradicted claims were sent back to the agent: it revises the answer]")
            case AnswerRevised():
                self._end_line()
                print("[Answer revised after verification]")
            case RunFinished(result=result):
                self._end_line()
                if result.status == "api_error":
                    print(f"[API error] {result.error_message}")
                if result.revision_failed:
                    print(f"[Warning] {result.verification_warning} (revision stopped: {result.status})")
                print(
                    f"\n{SEPARATOR}\n"
                    f"Tokens used: input={result.input_tokens}, output={result.output_tokens}"
                )
                print(
                    f"Time: total={result.duration_s:.1f}s, API={event.api_duration_s:.1f}s, "
                    f"code={event.code_duration_s:.1f}s"
                )
                if result.verification is not None:
                    print(
                        f"  of them verification: input={result.verification.input_tokens}, "
                        f"output={result.verification.output_tokens}"
                    )
                print(f"Saved charts ({len(result.charts)}):")
                for chart in result.charts:
                    print(f"  {chart}")
                print(f"Run log: {result.log_path}")

    def _end_line(self) -> None:
        """Finish a line left open by streamed text."""
        if self.line_open:
            print()
            self.line_open = False
