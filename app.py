"""Streamlit web interface for the data analysis agent."""

import ast
import atexit
import base64
import json
import os
import re
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import streamlit as st
from dotenv import load_dotenv
from streamlit.delta_generator import DeltaGenerator
from streamlit.elements.lib.mutable_status_container import StatusContainer
from streamlit.runtime.uploaded_file_manager import UploadedFile

from agent.events import (
    AnswerRevised,
    ApiCallFinished,
    ApprovalResolved,
    AssistantText,
    Event,
    RunFinished,
    SkillLoaded,
    StepStarted,
    StreamReset,
    TextDelta,
    PlanUpdated,
    ToolCall,
    ToolResult,
    UserAnswered,
    VerificationFinished,
    VerificationStarted,
    VerifierEvent,
)
from agent.loop import HistoryTurn, RunResult, action_record, create_client, resume_agent, run_agent
from agent.memory import DatasetMemory, MemoryStore, dataset_fingerprint
from agent.sandbox import Sandbox
from agent.skills_loader import Skill, load_skills
from agent.tools import plan_summary
from agent.verifier import VerificationReport

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOADS_DIR = PROJECT_ROOT / "uploads"
PLACEHOLDER_KEY = "your-api-key-here"

TOOL_ICONS = {
    "get_schema": ":material/table_view:",
    "update_plan": ":material/checklist:",
    "run_code": ":material/code:",
    "save_chart": ":material/bar_chart:",
    "export_data": ":material/download:",
    "remember": ":material/bookmark_add:",
    "load_skill": ":material/school:",
    "ask_user": ":material/help:",
    "report_verification": ":material/fact_check:",
}
ANSWER_ICON = ":material/check:"
STREAM_REFRESH_S = 0.1  # how often streamed text is redrawn
PLAN_ICONS = {
    "pending": ":material/radio_button_unchecked:",
    "in_progress": ":material/play_circle:",
    "done": ":material/check_circle:",
    "skipped": ":material/cancel:",
}
CLAIM_ICONS = {"supported": "✔", "contradicted": "✖", "unverifiable": "?"}

# Code blocks and inline code are shown literally, so $ inside them is left alone.
CODE_SPAN = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
UNESCAPED_DOLLAR = re.compile(r"(?<!\\)\$")


def escape_dollars(text: str) -> str:
    """Escape $ so Streamlit markdown shows money amounts instead of LaTeX math."""
    parts = CODE_SPAN.split(text)
    # split() with one capture group puts code spans at odd indexes.
    return "".join(
        part if index % 2 else UNESCAPED_DOLLAR.sub(r"\\$", part)
        for index, part in enumerate(parts)
    )


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    """One API client for the whole server process."""
    return create_client()


@st.cache_resource
def get_memory_store() -> MemoryStore:
    """One memory database for the whole server process."""
    return MemoryStore()


@st.cache_resource
def get_skills() -> tuple[dict[str, Skill], list[str]]:
    """Skills and loader warnings, read once per server process (restart to reload)."""
    return load_skills()


def get_sandbox() -> Sandbox:
    """One kernel per browser session, created on first use and reused on reruns."""
    if "sandbox" not in st.session_state:
        sandbox = Sandbox()
        sandbox.start()
        # Stop the kernel when the Streamlit process exits normally.
        atexit.register(sandbox.shutdown)
        st.session_state.sandbox = sandbox
    return st.session_state.sandbox


def save_upload(uploaded: UploadedFile) -> Path:
    """Save the uploaded file into uploads/ under a safe name."""
    UPLOADS_DIR.mkdir(exist_ok=True)
    name = Path(uploaded.name)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.stem).strip("_.") or "dataset"
    path = UPLOADS_DIR / f"{stem}{name.suffix.lower()}"
    path.write_bytes(uploaded.getvalue())
    return path


def load_dataset(uploaded: UploadedFile) -> None:
    """Load a newly uploaded file into the kernel; do nothing on plain reruns."""
    if st.session_state.get("file_id") == uploaded.file_id:
        return
    sandbox = get_sandbox()
    path = save_upload(uploaded)
    with st.spinner(f"Loading {uploaded.name}..."):
        if "file_id" in st.session_state:
            # A different file: start from a clean kernel so variables
            # from the previous dataset cannot leak into the new analysis.
            sandbox.restart()
        sandbox.load_dataframe(path)
        shape = sandbox.execute("df.shape").result or "(0, 0)"
        fingerprint = dataset_fingerprint(sandbox)
    st.session_state.file_id = uploaded.file_id
    st.session_state.fingerprint = fingerprint
    st.session_state.dataset_path = path
    st.session_state.shape = ast.literal_eval(shape)
    # A different file means a different conversation.
    new_conversation()


def plan_markdown(items: list[dict[str, str]]) -> str:
    """The plan as a checklist with progress in the header."""
    lines = [f"**Plan** · {plan_summary(items)}"]
    for item in items:
        note = f" _({item['note']})_" if item.get("note") else ""
        text = f"~~{item['text']}~~" if item["status"] == "skipped" else item["text"]
        lines.append(f"- {PLAN_ICONS.get(item['status'], '')} {text}{note}")
    return escape_dollars("\n".join(lines))


class StepRenderer:
    """Draws agent events as one status block per step.

    Live mode keeps the current step expanded; replay mode draws everything
    collapsed. Finished steps collapse and are marked complete or failed.
    """

    def __init__(
        self,
        parent: DeltaGenerator,
        live: bool,
        plan_slot: DeltaGenerator | None = None,
        label: str = "Step",
    ) -> None:
        self.parent = parent
        self.live = live
        self.label = label
        # The plan is drawn above the steps and redrawn whenever it changes.
        self.plan_slot = plan_slot
        self.box: StatusContainer | None = None
        self.step = 0
        self.tools: list[str] = []
        self.failed = False
        self.api_error = False
        # Placeholders created when a step starts, so the layout is the same
        # whether text arrives as live deltas or is replayed in full.
        self.caption_slot: DeltaGenerator | None = None
        self.text_slot: DeltaGenerator | None = None
        self.streamed_text = ""
        self.last_refresh = 0.0
        # The verifier's steps are drawn by a second renderer inside their own block.
        self.verifier: StepRenderer | None = None
        self.verify_header: DeltaGenerator | None = None

    def __call__(self, event: Event) -> None:
        match event:
            case StepStarted(step=step):
                self._finish()
                self.step, self.tools, self.failed = step, [], False
                self.box = self.parent.status(
                    f"{self.label} {step} · thinking...", expanded=self.live, state="running"
                )
                self.caption_slot = self.box.empty()
                self.text_slot = self.box.empty()
                self.streamed_text = ""
            case TextDelta(text=text) if self.text_slot:
                self.streamed_text += text
                # Redrawing markdown on every token is slow; refresh a few times a second.
                now = time.monotonic()
                if now - self.last_refresh >= STREAM_REFRESH_S:
                    self.text_slot.markdown(escape_dollars(self.streamed_text) + " ▌")
                    self.last_refresh = now
            case StreamReset() if self.text_slot:
                self.streamed_text = ""
                self.text_slot.empty()
            case ApiCallFinished() if self.caption_slot:
                self.caption_slot.caption(
                    f"API call {event.duration_s:.1f} s · tokens in {event.input_tokens:,} "
                    f"· out {event.output_tokens:,}"
                )
            case AssistantText(text=text) if self.box:
                # The first text block replaces the streamed preview with the full text.
                if self.text_slot is not None:
                    self.text_slot.markdown(escape_dollars(text))
                    self.text_slot = None
                else:
                    self.box.markdown(escape_dollars(text))
            case ToolCall(name=name, input=tool_input) if self.box:
                self._clear_preview()
                self.tools.append(name)
                self.box.update(label=f"{self.label} {self.step} · {self._what()}")
                code = tool_input.get("code")
                if name == "update_plan":
                    # The plan itself is shown above the steps.
                    self.box.caption(":material/checklist: Plan updated")
                elif name == "ask_user":
                    options = "".join(f"\n- {option}" for option in tool_input.get("options") or [])
                    self.box.markdown(escape_dollars(f"**Question:** {tool_input.get('question')}{options}"))
                elif isinstance(code, str):
                    self.box.code(code, language="python")
                else:
                    self.box.code(json.dumps(tool_input, ensure_ascii=False, indent=2), language="json")
            case ToolResult() if self.box:
                if event.is_error:
                    self.failed = True
                    self.box.error(f"{event.name} failed ({event.duration_s:.1f} s)")
                self.box.code(event.output, language=None)
                for image in event.images:
                    self.box.image(base64.b64decode(image))
                for path in event.chart_paths:
                    self.box.image(path, caption=Path(path).name)
            case PlanUpdated(items=items) if self.plan_slot:
                self.plan_slot.container(border=True).markdown(plan_markdown(items))
            case SkillLoaded(name=name) if self.box:
                self.box.caption(f":material/school: Skill loaded: {name}")
            case ApprovalResolved(request=request, approved=approved, automatic=automatic):
                # The request and the decision stay visible between the steps.
                decision = "Approved" if approved else ("Denied automatically (no user)" if automatic else "Denied")
                self.parent.container(border=True).markdown(
                    escape_dollars(f":material/verified_user: **Approval** · {request.action}\n\n**Your decision:** {decision}")
                )
            case UserAnswered(question=question, answer=answer):
                # The question and the human's answer stay visible between the steps.
                shown = answer or "_left to the agent_"
                self.parent.container(border=True).markdown(
                    escape_dollars(f":material/help: **Question** · {question}\n\n**Your answer:** {shown}")
                )
            case VerificationStarted():
                self._finish()
                area = self.parent.container(border=True)
                self.verify_header = area.empty()
                self.verify_header.markdown(":material/fact_check: **Verification** · checking the answer...")
                self.verifier = StepRenderer(area, self.live, label="Check")
            case VerifierEvent(event=inner) if self.verifier:
                self.verifier(inner)
            case VerificationFinished(report=report) if self.verify_header:
                summary = claim_counts(report) if report else "not completed"
                if event.revision_requested:
                    summary += " · the agent revises its answer"
                # A warning, if any, is shown with the answer.
                self.verify_header.markdown(f":material/fact_check: **Verification** · {summary}")
                self.verifier = None
            case AnswerRevised():
                self._finish()
                self.parent.caption(":material/edit_note: Answer revised after verification")
            case RunFinished(result=result):
                self.api_error = result.status == "api_error"
                self.failed = self.failed or self.api_error
                self._finish()

    def _what(self) -> str:
        """Short description of what the step did, for its label."""
        if self.tools:
            return ", ".join(f"{TOOL_ICONS.get(name, '')} {name}" for name in self.tools)
        if self.api_error:
            return ":material/error: API error"
        return f"{ANSWER_ICON} answer"

    def _clear_preview(self) -> None:
        """Release the text placeholder if no full text replaced it."""
        if self.text_slot is not None and self.streamed_text:
            # E.g. the API failed mid-stream: keep the partial text, drop the cursor.
            self.text_slot.markdown(escape_dollars(self.streamed_text))
        self.text_slot = None

    def _finish(self) -> None:
        if self.box is None:
            return
        self._clear_preview()
        self.box.update(
            label=f"{self.label} {self.step} · {self._what()}",
            state="error" if self.failed else "complete",
            expanded=False,
        )
        self.box = None


def ask_question(slot: DeltaGenerator, result: RunResult, key: str) -> str | None:
    """Show the agent's question; return the answer once given ("" = agent decides)."""
    answer: str | None = None
    with slot.container(border=True):
        st.info(escape_dollars(f"The agent asks: {result.question}"), icon=":material/help:")
        options = result.options or []
        for column, (index, option) in zip(st.columns(len(options) or 1), enumerate(options)):
            if column.button(escape_dollars(option), key=f"{key}_option_{index}", use_container_width=True):
                answer = option
        other = st.text_input("Other answer", key=f"{key}_other")
        send, decide = st.columns(2)
        if send.button("Send answer", key=f"{key}_send", disabled=not other.strip()):
            answer = other.strip()
        if decide.button("Let the agent decide", key=f"{key}_decide"):
            answer = ""
    return answer


def ask_approval(slot: DeltaGenerator, result: RunResult, key: str) -> bool | None:
    """Show what the agent wants to do; return True/False once the human decides."""
    request = result.approval
    if request is None:
        return None
    decision: bool | None = None
    with slot.container(border=True):
        st.warning(escape_dollars(f"The agent asks for approval: {request.action}"), icon=":material/verified_user:")
        st.markdown(escape_dollars("\n".join(f"- **{name}:** {value}" for name, value in request.details.items())))
        approve, deny = st.columns(2)
        if approve.button("Approve", key=f"{key}_approve", type="primary", use_container_width=True):
            decision = True
        if deny.button("Deny", key=f"{key}_deny", use_container_width=True):
            decision = False
    return decision


def show_result(result: RunResult) -> None:
    """Render the outcome of one run inside its chat message."""
    # st.error and st.warning render markdown too, so their text is escaped as well.
    if result.revision_failed:
        # The answer below is the draft with known errors: say so loudly.
        st.warning(escape_dollars(result.verification_warning or ""), icon=":material/warning:")
        st.caption(escape_dollars(f"The revision stopped ({result.status}): {result.error_message}"))
    elif result.status == "api_error":
        st.error(escape_dollars(result.error_message or "API error"))
    elif result.status != "completed":
        st.warning(escape_dollars(f"Stopped ({result.status}): {result.error_message}"))

    if result.plan_incomplete:
        st.caption(":material/info: The agent answered before finishing every item of its plan.")

    if result.verification_warning and not result.revision_failed:
        st.caption(escape_dollars(f":material/warning: {result.verification_warning}"))
    report = result.verification
    if report is not None and report.status != "incomplete":
        st.markdown(verification_badge(report, result.revised))
        with st.expander("Verification details", expanded=result.revision_failed):
            st.dataframe(
                [
                    {
                        "Status": f"{CLAIM_ICONS.get(claim.status, '')} {claim.status}",
                        "Claim": claim.claim,
                        "Evidence": claim.evidence,
                        "Correction": claim.correction or "",
                    }
                    for claim in report.claims
                ]
                + [
                    {
                        "Status": f"{CLAIM_ICONS.get(check.status, '')} {check.status}",
                        "Claim": f"[memory note] {check.claim}",
                        "Evidence": check.evidence,
                        "Correction": check.correction or "",
                    }
                    for check in report.note_checks
                ],
                hide_index=True,
                width="stretch",
            )
            st.caption(
                f"Check: {report.steps} steps · tokens {report.input_tokens:,} in / "
                f"{report.output_tokens:,} out · {report.duration_s:.1f} s · log: {report.log_path}"
            )

    if result.final_answer:
        st.markdown(escape_dollars(result.final_answer))

    if result.remembered:
        st.caption(escape_dollars(":material/bookmark_add: Remembered: " + " · ".join(result.remembered)))
    if result.not_remembered:
        st.caption(escape_dollars(":material/bookmark_remove: Not remembered: " + " · ".join(result.not_remembered)))
    if result.memory_not_saved:
        st.caption(escape_dollars(f":material/warning: {result.memory_not_saved}"))

    if result.revised and result.draft_answer is not None:
        with st.expander("Draft answer before verification"):
            st.markdown(escape_dollars(result.draft_answer))

    for chart in result.charts:
        st.image(chart, caption=Path(chart).name)

    for export in result.exports:
        path = Path(export)
        if path.exists():
            st.download_button(
                f"Download {path.name}",
                data=path.read_bytes(),
                file_name=path.name,
                key=f"download_{export}",
                icon=":material/download:",
            )

    st.caption(
        f"{result.steps} steps · tokens {result.input_tokens:,} in / {result.output_tokens:,} out "
        f"· {result.duration_s:.1f} s · log: {result.log_path}"
    )


def claim_counts(report: VerificationReport, revised: bool = False) -> str:
    """E.g. '7 supported · 1 corrected · 1 unverifiable'; zero counts are left out."""
    names = {"contradicted": "corrected"} if revised else {}
    parts = [f"{count} {names.get(status, status)}" for status, count in report.counts.items() if count]
    return " · ".join(parts) or "no claims"


def verification_badge(report: VerificationReport, revised: bool) -> str:
    """One line above the answer: how the check went."""
    if revised:
        badge = ":orange-badge[:material/edit_note: Revised after verification]"
    elif report.status == "ok":
        badge = ":green-badge[:material/verified: Verified]"
    else:
        badge = ":red-badge[:material/error: Issues found]"
    return f"{badge} · {claim_counts(report, revised)}"


def memory_sidebar(store: MemoryStore, fingerprint: str) -> None:
    """The dataset's notes with a delete button each, and a button to forget them all."""
    st.subheader("Memory for this dataset")
    notes = store.list_notes(fingerprint)
    if not notes:
        st.caption("No notes yet. The agent saves lasting facts and your definitions here.")
    for note in notes:
        with st.container(border=True):
            st.markdown(escape_dollars(f"**{note.kind}** · {note.text}"))
            info, delete = st.columns([3, 1])
            info.caption(f"#{note.id} · from {note.source} · {note.updated_at[:10]}")
            if delete.button("Delete", key=f"delete_note_{note.id}", icon=":material/delete:"):
                store.delete_note(note.id)
                st.rerun()
    if st.button("Forget everything about this dataset", disabled=not notes, use_container_width=True):
        store.forget_dataset(fingerprint)
        st.rerun()


def turn_tokens(turns: list[dict[str, Any]]) -> int:
    return sum(t["result"].input_tokens + t["result"].output_tokens for t in turns if t["result"])


def new_conversation() -> None:
    """Start an empty chat; the totals of the old one are kept for the session counter."""
    old_turns = st.session_state.get("turns", [])
    st.session_state.past_questions = st.session_state.get("past_questions", 0) + len(old_turns)
    st.session_state.past_tokens = st.session_state.get("past_tokens", 0) + turn_tokens(old_turns)
    st.session_state.turns = []
    st.session_state.conversation_id = uuid.uuid4().hex[:12]


def chat_history(turns: list[dict[str, Any]]) -> list[HistoryTurn]:
    """Question, final answer and the record of actions of the finished questions of this chat."""
    return [
        HistoryTurn(turn["question"], turn["result"].final_answer, action_record(turn["result"]))
        for turn in turns
        if turn["result"] and turn["result"].status == "completed" and turn["result"].final_answer
    ]


def live_handler(events: list[Event], renderer: StepRenderer) -> Callable[[Event], None]:
    """Draw events as they come and keep them for redrawing on later reruns."""

    def on_event(event: Event) -> None:
        # Deltas are not kept, also the verifier's: AssistantText carries the full text.
        inner = event.event if isinstance(event, VerifierEvent) else event
        if not isinstance(inner, (TextDelta, StreamReset)):
            events.append(event)
        renderer(event)

    return on_event


def replay(events: list[Event], parent: DeltaGenerator, plan_slot: DeltaGenerator) -> None:
    renderer = StepRenderer(parent, live=False, plan_slot=plan_slot)
    for event in events:
        renderer(event)


st.set_page_config(page_title="Data Agent", page_icon="📊", layout="wide")
st.title("Data Agent")
st.write("Upload a CSV or Excel file and ask questions about it in a chat.")

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key or api_key == PLACEHOLDER_KEY:
    st.error(
        "ANTHROPIC_API_KEY is not set. Put your API key into the .env file in the "
        "project folder and restart the app."
    )
    st.stop()

if "turns" not in st.session_state:
    new_conversation()

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("CSV or Excel file", type=["csv", "xlsx"])
    if uploaded is not None:
        load_dataset(uploaded)
        rows, cols = st.session_state.shape
        st.write(f"**{uploaded.name}**")
        st.caption(f"{uploaded.size / 1024 / 1024:.2f} MB · {rows:,} rows · {cols} columns")

    if st.button("New conversation", disabled=uploaded is None, use_container_width=True):
        # A clean kernel, so variables from the old chat cannot leak into the new one.
        with st.spinner("Restarting the kernel..."):
            sandbox = get_sandbox()
            sandbox.restart()
            sandbox.load_dataframe(st.session_state.dataset_path)
        new_conversation()

    verify = st.toggle(
        "Verify answers",
        value=True,
        help="A second model rechecks the numbers of each answer with code; "
        "contradicted claims are sent back to the agent to fix.",
    )

    use_memory = st.toggle(
        "Use memory",
        value=True,
        help="Load this dataset's notes from earlier sessions and let the agent save new ones.",
    )
    use_skills = st.toggle(
        "Use skills",
        value=True,
        help="Offer the agent methodology guides (skills) it can load for the question.",
    )
    skills, skill_warnings = get_skills()
    with st.expander(f"Skills ({len(skills)})"):
        for skill in skills.values():
            st.markdown(f"**{skill.name}** · {escape_dollars(skill.description)}")
        for warning in skill_warnings:
            st.caption(escape_dollars(f":material/warning: {warning}"))
    if uploaded is not None:
        memory_sidebar(get_memory_store(), st.session_state.fingerprint)

    turns: list[dict[str, Any]] = st.session_state.turns
    questions = st.session_state.past_questions + len(turns)
    tokens = st.session_state.past_tokens + turn_tokens(turns)
    st.caption(f"This session: {questions} questions · {tokens:,} tokens")

if not turns:
    st.info("Upload a dataset in the sidebar, then ask your first question below.")

# The whole chat is redrawn from session_state on every rerun.
for index, turn in enumerate(turns):
    with st.chat_message("user"):
        st.markdown(escape_dollars(turn["question"]))
    with st.chat_message("assistant"):
        plan_slot = st.empty()
        steps_area = st.container()
        replay(turn["events"], steps_area, plan_slot)
        result: RunResult | None = turn["result"]
        if result is None:
            st.warning("This question did not finish.")
        elif result.status != "waiting_for_user":
            show_result(result)
        elif index == len(turns) - 1 and result.state is not None:
            # The agent asked a question or an approval inside this answer: ask the human here.
            slot = st.empty()
            key = f"turn{index}_{result.state.questions_asked}_{result.state.approvals_requested}"
            reply: str | bool | None
            if result.waiting_for == "approval":
                reply = ask_approval(slot, result, key)
            else:
                reply = ask_question(slot, result, key)
            if reply is not None:
                slot.empty()
                with st.spinner("The agent continues..."):
                    turn["result"] = resume_agent(
                        result.state,
                        reply,
                        get_sandbox(),
                        get_client(),
                        on_event=live_handler(
                            turn["events"], StepRenderer(steps_area, live=True, plan_slot=plan_slot)
                        ),
                    )
                st.rerun()

waiting = bool(turns) and turns[-1]["result"] is not None and turns[-1]["result"].status == "waiting_for_user"
if uploaded is None:
    hint = "Upload a dataset to start"
elif waiting:
    hint = "Respond to the agent above first"
else:
    hint = "Ask a question about the data"
prompt = st.chat_input(hint, disabled=uploaded is None or waiting)

if prompt:
    history = chat_history(turns)
    turn = {"question": prompt, "events": [], "result": None}
    turns.append(turn)
    with st.chat_message("user"):
        st.markdown(escape_dollars(prompt))
    with st.chat_message("assistant"):
        plan_slot = st.empty()
        steps_area = st.container()
        with st.spinner("The agent is analyzing the data..."):
            turn["result"] = run_agent(
                prompt,
                get_sandbox(),
                get_client(),
                dataset_file=str(st.session_state.dataset_path),
                on_event=live_handler(
                    turn["events"], StepRenderer(steps_area, live=True, plan_slot=plan_slot)
                ),
                pause_on_ask=True,
                history=history,
                conversation_id=st.session_state.conversation_id,
                turn=len(turns),
                verify=verify,
                memory=DatasetMemory(get_memory_store(), st.session_state.fingerprint) if use_memory else None,
                skills=skills if use_skills else None,
            )
    # Rebuild the page from the new state (answer, question widget, input state).
    st.rerun()
