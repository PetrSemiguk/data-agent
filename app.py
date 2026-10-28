"""Streamlit web interface for the data analysis agent."""

import ast
import atexit
import base64
import json
import os
import re
import time
from pathlib import Path

import anthropic
import streamlit as st
from dotenv import load_dotenv
from streamlit.delta_generator import DeltaGenerator
from streamlit.elements.lib.mutable_status_container import StatusContainer
from streamlit.runtime.uploaded_file_manager import UploadedFile

from agent.events import (
    ApiCallFinished,
    AssistantText,
    Event,
    RunFinished,
    StepStarted,
    StreamReset,
    TextDelta,
    ToolCall,
    ToolResult,
)
from agent.loop import RunResult, create_client, run_agent
from agent.sandbox import Sandbox

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOADS_DIR = PROJECT_ROOT / "uploads"
PLACEHOLDER_KEY = "your-api-key-here"

TOOL_ICONS = {
    "get_schema": ":material/table_view:",
    "run_code": ":material/code:",
    "save_chart": ":material/bar_chart:",
    "ask_user": ":material/help:",
}
ANSWER_ICON = ":material/check:"
STREAM_REFRESH_S = 0.1  # how often streamed text is redrawn

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
    st.session_state.file_id = uploaded.file_id
    st.session_state.dataset_path = path
    st.session_state.shape = ast.literal_eval(shape)
    st.session_state.pop("result", None)
    st.session_state.pop("events", None)


class StepRenderer:
    """Draws agent events as one status block per step.

    Live mode keeps the current step expanded; replay mode draws everything
    collapsed. Finished steps collapse and are marked complete or failed.
    """

    def __init__(self, parent: DeltaGenerator, live: bool) -> None:
        self.parent = parent
        self.live = live
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

    def __call__(self, event: Event) -> None:
        match event:
            case StepStarted(step=step):
                self._finish()
                self.step, self.tools, self.failed = step, [], False
                self.box = self.parent.status(
                    f"Step {step} · thinking...", expanded=self.live, state="running"
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
                self.box.update(label=f"Step {self.step} · {self._what()}")
                code = tool_input.get("code")
                if isinstance(code, str):
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
            label=f"Step {self.step} · {self._what()}",
            state="error" if self.failed else "complete",
            expanded=False,
        )
        self.box = None


def show_result(result: RunResult) -> None:
    """Render the outcome of one agent run."""
    # st.error and st.warning render markdown too, so their text is escaped as well.
    if result.status == "completed":
        st.success("Analysis completed")
    elif result.status == "api_error":
        st.error(escape_dollars(result.error_message or "API error"))
    else:
        st.warning(escape_dollars(f"Stopped ({result.status}): {result.error_message}"))

    if result.final_answer:
        st.markdown(escape_dollars(result.final_answer))

    for chart in result.charts:
        st.image(chart, caption=Path(chart).name)

    steps, tokens_in, tokens_out, duration = st.columns(4)
    steps.metric("Steps", result.steps)
    tokens_in.metric("Tokens in", f"{result.input_tokens:,}")
    tokens_out.metric("Tokens out", f"{result.output_tokens:,}")
    duration.metric("Time", f"{result.duration_s:.1f} s")
    st.caption(f"Run log: {result.log_path}")


st.set_page_config(page_title="Data Agent", page_icon="📊", layout="wide")
st.title("Data Agent")
st.write("Upload a CSV or Excel file, ask a question, and get an answer backed by numbers and charts.")

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key or api_key == PLACEHOLDER_KEY:
    st.error(
        "ANTHROPIC_API_KEY is not set. Put your API key into the .env file in the "
        "project folder and restart the app."
    )
    st.stop()

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("CSV or Excel file", type=["csv", "xlsx"])
    if uploaded is not None:
        load_dataset(uploaded)
        rows, cols = st.session_state.shape
        st.write(f"**{uploaded.name}**")
        st.caption(f"{uploaded.size / 1024 / 1024:.2f} MB · {rows:,} rows · {cols} columns")

question = st.text_area("Question", placeholder="Which product categories bring the most revenue?")
analyze = st.button("Analyze", type="primary", disabled=uploaded is None)

# Steps are drawn here, above the final answer.
steps_area = st.container()

if analyze and not question.strip():
    st.warning("Please enter a question.")
elif analyze:
    st.session_state.events = []
    st.session_state.pop("result", None)
    renderer = StepRenderer(steps_area, live=True)

    def on_event(event: Event) -> None:
        # Keep events so the steps can be redrawn on later reruns; deltas are
        # not needed for that, because AssistantText carries the full text.
        if not isinstance(event, (TextDelta, StreamReset)):
            st.session_state.events.append(event)
        renderer(event)

    with st.spinner("The agent is analyzing the data..."):
        st.session_state.result = run_agent(
            question.strip(),
            get_sandbox(),
            get_client(),
            ask_user_fn=None,
            dataset_file=str(st.session_state.dataset_path),
            on_event=on_event,
        )
elif st.session_state.get("events"):
    replay = StepRenderer(steps_area, live=False)
    for saved_event in st.session_state.events:
        replay(saved_event)

# Kept in session_state so the answer survives later reruns.
if "result" in st.session_state:
    show_result(st.session_state.result)
