"""Streamlit web interface for the data analysis agent."""

import ast
import atexit
import os
import re
from pathlib import Path

import anthropic
import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.uploaded_file_manager import UploadedFile

from agent.loop import RunResult, create_client, run_agent
from agent.sandbox import Sandbox

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOADS_DIR = PROJECT_ROOT / "uploads"
PLACEHOLDER_KEY = "your-api-key-here"

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

if analyze:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("The agent is analyzing the data..."):
            st.session_state.result = run_agent(
                question.strip(),
                get_sandbox(),
                get_client(),
                ask_user_fn=None,
                dataset_file=str(st.session_state.dataset_path),
            )

# Kept in session_state so the answer survives later reruns.
if "result" in st.session_state:
    show_result(st.session_state.result)
