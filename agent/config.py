"""Central settings for the agent."""

# Tool output sent to the model
MAX_OUTPUT_CHARS = 4000
OUTPUT_HEAD_CHARS = 2500
OUTPUT_TAIL_CHARS = 1000

# Sandbox
CODE_TIMEOUT_SECONDS = 30
PANDAS_DISPLAY_OPTIONS: dict[str, int] = {
    "display.max_rows": 50,
    "display.max_columns": 30,
    "display.width": 200,
    "display.max_colwidth": 50,
}
# Readable default charts; autolayout applies tight layout to every figure.
MATPLOTLIB_RC_PARAMS: dict[str, object] = {
    "figure.figsize": (10, 5),
    "figure.dpi": 120,
    "figure.autolayout": True,
}

# Saved artifacts: each run gets its own subfolder
OUTPUTS_DIR = "outputs"

# Agent loop
MAX_STEPS = 15
MAX_USER_QUESTIONS = 2
MAX_HISTORY_TURNS = 5  # earlier (question, answer) pairs of a chat sent to the model
MAX_TOTAL_TOKENS = 200_000  # input + output for the whole run

# Verifier: a separate run that fact-checks a final answer
VERIFY_MAX_STEPS = 6
VERIFY_ANSWERS = True  # check final answers that contain numbers
MAX_REVISIONS = 1  # times the agent may fix an answer the verifier contradicted

# Claude API client; the SDK itself retries 408/409/429/5xx and connection errors
API_MAX_RETRIES = 4
API_TIMEOUT_SECONDS = 120

# JSON run logs, one file per run
LOGS_DIR = "logs/runs"
