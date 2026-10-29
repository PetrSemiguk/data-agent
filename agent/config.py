"""Central settings for the agent."""

# Tool output sent to the model
MAX_OUTPUT_CHARS = 4000
OUTPUT_HEAD_CHARS = 2500
OUTPUT_TAIL_CHARS = 1000

# Sandbox
CODE_TIMEOUT_SECONDS = 30
# The only project folders model code may read files from: the datasets
# (data/) and files uploaded in the web app (uploads/).
READABLE_DIRS = ("data", "uploads")
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
MAX_APPROVALS = 3  # export requests per run that are shown to the human
MAX_HISTORY_TURNS = 5  # earlier (question, answer) pairs of a chat sent to the model
MAX_TOTAL_TOKENS = 200_000  # input + output for the whole run

# Verifier: a separate run that fact-checks a final answer
VERIFY_MAX_STEPS = 6
VERIFY_ANSWERS = True  # check final answers that contain numbers
MAX_REVISIONS = 1  # times the agent may fix an answer the verifier contradicted

# Memory between sessions, per dataset (SQLite)
MEMORY_DB = "memory/agent_memory.db"
MEMORY_MAX_NOTES = 20  # notes kept per dataset
MEMORY_MAX_CHARS = 2000  # memory text added to the model's context
MEMORY_NOTE_MAX_CHARS = 160  # one note: one fact in a short sentence
FINGERPRINT_SAMPLE_ROWS = 100  # first and last rows hashed into the dataset fingerprint

# Skills: methodology guides the agent can load
SKILLS_DIR = "skills"

# Prompt caching: tools, system prompt and the conversation so far are read
# from the cache on later steps instead of being processed again.
PROMPT_CACHE = True

# Prices in USD per 1M tokens, per model, as (max prompt tokens, prices) tiers:
# a request uses the first tier its whole prompt (input + cache write + cache read)
# fits in; None means no limit. cache_write is the 5-minute cache write price.
# Source: https://platform.claude.com/docs/en/about-claude/pricing (checked 2026-10-29)
# A model missing here gets an unknown cost, never another model's prices.
MODEL_PRICES: dict[str, list[tuple[int | None, dict[str, float]]]] = {
    # Cache hits on Sonnet 5.5 cost 0.05x input.
    "claude-sonnet-5-5": [
        (None, {"input": 2.00, "cache_write": 2.50, "cache_read": 0.10, "output": 10.00}),
    ],
    # Haiku 5.5 charges more for requests whose prompt is over 100,000 tokens.
    "claude-haiku-5-5": [
        (100_000, {"input": 0.10, "cache_write": 0.125, "cache_read": 0.01, "output": 0.50}),
        (None, {"input": 0.50, "cache_write": 0.625, "cache_read": 0.05, "output": 2.50}),
    ],
}

# Claude API client; the SDK itself retries 408/409/429/5xx and connection errors
API_MAX_RETRIES = 4
API_TIMEOUT_SECONDS = 120

# JSON run logs, one file per run
LOGS_DIR = "logs/runs"
