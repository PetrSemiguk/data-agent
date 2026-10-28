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
