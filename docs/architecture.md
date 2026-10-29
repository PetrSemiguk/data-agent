# Architecture

## Modules

| Module | Role |
|---|---|
| `app.py` | Streamlit chat UI: datasets, live steps, answers, approvals, visitor keys |
| `agent/loop.py` | the agent loop: calls Claude, runs tools, pauses for the human, verifies, logs |
| `agent/tools.py` | tool definitions and handlers |
| `agent/sandbox.py` | Jupyter kernel that runs the model's code (encrypted channel, no secrets in its environment) |
| `agent/guard.py` | checks code with `ast` before it runs (writes, deletes, shell, network, reads outside allowed folders) |
| `agent/verifier.py` | a second model run that recomputes the claims of a final answer |
| `agent/memory.py` | per-dataset notes in SQLite (local mode) |
| `agent/skills_loader.py`, `skills/` | methodology guides the agent can load (off by default) |
| `agent/mode.py` | everything that differs between local and public mode |
| `agent/sessions.py` | public mode: per-session folders, kernel limit, idle kernels, cleanup, key check |
| `agent/redact.py` | masks API keys in logs, events and outputs |
| `agent/events.py` | events the loop emits; the UI and the console draw them |

## Tools

| Tool | What it does |
|---|---|
| `get_schema` | compact overview of `df`: columns, types, missing values, examples |
| `run_code` | runs pandas code in the kernel; output is truncated to fit the context |
| `update_plan` | the agent's plan for multi-step questions, shown live above the steps |
| `save_chart` | saves the current matplotlib figure into the run's folder |
| `export_data` | writes a DataFrame to CSV/XLSX, only after the human approves |
| `ask_user` | asks the human a clarifying question; the run pauses until the answer |
| `remember` | proposes a note for the dataset's memory (saved only after a verified answer) |
| `load_skill` | loads a methodology guide (when skills are on) |
| `report_verification` | the verifier's final report: one verdict and evidence per claim |

## One question, end to end

1. The question, recent chat turns (with the app's own record of earlier actions) and
   memory notes become the first messages. System prompt and tools are cached.
2. The loop streams Claude's reply. Text goes to the UI as it arrives; tool calls run
   one by one, and their results go back to the model.
3. A pause (`ask_user`, an export approval) ends the run with "waiting for user"; the
   saved state continues it after the human's reply.
4. The run ends with a final answer, or at a limit: 15 steps, 200,000 tokens, two
   questions to the human, three approval requests.
5. A final answer with numbers goes to the verifier: a fresh context with only the
   question and the answer, in its own kernel namespace with its own copy of `df`.
   It recomputes each claim. If it contradicts one, the agent gets the remarks once
   and revises; if the revision fails, the draft is shown with a warning.
6. The run's JSON log is written under `logs/runs/`, charts under `outputs/`. Memory
   notes are saved only if the run succeeded, and each fact about the data only if
   the verifier supported it (its evidence is stored with the note).

## Local and public mode

`DATA_AGENT_MODE` selects the mode; `agent/mode.py` holds every difference.

| | local | public |
|---|---|---|
| API key | server's, from `.env` | each visitor's own, kept in their browser tab only |
| Memory between sessions | on | off |
| Files | shared `uploads/`, `outputs/`, `logs/` | a folder per session, deleted after 2 hours of inactivity |
| Code may read | `data/`, `uploads/`, `demo/` | own session's uploads and `demo/` |
| Kernels | one per browser session | at most 3 at once; closed after 15 idle minutes |
| Uploads | Streamlit's limit | 50 MB |
| Log paths, skills toggle | shown | hidden |
