# data-agent

An AI agent for data analysis. The user uploads a CSV/Excel file and asks a question in natural language. The agent explores the data, makes an analysis plan, writes and runs pandas code in a sandbox, fixes its own errors, builds charts, and returns a conclusion backed by numbers.

## Stack
- Python 3.14, virtual environment in .venv
- Claude API (anthropic SDK), model claude-sonnet-5-5, tool use
- Custom agent loop, NO LangChain/LangGraph
- Sandbox: jupyter_client (local kernel), Docker later
- pandas, matplotlib/plotly, Streamlit, SQLite/JSON

## Environment
- OS: Windows, PowerShell
- Run Python only from the venv: .venv\Scripts\python.exe (or after activating .venv)
- Install packages only into .venv and pin them in requirements.txt

## Language
- Talk to me in Russian.
- Everything written to the repository must be in English: code, comments, docstrings, docs, file contents, suggested commit messages.

## Working rules
- I am learning. Do only what the current task asks. Do not add features "for later".
- After changes, briefly explain in simple terms what you did and why.
- If you see a better approach, suggest it with pros and cons, but do not implement it without my approval.

## Git
- NEVER run git commit or git push. I make all commits manually.
- When a logical piece of work is finished and verified, explicitly tell me it is time to commit: list the files to stage and suggest a commit message in English following Conventional Commits (feat:, fix:, chore:, docs:, refactor:, test:).
- Never add "Co-Authored-By" or any mention of Claude to suggested commit messages.

## Security
- NEVER read, print, or edit .env. The API key reaches the code only via the ANTHROPIC_API_KEY environment variable (loaded with python-dotenv).
- Never put keys or tokens into code, logs, or commit messages.
