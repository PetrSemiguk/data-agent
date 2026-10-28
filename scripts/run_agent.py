"""Run the data analysis agent on a dataset from the command line."""

import argparse
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Make the project root importable when running this file as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.loop import run_agent  # noqa: E402
from agent.sandbox import Sandbox  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the data agent a question about a dataset.")
    parser.add_argument("--file", required=True, help="Path to a CSV or XLSX file")
    parser.add_argument("--question", required=True, help="Question in natural language")
    args = parser.parse_args()

    load_dotenv()
    client = anthropic.Anthropic()

    with Sandbox() as sb:
        sb.load_dataframe(args.file)
        answer = run_agent(args.question, sb, client)

    print(f"\n{'=' * 30} Final answer {'=' * 30}\n{answer}")


if __name__ == "__main__":
    main()
