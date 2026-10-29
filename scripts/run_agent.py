"""Run the data analysis agent on a dataset from the command line."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Make the project root importable when running this file as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.console import ConsolePrinter  # noqa: E402
from agent.loop import create_client, run_agent  # noqa: E402
from agent.sandbox import Sandbox  # noqa: E402


def console_ask_user(question: str, options: list[str] | None) -> str:
    """Ask the human in the terminal; accepts an option number or free text."""
    print(f"\n[Agent asks] {question}")
    for number, option in enumerate(options or [], start=1):
        print(f"  {number}. {option}")
    hint = "Enter a number or your own answer" if options else "Your answer"
    return input(f"{hint} (empty = agent decides): ")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the data agent a question about a dataset.")
    parser.add_argument("--file", required=True, help="Path to a CSV or XLSX file")
    parser.add_argument("--question", required=True, help="Question in natural language")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never ask the user; the agent must make and state its own assumptions",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Do not fact-check the final answer with the verifier",
    )
    args = parser.parse_args()
    # Symbols such as ✔ may not exist in the console encoding; print '?' instead of failing.
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]

    # Does not override variables already set in the environment.
    load_dotenv()
    client = create_client()
    ask_user_fn = None if args.non_interactive else console_ask_user

    with Sandbox() as sb:
        sb.load_dataframe(args.file)
        result = run_agent(
            args.question,
            sb,
            client,
            ask_user_fn=ask_user_fn,
            dataset_file=args.file,
            on_event=ConsolePrinter(),
            verify=False if args.no_verify else None,
        )

    print(f"\n{'=' * 30} Result: {result.status} {'=' * 30}")
    if result.error_message:
        print(f"Error: {result.error_message}")
    if result.final_answer:
        if result.revised:
            print("(Revised after verification; the draft is in the log.)")
        print(f"\n{result.final_answer}")
    print(
        f"\nSteps: {result.steps}, tokens in={result.input_tokens} out={result.output_tokens}, "
        f"time {result.duration_s:.1f}s"
    )
    print(f"Log: {result.log_path}")
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
