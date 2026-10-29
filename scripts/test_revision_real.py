"""The revision path on the REAL Claude API (costs money).

The agent's first reply is replaced by a prepared draft with a planted error;
everything after it is the production path with real model calls: the
verifier checks the draft, the agent gets the remarks and revises its answer.
The full output is also saved to logs/revision_real.txt.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO

import pandas as pd
from anthropic.types import TextBlock
from dotenv import load_dotenv

# Make the project root importable when running this file as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.console import ConsolePrinter  # noqa: E402
from agent.events import Event, VerificationFinished  # noqa: E402
from agent.loop import create_client, run_agent  # noqa: E402
from agent.prompts import SYSTEM_PROMPT  # noqa: E402
from agent.sandbox import Sandbox  # noqa: E402

DATASET = PROJECT_ROOT / "data" / "Sample - Superstore.csv"
OUTPUT_FILE = PROJECT_ROOT / "logs" / "revision_real.txt"
QUESTION = "Which months are the strongest for sales in each year? Rank the top 3 months per year."

# claude-sonnet-5-5 prices, USD per 1M tokens
INPUT_PRICE = 2.00
OUTPUT_PRICE = 10.00

# Correct except 2017: October ($77.8K, really 4th) replaces September ($87.9K)
# and is placed above December.
DRAFT = """The top 3 sales months in each year, by order date (sales = sum of `Sales`, no filters, \
data from 2014-01-03 to 2017-12-30):

| Year | #1 | #2 | #3 |
|---|---|---|---|
| 2014 | September ($81.8K) | November ($78.6K) | December ($69.5K) |
| 2015 | November ($76.0K) | December ($74.9K) | September ($64.6K) |
| 2016 | December ($97.0K) | November ($79.4K) | September ($73.4K) |
| 2017 | 1. November ($118.4K) | 2. October ($77.8K) | 3. December ($83.8K) |

November and December are in the top 3 every year; the strongest single month is \
November 2017 ($118.4K)."""


class DraftStream:
    """Replays the prepared draft like the SDK's message stream."""

    def __init__(self, draft: str) -> None:
        self.draft = draft

    def __enter__(self) -> "DraftStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def __iter__(self) -> Any:
        yield SimpleNamespace(type="text", text=self.draft)

    def get_final_message(self) -> Any:
        # A real SDK block: it is sent back to the API in the following requests.
        return SimpleNamespace(
            content=[TextBlock(type="text", text=self.draft)],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
        )


class DraftFirstClient:
    """The first model call returns the draft; every other call goes to the real API."""

    def __init__(self, client: Any, draft: str) -> None:
        self.client = client
        self.draft: str | None = draft
        self.requests: list[tuple[str, list[Any]]] = []
        self.messages = self

    def stream(self, **kwargs: Any) -> Any:
        self.requests.append((kwargs["system"], list(kwargs["messages"])))
        if self.draft is not None:
            draft, self.draft = self.draft, None
            return DraftStream(draft)
        return self.client.messages.stream(**kwargs)


class Tee:
    """Write everything printed to the console and to a file."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * INPUT_PRICE + output_tokens * OUTPUT_PRICE) / 1_000_000


def top3_2017() -> list[tuple[str, int]]:
    """The true top 3 months of 2017, computed here independently of the agent."""
    data = pd.read_csv(DATASET, encoding="latin-1")
    dates = pd.to_datetime(data["Order Date"], format="%m/%d/%Y")
    months = data[dates.dt.year == 2017].groupby(dates.dt.month)["Sales"].sum().nlargest(3)
    return [(pd.Timestamp(2017, month, 1).strftime("%b"), round(value)) for month, value in months.items()]


def line_2017_is_correct(answer: str, truth: list[tuple[str, int]]) -> tuple[bool, str]:
    """Some line about 2017 names the true months in the true order, with their values."""
    for line in answer.splitlines():
        if "2017" not in line or "Oct" in line:
            continue
        positions = [line.find(month) for month, _ in truth]
        values_ok = all(f"{value:,}" in line or f"{value / 1000:.1f}K" in line for _, value in truth)
        if all(p >= 0 for p in positions) and positions == sorted(positions) and values_ok:
            return True, line.strip()
    return False, "no line about 2017 with the true months, order and values"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(sys.__stdout__, log_file)  # type: ignore[assignment]
        try:
            return run(log_file)
        finally:
            sys.stdout = sys.__stdout__


def run(log_file: TextIO) -> int:
    load_dotenv()
    client = DraftFirstClient(create_client(), DRAFT)
    events: list[Event] = []
    printer = ConsolePrinter()

    def on_event(event: Event) -> None:
        events.append(event)
        printer(event)

    print(f"Question: {QUESTION}\n\nPrepared draft (the agent's first reply):\n{DRAFT}\n")
    with Sandbox() as sandbox:
        sandbox.load_dataframe(DATASET)
        result = run_agent(QUESTION, sandbox, client, dataset_file=str(DATASET), on_event=on_event, verify=True)

    # What the agent received: the last message of its first request after the draft.
    agent_requests = [messages for system, messages in client.requests if system == SYSTEM_PROMPT]
    remarks = agent_requests[1][-1]["content"] if len(agent_requests) > 1 else "(no remarks were sent)"
    finished = next((e for e in events if isinstance(e, VerificationFinished)), None)
    report = result.verification
    truth = top3_2017()
    correct, line = line_2017_is_correct(result.final_answer or "", truth)

    print(f"\n{'=' * 30} Summary {'=' * 30}")
    print(f"Verifier status: {report.status if report else None}, counts: {report.counts if report else None}")
    for claim in report.claims if report else []:
        if claim.status != "supported":
            print(f"  [{claim.status}] {claim.claim}\n      evidence: {claim.evidence}\n      correction: {claim.correction}")
    print(f"\nRemarks the agent received:\n{remarks}\n")
    print(f"Revision requested: {finished.revision_requested if finished else None}")
    print(f"status={result.status}, revised={result.revised}, revision_failed={result.revision_failed}")
    print(f"\nFinal (revised) answer:\n{result.final_answer}\n")
    print(f"Independent check of 2017: true top 3 = {truth}")
    print(f"  {'PASS' if correct else 'FAIL'}: {line}")

    verify_in = report.input_tokens if report else 0
    verify_out = report.output_tokens if report else 0
    agent_in, agent_out = result.input_tokens - verify_in, result.output_tokens - verify_out
    total = cost(result.input_tokens, result.output_tokens)
    print(
        f"\nCost: verification {verify_in} in / {verify_out} out = ${cost(verify_in, verify_out):.3f}; "
        f"revision by the agent {agent_in} in / {agent_out} out = ${cost(agent_in, agent_out):.3f}; "
        f"total ${total:.3f}, {result.duration_s:.1f}s (the draft itself cost nothing)"
    )
    print(f"Log: {result.log_path}")
    ok = result.revised and correct and report is not None and report.status == "issues_found"
    print(f"\n{'PASS' if ok else 'FAIL'}: revision path")
    log_file.flush()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
