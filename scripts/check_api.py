"""Check that the Claude API key works by sending one short request."""

import os
import sys

import anthropic
from dotenv import load_dotenv

MODEL = "claude-sonnet-5-5"
PLACEHOLDER_KEY = "your-api-key-here"
PROMPT = "Reply with one short sentence: are you ready to analyze data?"


def main() -> int:
    # Load variables from .env into the process environment.
    load_dotenv()

    # Validate the key without ever printing it.
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == PLACEHOLDER_KEY:
        print(
            "Error: ANTHROPIC_API_KEY is not set. "
            "Put your real API key into the .env file.",
            file=sys.stderr,
        )
        return 1

    # The client reads ANTHROPIC_API_KEY from the environment.
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": PROMPT}],
    )

    # The response may contain several blocks; keep only the text ones.
    text = "".join(block.text for block in response.content if block.type == "text")

    print(f"Model: {response.model}")
    print(f"Stop reason: {response.stop_reason}")
    print(f"Answer: {text or '(no text in response)'}")
    print(
        f"Usage: input_tokens={response.usage.input_tokens}, "
        f"output_tokens={response.usage.output_tokens}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
