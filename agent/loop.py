"""The agent loop: ask Claude, run the tools it requests, repeat until done."""

import json
import time
from typing import Any

import anthropic

from agent.config import MAX_STEPS
from agent.prompts import SYSTEM_PROMPT
from agent.sandbox import Sandbox
from agent.tools import TOOLS, execute_tool

SEPARATOR = "-" * 80


def run_agent(
    question: str,
    sandbox: Sandbox,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-5-5",
    max_steps: int = MAX_STEPS,
) -> str:
    """Answer a question about the data in the sandbox and return the final text."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    input_tokens = 0
    output_tokens = 0
    api_seconds = 0.0
    code_seconds = 0.0
    run_start = time.perf_counter()

    try:
        for step in range(1, max_steps + 1):
            print(f"\n{'=' * 30} Step {step} {'=' * 30}")

            start = time.perf_counter()
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
            elapsed = time.perf_counter() - start
            api_seconds += elapsed
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens
            print(
                f"[API call: {elapsed:.1f}s, tokens in={response.usage.input_tokens} "
                f"out={response.usage.output_tokens}]"
            )

            # Keep the whole assistant turn, including tool_use blocks.
            messages.append({"role": "assistant", "content": response.content})
            print_assistant_blocks(response.content)

            if response.stop_reason == "end_turn":
                return extract_text(response.content)

            if response.stop_reason != "tool_use":
                return (
                    f"Agent stopped unexpectedly (stop_reason={response.stop_reason}).\n"
                    f"{extract_text(response.content)}"
                )

            # Run every requested tool and send all results back in one message.
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                start = time.perf_counter()
                output, is_error = execute_tool(block.name, block.input, sandbox)
                elapsed = time.perf_counter() - start
                code_seconds += elapsed
                # Let the model see slow operations.
                output = f"{output}\n[executed in {elapsed:.1f}s]"
                print_tool_result(output, is_error)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        return f"Agent stopped: reached the limit of {max_steps} steps without a final answer."
    finally:
        total_seconds = time.perf_counter() - run_start
        print(f"\n{SEPARATOR}\nTokens used: input={input_tokens}, output={output_tokens}")
        print(
            f"Time: total={total_seconds:.1f}s, API={api_seconds:.1f}s, "
            f"code={code_seconds:.1f}s"
        )


def extract_text(content: list[Any]) -> str:
    """Join all text blocks of an assistant message."""
    return "\n".join(block.text for block in content if block.type == "text").strip()


def print_assistant_blocks(content: list[Any]) -> None:
    """Show the agent's thoughts and the code it wants to run."""
    for block in content:
        if block.type == "text" and block.text.strip():
            print(f"[Agent]\n{block.text.strip()}")
        elif block.type == "tool_use":
            code = block.input.get("code")
            if isinstance(code, str):
                print(f"[Tool: {block.name}]\n{SEPARATOR}\n{code}\n{SEPARATOR}")
            else:
                print(f"[Tool: {block.name}] args: {json.dumps(block.input)}")


def print_tool_result(output: str, is_error: bool) -> None:
    """Show what the tool returned."""
    label = "Error" if is_error else "Result"
    print(f"[{label}]\n{output}")
