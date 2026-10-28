"""Tool definitions for Claude and the code that executes them."""

from typing import Any

from agent.sandbox import Sandbox

RUN_CODE_TOOL: dict[str, Any] = {
    "name": "run_code",
    "description": (
        "Execute Python code in a persistent Jupyter kernel and return its output. "
        "The dataset is already loaded into a pandas DataFrame named `df`. "
        "Variables, imports and functions persist between calls, so build the "
        "analysis step by step instead of repeating earlier work. "
        "To see a value, print() it or leave the expression as the last line of "
        "the code. Matplotlib charts are captured automatically; you will be told "
        "how many charts were generated, but you cannot see them. "
        "Errors are returned with the exception type and traceback."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute.",
            },
        },
        "required": ["code"],
    },
}


def execute_tool(name: str, tool_input: dict[str, Any], sandbox: Sandbox) -> tuple[str, bool]:
    """Run a tool requested by the model.

    Returns the text to send back to the model and whether it is an error.
    """
    if name != RUN_CODE_TOOL["name"]:
        return f"Unknown tool: {name}. Available tools: run_code.", True

    code = tool_input.get("code")
    if not isinstance(code, str) or not code.strip():
        return "Missing required input: code (a non-empty string).", True

    result = sandbox.execute(code)

    parts: list[str] = []
    if result.stdout:
        parts.append(result.stdout.rstrip())
    if result.result is not None:
        parts.append(result.result)
    if result.images:
        parts.append(f"[{len(result.images)} chart(s) generated]")
    if result.error:
        parts.append(result.error)

    text = "\n".join(parts) if parts else "[code executed, no output]"
    return text, not result.success
