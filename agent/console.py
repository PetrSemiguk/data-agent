"""Print agent loop events to the terminal."""

import json

from agent.events import (
    ApiCallFinished,
    AssistantText,
    Event,
    RunFinished,
    RunStarted,
    StepStarted,
    ToolCall,
    ToolResult,
)

SEPARATOR = "-" * 80


class ConsolePrinter:
    """Event handler that reproduces the classic console output of the loop."""

    def __call__(self, event: Event) -> None:
        match event:
            case RunStarted(output_dir=output_dir):
                print(f"Output folder: {output_dir}")
            case StepStarted(step=step):
                print(f"\n{'=' * 30} Step {step} {'=' * 30}")
            case ApiCallFinished():
                print(
                    f"[API call: {event.duration_s:.1f}s, tokens in={event.input_tokens} "
                    f"out={event.output_tokens}]"
                )
            case AssistantText(text=text):
                print(f"[Agent]\n{text}")
            case ToolCall(name=name, input=tool_input):
                code = tool_input.get("code")
                if isinstance(code, str):
                    print(f"[Tool: {name}]\n{SEPARATOR}\n{code}\n{SEPARATOR}")
                else:
                    print(f"[Tool: {name}] args: {json.dumps(tool_input)}")
            case ToolResult(output=output, is_error=is_error):
                print(f"[{'Error' if is_error else 'Result'}]\n{output}")
            case RunFinished(result=result):
                if result.status == "api_error":
                    print(f"[API error] {result.error_message}")
                print(
                    f"\n{SEPARATOR}\n"
                    f"Tokens used: input={result.input_tokens}, output={result.output_tokens}"
                )
                print(
                    f"Time: total={result.duration_s:.1f}s, API={event.api_duration_s:.1f}s, "
                    f"code={event.code_duration_s:.1f}s"
                )
                print(f"Saved charts ({len(result.charts)}):")
                for chart in result.charts:
                    print(f"  {chart}")
                print(f"Run log: {result.log_path}")
