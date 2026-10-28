"""Checks for agent.tools.truncate_output (no API calls)."""

import sys
from pathlib import Path

# Make the project root importable when running this file as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.config import MAX_OUTPUT_CHARS, OUTPUT_HEAD_CHARS, OUTPUT_TAIL_CHARS  # noqa: E402
from agent.tools import truncate_output  # noqa: E402

passed = 0
failed = 0


def report(name: str, ok: bool, details: str) -> None:
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"       {details}")


def main() -> int:
    # a) Short text is returned unchanged
    short = "x" * MAX_OUTPUT_CHARS
    report("a) short text unchanged", truncate_output(short) == short,
           f"input {len(short)} chars")

    # Long text with recognizable start and end
    long_text = "START" + "a" * 50_000 + "b" * 50_000 + "END"
    out = truncate_output(long_text)

    # b) Long text is cut to about MAX_OUTPUT_CHARS
    report("b) long text truncated", len(out) <= MAX_OUTPUT_CHARS,
           f"input {len(long_text)} chars -> output {len(out)} chars (limit {MAX_OUTPUT_CHARS})")

    # c) Marker reports the exact number of removed chars
    omitted = len(long_text) - OUTPUT_HEAD_CHARS - OUTPUT_TAIL_CHARS
    marker = f"[truncated {omitted} chars]"
    report("c) marker present", marker in out and "Tip: print less" in out,
           f"looking for '{marker}'")

    # d) Head and tail are preserved
    ok = (out.startswith(long_text[:OUTPUT_HEAD_CHARS])
          and out.endswith(long_text[-OUTPUT_TAIL_CHARS:]))
    report("d) head and tail kept", ok,
           f"starts with {out[:5]!r}, ends with {out[-3:]!r}")

    print(f"\nTotal: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
