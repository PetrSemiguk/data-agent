"""Checks for agent.skills_loader and the load_skill tool (no kernel, no API)."""

import shutil
import sys
import tempfile
from pathlib import Path

# Make the project root importable when running this file as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.config import MAX_OUTPUT_CHARS  # noqa: E402
from agent.skills_loader import load_skills  # noqa: E402
from agent.tools import ToolContext, execute_tool  # noqa: E402

VALID = "---\nname: {name}\ndescription: Use when testing the loader.\n---\n\n# Title\n\nStep one.\n"

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


def write(root: Path, folder: str, text: str) -> None:
    (root / folder).mkdir(parents=True)
    (root / folder / "SKILL.md").write_text(text, encoding="utf-8")


def main() -> int:
    # a) The project's skills load without warnings
    skills, warnings = load_skills()
    report("a) project skills load", len(skills) == 4 and not warnings
           and all(s.description and s.body for s in skills.values()),
           f"{list(skills)}, warnings={warnings}")

    root = Path(tempfile.mkdtemp())
    try:
        write(root, "good", VALID.format(name="good"))
        write(root, "no-frontmatter", "# Just a title\n\nNo YAML here.\n")
        write(root, "dup", VALID.format(name="good"))  # same name as 'good'
        write(root, "empty-body", "---\nname: empty\ndescription: Has no body.\n---\n\n   \n")
        write(root, "no-description", "---\nname: nodesc\n---\nBody.\n")
        write(root, "bad-yaml", "---\nname: [unclosed\ndescription: x\n---\nBody.\n")
        write(root, "bad-name", "---\nname: Bad Name!\ndescription: x\n---\nBody.\n")
        skills, warnings = load_skills(root)
        text = " | ".join(warnings)

        # b) A valid skill is loaded with its fields
        good = skills.get("good")
        report("b) valid skill", good is not None and good.description == "Use when testing the loader."
               and good.body.startswith("# Title"), f"loaded: {list(skills)}")

        # c)-g) Broken skills are skipped with a warning; the loader does not fail
        cases = {
            "c) no frontmatter": "'no-frontmatter': no YAML frontmatter",
            "d) duplicate name": "duplicate name 'good'",  # whichever folder comes second
            "e) empty body": "'empty-body': empty body",
            "f) missing description": "'no-description': missing 'description'",
            "g) invalid YAML / name": "'bad-yaml': invalid YAML",
        }
        for name, expected in cases.items():
            report(name, expected in text and len(skills) == 1, expected)
        report("h) invalid name skipped", "'bad-name': missing or invalid 'name'" in text, f"{len(warnings)} warnings")

        # i) No skills folder at all: no skills, no crash
        none, none_warnings = load_skills(root / "missing")
        report("i) missing folder", none == {} and none_warnings == [], "empty result")

        # load_skill: the body comes back whole, even past the tool output limit
        long_body = "Step.\n" * (MAX_OUTPUT_CHARS // 3)
        write(root, "long", f"---\nname: long\ndescription: A long guide.\n---\n{long_body}")
        skills, _ = load_skills(root)
        ctx = ToolContext(sandbox=None, skills=skills)  # type: ignore[arg-type]
        output, is_error = execute_tool("load_skill", {"name": "long"}, ctx)
        report("j) load_skill returns the whole body",
               not is_error and long_body.strip() in output and len(output) > MAX_OUTPUT_CHARS
               and ctx.loaded_skills == ["long"],
               f"{len(output)} chars returned (tool output limit {MAX_OUTPUT_CHARS}), loaded={ctx.loaded_skills}")

        output, is_error = execute_tool("load_skill", {"name": "long"}, ctx)
        report("k) loading twice is recorded once", not is_error and ctx.loaded_skills == ["long"],
               f"loaded={ctx.loaded_skills}")

        output, is_error = execute_tool("load_skill", {"name": "nope"}, ctx)
        report("l) unknown skill -> error with the list",
               is_error and "Unknown skill 'nope'" in output and "good" in output and "long" in output, output)

        output, is_error = execute_tool("load_skill", {"name": "good"}, ToolContext(sandbox=None))  # type: ignore[arg-type]
        report("m) skills off -> error", is_error and "off" in output, output)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"\nTotal: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
