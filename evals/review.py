"""Manual check of the grader: review graded answers, then score the human's verdicts.

    .venv\\Scripts\\python.exe evals\\review.py --name dev_baseline --sample 10
        writes evals/results/<name>/review.md with EVERY answer that has a failed
        check plus 10 random fully passed answers; after every check write
        "Human: agree" or "Human: disagree <comment>"
    .venv\\Scripts\\python.exe evals\\review.py --name dev_baseline --score
        reads the filled file and reports how often the human agrees with the grader

Only the tuning splits (dev, dev_hard, dev_open) are reviewed: holdout and final are not
looked at in detail.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS_DIR))

from report import QUESTIONS_FILE, TUNING_SPLITS, load_run  # noqa: E402
from run_evals import RESULTS_DIR  # noqa: E402

CHECK_HEADER = re.compile(r"^### Check (\S+)")
GRADER_LINE = re.compile(r"^- Grader: (PASS|FAIL)")
HUMAN_LINE = re.compile(r"^Human:\s*(.*)$")


def quoted(text: str | None) -> str:
    return "\n".join(f"> {line}" for line in (text or "(no answer)").splitlines())


def render_review(name: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Grader review: {name}",
        "",
        "For every check, read the answer and write after `Human:` either `agree` or `disagree`, "
        "optionally followed by a comment. Then run `evals/review.py --name "
        f"{name} --score`.",
        "",
    ]
    for number, row in enumerate(rows, start=1):
        failed = not row["grade"]["strict_pass"]
        lines += [f"## {number}. {row['id']} r{row['repeat']} ({row['category']})"
                  + (" - HAS FAILED CHECKS" if failed else " - random passed answer"), "",
                  f"**Question:** {row['question']}", "", "**Agent's answer:**", "",
                  quoted(row["record"].get("final_answer")), ""]
        for check in row["grade"]["checks"]:
            lines += [
                f"### Check {row['id']}__r{row['repeat']}#{check['index']} ({check['type']})",
                "",
                f"- Check: {check['description']}",
                f"- Expected: {check['expected'] if check['expected'] is not None else '-'}",
                f"- Extracted by the judge: {check['extracted']}",
                *other_values_lines(check),
                f"- Judge's quote: {check['quote']!r}",
                f"- Grader: {'PASS' if check['passed'] else 'FAIL'}",
                "",
                "Human: ",
                "",
            ]
    return "\n".join(lines)


def other_values_lines(check: dict[str, Any]) -> list[str]:
    """Marks other values the answer gives for the same thing, and a flagged contradiction."""
    lines = []
    if check.get("other_values"):
        lines.append(f"- **Other values for the same thing in the answer:** {', '.join(map(str, check['other_values']))}")
    if check.get("conflicting"):
        lines.append("- **The judge found the answer contradicts itself on this value.**")
    return lines


def select_answers(rows: list[dict[str, Any]], sample: int, seed: int) -> list[dict[str, Any]]:
    """Every graded tuning-split answer with a failed check, plus `sample` random fully passed ones."""
    graded = [r for r in rows if r["grade"] and r["split"] in TUNING_SPLITS]
    failed = sorted((r for r in graded if not r["grade"]["strict_pass"]), key=lambda r: (r["id"], r["repeat"]))
    passed = [r for r in graded if r["grade"]["strict_pass"]]
    picked = random.Random(seed).sample(passed, min(sample, len(passed)))
    return failed + sorted(picked, key=lambda r: (r["id"], r["repeat"]))


def score_review(text: str) -> dict[str, Any]:
    """Agreement of the human with the grader over the filled checks."""
    result: dict[str, Any] = {"checks": 0, "filled": 0, "agree": 0, "disagree": [], "unclear": []}
    current, grader = None, None
    for line in text.splitlines():
        if match := CHECK_HEADER.match(line):
            current, grader = match.group(1), None
            result["checks"] += 1
        elif match := GRADER_LINE.match(line):
            grader = match.group(1)
        elif (match := HUMAN_LINE.match(line)) and current:
            verdict = match.group(1).strip()
            word = verdict.split()[0].lower().strip(".,:;") if verdict else ""
            if word == "agree":
                result["filled"] += 1
                result["agree"] += 1
            elif word == "disagree":
                result["filled"] += 1
                result["disagree"].append(f"{current} (grader {grader}): {verdict[len('disagree'):].strip(' :-')}")
            elif verdict:
                result["unclear"].append(f"{current}: {verdict}")
            current = None
    result["agreement"] = result["agree"] / result["filled"] if result["filled"] else None
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample", type=int,
                      help="Number of random fully passed answers to add to every answer with a failed check")
    mode.add_argument("--score", action="store_true", help="Score the filled review.md")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the sample")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # not when the output is captured
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    path = Path(args.results_dir) / args.name / "review.md"

    if args.score:
        if not path.exists():
            print(f"No {path}; create it with --sample first.")
            return 2
        result = score_review(path.read_text(encoding="utf-8"))
        if result["agreement"] is None:
            print(f"{result['checks']} checks, none filled in yet.")
            return 1
        print(f"Human agrees with the grader on {result['agree']}/{result['filled']} filled checks "
              f"({result['agreement']:.0%}); {result['checks'] - result['filled']} not filled.")
        for item in result["disagree"]:
            print(f"  disagree: {item}")
        for item in result["unclear"]:
            print(f"  unclear (write agree or disagree): {item}")
        return 0

    questions = {q["id"]: q for q in json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))}
    sample = select_answers(load_run(Path(args.results_dir) / args.name, questions), args.sample, args.seed)
    if not sample:
        print("No graded dev or dev_hard answers in this run.")
        return 2
    if path.exists():
        print(f"{path} exists; delete it to make a new sample (it may contain your verdicts).")
        return 2
    path.write_text(render_review(args.name, sample), encoding="utf-8")
    checks = sum(len(r["grade"]["checks"]) for r in sample)
    failed = sum(not r["grade"]["strict_pass"] for r in sample)
    print(f"Wrote {path}: {len(sample)} answers ({failed} with failed checks, {len(sample) - failed} random passed), "
          f"{checks} checks to review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
