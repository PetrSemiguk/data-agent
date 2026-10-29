"""Compute the expected values of every eval question and write them into questions.json.

Run from the project root:  .venv\\Scripts\\python.exe evals\\build_expected.py
Values of the final split are not printed unless --show-final is given.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS_DIR))

from reference import DATA_DIR, REFERENCES  # noqa: E402

QUESTIONS_FILE = EVALS_DIR / "questions.json"
SPLITS = {"dev", "dev_hard", "holdout", "final"}
CATEGORIES = {"lookup", "aggregation", "time_series", "group_comparison", "data_quality_trap", "multi_step"}
CHECK_TYPES = {"number", "label", "ranking", "mention", "not_claim"}
ORDERS = {"descending", "ascending"}


def validate(question: dict[str, Any]) -> list[str]:
    """Problems with one question's structure; empty means it is valid."""
    qid = question.get("id", "?")
    problems = []
    for key in ("id", "dataset", "split", "question", "category", "traps", "checks", "reference"):
        if key not in question:
            problems.append(f"{qid}: missing field '{key}'")
    if question.get("split") not in SPLITS:
        problems.append(f"{qid}: invalid split {question.get('split')!r}")
    if question.get("category") not in CATEGORIES:
        problems.append(f"{qid}: invalid category {question.get('category')!r}")
    if not (DATA_DIR / str(question.get("dataset"))).exists():
        problems.append(f"{qid}: dataset {question.get('dataset')!r} is not in data/")
    if question.get("reference") not in REFERENCES:
        problems.append(f"{qid}: no reference function {question.get('reference')!r}")
    for check in question.get("checks", []):
        kind = check.get("type")
        if kind not in CHECK_TYPES or not check.get("description"):
            problems.append(f"{qid}: invalid check {check}")
        if kind in ("number", "label", "ranking") and not check.get("key"):
            problems.append(f"{qid}: {kind} check without 'key'")
        if kind == "number" and "abs_tol" not in check and "rel_tol" not in check:
            problems.append(f"{qid}: number check '{check.get('key')}' without abs_tol or rel_tol")
        if kind == "ranking" and not isinstance(check.get("top_k"), int):
            problems.append(f"{qid}: ranking check '{check.get('key')}' without top_k")
        if kind == "ranking" and check.get("order") not in ORDERS:
            problems.append(f"{qid}: ranking check '{check.get('key')}' without order ({' or '.join(sorted(ORDERS))})")
    return problems


def fill(question: dict[str, Any]) -> dict[str, Any]:
    """Run the question's reference function and put its values into the checks."""
    values = REFERENCES[question["reference"]]()
    for check in question["checks"]:
        key = check.get("key")
        if key is None:
            continue
        if key not in values:
            raise KeyError(f"{question['id']}: reference {question['reference']} returned no '{key}'")
        if check["type"] == "number":
            check["value"] = round(float(values[key]), 6)
        elif check["type"] == "label":
            check["value"] = str(values[key])
        else:
            check["items"] = list(values[key])[: check["top_k"]]
    return values


def shown(value: Any) -> str:
    if isinstance(value, list):
        return " > ".join(map(str, value))
    return f"{value:,.4f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-final", action="store_true", help="Also print the values of the final split")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    problems = [p for q in questions for p in validate(q)]
    ids = [q.get("id") for q in questions]
    problems += [f"duplicate id {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    if problems:
        print("Invalid questions.json:\n" + "\n".join(f"- {p}" for p in problems))
        return 1

    print(f"{'id':4} {'split':8} {'category':18} computed")
    for question in questions:
        values = fill(question)
        hide = question["split"] == "final" and not args.show_final
        used = {check["key"] for check in question["checks"] if check.get("key")}
        text = "; ".join(
            f"{key}={'(hidden)' if hide else shown(value)}{'' if key in used else ' [info]'}"
            for key, value in values.items()
        )
        print(f"{question['id']:4} {question['split']:8} {question['category']:18} {text}")

    QUESTIONS_FILE.write_text(json.dumps(questions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = {split: sum(q["split"] == split for q in questions) for split in ("dev", "dev_hard", "holdout", "final")}
    print(f"\nWrote {QUESTIONS_FILE.name}: {len(questions)} questions {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
