"""Grade the answers of an eval run against the checks in questions.json.

Run from the project root:  .venv\\Scripts\\python.exe evals\\grade.py --name smoke

For every <id>__r<k>.json in evals/results/<name>/ it writes <id>__r<k>.grade.json
next to it; existing grades are kept unless --force is given.

How a check is graded:
- number and ranking: the judge model only EXTRACTS what the answer states
  (with a verbatim quote); this code compares it with the expected value;
- mention and not_claim: the judge answers yes/no with a quote.
If the agent revised its answer after the verifier's remarks, the draft is graded
the same way (under "draft"), to measure what the verifier changed.
The judge never sees the expected values. Expected values of the final split
are never printed, only pass/fail.

Notes for the API: temperature is not set, because Claude Sonnet 5.5 rejects a
non-default temperature (and Claude Haiku 5.5 accepts only 1); the structured
tool output keeps the judge consistent instead. A forced tool_choice is also
rejected on Sonnet 5.5, so the prompt asks for the tool, with one retry.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVALS_DIR))

from dotenv import load_dotenv  # noqa: E402

from agent.loop import create_client  # noqa: E402
from agent.usage import add_costs, format_cost, request_cost, usage_counts  # noqa: E402
from judge_prompt import JUDGE_PROMPT, SUBMIT_GRADES_TOOL, judge_request  # noqa: E402
from run_evals import result_files  # noqa: E402

RESULTS_DIR = EVALS_DIR / "results"
QUESTIONS_FILE = EVALS_DIR / "questions.json"
DEFAULT_JUDGE_MODEL = "claude-sonnet-5-5"
# A rounded number ("74%", "$118K") passes if rounding the expected value could give
# it, but the rounding allowance is capped at 1% of the expected value, so a coarse
# number ("$1M" for 836,154) cannot pass. Counts with abs_tol 0 must be exact.
MAX_ROUNDING_SHARE = 0.01
MIN_PREFIX = 3  # "Nov" matches "November"


# ------------------------------------------------------------ comparison (no API)

def tolerance(check: dict[str, Any]) -> float:
    if "abs_tol" in check:
        return float(check["abs_tol"])
    return float(check["rel_tol"]) * abs(float(check["value"]))


def compare_number(check: dict[str, Any], grade: dict[str, Any]) -> tuple[bool, str]:
    """Pass if the stated value is within the check's tolerance of the expected one.

    - rounding: a number written with precision p also passes within p/2 (capped,
      see MAX_ROUNDING_SHARE), except for exact counts (abs_tol 0);
    - percent vs fraction: for a check "in percent", a value 0.742 is read as 74.2;
    - range: passes only if both ends are within the tolerance, i.e. every value
      in the range would pass; a wide range that merely contains the expected
      value does not.
    """
    expected = float(check["value"])
    tol = tolerance(check)
    precision = grade.get("precision")
    allowance = 0.0
    if tol > 0 and precision:
        allowance = min(float(precision) / 2, MAX_ROUNDING_SHARE * abs(expected))
    limit = max(tol, allowance) + 1e-9

    value, low, high = grade.get("value"), grade.get("low"), grade.get("high")
    if value is None and (low is None or high is None):
        return False, "the answer does not state this value"
    if value is None:
        ok = abs(float(low) - expected) <= limit and abs(float(high) - expected) <= limit
        return ok, f"range {low}-{high}: {'both ends' if ok else 'not both ends'} within {limit:g} of the expected value"
    value = float(value)
    if abs(value - expected) <= limit:
        return True, f"within {limit:g} of the expected value"
    if "percent" in check["description"].lower() and abs(value) <= 1 < abs(expected) and abs(value * 100 - expected) <= limit:
        return True, f"a fraction ({value}) of the expected percentage"
    return False, f"off by {abs(value - expected):g} (allowed {limit:g})"


def same_item(expected: str, stated: str) -> bool:
    a, b = expected.strip().lower(), stated.strip().lower()
    if a == b:
        return True
    shorter = min(a, b, key=len)
    return len(shorter) >= MIN_PREFIX and (a.startswith(b) or b.startswith(a))


def compare_ranking(check: dict[str, Any], grade: dict[str, Any]) -> tuple[bool, str]:
    """Pass if the first top_k stated items match the expected ones in order."""
    stated = grade.get("items")
    if not stated:
        return False, "the answer gives no ranking for this"
    want = check["items"][: check["top_k"]]
    got = list(stated)[: check["top_k"]]
    if len(got) < len(want):
        return False, f"only {len(got)} of {len(want)} ranked items stated"
    ok = all(same_item(w, g) for w, g in zip(want, got))
    return ok, "same order" if ok else "different order or items"


def compare_label(check: dict[str, Any], grade: dict[str, Any]) -> tuple[bool, str]:
    """Pass if the named answer is the expected one (case and abbreviations as in rankings)."""
    stated = grade.get("label")
    if not stated:
        return False, "the answer names no value for this"
    ok = same_item(str(check["value"]), str(stated))
    return ok, "same value" if ok else "a different value"


def compare(check: dict[str, Any], grade: dict[str, Any] | None) -> tuple[bool, str]:
    if grade is None:
        return False, "the judge did not report this check"
    if check["type"] in ("number", "label", "ranking") and grade.get("conflicting"):
        return False, "answer contradicts itself: it states different values and its result is unclear"
    if check["type"] == "number":
        return compare_number(check, grade)
    if check["type"] == "label":
        return compare_label(check, grade)
    if check["type"] == "ranking":
        return compare_ranking(check, grade)
    verdict = grade.get("verdict")
    if verdict not in ("yes", "no"):
        return False, "no verdict from the judge"
    passed = verdict == "yes" if check["type"] == "mention" else verdict == "no"
    return passed, f"judge: {verdict}"


def extracted_of(check: dict[str, Any], grade: dict[str, Any] | None) -> Any:
    if grade is None:
        return None
    if check["type"] == "number":
        return grade.get("value") if grade.get("value") is not None else (
            [grade.get("low"), grade.get("high")] if grade.get("low") is not None else None)
    if check["type"] == "ranking":
        return grade.get("items")
    if check["type"] == "label":
        return grade.get("label")
    return grade.get("verdict")


def expected_of(check: dict[str, Any]) -> Any:
    return check.get("items") if check["type"] == "ranking" else check.get("value")


# ------------------------------------------------------------------ the judge

def missing_fields(checks: list[dict[str, Any]], grades: dict[int, dict]) -> list[str]:
    """Checks the judge left out or reported without the field its type needs."""
    problems = []
    for index, check in enumerate(checks, start=1):
        grade = grades.get(index)
        if grade is None:
            problems.append(f"check {index} is missing")
        elif check["type"] in ("mention", "not_claim") and grade.get("verdict") not in ("yes", "no"):
            problems.append(f"check {index} ({check['type']}) needs verdict \"yes\" or \"no\"")
    return problems


def ask_judge(client: Any, model: str, question: str, answer: str, checks: list[dict[str, Any]]) -> tuple[dict[int, dict], float | None, dict[str, int]]:
    """One judge call; it is asked again (at most twice) if it answers without the tool
    or leaves a check out. A report that stays incomplete is returned as it is, and
    its incomplete checks fail."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": judge_request(question, answer, checks)}]
    cost: float | None = 0.0
    tokens = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0}
    grades: dict[int, dict] | None = None
    for _ in range(3):
        response = client.messages.create(
            model=model, max_tokens=4096, system=JUDGE_PROMPT, tools=[SUBMIT_GRADES_TOOL], messages=messages
        )
        counts = usage_counts(response.usage)
        for name, count in zip(tokens, counts):
            tokens[name] += count
        cost = add_costs(cost, request_cost(model, *counts))
        call = next((b for b in response.content if b.type == "tool_use" and b.name == "submit_grades"), None)
        messages.append({"role": "assistant", "content": response.content})
        if call is None:
            messages.append({"role": "user", "content": "Report the grades by calling submit_grades."})
            continue
        grades = {int(g["index"]): g for g in call.input.get("grades", []) if "index" in g}
        problems = missing_fields(checks, grades)
        if not problems:
            return grades, cost, tokens
        messages.append({"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": getattr(call, "id", "submit_grades"),
            "is_error": True,
            "content": "Incomplete report: " + "; ".join(problems)
                       + ". Call submit_grades again with every check.",
        }]})
    if grades is None:
        raise RuntimeError("The judge did not call submit_grades.")
    return grades, cost, tokens


def grade_answer(client: Any, model: str, question: dict[str, Any], answer: str | None) -> dict[str, Any]:
    """Grades of one answer: per check and for the question as a whole."""
    checks = question["checks"]
    cost: float | None = 0.0
    tokens = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0}
    if answer:
        grades, cost, tokens = ask_judge(client, model, question["question"], answer, checks)
    else:
        grades = {}  # no answer at all: every check fails, the judge is not called
    results = []
    for index, check in enumerate(checks, start=1):
        grade = grades.get(index)
        passed, reason = compare(check, grade) if answer else (False, "no final answer")
        results.append({
            "index": index,
            "type": check["type"],
            "key": check.get("key"),
            "description": check["description"],
            "passed": passed,
            "extracted": extracted_of(check, grade),
            "quote": grade.get("quote") if grade else None,
            "expected": expected_of(check),
            "conflicting": bool(grade.get("conflicting")) if grade else False,
            "explanation": f"{reason}; judge: {grade.get('explanation', '')}" if grade else reason,
        })
    passed_count = sum(r["passed"] for r in results)
    return {
        "checks": results,
        "strict_pass": passed_count == len(results),
        "partial_score": passed_count / len(results) if results else 0.0,
        "judge_model": model,
        "judge_tokens": tokens,
        "judge_cost_usd": cost,
    }


# --------------------------------------------------------------------- output

def shown_expected(question: dict[str, Any], expected: Any) -> str:
    """Expected values of the final split are never printed."""
    if question["split"] == "final":
        return "(hidden)"
    return " > ".join(map(str, expected)) if isinstance(expected, list) else f"{expected:g}" if isinstance(expected, float) else str(expected)


def print_grade(question: dict[str, Any], repeat: int, grade: dict[str, Any]) -> None:
    print(f"\n{question['id']} r{repeat}: strict_pass={grade['strict_pass']}, partial={grade['partial_score']:.2f}, "
          f"judge {format_cost(grade['judge_cost_usd'])}")
    for c in grade["checks"]:
        extracted = c["extracted"]
        if question["split"] == "final" and c["type"] in ("number", "label", "ranking"):
            extracted = "(hidden)" if extracted is not None else None
        quote = (c["quote"] or "")[:70]
        print(f"  {'PASS' if c['passed'] else 'FAIL'} {c['index']}. {c['type']:9} expected={shown_expected(question, c['expected']):28} "
              f"extracted={extracted!s:28} quote: {quote!r}")


def draft_of(record: dict[str, Any]) -> str | None:
    """The answer before revision, for revised runs; older result files keep it only in the run log."""
    verification = record.get("verification") or {}
    if not verification.get("revised"):
        return None
    if record.get("draft_answer"):
        return record["draft_answer"]
    log_path = Path(record.get("log_path") or "")
    if log_path.is_file():
        return json.loads(log_path.read_text(encoding="utf-8")).get("draft_answer")
    return None


def main(argv: list[str] | None = None, client: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Grade the answers of an eval run.")
    parser.add_argument("--name", required=True, help="Name of the run in evals/results/")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--force", action="store_true", help="Grade again even if a grade file exists")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    run_dir = Path(args.results_dir) / args.name
    if not run_dir.is_dir():
        print(f"No results in {run_dir}")
        return 2
    questions = {q["id"]: q for q in json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))}
    load_dotenv(PROJECT_ROOT / ".env")
    client = client or create_client()
    total_cost: float | None = 0.0
    graded = skipped = 0
    for result_path in result_files(run_dir):
        grade_path = result_path.with_name(result_path.stem + ".grade.json")
        if grade_path.exists() and not args.force:
            skipped += 1
            continue
        record = json.loads(result_path.read_text(encoding="utf-8"))
        question = questions[record["id"]]
        grade = {"id": record["id"], "repeat": record["repeat"], "split": question["split"],
                 **grade_answer(client, args.judge_model, question, record.get("final_answer"))}
        draft = draft_of(record)
        grade["draft"] = grade_answer(client, args.judge_model, question, draft) if draft else None
        if grade["draft"]:
            grade["judge_cost_usd"] = add_costs(grade["judge_cost_usd"], grade["draft"]["judge_cost_usd"])
        grade_path.write_text(json.dumps(grade, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        total_cost = add_costs(total_cost, grade["judge_cost_usd"])
        graded += 1
        print_grade(question, record["repeat"], grade)
        if grade["draft"]:
            print(f"  draft before revision: strict_pass={grade['draft']['strict_pass']}, "
                  f"partial={grade['draft']['partial_score']:.2f}")
    print(f"\nGraded {graded} answers ({skipped} already graded; --force grades again), judge cost {format_cost(total_cost)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
