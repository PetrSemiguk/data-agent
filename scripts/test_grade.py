"""Checks for evals/grade.py without the API: comparison code, hiding final values, the judge request."""

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evals"))

import grade as g  # noqa: E402

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


def number(value: float, description: str = "Survival rate of women, in percent", **tol: float) -> dict[str, Any]:
    return {"type": "number", "description": description, "value": value, **(tol or {"abs_tol": 0.1})}


class FakeJudge:
    """client.messages.create that returns a submit_grades call and records the request."""

    def __init__(self, grades: list[dict[str, Any]], first_without_tool: bool = False) -> None:
        self.grades = grades
        self.first_without_tool = first_without_tool
        self.requests: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        usage = SimpleNamespace(input_tokens=1000, output_tokens=200, cache_creation_input_tokens=0,
                                cache_read_input_tokens=0)
        if self.first_without_tool and len(self.requests) == 1:
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="Here are my grades...")], usage=usage)
        call = SimpleNamespace(type="tool_use", name="submit_grades", input={"grades": self.grades})
        return SimpleNamespace(content=[call], usage=usage)


def main() -> int:
    # Numbers: absolute and relative tolerance
    cases = [
        ("a) within abs_tol", number(74.2038), {"value": 74.2, "precision": 0.1}, True),
        ("b) outside abs_tol", number(74.2038), {"value": 75.2, "precision": 0.1}, False),
        ("c) rounded to the answer's precision", number(74.2038), {"value": 74, "precision": 1}, True),
        ("d) wrong at that precision", number(74.2038), {"value": 73, "precision": 1}, False),
        ("e) fraction for a percent", number(74.2038), {"value": 0.742, "precision": 0.001}, True),
        ("f) rel_tol, $836K", number(836154.033, "Total sales", rel_tol=0.002), {"value": 836000, "precision": 1000}, True),
        ("g) coarse rounding capped ($1M)", number(836154.033, "Total sales", rel_tol=0.002), {"value": 1_000_000, "precision": 1_000_000}, False),
        ("h) exact count must be exact", number(34335, "Rows removed", abs_tol=0), {"value": 34300, "precision": 100}, False),
        ("i) exact count", number(34335, "Rows removed", abs_tol=0), {"value": 34335, "precision": 1}, True),
        ("j) narrow range inside tolerance", number(3.1022, "Growth in percent", abs_tol=2.5), {"low": 3, "high": 4}, True),
        ("k) wide range containing the value", number(3.1022, "Growth in percent", abs_tol=2.5), {"low": -1, "high": 6}, False),
        ("l) value not stated", number(74.2038), {"value": None}, False),
        ("m) negative sign kept", number(3.1022, "Growth in percent", abs_tol=2.5), {"value": -1.1, "precision": 0.1}, False),
    ]
    for name, check, grade, expected in cases:
        ok, why = g.compare_number(check, grade)
        report(name, ok == expected, f"{grade} -> passed={ok} ({why})")

    # Rankings
    ranking = {"type": "ranking", "description": "Top months", "items": ["November", "September", "December"],
               "top_k": 3, "order": "descending"}
    for name, items, expected in [
        ("n) same order", ["November", "September", "December", "October"], True),
        ("o) abbreviations", ["Nov", "Sep", "Dec"], True),
        ("p) swapped", ["November", "December", "September"], False),
        ("q) too short", ["November", "September"], False),
        ("r) no ranking", None, False),
    ]:
        ok, why = g.compare_ranking(ranking, {"items": items})
        report(name, ok == expected, f"{items} -> passed={ok} ({why})")
    top1 = {"type": "ranking", "description": "Weakest region", "items": ["Central"], "top_k": 1, "order": "ascending"}
    report("s) top_k=1 uses the first item", g.compare_ranking(top1, {"items": ["Central", "South"]})[0]
           and not g.compare_ranking(top1, {"items": ["South", "Central"]})[0], "Central first passes, South first fails")

    # label: one named value, case and abbreviations as in rankings
    label = {"type": "label", "description": "Region with the lowest profit margin", "value": "Central"}
    report("s2) label compare",
           g.compare(label, {"label": "central"})[0] and g.compare(label, {"label": "CENTRAL region"})[0]
           and not g.compare(label, {"label": "West"})[0] and not g.compare(label, {"label": None})[0]
           and g.compare({**label, "value": "November"}, {"label": "Nov"})[0],
           "'central', 'CENTRAL region', 'Nov' pass; 'West' and null fail")

    # conflicting: number, label and ranking fail; mention and not_claim are not affected
    conflict_number = g.compare(number(74.2038), {"value": 74.2, "precision": 0.1, "conflicting": True})
    report("s3) conflicting answers fail",
           not conflict_number[0] and "contradicts itself" in conflict_number[1]
           and not g.compare(label, {"label": "Central", "conflicting": True})[0]
           and not g.compare(ranking, {"items": ["November", "September", "December"], "conflicting": True})[0]
           and g.compare({"type": "mention", "description": "x"}, {"verdict": "yes", "conflicting": True})[0],
           conflict_number[1])

    # The judge sees the ranking direction and the label type in the request
    from judge_prompt import judge_request  # noqa: E402
    request = judge_request("Q?", "A.", [
        {"type": "ranking", "order": "descending", "description": "Top months"},
        {"type": "ranking", "order": "ascending", "description": "Bottom months"},
        {"type": "label", "description": "Weakest region"},
    ])
    report("s4) order and label in the judge request",
           "1. [ranking, highest first] Top months" in request and "2. [ranking, lowest first] Bottom months" in request
           and "3. [label] Weakest region" in request, request.split("Checks:\n")[1].replace("\n", " | "))

    # mention / not_claim semantics
    mention, not_claim = {"type": "mention", "description": "x"}, {"type": "not_claim", "description": "y"}
    report("t) mention yes/no and not_claim yes/no",
           g.compare(mention, {"verdict": "yes"})[0] and not g.compare(mention, {"verdict": "no"})[0]
           and g.compare(not_claim, {"verdict": "no"})[0] and not g.compare(not_claim, {"verdict": "yes"})[0]
           and not g.compare(mention, None)[0], "mention: yes passes; not_claim: no passes; missing grade fails")

    # The judge request has no expected values; grades are combined in code
    questions = {q["id"]: q for q in json.loads(g.QUESTIONS_FILE.read_text(encoding="utf-8"))}
    q = questions["d06"]
    judge = FakeJudge([
        {"index": 1, "items": ["November", "September", "December"], "quote": "Nov, Sep, Dec", "explanation": "e"},
        {"index": 2, "value": 118400, "precision": 100, "quote": "$118.4K", "explanation": "e"},
        {"index": 3, "verdict": "no", "quote": None, "explanation": "e"},
    ], first_without_tool=True)
    result = g.grade_answer(judge, "claude-sonnet-5-5", q, "Some answer.")
    sent = json.dumps(judge.requests[0]["messages"], default=str) + judge.requests[0]["system"]
    secret = [str(round(q["checks"][1]["value"])), "118447", "118,447"]
    report("u) judge sees no expected values; retry without tool; scores",
           not any(s in sent for s in secret) and "temperature" not in judge.requests[0]
           and len(judge.requests) == 2 and result["strict_pass"] and result["partial_score"] == 1.0
           and result["judge_cost_usd"] == 2 * (1000 * 2 + 200 * 10) / 1e6,
           f"requests={len(judge.requests)}, strict={result['strict_pass']}, cost={result['judge_cost_usd']}")

    # An incomplete report (a mention without verdict) is sent back once and then complete
    class IncompleteFirst(FakeJudge):
        def create(self, **kwargs: Any) -> Any:
            response = super().create(**kwargs)
            if len(self.requests) == 1:
                call = response.content[0]
                call.input = {"grades": [dict(call.input["grades"][0]), {"index": 2, "verdict": None, "explanation": "e"}]}
                call.id = "toolu_1"
            return response

    mention_q = {"question": "Q?", "checks": [number(74.2038), {"type": "mention", "description": "Says x"}]}
    judge = IncompleteFirst([{"index": 1, "value": 74.2, "precision": 0.1, "explanation": "e"},
                             {"index": 2, "verdict": "yes", "explanation": "e"}])
    result = g.grade_answer(judge, "claude-sonnet-5-5", mention_q, "An answer.")
    retry = next(m["content"][0] for m in judge.requests[1]["messages"]
                 if m["role"] == "user" and isinstance(m["content"], list))
    report("u2) incomplete judge report is asked again",
           len(judge.requests) == 2 and retry["is_error"] and "check 2 (mention) needs verdict" in retry["content"]
           and result["strict_pass"],
           retry["content"])

    # No answer: every check fails without calling the judge
    judge = FakeJudge([])
    result = g.grade_answer(judge, "claude-sonnet-5-5", q, None)
    report("v) no answer -> all fail, no judge call",
           not judge.requests and result["partial_score"] == 0.0 and not result["strict_pass"], "0 calls")

    # Final split: expected values and extracted numbers are not printed
    final_q = questions["f04"]
    judge = FakeJudge([{"index": i, "value": final_q["checks"][0]["value"], "precision": 1, "verdict": "yes",
                        "quote": "q", "explanation": "e"} for i in range(1, len(final_q["checks"]) + 1)])
    result = g.grade_answer(judge, "claude-sonnet-5-5", final_q, "An answer.")
    out = io.StringIO()
    with redirect_stdout(out):
        g.print_grade(final_q, 1, result)
    printed = out.getvalue()
    hidden_value = f"{final_q['checks'][0]['value']:g}"
    report("w) final values are hidden in the console",
           hidden_value not in printed and "(hidden)" in printed and "PASS" in printed,
           f"'(hidden)' printed, expected value absent: {hidden_value not in printed}")

    print(f"\nTotal: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
