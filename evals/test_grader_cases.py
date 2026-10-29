"""Check the grader on hand-written answers whose correct grades are known (real judge API).

Each case is a dev question, an artificial answer and the expected verdict of
every check (in check order). Output is also saved to logs/grader_cases.txt.
Run from the project root:  .venv\\Scripts\\python.exe evals\\test_grader_cases.py
"""

import argparse
import json
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVALS_DIR))

from dotenv import load_dotenv  # noqa: E402

from agent.loop import create_client  # noqa: E402
from agent.usage import add_costs, format_cost  # noqa: E402
from grade import DEFAULT_JUDGE_MODEL, QUESTIONS_FILE, grade_answer  # noqa: E402

OUTPUT_FILE = PROJECT_ROOT / "logs" / "grader_cases.txt"

# (question id, case name, answer, expected passed per check)
CASES = [
    # d02 checks: female rate (%), male rate (%), mention of group sizes
    ("d02", "correct, human format",
     "Yes. Women survived far more often: 74.2% of the 314 women survived, against 18.9% of the 577 men.",
     [True, True, True]),
    ("d02", "number close but wrong",
     "Yes. 75.2% of the 314 women survived, against 18.9% of the 577 men.",
     [False, True, True]),
    ("d02", "fractions instead of percent",
     "Yes. The survival rate was 0.742 for women (n = 314) and 0.189 for men (n = 577).",
     [True, True, True]),
    ("d02", "group sizes missing",
     "Yes. Women's survival rate was 74.2% and men's was 18.9%.",
     [True, True, False]),
    ("d02", "number about another group",
     "Yes. In first class, 96.8% of women survived. Overall, 18.9% of the 577 men survived; there were 314 women.",
     [False, True, True]),
    # d06 checks: top months ranking, November sales, not_claim October above December
    ("d06", "correct, human format",
     "In 2017 the strongest months were November ($118.4K), September ($87.9K) and December ($83.8K). "
     "October ($77.8K) came fourth.",
     [True, True, True]),
    ("d06", "ranking swapped",
     "The top three months of 2017 were November ($118.4K), December ($83.8K) and September ($87.9K), in that order.",
     [False, True, True]),
    ("d06", "forbidden claim in other words",
     "November led 2017 with $118.4K, followed by September at $87.9K. October's $77.8K put it ahead of "
     "December's $83.8K for third place.",
     [False, True, False]),
    ("d06", "November close but wrong",
     "In 2017 the top months were November ($120.4K), September ($87.9K) and December ($83.8K).",
     [True, False, True]),
    ("d06", "November value from another year",
     "For 2017 the top three months were November, September and December. For comparison, November 2016 "
     "had sales of $79.4K.",
     [True, False, True]),
    # d10 checks: growth (%), mention of the 9 December end, not_claim decline
    ("d10", "correct, human format",
     "Yes. The 2011 data stops on 9 December, so I compared 1 January to 9 December in both years: net "
     "revenue grew by about 3%.",
     [True, True, True]),
    ("d10", "naive decline",
     "No. Net revenue fell from 9.08 million in 2010 to 8.98 million in 2011, a 1.1% decline.",
     [False, False, False]),
    ("d10", "per-day variant within tolerance",
     "Yes. The data for 2011 ends on 9 December, so I compared revenue per day of data: 2011 was 5.2% "
     "higher than 2010.",
     [True, True, True]),
    ("d10", "growth outside tolerance",
     "Yes. Because 2011 ends on 9 December, I compared the same dates in both years: revenue grew by about 6%.",
     [False, True, True]),
    ("d10", "incomplete year not mentioned",
     "Yes, net revenue grew by about 3% from 2010 to 2011.",
     [True, False, True]),
    # d11 checks: net revenue 2010, duplicates removed
    ("d11", "correct, human format",
     "After removing 34,335 exact duplicate rows, net revenue in 2010 was about 9.08 million.",
     [True, True]),
    ("d11", "revenue of another year",
     "After removing 34,335 exact duplicate rows, net revenue in 2011 was 8.98 million.",
     [False, True]),
    ("d11", "revenue close but wrong",
     "After removing 34,335 duplicate rows, net revenue in 2010 was 9.18 million.",
     [False, True]),
    ("d11", "duplicate count wrong",
     "After removing 34,353 exact duplicates, net revenue in 2010 was £9,079,584.",
     [True, False]),
    # Answers that state different values for one quantity
    ("d11", "contradiction, result is clear",
     "My first estimate was 9.2 million, before removing duplicates. After removing the 34,335 exact duplicate "
     "rows, the final net revenue for 2010 is 9.08 million.",
     [True, True]),
    ("d02", "contradiction, result unclear",
     "Women survived at a rate of 74.2%, and a second calculation gives 68.1% for women. Men survived at 18.9%. "
     "There were 314 women and 577 men.",
     [False, True, True]),
    ("d09", "headline plus a contradicting side remark",
     "**5,881 distinct customers made at least one purchase.** I left out rows without a Customer ID and "
     "cancellation invoices. All 5,942 customer IDs in the data appear on at least one non-cancelled row.",
     [True, True]),
    # d08 checks: weakest region (label), its margin, sub-category with the largest loss (label), that loss
    ("d08", "label with a table listed highest first",
     "Central has the lowest profit margin, and Furnishings loses the most money there.\n\n"
     "| Region | Margin |\n|---|---|\n| West | 14.9% |\n| East | 13.5% |\n| South | 11.9% |\n| Central | 7.9% |\n\n"
     "In Central, Furnishings lost $3,906, followed by Tables ($3,560).",
     [True, True, True, True]),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the grader on answers with known grades.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    load_dotenv(PROJECT_ROOT / ".env")
    client = create_client()
    questions = {q["id"]: q for q in json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))}
    lines: list[str] = []
    right = total = 0
    cost = 0.0
    for qid, name, answer, expected in CASES:
        question = questions[qid]
        assert len(expected) == len(question["checks"]), (qid, name)
        grade = grade_answer(client, args.judge_model, question, answer)
        cost = add_costs(cost, grade["judge_cost_usd"])
        lines.append(f"\n{qid} | {name}\n  answer: {answer}")
        for want, check in zip(expected, grade["checks"]):
            ok = want == check["passed"]
            right += ok
            total += 1
            lines.append(
                f"  {'ok ' if ok else 'XX '} {check['index']}. {check['type']:9} expected {'PASS' if want else 'FAIL'}, "
                f"got {'PASS' if check['passed'] else 'FAIL'} | extracted={check['extracted']!s} | "
                f"quote={check['quote']!r}"
                + ("" if ok else f"\n       why: {check['explanation']}")
            )
        print("\n".join(lines[-1 - len(expected):]), flush=True)
    summary = (f"\nGrader accuracy ({args.judge_model}): {right}/{total} checks ({right / total:.1%}), "
               f"judge cost {format_cost(cost)}")
    lines.append(summary)
    print(summary)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if right == total else 1


if __name__ == "__main__":
    sys.exit(main())
