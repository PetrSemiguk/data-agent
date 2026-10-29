"""Regression cases for the verifier on the REAL Claude API (costs money).

Each case is a short draft answer with a known error (or none), the question,
and the dataset. The full output is also saved to logs/verifier_cases.txt.
"""

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Make the project root importable when running this file as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.loop import create_client  # noqa: E402
from agent.sandbox import Sandbox  # noqa: E402
from agent.usage import format_cost  # noqa: E402
from agent.verifier import VerificationReport, verify_answer  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = PROJECT_ROOT / "logs" / "verifier_cases_v2.txt"
SUPERSTORE = "Sample - Superstore.csv"
ONLINE_RETAIL = "online_retail_II.csv"


# Runs in the kernel before and after a check: a fingerprint of df, the
# kernel's own variable names, and how many separate namespaces still exist.
SNAPSHOT_CODE = (
    "print(int(pd.util.hash_pandas_object(df, index=True).sum()))\n"
    "print(sorted(k for k in globals() if not k.startswith('_')))\n"
    "print(len(globals().get('_sandbox_namespaces', {})))"
)


@dataclass
class Case:
    name: str
    dataset: str
    question: str
    answer: str
    # "contradicted": a claim containing one of `fragments` must be contradicted.
    # "supported": that claim must be supported and nothing contradicted.
    # "ok": every claim supported and status ok.
    expect: str
    fragments: list[str]
    # Borderline case: PASS with either verdict on the key claim, as long as its
    # evidence cites this real value.
    borderline_value: str | None = None


CASES = [
    Case(
        name="a) Q1 share range",
        dataset=SUPERSTORE,
        question="How seasonal are sales by quarter?",
        answer=(
            "The data covers orders from January 2014 to December 2017. Sales are strongly "
            "back-loaded: Q4 is the largest quarter in every year. Q1 accounts for only "
            "15–17% of annual sales in every year."
        ),
        expect="contradicted",
        fragments=["15-17%"],
        # 2015 is 14.6%: outside 15-17% exactly, inside it after rounding to whole percent.
        borderline_value="14.6",
    ),
    Case(
        name="b) wrong month comparison",
        dataset=SUPERSTORE,
        question="Which months had the highest sales in 2017?",
        answer=(
            "In 2017, November was the strongest month ($118.4K), followed by September "
            "($87.9K). In 2017, October ($77.8K) edged out December ($83.8K) for third "
            "place among months."
        ),
        expect="contradicted",
        fragments=["edged out"],
    ),
    Case(
        name="c) mixed-up row sets",
        dataset=ONLINE_RETAIL,
        question="How large are the bad-debt adjustments in the data?",
        answer=(
            "Bad-debt adjustments (StockCode 'B', description 'Adjust bad debt', revenue "
            "computed as Quantity × Price) total about −£147.6K, which are the 5 lines "
            "with negative prices."
        ),
        expect="contradicted",
        fragments=["147.6k", "5 lines"],
    ),
    Case(
        name="d) true trend (false alarm check)",
        dataset=SUPERSTORE,
        question="How has the sales trend developed recently?",
        answer=(
            "The 12-month rolling average of monthly sales rises from about $39K in "
            "January 2016 to about $61K in December 2017 (+56%)."
        ),
        expect="supported",
        fragments=["rolling average", "39k", "61k"],
    ),
    Case(
        name="e) fully correct answer",
        dataset=SUPERSTORE,
        question="Which product categories bring the most revenue?",
        answer=(
            "Total sales over 2014–2017 are $2.30M. Technology brings the most revenue: "
            "$836K (36.4%), followed by Furniture with $742K (32.3%) and Office Supplies "
            "with $719K (31.3%)."
        ),
        expect="ok",
        fragments=[],
    ),
]

lines_out: list[str] = []


def out(text: str = "") -> None:
    print(text, flush=True)
    lines_out.append(text)


def normalize(text: str) -> str:
    """Lower case with all dash variants as '-', so fragments match reliably."""
    for dash in "–—−":
        text = text.replace(dash, "-")
    return text.lower()


def judge(case: Case, report: VerificationReport) -> tuple[bool, str]:
    """PASS/FAIL for a case and a short reason."""
    if report.status == "incomplete":
        return False, f"no report: {report.error_message}"
    key = [c for c in report.claims if any(f in normalize(c.claim) for f in case.fragments)]
    contradicted = [c for c in report.claims if c.status == "contradicted"]
    if case.borderline_value is not None:
        ok = any(case.borderline_value in c.evidence for c in key)
        verdicts = ", ".join(c.status for c in key) or "no key claim"
        return ok, f"BORDERLINE ({verdicts}; {case.borderline_value} {'cited' if ok else 'NOT cited'} in evidence)"
    if case.expect == "contradicted":
        ok = any(c.status == "contradicted" for c in key)
        return ok, "key claim contradicted" if ok else "key claim not contradicted"
    if case.expect == "supported":
        ok = bool(key) and all(c.status == "supported" for c in key) and not contradicted
        return ok, "key claim supported, nothing contradicted" if ok else "false alarm or key claim missing"
    ok = report.status == "ok" and all(c.status == "supported" for c in report.claims)
    return ok, "all claims supported" if ok else "not every claim supported"


def snapshot(sandbox: Sandbox) -> tuple[str, set[str], str]:
    result = sandbox.execute(SNAPSHOT_CODE)
    fingerprint, names, namespaces = result.stdout.strip().splitlines()
    return fingerprint, set(ast.literal_eval(names)), namespaces




def main() -> int:
    parser = argparse.ArgumentParser(description="Verifier regression cases on the real API.")
    parser.add_argument("--model", default="claude-sonnet-5-5", help="Model of the verifier")
    parser.add_argument("--output", default=str(OUTPUT_FILE), help="Where to save the output")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    load_dotenv()
    out(f"Verifier model: {args.model}")
    client = create_client()
    passed = 0
    false_alarms = 0
    total_in = total_out = 0
    total_cost = 0.0
    total_time = 0.0

    with Sandbox() as sandbox:
        loaded = None
        for case in CASES:
            if case.dataset != loaded:
                sandbox.load_dataframe(DATA_DIR / case.dataset)
                loaded = case.dataset
            before_hash, before_names, _ = snapshot(sandbox)

            report = verify_answer(case.question, case.answer, sandbox, client, model=args.model)

            after_hash, after_names, namespaces_left = snapshot(sandbox)
            ok, reason = judge(case, report)
            passed += ok
            total_in += report.input_tokens
            total_out += report.output_tokens
            total_cost += report.cost_usd or 0.0
            total_time += report.duration_s

            out("=" * 78)
            out(f"[{'PASS' if ok else 'FAIL'}] {case.name}: {reason}")
            out(f"  dataset: {case.dataset}")
            out(f"  answer:  {case.answer}")
            out(f"  status:  {report.status}  counts: {report.counts}")
            for claim in report.claims:
                out(f"  - [{claim.status}] {claim.claim}")
                out(f"      evidence:   {claim.evidence}")
                if claim.correction:
                    out(f"      correction: {claim.correction}")
                expected_key = any(f in normalize(claim.claim) for f in case.fragments)
                if claim.status == "contradicted" and not expected_key:
                    false_alarms += 1
                    out("      ^ contradicted outside the planted error: review by hand")
            new_names = sorted(after_names - before_names)
            out(
                f"  isolation: df {'unchanged' if before_hash == after_hash else 'CHANGED'}, "
                f"new kernel variables: {new_names or 'none'}, namespaces left: {namespaces_left}"
            )
            out(
                f"  steps {report.steps}, tokens {report.input_tokens} in / {report.output_tokens} out "
                f"(cache write {report.cache_write_tokens}, read {report.cache_read_tokens}; "
                f"{format_cost(report.cost_usd, 4)}), {report.duration_s:.1f}s"
            )
            out(f"  log: {report.log_path}")

    count = len(CASES)
    out("=" * 78)
    out(f"Total: {passed}/{count} passed, contradicted outside planted errors: {false_alarms}")
    out(
        f"Average per check: {total_in // count} in / {total_out // count} out tokens, "
        f"${total_cost / count:.4f} (total ${total_cost:.4f}), {total_time / count:.1f}s"
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return 0 if passed == count else 1


if __name__ == "__main__":
    sys.exit(main())
