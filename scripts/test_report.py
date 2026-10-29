"""Checks for evals/report.py and evals/review.py on synthetic results (no API calls)."""

import io
import json
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evals"))

import report  # noqa: E402
import review  # noqa: E402

passed = 0
failed = 0
SECRET_QUOTE = "holdout quote that must stay hidden"


def check(name: str, ok: bool, details: str) -> None:
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"       {details}")


def write_run(root: Path, name: str, split: str, runs: list[dict[str, Any]]) -> None:
    """Synthetic result and grade files; every run has two checks."""
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"split": split, "repeats": 3, "config": {}}), encoding="utf-8")
    for r in runs:
        stem = f"{r['id']}__r{r['repeat']}"
        record = {
            "id": r["id"], "repeat": r["repeat"], "final_answer": "Answer.", "status": r.get("status", "completed"),
            "steps": r.get("steps", 4), "duration_s": 10.0, "loaded_skills": r.get("skills", []),
            "plan_incomplete": False, "tool_errors": r.get("tool_errors", 0), "guard_blocks": r.get("guard_blocks", 0),
            "integrity_quotes": [], "cost": {"agent": 0.02, "verification": 0.01, "total": 0.03},
            "verification": {"status": "ok", "counts": {"supported": 3, "contradicted": int(r.get("revised", False)),
                                                        "unverifiable": 0},
                             "revised": r.get("revised", False), "revision_failed": False},
        }
        checks = [
            {"index": 1, "type": "number", "description": "Value one", "passed": True, "expected": 1.0,
             "extracted": 1.0, "quote": "one"},
            {"index": 2, "type": "mention", "description": "Mentions two", "passed": r["strict"],
             "expected": None if split == "dev" else 42.4242, "extracted": "no" if not r["strict"] else "yes",
             "quote": "two" if split == "dev" else SECRET_QUOTE},
        ]
        grade = {"id": r["id"], "repeat": r["repeat"], "checks": checks, "strict_pass": r["strict"],
                 "partial_score": 1.0 if r["strict"] else 0.5, "judge_cost_usd": 0.005,
                 "draft": None if "draft" not in r else {"strict_pass": r["draft"], "partial_score": 1.0}}
        (run_dir / f"{stem}.json").write_text(json.dumps(record), encoding="utf-8")
        (run_dir / f"{stem}.grade.json").write_text(json.dumps(grade), encoding="utf-8")


def main() -> int:
    # a) Wilson interval
    low, high = report.wilson(8, 10)
    check("a) Wilson 8/10", abs(low - 0.4902) < 1e-3 and abs(high - 0.9433) < 1e-3, f"({low:.4f}, {high:.4f})")
    check("b) Wilson edge cases", report.wilson(0, 0) == (0.0, 1.0) and report.wilson(10, 10)[1] == 1.0
          and report.wilson(0, 10)[0] == 0.0, f"0/0 -> {report.wilson(0, 0)}, 10/10 -> {report.wilson(10, 10)}")

    root = Path(tempfile.mkdtemp())
    try:
        questions = {q["id"]: q for q in json.loads(report.QUESTIONS_FILE.read_text(encoding="utf-8"))}
        write_run(root, "dev_run", "dev", [
            {"id": "d01", "repeat": 1, "strict": True, "revised": True, "draft": True, "skills": ["data-quality"]},
            {"id": "d01", "repeat": 2, "strict": True},
            {"id": "d01", "repeat": 3, "strict": True},
            {"id": "d02", "repeat": 1, "strict": True, "revised": True, "draft": False},  # helped
            {"id": "d02", "repeat": 2, "strict": False, "revised": True, "draft": True, "tool_errors": 2},  # harmed
            {"id": "d02", "repeat": 3, "strict": True, "guard_blocks": 1},
            {"id": "d03", "repeat": 1, "strict": False, "status": "max_steps", "tool_errors": 1},
        ])
        rows = report.load_run(root / "dev_run", questions)
        s = report.summary(rows)

        # c) accuracy overall, by category and traps (d01 has no traps, d03 has)
        check("c) accuracy and groups",
              s["overall"]["passes"] == 5 and s["overall"]["n"] == 7
              and abs(s["overall"]["partial"] - (5 + 2 * 0.5) / 7) < 1e-9
              and s["by_traps"][False]["n"] == 6 and s["by_traps"][True]["passes"] == 0
              and set(s["by_category"]) == {"lookup", "group_comparison", "data_quality_trap"},
              f"overall {s['overall']['passes']}/{s['overall']['n']}, partial {s['overall']['partial']:.3f}, "
              f"traps {s['by_traps'][True]['passes']}/{s['by_traps'][True]['n']}")

        # d) stability: d02 is unstable; d03 has one repeat and is not counted
        check("d) stability", s["stability"] == {"questions": 2, "stable": 1, "unstable": {"d02": "2/3 passed"}},
              f"{s['stability']}")

        # e) verifier effect
        v = s["verifier"]
        check("e) verifier helped / harmed / unchanged",
              (v["revised"], v["helped"], v["harmed"], v["unchanged"]) == (3, 1, 1, 1) and v["claims"]["contradicted"] == 3,
              f"{v}")

        # f) self-correction, guard blocks, skills, costs
        st = s["stats"]
        check("f) run stats",
              st["runs_with_tool_errors"] == 2 and st["self_corrected"] == 1 and st["guard_blocks"] == 1
              and abs(st["skills"]["data-quality"] - 1 / 7) < 1e-9 and abs(st["cost_total"] - 0.035) < 1e-9
              and abs(st["cost_run"] - 7 * 0.035) < 1e-9 and st["status"] == {"completed": 6, "max_steps": 1},
              f"tool errors in {st['runs_with_tool_errors']} runs, self-corrected {st['self_corrected']}, "
              f"cost/run ${st['cost_total']:.3f}")

        # g) dev failures carry details
        md = report.render_report("dev_run", s, {})
        check("g) dev failures with details",
              len(s["failures"]) == 2 and "- Expected:" in md and "- Quote: 'no'" not in md and "Mentions two" in md,
              f"{len(s['failures'])} failed checks listed with question, check and quote")

        # h) holdout failures without expected values or quotes
        write_run(root, "holdout_run", "holdout", [
            {"id": "h01", "repeat": 1, "strict": False},
            {"id": "h01", "repeat": 2, "strict": True},
        ])
        hs = report.summary(report.load_run(root / "holdout_run", questions))
        hmd = report.render_report("holdout_run", hs, {})
        check("h) holdout failures hide details",
              len(hs["failures"]) == 1 and "42.4242" not in hmd and SECRET_QUOTE not in hmd
              and "h01 r1 (holdout): group_comparison, failed mention check" in hmd
              and "expected" not in hs["failures"][0],
              hmd.split("## Failures")[1].strip())

        # i) compare two runs
        out = io.StringIO()
        with redirect_stdout(out):
            code = report.main(["--compare", "dev_run", "holdout_run", "--results-dir", str(root)])
        check("i) compare table", code == 0 and "| Strict pass rate | 71.4% (5/7" in out.getvalue()
              and (root / "compare_dev_run_vs_holdout_run.md").exists(), out.getvalue().splitlines()[2])

        # j) review: only dev answers are sampled, and the human's verdicts are scored
        with redirect_stdout(io.StringIO()):
            review.main(["--name", "dev_run", "--sample", "3", "--results-dir", str(root)])
            no_dev = review.main(["--name", "holdout_run", "--sample", "3", "--results-dir", str(root)])
        text = (root / "dev_run" / "review.md").read_text(encoding="utf-8")
        filled = text.replace("Human: ", "Human: agree", 4).replace("Human: \n", "Human: disagree the quote is wrong\n", 1)
        result = review.score_review(filled)
        check("j) review sample and score",
              no_dev == 2 and text.count("### Check") == 6 and result["filled"] == 5 and result["agree"] == 4
              and "the quote is wrong" in result["disagree"][0] and abs(result["agreement"] - 0.8) < 1e-9,
              f"6 checks sampled; agreement {result['agree']}/{result['filled']}; {result['disagree']}")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"\nTotal: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
