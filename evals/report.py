"""Metrics and a Markdown report for a graded eval run.

Run from the project root:
    .venv\\Scripts\\python.exe evals\\report.py --name dev_baseline
    .venv\\Scripts\\python.exe evals\\report.py --compare dev_baseline dev_no_skills

Writes evals/results/<name>/report.md and prints a short summary. The Failures
section shows expected values and quotes only for the tuning splits (dev,
dev_hard, dev_open): holdout and final failures list only the question id, category and
check type, so the agent is never tuned on them.
"""

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS_DIR))

from run_evals import RESULTS_DIR, result_files  # noqa: E402

QUESTIONS_FILE = EVALS_DIR / "questions.json"
SPLITS = ("dev", "dev_hard", "dev_open", "holdout", "final")
# Splits whose failures may be looked at in detail (the agent may be tuned on them).
TUNING_SPLITS = ("dev", "dev_hard", "dev_open")


# ------------------------------------------------------------------- loading

def load_run(run_dir: Path, questions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per run (question x repeat): the result, its grade (or None) and question info."""
    rows = []
    for path in result_files(run_dir):
        record = json.loads(path.read_text(encoding="utf-8"))
        grade_path = path.with_name(path.stem + ".grade.json")
        grade = json.loads(grade_path.read_text(encoding="utf-8")) if grade_path.exists() else None
        question = questions[record["id"]]
        rows.append({
            "record": record,
            "grade": grade,
            "id": record["id"],
            "repeat": record["repeat"],
            "split": question["split"],
            "category": question["category"],
            "has_traps": bool(question["traps"]),
            "question": question["question"],
        })
    return rows


# ------------------------------------------------------------------- metrics

def wilson(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a pass rate; (0, 1) when n is 0."""
    if n == 0:
        return 0.0, 1.0
    p = passes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [r for r in rows if r["grade"]]
    passes = sum(r["grade"]["strict_pass"] for r in graded)
    n = len(graded)
    return {
        "n": n,
        "passes": passes,
        "pass_rate": passes / n if n else 0.0,
        "ci": wilson(passes, n),
        "partial": statistics.fmean(r["grade"]["partial_score"] for r in graded) if graded else 0.0,
    }


def grouped_accuracy(rows: list[dict[str, Any]], key: str) -> dict[Any, dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {value: accuracy(group) for value, group in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Questions whose repeats all gave the same strict result, and the unstable ones."""
    by_question: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if row["grade"]:
            by_question[row["id"]].append(row["grade"]["strict_pass"])
    repeated = {qid: results for qid, results in by_question.items() if len(results) > 1}
    unstable = {qid: f"{sum(results)}/{len(results)} passed" for qid, results in sorted(repeated.items())
                if len(set(results)) > 1}
    return {"questions": len(repeated), "stable": len(repeated) - len(unstable), "unstable": unstable}


def verifier_effect(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How the verifier's revisions changed the strict result, and the claim totals."""
    effect = {"revised": 0, "helped": 0, "harmed": 0, "unchanged": 0, "draft_not_graded": 0, "revision_failed": 0}
    claims: Counter[str] = Counter()
    for row in rows:
        verification = row["record"].get("verification")
        if not verification:
            continue
        claims.update(verification.get("counts") or {})
        effect["revision_failed"] += bool(verification.get("revision_failed"))
        if not verification.get("revised"):
            continue
        effect["revised"] += 1
        draft = (row["grade"] or {}).get("draft")
        if not row["grade"] or not draft:
            effect["draft_not_graded"] += 1
            continue
        before, after = draft["strict_pass"], row["grade"]["strict_pass"]
        key = "helped" if after and not before else "harmed" if before and not after else "unchanged"
        effect[key] += 1
    effect["claims"] = dict(claims)
    return effect


def run_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Steps, time, cost per run, tool errors, guard blocks, plans, skills, integrity."""
    records = [r["record"] for r in rows]
    n = len(records)

    def mean(values: list[float]) -> float:
        return statistics.fmean(values) if values else 0.0

    judge = [(r["grade"] or {}).get("judge_cost_usd") or 0.0 for r in rows]
    agent = [r["cost"]["agent"] or 0.0 for r in records]
    check = [r["cost"]["verification"] or 0.0 for r in records]
    skills = Counter(skill for r in records for skill in r["loaded_skills"])
    return {
        "runs": n,
        "steps_mean": mean([r["steps"] for r in records]),
        "steps_median": statistics.median([r["steps"] for r in records]) if records else 0,
        "time_mean": mean([r["duration_s"] for r in records]),
        "cost_agent": mean(agent),
        "cost_verification": mean(check),
        "cost_judge": mean(judge),
        "cost_total": mean([a + c + j for a, c, j in zip(agent, check, judge)]),
        "cost_run": sum(agent) + sum(check) + sum(judge),
        "unknown_cost_runs": sum(r["cost"]["total"] is None for r in records),
        "status": dict(Counter(r["status"] for r in records)),
        "self_corrected": sum(r["tool_errors"] > 0 and r["status"] == "completed" for r in records),
        "runs_with_tool_errors": sum(r["tool_errors"] > 0 for r in records),
        "guard_blocks": sum(r["guard_blocks"] for r in records),
        "plan_incomplete": sum(bool(r["plan_incomplete"]) for r in records),
        "skills": {name: count / n for name, count in skills.most_common()} if n else {},
        "integrity": [f"{r['id']} r{r['repeat']}: {q}" for r in records for q in r["integrity_quotes"]],
        "ungraded": sum(row["grade"] is None for row in rows),
    }


def failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every failed check; details only for the tuning splits, where looking at failures is allowed."""
    found = []
    for row in rows:
        if not row["grade"]:
            continue
        for check in row["grade"]["checks"]:
            if check["passed"]:
                continue
            item = {"id": row["id"], "repeat": row["repeat"], "split": row["split"],
                    "category": row["category"], "type": check["type"]}
            if row["split"] in TUNING_SPLITS:
                item.update(question=row["question"], description=check["description"], expected=check["expected"],
                            extracted=check["extracted"], quote=check["quote"])
            found.append(item)
    return found


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": accuracy(rows),
        "by_split": grouped_accuracy(rows, "split"),
        "by_category": grouped_accuracy(rows, "category"),
        "by_traps": grouped_accuracy(rows, "has_traps"),
        "stability": stability(rows),
        "verifier": verifier_effect(rows),
        "stats": run_stats(rows),
        "failures": failures(rows),
    }


# -------------------------------------------------------------------- output

def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def rate_cell(acc: dict[str, Any]) -> str:
    low, high = acc["ci"]
    return f"{pct(acc['pass_rate'])} ({acc['passes']}/{acc['n']}, CI {pct(low)}-{pct(high)})"


def accuracy_table(title: str, groups: dict[Any, dict[str, Any]]) -> list[str]:
    lines = [f"### {title}", "", "| Group | Strict pass rate (95% CI) | Mean partial score |", "|---|---|---|"]
    lines += [f"| {name} | {rate_cell(acc)} | {acc['partial']:.2f} |" for name, acc in groups.items()]
    return lines + [""]


def render_report(name: str, s: dict[str, Any], manifest: dict[str, Any]) -> str:
    stats, verifier, stab = s["stats"], s["verifier"], s["stability"]
    lines = [
        f"# Eval report: {name}",
        "",
        f"Split: {manifest.get('split')}, repeats: {manifest.get('repeats')}, commit: {manifest.get('git_commit')}, "
        f"started: {manifest.get('started_at')}",
        f"Config: `{json.dumps(manifest.get('config'))}`",
        "",
        "## Accuracy",
        "",
        f"Overall strict pass rate: **{rate_cell(s['overall'])}**, mean partial score **{s['overall']['partial']:.2f}**.",
        "",
        "Strict pass = every check of the question passed; partial score = share of checks passed. "
        "Rates are over runs (question x repeat). The 95% Wilson interval shows how uncertain a rate is: "
        "with a small n it is wide (with 10 runs, 8 passes means anything from about 49% to 94%). Repeats "
        "of the same question are not independent, so the real uncertainty is somewhat larger than shown.",
        "",
        *accuracy_table("By split", s["by_split"]),
        *accuracy_table("By category", s["by_category"]),
        *accuracy_table("With and without traps", {("with traps" if k else "no traps"): v for k, v in s["by_traps"].items()}),
        "## Stability across repeats",
        "",
        f"{stab['stable']}/{stab['questions']} questions gave the same strict result in every repeat.",
    ]
    if stab["unstable"]:
        lines += ["", "Unstable: " + ", ".join(f"{qid} ({text})" for qid, text in stab["unstable"].items())]
    lines += [
        "",
        "## Verifier",
        "",
        f"- Revised answers: {verifier['revised']} (revision failed: {verifier['revision_failed']})",
        f"- Draft failed -> final passed (helped): {verifier['helped']}",
        f"- Draft passed -> final failed (harmed): {verifier['harmed']}",
        f"- Same strict result: {verifier['unchanged']}; drafts not graded: {verifier['draft_not_graded']}",
        f"- Claims checked: {json.dumps(verifier['claims'])}",
        "",
        "## Runs",
        "",
        f"- Status: {json.dumps(stats['status'])}",
        f"- Steps: mean {stats['steps_mean']:.1f}, median {stats['steps_median']}; time per run {stats['time_mean']:.1f} s",
        f"- Runs with a failed tool call: {stats['runs_with_tool_errors']}, of them completed anyway "
        f"(self-corrected): {stats['self_corrected']}; guard blocks: {stats['guard_blocks']}",
        f"- Plan incomplete: {stats['plan_incomplete']}",
        "- Skills loaded (share of runs): " + (", ".join(f"{k} {pct(v)}" for k, v in stats["skills"].items()) or "none"),
        f"- Integrity violations: {len(stats['integrity'])}" + "".join(f"\n  - {q}" for q in stats["integrity"]),
        "",
        "## Cost",
        "",
        "| Per run | Agent | Verification | Judge | Total |",
        "|---|---|---|---|---|",
        f"| mean | ${stats['cost_agent']:.4f} | ${stats['cost_verification']:.4f} | ${stats['cost_judge']:.4f} | "
        f"${stats['cost_total']:.4f} |",
        "",
        f"Whole run: **${stats['cost_run']:.3f}** for {stats['runs']} runs"
        + (f" ({stats['unknown_cost_runs']} runs with unknown cost not included)" if stats["unknown_cost_runs"] else "")
        + (f"; {stats['ungraded']} runs not graded yet" if stats["ungraded"] else "") + ".",
        "",
        "## Failures",
        "",
    ]
    fails = s["failures"]
    if not fails:
        lines.append("No failed checks.")
    for f in fails:
        if f["split"] in TUNING_SPLITS:
            lines += [
                f"### {f['id']} r{f['repeat']} - {f['type']} ({f['category']})",
                "",
                f"- Question: {f['question']}",
                f"- Check: {f['description']}",
                f"- Expected: {f['expected']}",
                f"- Extracted: {f['extracted']}",
                f"- Quote: {f['quote']!r}",
                "",
            ]
        else:
            lines.append(f"- {f['id']} r{f['repeat']} ({f['split']}): {f['category']}, failed {f['type']} check")
    return "\n".join(lines) + "\n"


def print_summary(name: str, s: dict[str, Any]) -> None:
    stats, verifier, stab = s["stats"], s["verifier"], s["stability"]
    print(f"Run {name}: {stats['runs']} runs, strict pass {rate_cell(s['overall'])}, partial {s['overall']['partial']:.2f}")
    for title, groups in (("split", s["by_split"]), ("category", s["by_category"]), ("traps", s["by_traps"])):
        print(f"  by {title}: " + "; ".join(f"{k}: {pct(v['pass_rate'])} ({v['passes']}/{v['n']})" for k, v in groups.items()))
    print(f"  stable questions: {stab['stable']}/{stab['questions']}"
          + (f"; unstable: {', '.join(f'{k} ({v})' for k, v in stab['unstable'].items())}" if stab["unstable"] else ""))
    print(f"  verifier: {verifier['revised']} revised -> helped {verifier['helped']}, harmed {verifier['harmed']}, "
          f"unchanged {verifier['unchanged']}; claims {json.dumps(verifier['claims'])}")
    print(f"  steps mean {stats['steps_mean']:.1f} / median {stats['steps_median']}; time {stats['time_mean']:.1f}s; "
          f"tool errors in {stats['runs_with_tool_errors']} runs (self-corrected {stats['self_corrected']}); "
          f"guard blocks {stats['guard_blocks']}; plan incomplete {stats['plan_incomplete']}; "
          f"integrity violations {len(stats['integrity'])}")
    print(f"  cost per run: agent ${stats['cost_agent']:.4f} + verification ${stats['cost_verification']:.4f} + "
          f"judge ${stats['cost_judge']:.4f} = ${stats['cost_total']:.4f}; whole run ${stats['cost_run']:.3f}")
    print("  skills: " + (", ".join(f"{k} {pct(v)}" for k, v in stats["skills"].items()) or "none"))
    print(f"  failed checks: {len(s['failures'])}")


def compare_table(a_name: str, a: dict[str, Any], b_name: str, b: dict[str, Any]) -> str:
    def row(label: str, va: str, vb: str) -> str:
        return f"| {label} | {va} | {vb} |"

    lines = [f"| Metric | {a_name} | {b_name} |", "|---|---|---|",
             row("Strict pass rate", rate_cell(a["overall"]), rate_cell(b["overall"])),
             row("Mean partial score", f"{a['overall']['partial']:.2f}", f"{b['overall']['partial']:.2f}")]
    for split in SPLITS:
        if split in a["by_split"] or split in b["by_split"]:
            cell = lambda s: rate_cell(s["by_split"][split]) if split in s["by_split"] else "-"  # noqa: E731
            lines.append(row(f"Strict pass, {split}", cell(a), cell(b)))
    lines += [
        row("Stable questions", f"{a['stability']['stable']}/{a['stability']['questions']}",
            f"{b['stability']['stable']}/{b['stability']['questions']}"),
        row("Revised (helped / harmed)", f"{a['verifier']['revised']} ({a['verifier']['helped']} / {a['verifier']['harmed']})",
            f"{b['verifier']['revised']} ({b['verifier']['helped']} / {b['verifier']['harmed']})"),
        row("Steps, mean", f"{a['stats']['steps_mean']:.1f}", f"{b['stats']['steps_mean']:.1f}"),
        row("Time per run, s", f"{a['stats']['time_mean']:.1f}", f"{b['stats']['time_mean']:.1f}"),
        row("Cost per run, total", f"${a['stats']['cost_total']:.4f}", f"${b['stats']['cost_total']:.4f}"),
        row("Cost per run, agent", f"${a['stats']['cost_agent']:.4f}", f"${b['stats']['cost_agent']:.4f}"),
        row("Cost per run, verification", f"${a['stats']['cost_verification']:.4f}", f"${b['stats']['cost_verification']:.4f}"),
        row("Whole run cost", f"${a['stats']['cost_run']:.3f}", f"${b['stats']['cost_run']:.3f}"),
    ]
    return "\n".join(lines) + "\n"


def load_summary(results_dir: Path, name: str, questions: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = results_dir / name
    if not run_dir.is_dir():
        raise SystemExit(f"No results in {run_dir}")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return summary(load_run(run_dir, questions)), manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report the metrics of a graded eval run.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", help="Run name in evals/results/")
    group.add_argument("--compare", nargs=2, metavar=("A", "B"), help="Compare the main metrics of two runs")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # not when the output is captured
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    results_dir = Path(args.results_dir)
    questions = {q["id"]: q for q in json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))}

    if args.compare:
        (a, _), (b, _) = (load_summary(results_dir, n, questions) for n in args.compare)
        table = compare_table(args.compare[0], a, args.compare[1], b)
        path = results_dir / f"compare_{args.compare[0]}_vs_{args.compare[1]}.md"
        path.write_text(f"# {args.compare[0]} vs {args.compare[1]}\n\n{table}", encoding="utf-8")
        print(table + f"\nWritten to {path}")
        return 0

    s, manifest = load_summary(results_dir, args.name, questions)
    path = results_dir / args.name / "report.md"
    path.write_text(render_report(args.name, s, manifest), encoding="utf-8")
    print_summary(args.name, s)
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
