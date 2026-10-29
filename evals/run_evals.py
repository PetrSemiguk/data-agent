"""Run the agent on the eval questions and save one result file per run.

Run from the project root, e.g.:
    .venv\\Scripts\\python.exe evals\\run_evals.py --split dev --ids d01,d06 --name smoke

How every question is run:
- a new sandbox (kernel) per question and repeat, so no run sees variables of another;
- memory is always OFF: notes saved by one eval run would leak facts (and sometimes
  answers) into the next, so results would depend on the order of the runs and
  repeats would not be independent;
- no human: ask_user is answered "user not available" and approvals (exports) are
  denied automatically;
- everything else is the production path: the same run_agent with skills, the
  verifier and prompt caching unless a flag turns them off.

Grading the answers is a separate step; this script only collects them.
"""

import argparse
import ast
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

from agent.loop import RunResult, create_client, run_agent  # noqa: E402
from agent.sandbox import Sandbox  # noqa: E402
from agent.skills_loader import load_skills  # noqa: E402
from agent.usage import format_cost  # noqa: E402

QUESTIONS_FILE = EVALS_DIR / "questions.json"
RESULTS_DIR = EVALS_DIR / "results"
DATA_DIR = PROJECT_ROOT / "data"
MODEL = "claude-sonnet-5-5"
# Used when no earlier results with the same settings exist. Measured on Superstore
# and Titanic with caching, skills and the verifier on: $0.04-0.10 per run
# (logs/run_cost.txt); the upper end is taken so the estimate errs on the safe side.
DEFAULT_COST_PER_RUN = 0.10
FINAL_FLAG = "--i-understand-this-is-final"
GUARD_BLOCKED = "Blocked before running"

# Model code that touches the eval set, or reads files outside data/, is cheating.
# "evals" only as a whole word, so names like "retrievals" do not match.
EVAL_PATH = re.compile(r"\bevals\b|questions\.json", re.IGNORECASE)
FILE_READERS = {"open", "read_csv", "read_excel", "read_json", "read_parquet", "read_table", "loadtxt", "genfromtxt", "Path"}
FILE_BROWSERS = {"listdir", "scandir", "walk", "glob", "iglob", "rglob", "iterdir"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agent on the eval questions.")
    parser.add_argument("--split", required=True, choices=["dev", "holdout", "final"])
    parser.add_argument(FINAL_FLAG, dest="final_confirmed", action="store_true",
                        help="Required for --split final: the final split is run once, at the very end")
    parser.add_argument("--ids", help="Comma-separated question ids, e.g. d01,d06 (default: the whole split)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--name", default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    parser.add_argument("--no-skills", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--verifier-model", default=None, help="Model of the verifier (default: the agent's model)")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--max-cost", type=float, default=3.0, help="Hard limit in USD for the whole run")
    parser.add_argument("--yes", action="store_true", help="Start without asking for confirmation")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def select_questions(split: str, ids: str | None) -> list[dict[str, Any]]:
    questions = [q for q in json.loads(QUESTIONS_FILE.read_text(encoding="utf-8")) if q["split"] == split]
    if ids:
        wanted = [i.strip() for i in ids.split(",") if i.strip()]
        known = {q["id"] for q in questions}
        unknown = [i for i in wanted if i not in known]
        if unknown:
            raise SystemExit(f"Unknown ids for split {split}: {', '.join(unknown)}")
        questions = [q for q in questions if q["id"] in wanted]
    return questions


def config_of(args: argparse.Namespace) -> dict[str, Any]:
    """The settings that change the results; a resumed run must use the same ones."""
    return {
        "model": MODEL,
        "verifier_model": None if args.no_verify else (args.verifier_model or MODEL),
        "skills": not args.no_skills,
        "verify": not args.no_verify,
        "cache": not args.no_cache,
        "memory": False,
        "human": False,
    }


def git_commit() -> str:
    """Current commit, with '+dirty' if there are uncommitted changes."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                                cwd=PROJECT_ROOT).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               cwd=PROJECT_ROOT).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def estimate_cost_per_run(results_root: Path, config: dict[str, Any]) -> tuple[float, str]:
    """Average $ per run from earlier result files with the same settings, else the constant."""
    costs = []
    for manifest_path in results_root.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config") != config:
            continue
        for result in manifest_path.parent.glob("*__r*.json"):
            cost = json.loads(result.read_text(encoding="utf-8"))["cost"]["total"]
            if cost is not None:
                costs.append(cost)
    if costs:
        return sum(costs) / len(costs), f"average of {len(costs)} earlier runs with the same settings"
    return DEFAULT_COST_PER_RUN, "default constant (no earlier runs with the same settings)"


def integrity_check(log: dict[str, Any]) -> list[str]:
    """Quotes of model code that reads the eval set or files outside data/."""
    quotes = []
    for step in log.get("steps", []):
        for call in step.get("tool_calls", []):
            code = call.get("input", {}).get("code") if call.get("name") in ("run_code", "save_chart") else None
            if isinstance(code, str):
                quotes += [f"step {step['step']} {call['name']}: {q}" for q in suspicious_lines(code)]
    return quotes


def suspicious_lines(code: str) -> list[str]:
    found = [line.strip() for line in code.splitlines() if EVAL_PATH.search(line)]
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return found
    lines = code.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
        literal_paths = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        outside_data = [p for p in literal_paths if not is_in_data(p)]
        if name in FILE_BROWSERS or (name in FILE_READERS and outside_data):
            line = lines[node.lineno - 1].strip()
            if line not in found:
                found.append(line)
    return found


def is_in_data(path: str) -> bool:
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else PROJECT_ROOT / candidate).resolve()
    return resolved == DATA_DIR or DATA_DIR in resolved.parents


def tool_counts(log: dict[str, Any]) -> tuple[int, int]:
    """(failed tool calls, calls blocked by the guard) of the agent, not the verifier."""
    calls = [c for step in log.get("steps", []) for c in step.get("tool_calls", [])]
    errors = sum(bool(c.get("is_error")) for c in calls)
    blocked = sum(str(c.get("output", "")).startswith(GUARD_BLOCKED) for c in calls)
    return errors, blocked


def record_of(question: dict[str, Any], repeat: int, result: RunResult) -> dict[str, Any]:
    log = json.loads(Path(result.log_path or "").read_text(encoding="utf-8")) if result.log_path else {}
    errors, blocked = tool_counts(log)
    quotes = integrity_check(log)
    report = result.verification
    verify_cost = report.cost_usd if report else 0.0
    # None when a model has no prices in config.py; the run then has an unknown cost.
    agent_cost = None if result.cost_usd is None or verify_cost is None else result.cost_usd - verify_cost
    return {
        "id": question["id"],
        "repeat": repeat,
        "split": question["split"],
        "dataset": question["dataset"],
        "question": question["question"],
        "final_answer": result.final_answer,
        "status": result.status,
        "error_message": result.error_message,
        "steps": result.steps,
        "tokens": {
            "input": result.input_tokens,
            "cache_write": result.cache_write_tokens,
            "cache_read": result.cache_read_tokens,
            "output": result.output_tokens,
        },
        "cost": {"agent": agent_cost, "verification": verify_cost, "total": result.cost_usd},
        "duration_s": result.duration_s,
        "loaded_skills": result.loaded_skills,
        "plan_incomplete": result.plan_incomplete,
        "verification": {
            "status": report.status,
            "counts": report.counts,
            "revised": result.revised,
            "revision_failed": result.revision_failed,
        } if report else None,
        "tool_errors": errors,
        "guard_blocks": blocked,
        "log_path": result.log_path,
        "integrity_violation": bool(quotes),
        "integrity_quotes": quotes,
    }


def run_one(
    question: dict[str, Any],
    repeat: int,
    args: argparse.Namespace,
    client: Any,
    skills: dict[str, Any] | None,
    sandbox_factory: Callable[[], Any],
) -> dict[str, Any]:
    """One question in a fresh sandbox, as in production but without memory or a human."""
    with sandbox_factory() as sandbox:
        sandbox.load_dataframe(DATA_DIR / question["dataset"])
        result = run_agent(
            question["question"],
            sandbox,
            client,
            model=MODEL,
            dataset_file=question["dataset"],
            ask_user_fn=None,
            approve_fn=None,
            memory=None,
            skills=skills,
            verify=not args.no_verify,
            verifier_model=args.verifier_model,
            cache=not args.no_cache,
        )
    return record_of(question, repeat, result)


def main(
    argv: list[str] | None = None,
    client: Any = None,
    sandbox_factory: Callable[[], Any] = Sandbox,
    input_fn: Callable[[str], str] = input,
) -> int:
    args = parse_args(argv)
    if args.split == "final" and not args.final_confirmed:
        print(
            "Refusing to run the final split. It is the exam: run it once, at the very end, after "
            f"all changes. If that is now, add {FINAL_FLAG}."
        )
        return 2
    questions = select_questions(args.split, args.ids)
    config = config_of(args)
    results_root = Path(args.results_dir)
    out_dir = results_root / args.name
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("config") != config or old.get("split") != args.split:
            print(f"Results '{args.name}' were made with other settings; use another --name.")
            return 2

    runs = [(q, r) for q in questions for r in range(1, args.repeats + 1)]
    done = {(q["id"], r) for q, r in runs if (out_dir / f"{q['id']}__r{r}.json").exists()}
    todo = [(q, r) for q, r in runs if (q["id"], r) not in done]
    per_run, basis = estimate_cost_per_run(results_root, config)

    print(f"Eval run '{args.name}': split {args.split}, {len(questions)} questions x {args.repeats} repeats = {len(runs)} runs")
    if done:
        print(f"  {len(done)} runs already done (results found), {len(todo)} to go")
    print(f"  config: {json.dumps(config)}")
    print(f"  estimated cost: ~${per_run * len(todo):.2f} (${per_run:.3f} per run, {basis}); limit ${args.max_cost:.2f}")
    print(f"  results: {out_dir}")
    if not todo:
        print("Nothing to do.")
        return 0
    if not args.yes and input_fn("Start? [y/N]: ").strip().lower() not in ("y", "yes"):
        print("Cancelled.")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": args.name,
        "split": args.split,
        "ids": [q["id"] for q in questions],
        "repeats": args.repeats,
        "config": config,
        "git_commit": git_commit(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    load_dotenv(PROJECT_ROOT / ".env")
    client = client or create_client()
    skills = None if args.no_skills else load_skills()[0]
    # A run with an unknown cost (no prices for its model) cannot count towards the limit.
    spent = sum(
        json.loads((out_dir / f"{i}__r{r}.json").read_text(encoding="utf-8"))["cost"]["total"] or 0.0 for i, r in done
    )
    finished = 0
    for question, repeat in todo:
        # Stop before a run that would likely cross the limit, using the average so far.
        expected = spent / (len(done) + finished) if done or finished else per_run
        if spent + expected > args.max_cost:
            print(f"Stopping: ${spent:.3f} spent, the next run (~${expected:.3f}) would exceed --max-cost ${args.max_cost:.2f}.")
            break
        start = time.perf_counter()
        record = run_one(question, repeat, args, client, skills, sandbox_factory)
        path = out_dir / f"{question['id']}__r{repeat}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        spent += record["cost"]["total"] or 0.0
        if record["cost"]["total"] is None:
            print("  Warning: no prices for a model of this run in config.py; --max-cost cannot count its cost.")
        finished += 1
        flag = "  INTEGRITY VIOLATION" if record["integrity_violation"] else ""
        print(
            f"[{finished}/{len(todo)}] {question['id']} r{repeat}: {record['status']}, {record['steps']} steps, "
            f"{format_cost(record['cost']['total'], 3)}, {time.perf_counter() - start:.1f}s, total ${spent:.3f}{flag}",
            flush=True,
        )

    print_summary(out_dir)
    return 0


def print_summary(out_dir: Path) -> None:
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("*__r*.json"))]
    print(f"\n{'id':4} {'rep':>3} {'status':16} {'steps':>5} {'$':>7} {'time,s':>7} {'revised':8} skills")
    for r in records:
        revised = "-" if not r["verification"] else ("failed" if r["verification"]["revision_failed"] else str(r["verification"]["revised"]))
        flag = "  INTEGRITY VIOLATION" if r["integrity_violation"] else ""
        print(f"{r['id']:4} {r['repeat']:>3} {r['status']:16} {r['steps']:>5} {format_cost(r['cost']['total'], 3):>7} "
              f"{r['duration_s']:>7.1f} {revised:8} {','.join(r['loaded_skills']) or '-'}{flag}")
    total = sum(r["cost"]["total"] or 0.0 for r in records)
    print(f"Total: {len(records)} runs, ${total:.3f}")


if __name__ == "__main__":
    sys.exit(main())
