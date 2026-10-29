"""Checks for evals/run_evals.py with a fake Claude client (no real API calls)."""

import io
import json
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

# Make the project root and evals/ importable when running this file as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evals"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import agent.tools  # noqa: E402
import run_evals  # noqa: E402
from test_loop import FakeClient, make_response, text, tool_use  # noqa: E402

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


def run(argv: list[str], client: Any = None, answer: str = "y") -> tuple[int, str]:
    """Run run_evals.main and capture what it prints."""
    out = io.StringIO()
    with redirect_stdout(out):
        code = run_evals.main(argv, client=client, input_fn=lambda prompt: answer)
    return code, out.getvalue()


def answering_client(**usage: int) -> FakeClient:
    """Answers every question at once, without tools."""
    return FakeClient(repeat=make_response([text("There are 891 passengers.")], "end_turn", **usage))


def main() -> int:
    root = Path(tempfile.mkdtemp())
    base = ["--results-dir", str(root), "--no-verify"]
    try:
        # a) The final split is refused without the extra flag
        code, out = run(["--split", "final", *base, "--name", "f"])
        report("a) final refused without flag", code == 2 and "Refusing" in out and not (root / "f").exists(),
               out.strip().splitlines()[0])

        # b) The plan is printed and nothing runs when the user does not confirm
        client = answering_client()
        code, out = run(["--split", "dev", "--ids", "d01,d02", "--repeats", "2", *base, "--name", "plan"], client, "n")
        report("b) plan, then cancel",
               code == 1 and "2 questions x 2 repeats = 4 runs" in out and "estimated cost: ~$0.40" in out
               and "default constant" in out and client.calls == 0 and not (root / "plan").exists(),
               " | ".join(line.strip() for line in out.splitlines()[:3]))

        # c) Results go to one file per run, with the manifest; memory is off in every run
        client = answering_client(tokens_in=1000, tokens_out=100, cache_write=500, cache_read=2000)
        code, out = run(["--split", "dev", "--ids", "d01", "--repeats", "2", *base, "--name", "c", "--yes"], client)
        out_dir = root / "c"
        files = sorted(p.name for p in out_dir.glob("*.json"))
        record = json.loads((out_dir / "d01__r2.json").read_text(encoding="utf-8"))
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        agent_log = json.loads(Path(record["log_path"]).read_text(encoding="utf-8"))
        report("c) results, manifest, memory off",
               code == 0 and files == ["d01__r1.json", "d01__r2.json", "manifest.json"]
               and record["id"] == "d01" and record["repeat"] == 2 and record["status"] == "completed"
               and record["final_answer"] == "There are 891 passengers."
               and record["tokens"] == {"input": 1000, "cache_write": 500, "cache_read": 2000, "output": 100}
               and abs(record["cost"]["total"] - (1000 * 2 + 500 * 2.5 + 2000 * 0.1 + 100 * 10) / 1e6) < 1e-12
               and manifest["config"]["memory"] is False and manifest["ids"] == ["d01"] and manifest["repeats"] == 2
               and manifest["git_commit"] and agent_log["memory"] is None
               and all("remember" not in tools for tools in client.tools),
               f"files={files}, cost=${record['cost']['total']:.5f}, commit={manifest['git_commit'][:12]}")

        # d) Running again with the same name skips finished runs
        client = answering_client()
        code, out = run(["--split", "dev", "--ids", "d01", "--repeats", "2", *base, "--name", "c", "--yes"], client)
        report("d) resume skips finished runs", code == 0 and client.calls == 0 and "Nothing to do" in out,
               out.strip().splitlines()[1])

        # e) The cost limit stops the run, keeping what was done
        client = answering_client(tokens_in=250_000)  # about $0.50 per run
        code, out = run(["--split", "dev", "--ids", "d01", "--repeats", "3", *base, "--name", "e",
                         "--max-cost", "0.8", "--yes"], client)
        done = sorted(p.name for p in (root / "e").glob("d01__r*.json"))
        report("e) cost limit stops the run",
               code == 0 and done == ["d01__r1.json"] and "Stopping: $0.500 spent" in out,
               next(line for line in out.splitlines() if line.startswith("Stopping")))

        # f) Code that reads the eval set is blocked by the guard (and still flagged); reading data/ is not
        client = FakeClient([
            make_response([tool_use("import json\nq = json.load(open('evals/questions.json'))\nlen(q)")], "tool_use"),
            make_response([text("891.")], "end_turn"),
            make_response([tool_use("import pandas as pd\nlen(pd.read_csv('data/Titanic-Dataset.csv'))")], "tool_use"),
            make_response([text("891.")], "end_turn"),
        ])
        code, out = run(["--split", "dev", "--ids", "d01", "--repeats", "2", *base, "--name", "f2", "--yes"], client)
        bad = json.loads((root / "f2" / "d01__r1.json").read_text(encoding="utf-8"))
        good = json.loads((root / "f2" / "d01__r2.json").read_text(encoding="utf-8"))
        blocked_output = json.loads(Path(bad["log_path"]).read_text(encoding="utf-8"))["steps"][0]["tool_calls"][0]["output"]
        report("f) guard blocks reading the eval set",
               bad["guard_blocks"] == 1 and "Reading files outside the dataset folder is blocked" in blocked_output
               and bad["integrity_violation"] and "evals/questions.json" in bad["integrity_quotes"][0]
               and good["guard_blocks"] == 0 and not good["integrity_violation"] and "INTEGRITY VIOLATION" in out,
               f"guard: {blocked_output.splitlines()[1]}; integrity quote: {bad['integrity_quotes']}")

        # h) With the guard switched off, the integrity check is still the second line of defense
        guard = agent.tools.check_code
        agent.tools.check_code = lambda code, readable_dirs=None: None
        try:
            client = FakeClient([
                make_response([tool_use("import os\nos.listdir('.')")], "tool_use"),
                make_response([text("891.")], "end_turn"),
            ])
            code, out = run(["--split", "dev", "--ids", "d01", *base, "--name", "h", "--yes"], client)
        finally:
            agent.tools.check_code = guard
        record = json.loads((root / "h" / "d01__r1.json").read_text(encoding="utf-8"))
        report("h) guard off -> integrity check still catches",
               record["guard_blocks"] == 0 and record["integrity_violation"]
               and "os.listdir('.')" in record["integrity_quotes"][0],
               f"guard blocks={record['guard_blocks']}, quotes={record['integrity_quotes']}")

        # g) The checker on its own: browsing folders and reading outside data/ are flagged
        log = {"steps": [{"step": 1, "tool_calls": [
            {"name": "run_code", "input": {"code": "import os\nos.listdir('.')"}},
            {"name": "run_code", "input": {"code": "open('C:/Users/someone/notes.txt').read()"}},
            {"name": "run_code", "input": {"code": "retrievals = df.head()\nopen('data/x.csv')"}},
        ]}]}
        quotes = run_evals.integrity_check(log)
        report("g) browsing and outside reads flagged", len(quotes) == 2
               and "os.listdir('.')" in quotes[0] and "notes.txt" in quotes[1], f"{quotes}")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"\nTotal: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
