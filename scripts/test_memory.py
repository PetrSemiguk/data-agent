"""Checks for agent.memory: dataset fingerprint, notes in SQLite, limits (no API calls)."""

import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Make the project root importable when running this file as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.config import MEMORY_MAX_CHARS, MEMORY_MAX_NOTES  # noqa: E402
from agent.memory import MemoryFullError, MemoryStore, dataset_fingerprint, memory_context  # noqa: E402
from agent.sandbox import Sandbox  # noqa: E402

DATASET = PROJECT_ROOT / "data" / "Titanic-Dataset.csv"

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


def fingerprint_of(sb: Sandbox, path: Path) -> str:
    sb.load_dataframe(path)
    return dataset_fingerprint(sb)


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    try:
        # Variants of the same file, written with pandas so only the intended change differs
        data = pd.read_csv(DATASET)
        renamed = tmp / "completely_other_name.csv"
        shutil.copy(DATASET, renamed)
        changed_value = tmp / "changed_value.csv"
        edited = data.copy()
        edited.loc[0, "Fare"] = edited.loc[0, "Fare"] + 1
        edited.to_csv(changed_value, index=False)
        changed_tail = tmp / "changed_tail.csv"
        edited = data.copy()
        edited.loc[len(edited) - 1, "Name"] = "Someone Else"
        edited.to_csv(changed_tail, index=False)
        one_row_less = tmp / "one_row_less.csv"
        data.iloc[:-1].to_csv(one_row_less, index=False)

        with Sandbox() as sb:
            base = fingerprint_of(sb, DATASET)
            again = fingerprint_of(sb, DATASET)
            report("a) fingerprint stable on reload", base == again and len(base) == 16, f"{base} == {again}")
            other = fingerprint_of(sb, renamed)
            report("b) independent of the file name", other == base, f"{renamed.name}: {other}")
            first_row = fingerprint_of(sb, changed_value)
            last_row = fingerprint_of(sb, changed_tail)
            report("c) changes with a value in the sample", first_row != base and last_row != base,
                   f"first row Fare +1: {first_row}; last row Name: {last_row}")
            fewer = fingerprint_of(sb, one_row_less)
            report("d) changes with the row count", fewer != base, f"890 rows: {fewer}")

        store = MemoryStore(tmp / "memory.db")
        fp, other_fp = "fp_test", "fp_other"

        # e) add / list / replace / delete
        first = store.add_note(fp, "data_fact", "There are 177 missing ages.", "isna().sum() = 177", "agent", "run1")
        second = store.add_note(fp, "definition", "Survival rate means mean of Survived.", None, "user", "run1")
        store.add_note(other_fp, "preference", "Answer in short bullets.", None, "user", "run1")
        notes = store.list_notes(fp)
        replaced = store.replace_note(first, fp, "data_fact", "Age is missing in 177 of 891 rows.",
                                      "isna().sum() = 177", "agent", "run2")
        after = {n.id: n for n in store.list_notes(fp)}
        wrong_dataset = store.replace_note(first, other_fp, "data_fact", "x", "y", "agent", "run3")
        report("e) add, list, replace keeps the id",
               len(notes) == 2 and replaced and not wrong_dataset
               and after[first].text == "Age is missing in 177 of 891 rows." and after[first].run_id == "run2"
               and after[first].created_at == notes[-1].created_at and set(after) == {first, second},
               f"ids={sorted(after)}, #{first} -> {after[first].text!r}, other dataset's replace={wrong_dataset}")

        # f) delete one, forget a dataset (the other dataset keeps its notes)
        deleted = store.delete_note(second)
        forgotten = store.forget_dataset(fp)
        report("f) delete and forget",
               deleted and forgotten == 1 and store.list_notes(fp) == [] and len(store.list_notes(other_fp)) == 1,
               f"deleted #{second}: {deleted}, forgotten: {forgotten}, other dataset: {len(store.list_notes(other_fp))} note")

        # g) at most MEMORY_MAX_NOTES per dataset
        for i in range(MEMORY_MAX_NOTES):
            store.add_note(fp, "data_fact", f"Fact {i}.", f"value {i}", "agent", "run")
        try:
            store.add_note(fp, "data_fact", "One too many.", "x", "agent", "run")
            full = False
        except MemoryFullError:
            full = True
        report("g) note limit", full and len(store.list_notes(fp)) == MEMORY_MAX_NOTES,
               f"{len(store.list_notes(fp))} notes, the next one raised MemoryFullError: {full}")

        # h) the context block has ids, fits MEMORY_MAX_CHARS and says what was left out
        long_fp = "fp_long"
        for i in range(MEMORY_MAX_NOTES):
            store.add_note(long_fp, "data_fact", f"Fact number {i}: " + "x" * 150, "e" * 200, "agent", "run")
        block = memory_context(store.list_notes(long_fp)) or ""
        report("h) context block within the limit",
               block.startswith("[Memory for this dataset, written by the app") and len(block) <= MEMORY_MAX_CHARS
               and "#" in block and "older notes not shown" in block and memory_context([]) is None,
               f"{len(block)} chars (limit {MEMORY_MAX_CHARS}); last line: {block.splitlines()[-1]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\nTotal: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
