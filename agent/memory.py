"""Memory between sessions: short notes about a dataset, kept in SQLite.

Notes belong to a dataset fingerprint, not to a file name, so the same data
uploaded again (under any name) gets its notes back.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from agent.config import FINGERPRINT_SAMPLE_ROWS, MEMORY_DB, MEMORY_MAX_CHARS, MEMORY_MAX_NOTES
from agent.sandbox import Sandbox

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NoteKind = Literal["data_fact", "definition", "preference"]
NOTE_KINDS: tuple[str, ...] = ("data_fact", "definition", "preference")
MEMORY_EVIDENCE_CHARS = 150  # evidence shown in the context, per note

# Runs in the kernel, where the data lives. Hashes the column names and dtypes,
# the number of rows and the values of the first and last rows: fast on 1M
# rows and independent of the file name.
FINGERPRINT_CODE = f'''
def _agent_fingerprint():
    import hashlib
    import pandas as pd

    digest = hashlib.sha256()
    digest.update(repr([(str(c), str(t)) for c, t in df.dtypes.items()]).encode())
    digest.update(str(len(df)).encode())
    sample = pd.concat([df.head({FINGERPRINT_SAMPLE_ROWS}), df.tail({FINGERPRINT_SAMPLE_ROWS})])
    digest.update(pd.util.hash_pandas_object(sample, index=False).values.tobytes())
    return digest.hexdigest()[:16]
'''


def dataset_fingerprint(sandbox: Sandbox) -> str:
    """Fingerprint of `df` in the kernel.

    Changes when columns, dtypes, the row count or any value in the first or
    last FINGERPRINT_SAMPLE_ROWS rows change. An edit in the middle of the data
    that keeps the row count does NOT change it: the old notes are then shown
    for slightly different data, and the agent is told to report and update a
    note that the data contradicts.
    """
    result = sandbox.execute(f"{FINGERPRINT_CODE}\nprint(_agent_fingerprint())\ndel _agent_fingerprint")
    if not result.success:
        raise RuntimeError(f"Could not fingerprint the dataset:\n{result.error}")
    return result.stdout.strip()


@dataclass(frozen=True)
class Note:
    id: int
    dataset_fingerprint: str
    kind: str
    text: str
    evidence: str | None
    source: str
    run_id: str | None
    created_at: str
    updated_at: str


class MemoryFullError(Exception):
    """The dataset already has MEMORY_MAX_NOTES notes."""


class MemoryStore:
    """Notes in an SQLite file. Each call opens its own connection, so one
    store can be shared by Streamlit's script threads."""

    def __init__(self, path: str | Path = PROJECT_ROOT / MEMORY_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_fingerprint TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    evidence TEXT,
                    source TEXT NOT NULL,
                    run_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS notes_dataset ON notes (dataset_fingerprint)")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # closing() closes the connection; the inner `with` commits or rolls back.
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.row_factory = sqlite3.Row
            yield conn

    def list_notes(self, fingerprint: str) -> list[Note]:
        """Notes of a dataset, most recently updated first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM notes WHERE dataset_fingerprint = ? ORDER BY updated_at DESC, id DESC",
                (fingerprint,),
            ).fetchall()
        return [Note(**dict(row)) for row in rows]

    def add_note(
        self,
        fingerprint: str,
        kind: str,
        text: str,
        evidence: str | None,
        source: str,
        run_id: str | None,
    ) -> int:
        """Add a note and return its id; MemoryFullError if the dataset has no room."""
        now = timestamp()
        with self._connect() as conn:
            (count,) = conn.execute(
                "SELECT COUNT(*) FROM notes WHERE dataset_fingerprint = ?", (fingerprint,)
            ).fetchone()
            if count >= MEMORY_MAX_NOTES:
                raise MemoryFullError(f"This dataset already has {count} notes (limit {MEMORY_MAX_NOTES}).")
            cursor = conn.execute(
                "INSERT INTO notes (dataset_fingerprint, kind, text, evidence, source, run_id, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (fingerprint, kind, text, evidence, source, run_id, now, now),
            )
            return int(cursor.lastrowid or 0)

    def replace_note(
        self,
        note_id: int,
        fingerprint: str,
        kind: str,
        text: str,
        evidence: str | None,
        source: str,
        run_id: str | None,
    ) -> bool:
        """Update a note of this dataset in place (same id); False if there is no such note."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE notes SET kind = ?, text = ?, evidence = ?, source = ?, run_id = ?, updated_at = ?"
                " WHERE id = ? AND dataset_fingerprint = ?",
                (kind, text, evidence, source, run_id, timestamp(), note_id, fingerprint),
            )
            return cursor.rowcount == 1

    def delete_note(self, note_id: int) -> bool:
        with self._connect() as conn:
            return conn.execute("DELETE FROM notes WHERE id = ?", (note_id,)).rowcount == 1

    def forget_dataset(self, fingerprint: str) -> int:
        """Delete every note of a dataset; returns how many were deleted."""
        with self._connect() as conn:
            return conn.execute("DELETE FROM notes WHERE dataset_fingerprint = ?", (fingerprint,)).rowcount


@dataclass(frozen=True)
class DatasetMemory:
    """The memory of one dataset, as a run uses it."""

    store: MemoryStore
    fingerprint: str


def memory_context(notes: list[Note]) -> str | None:
    """The notes as one block for the model, within MEMORY_MAX_CHARS; None if there are none.

    Newest notes come first, so older ones are the ones left out when the
    block is full.
    """
    if not notes:
        return None
    header = (
        "[Memory for this dataset, written by the app: notes from earlier sessions. "
        "To correct or refine a note, call remember with replaces=<id>."
    )
    lines: list[str] = []
    used = len(header) + 1
    for note in notes:
        evidence = f" (evidence: {note.evidence[:MEMORY_EVIDENCE_CHARS]})" if note.evidence else ""
        line = f"- #{note.id} {note.kind}, from {note.source}: {note.text}{evidence}"
        if used + len(line) + 1 > MEMORY_MAX_CHARS:
            lines.append(f"- ... {len(notes) - len(lines)} older notes not shown")
            break
        lines.append(line)
        used += len(line) + 1
    return header + "\n" + "\n".join(lines) + "]"


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")
