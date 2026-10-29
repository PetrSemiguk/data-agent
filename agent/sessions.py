"""Visitor sessions on a shared server: folders, kernels, limits and API keys.

Each browser session gets a random id. In public mode its uploads, outputs and
logs live in folders named after that id, old folders are deleted, and only a
few kernels may run at once; idle ones are closed and started again on demand.
"""

from __future__ import annotations

import logging
import re
import secrets
import shutil
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import anthropic

from agent.config import LOGS_DIR, OUTPUTS_DIR
from agent.loop import RunDirs
from agent.redact import API_KEY_PATTERN, SecretFilter
from agent.sandbox import Sandbox

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOADS_ROOT = PROJECT_ROOT / "uploads"
DEMO_DIR = PROJECT_ROOT / "demo"
# The folders that get one subfolder per session.
SESSION_ROOTS = (UPLOADS_ROOT, PROJECT_ROOT / OUTPUTS_DIR, PROJECT_ROOT / LOGS_DIR)
SESSION_PREFIX = "session_"
SESSION_ID = re.compile(rf"{SESSION_PREFIX}[0-9a-f]{{32}}")

logger = logging.getLogger(__name__)
logger.addFilter(SecretFilter())


def new_session_id() -> str:
    return SESSION_PREFIX + secrets.token_hex(16)


@dataclass(frozen=True)
class SessionDirs:
    """Where one session reads uploads from and writes runs to."""

    uploads: Path
    runs: RunDirs


def session_dirs(session_id: str | None) -> SessionDirs:
    """The session's own folders, or the shared ones for None (local mode).

    A session's code may read only its own uploads and the demo dataset.
    """
    if session_id is None:
        return SessionDirs(UPLOADS_ROOT, RunDirs())
    uploads, outputs, logs = (root / session_id for root in SESSION_ROOTS)
    return SessionDirs(uploads, RunDirs(outputs=outputs, logs=logs, readable=(uploads, DEMO_DIR)))


def last_activity(folder: Path) -> float:
    """The newest modification time of the folder and everything in it."""
    times = [folder.stat().st_mtime]
    times += [path.stat().st_mtime for path in folder.rglob("*")]
    return max(times)


def cleanup_sessions(
    ttl_hours: float,
    keep: Iterable[str] = (),
    roots: Iterable[Path] = SESSION_ROOTS,
    now: float | None = None,
) -> list[Path]:
    """Delete session folders with no activity for `ttl_hours`; return them.

    Only folders named like a session id are touched, never those in `keep`
    (sessions with a running kernel).
    """
    now = time.time() if now is None else now
    keep = set(keep)
    deleted = []
    for root in roots:
        if not root.is_dir():
            continue
        for folder in root.iterdir():
            if not folder.is_dir() or not SESSION_ID.fullmatch(folder.name) or folder.name in keep:
                continue
            try:
                if now - last_activity(folder) > ttl_hours * 3600:
                    shutil.rmtree(folder)
                    deleted.append(folder)
            except OSError:
                logger.warning("Could not clean up %s", folder, exc_info=True)
    return deleted


def upload_error(size_bytes: int, limit_mb: int | None) -> str | None:
    """A message for a file over the upload limit, else None."""
    if limit_mb is not None and size_bytes > limit_mb * 1024 * 1024:
        return f"The file is {size_bytes / 1024 / 1024:.1f} MB; the limit is {limit_mb} MB. Upload a smaller file."
    return None


def check_api_key(api_key: str) -> str | None:
    """Check a visitor's key without spending tokens; a message if it is unusable.

    Listing models is a metadata request: it needs a valid key but runs no model.
    """
    if not API_KEY_PATTERN.fullmatch(api_key):
        return "This does not look like an Anthropic API key (it starts with sk-ant-)."
    client = anthropic.Anthropic(api_key=api_key, max_retries=1, timeout=15)
    try:
        client.models.list(limit=1)
    except anthropic.AuthenticationError:
        return "The key was rejected: it is invalid, revoked or disabled."
    except anthropic.PermissionDeniedError:
        return "This key is not allowed to use the API."
    except anthropic.APIConnectionError:
        return "Could not reach the Anthropic API to check the key. Try again."
    except anthropic.APIStatusError as exc:
        return f"The key could not be checked (HTTP {exc.status_code}). Try again later."
    return None


class ServerBusy(Exception):
    """All kernel slots are taken by other sessions."""


@dataclass
class _Kernel:
    sandbox: Sandbox
    last_used: float
    busy: int = 0


class KernelPool:
    """One kernel per session, with an optional limit and idle timeout.

    `max_active` None means no limit, `idle_seconds` None means kernels are
    never closed for being idle. `factory` and `clock` are replaced in tests.
    """

    def __init__(
        self,
        max_active: int | None = None,
        idle_seconds: float | None = None,
        factory: Callable[[], Sandbox] = Sandbox,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_active = max_active
        self.idle_seconds = idle_seconds
        self._factory = factory
        self._clock = clock
        self._kernels: dict[str, _Kernel] = {}
        self._lock = threading.Lock()

    @contextmanager
    def using(self, session_id: str) -> Iterator[tuple[Sandbox, bool]]:
        """The session's kernel while it is used; True if it was just started.

        A kernel in use is never closed as idle. Raises ServerBusy when a new
        kernel is needed and every slot is taken.
        """
        with self._lock:
            self._close_idle_locked()
            kernel = self._kernels.get(session_id)
            fresh = kernel is None
            if kernel is None:
                if self.max_active is not None and len(self._kernels) >= self.max_active:
                    raise ServerBusy("The server is busy: too many active sessions. Try again in a few minutes.")
                sandbox = self._factory()
                sandbox.start()
                kernel = _Kernel(sandbox, self._clock())
                self._kernels[session_id] = kernel
            kernel.busy += 1
        try:
            yield kernel.sandbox, fresh
        finally:
            with self._lock:
                kernel.busy -= 1
                kernel.last_used = self._clock()

    def close_idle(self) -> list[str]:
        """Close kernels unused for `idle_seconds`; return their session ids."""
        with self._lock:
            return self._close_idle_locked()

    def active_sessions(self) -> set[str]:
        with self._lock:
            return set(self._kernels)

    def close(self, session_id: str) -> None:
        with self._lock:
            kernel = self._kernels.pop(session_id, None)
        if kernel is not None:
            kernel.sandbox.shutdown()

    def close_all(self) -> None:
        for session_id in self.active_sessions():
            self.close(session_id)

    def _close_idle_locked(self) -> list[str]:
        if self.idle_seconds is None:
            return []
        now = self._clock()
        idle = [
            session_id
            for session_id, kernel in self._kernels.items()
            if not kernel.busy and now - kernel.last_used > self.idle_seconds
        ]
        for session_id in idle:
            self._kernels.pop(session_id).sandbox.shutdown()
        return idle


def start_housekeeping(pool: KernelPool, ttl_hours: float | None, interval_seconds: float) -> threading.Thread:
    """Clean up now, then every `interval_seconds` in a background thread."""

    def clean() -> None:
        pool.close_idle()
        if ttl_hours is not None:
            cleanup_sessions(ttl_hours, keep=pool.active_sessions())

    def loop() -> None:
        while True:
            time.sleep(interval_seconds)
            try:
                clean()
            except Exception:
                logger.exception("Housekeeping failed")

    clean()
    thread = threading.Thread(target=loop, name="session-housekeeping", daemon=True)
    thread.start()
    return thread
