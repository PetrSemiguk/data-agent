"""Local or public mode: every difference between them is decided here.

local (default): one person on their own computer; the API key comes from .env.
public: many visitors on one server; each brings their own API key, and
sessions do not share memory, files or kernels.
Set with the DATA_AGENT_MODE environment variable.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from agent.config import (
    IDLE_KERNEL_MINUTES,
    MAX_ACTIVE_SESSIONS,
    MAX_UPLOAD_MB,
    SESSION_TTL_HOURS,
)

MODE_VARIABLE = "DATA_AGENT_MODE"
# Credentials the SDK would read from the environment when no key is passed.
SERVER_CREDENTIAL_VARIABLES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


@dataclass(frozen=True)
class Mode:
    name: str
    # The visitor enters a key in the page; the server's key is never used.
    visitor_key: bool
    # Notes about a dataset are kept between sessions (shared by everyone).
    memory: bool
    # Uploads, outputs and logs go to a folder of their own per session.
    session_dirs: bool
    # The "Use demo dataset" button.
    demo_dataset: bool
    # Paths of run logs under answers: useful to the developer, internal details to a visitor.
    show_log_paths: bool
    # The "Use skills" toggle and the list of skills; when hidden, SKILLS_DEFAULT applies.
    show_skills: bool
    # None: Streamlit's own limit (200 MB).
    max_upload_mb: int | None
    # None: no limit.
    max_active_sessions: int | None
    idle_kernel_minutes: float | None
    session_ttl_hours: float | None


LOCAL = Mode(
    name="local",
    visitor_key=False,
    memory=True,
    session_dirs=False,
    demo_dataset=False,
    show_log_paths=True,
    show_skills=True,
    max_upload_mb=None,
    max_active_sessions=None,
    idle_kernel_minutes=None,
    session_ttl_hours=None,
)

PUBLIC = Mode(
    name="public",
    visitor_key=True,
    memory=False,
    session_dirs=True,
    demo_dataset=True,
    show_log_paths=False,
    show_skills=False,
    max_upload_mb=MAX_UPLOAD_MB,
    max_active_sessions=MAX_ACTIVE_SESSIONS,
    idle_kernel_minutes=IDLE_KERNEL_MINUTES,
    session_ttl_hours=SESSION_TTL_HOURS,
)

MODES = {mode.name: mode for mode in (LOCAL, PUBLIC)}


def get_mode() -> Mode:
    """The mode set by DATA_AGENT_MODE; local if it is not set."""
    name = os.environ.get(MODE_VARIABLE, "local").strip().lower()
    if name not in MODES:
        raise ValueError(f"{MODE_VARIABLE} must be one of {', '.join(MODES)}, not {name!r}")
    return MODES[name]


def load_server_environment(mode: Mode) -> None:
    """Local: read .env. Visitor keys: remove the server's key from this process.

    The client always gets the visitor's key explicitly; removing the server's
    makes sure no code path can fall back to it.
    """
    if not mode.visitor_key:
        load_dotenv()
        return
    for name in SERVER_CREDENTIAL_VARIABLES:
        os.environ.pop(name, None)
