"""Sandbox that runs agent code in a local Jupyter (IPython) kernel."""

from __future__ import annotations

import queue
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

from jupyter_client.blocking.client import BlockingKernelClient
from jupyter_client.manager import KernelManager

from agent.config import CODE_TIMEOUT_SECONDS, MATPLOTLIB_RC_PARAMS, PANDAS_DISPLAY_OPTIONS

# IPython colors tracebacks with ANSI escape codes; the model does not need them.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# How long to wait for the kernel to become idle after an interrupt.
INTERRUPT_GRACE_SECONDS = 5.0


@dataclass
class ExecutionResult:
    """Everything the kernel produced while running one piece of code."""

    success: bool
    stdout: str = ""
    result: str | None = None
    error: str | None = None
    images: list[str] = field(default_factory=list)


class Sandbox:
    """A persistent Python kernel: variables survive between execute() calls."""

    def __init__(self) -> None:
        self._km: KernelManager | None = None
        self._kc: BlockingKernelClient | None = None
        # Remembered so df can be restored if the kernel has to be restarted.
        self._df_path: Path | None = None

    def start(self) -> None:
        """Start a Python kernel and enable inline matplotlib output."""
        # The "python3" kernelspec comes from ipykernel inside .venv; jupyter_client
        # replaces its "python" command with sys.executable, i.e. the venv Python.
        # Encrypt kernel traffic with CurveZMQ; fail loudly if that is impossible.
        self._km = KernelManager(kernel_name="python3", transport_encryption="required")
        self._km.start_kernel()
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=60)
        self._setup_kernel()

    def load_dataframe(self, path: str | Path) -> ExecutionResult:
        """Load a CSV/XLSX file into the variable `df` inside the kernel."""
        path = Path(path).resolve()
        if path.suffix.lower() == ".xlsx":
            code = f"import pandas as pd\ndf = pd.read_excel({str(path)!r})"
        else:
            code = (
                "import pandas as pd\n"
                "try:\n"
                f"    df = pd.read_csv({str(path)!r}, encoding='utf-8')\n"
                "    print('Encoding: utf-8')\n"
                "except UnicodeDecodeError:\n"
                f"    df = pd.read_csv({str(path)!r}, encoding='latin-1')\n"
                "    print('Encoding: latin-1')\n"
            )
        result = self.execute(code, timeout=120)
        if not result.success:
            raise RuntimeError(f"Failed to load {path.name}:\n{result.error}")
        self._df_path = path
        return result

    def execute(self, code: str, timeout: float = CODE_TIMEOUT_SECONDS) -> ExecutionResult:
        """Run code in the kernel and collect its output until the kernel is idle."""
        kc = self._client()
        msg_id = kc.execute(code)
        deadline = time.monotonic() + timeout

        stdout_parts: list[str] = []
        result: str | None = None
        error: str | None = None
        images: list[str] = []

        while True:
            remaining = deadline - time.monotonic()
            try:
                msg = kc.get_iopub_msg(timeout=max(remaining, 0))
            except queue.Empty:
                if self._interrupt(msg_id):
                    note = "was interrupted. Variables defined earlier are kept."
                else:
                    # Interrupt did not work (e.g. a blocking C call on Windows).
                    self._restart_and_reload()
                    note = (
                        "could not be interrupted, so the kernel was restarted. "
                        "df was reloaded; other variables are lost."
                    )
                return ExecutionResult(
                    success=False,
                    stdout="".join(stdout_parts),
                    error=f"TimeoutError: execution exceeded {timeout} seconds and {note}",
                    images=images,
                )

            # iopub is a broadcast channel: skip messages from other requests.
            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = msg["msg_type"]
            content = msg["content"]

            if msg_type == "stream":
                stdout_parts.append(content["text"])
            elif msg_type == "execute_result":
                result = content["data"].get("text/plain")
                self._collect_image(content["data"], images)
            elif msg_type == "display_data":
                self._collect_image(content["data"], images)
            elif msg_type == "error":
                traceback = ANSI_ESCAPE.sub("", "\n".join(content["traceback"]))
                error = f"{content['ename']}: {content['evalue']}\n{traceback}"
            elif msg_type == "status" and content["execution_state"] == "idle":
                break

        return ExecutionResult(
            success=error is None,
            stdout="".join(stdout_parts),
            result=result,
            error=error,
            images=images,
        )

    def restart(self) -> None:
        """Restart the kernel. All variables, including df, are lost."""
        if self._km is None:
            raise RuntimeError("Sandbox is not started")
        self._km.restart_kernel(now=True)
        self._client().wait_for_ready(timeout=60)
        self._setup_kernel()

    def shutdown(self) -> None:
        """Stop the kernel and close the communication channels."""
        if self._kc is not None:
            self._kc.stop_channels()
            self._kc = None
        if self._km is not None:
            self._km.shutdown_kernel(now=True)
            self._km = None

    def __enter__(self) -> Sandbox:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.shutdown()

    def _client(self) -> BlockingKernelClient:
        if self._kc is None:
            raise RuntimeError("Sandbox is not started")
        return self._kc

    def _setup_kernel(self) -> None:
        # Compact pandas output is the first defense against huge tool results.
        options = "\n".join(
            f"pd.set_option({name!r}, {value!r})" for name, value in PANDAS_DISPLAY_OPTIONS.items()
        )
        # Imports the model expects, as in any notebook.
        imports = "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt"
        # Set after %matplotlib inline, which applies its own rc defaults.
        rc_params = f"plt.rcParams.update({MATPLOTLIB_RC_PARAMS!r})"
        result = self.execute(f"%matplotlib inline\n{imports}\n{options}\n{rc_params}")
        if not result.success:
            raise RuntimeError(f"Kernel setup failed:\n{result.error}")

    def _interrupt(self, msg_id: str) -> bool:
        """Stop the running code but keep the kernel and its variables alive.

        Returns True if the kernel became idle within the grace period.
        """
        assert self._km is not None
        self._km.interrupt_kernel()
        # Drain the interrupted request's messages until the kernel is idle,
        # so the next execute() starts from a clean state.
        kc = self._client()
        deadline = time.monotonic() + INTERRUPT_GRACE_SECONDS
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                msg = kc.get_iopub_msg(timeout=remaining)
            except queue.Empty:
                return False
            if (
                msg["parent_header"].get("msg_id") == msg_id
                and msg["msg_type"] == "status"
                and msg["content"]["execution_state"] == "idle"
            ):
                return True
        return False

    def _restart_and_reload(self) -> None:
        """Last resort after a failed interrupt: fresh kernel with df restored."""
        self.restart()
        if self._df_path is not None:
            self.load_dataframe(self._df_path)

    @staticmethod
    def _collect_image(data: dict[str, Any], images: list[str]) -> None:
        png = data.get("image/png")
        if png:
            images.append(png)
