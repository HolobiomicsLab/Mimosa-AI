"""
This module provides an asynchronous workflow execution engine for Python code.
"""

import asyncio
import fcntl
import logging
import os
import pty
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ExecutionStatus(Enum):
    """Lifecycle status of a workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Outcome of a subprocess execution managed by ``WorkflowRunner``.

    Attributes:
        status: Final :class:`ExecutionStatus` of the execution.
        return_code: Process exit code (``-1`` for runner-level failures).
        stdout: Captured standard output as a single string.
        stderr: Captured standard error as a single string.
        execution_time: Wall-clock duration in seconds.
        resource_usage: Optional dictionary of resource-usage metrics.
    """

    status: ExecutionStatus
    return_code: int
    stdout: str
    stderr: str
    execution_time: float
    resource_usage: dict[str, Any] | None = None


@dataclass
class RuntimeConfig:
    """Configuration controlling how the workflow runner spawns Python.

    Attributes:
        python_version: Target Python version string (e.g. ``"3.12"``). Used
            to resolve an interpreter from ``PATH`` when ``python_executable``
            is not set.
        python_executable: Explicit path to a Python interpreter (e.g.
            ``sys.executable``). When provided it overrides ``python_version``
            entirely: the runner uses this exact interpreter and skips the
            ``PATH``-based version resolution, so it never depends on a
            matching ``pythonX.Y`` being installed on ``PATH``.
        timeout: Maximum execution time per command, in seconds.
        max_memory_mb: Soft memory cap, in megabytes (advisory).
        max_cpu_percent: Soft CPU cap, as a percentage (advisory).
        temp_dir: Directory used to materialise generated scripts. Defaults to
            ``"./tmp"`` when not provided.
        requirements_file: Optional path to a pip requirements file used as a
            fallback when no explicit dependencies are passed.
        use_pty: When False, the runner skips the PTY/color path and uses
            plain pipes. Useful for short, non-interactive scripts (e.g.
            verifier checks) where ANSI colour propagation and TTY emulation
            are pure overhead.
    """

    python_version: str = "3.12"
    # Explicit interpreter path; when set, overrides python_version resolution.
    python_executable: str | None = None
    timeout: int = 1800
    max_memory_mb: int = 1024
    max_cpu_percent: int = 100
    temp_dir: Path | None = None
    # optional replacement for config requirements list
    requirements_file: Path | None = "requirements.txt"
    # When False, the runner skips the PTY/color path and uses plain pipes.
    # Useful for short, non-interactive scripts (e.g. verifier checks) where
    # ANSI colour propagation and TTY emulation are pure overhead.
    use_pty: bool = True

    def __post_init__(self) -> None:
        """Default ``temp_dir`` to ``./tmp`` when left unset."""
        if self.temp_dir is None:
            self.temp_dir = "./tmp"


class WorkflowRunner:
    """Async workflow execution engine for python code."""

    def __init__(self, config: RuntimeConfig | None = None, execution_dir: str = '.') -> None:
        """Initialize the runner and resolve its Python executable.

        Args:
            config: Runtime configuration to use. When ``None``, a default
                :class:`RuntimeConfig` is constructed.
            execution_dir: Working directory used when spawning subprocesses.

        Raises:
            RuntimeError: If the configured Python version cannot be found on
                the host.
        """
        self.config = config or RuntimeConfig()
        self.execution_dir = execution_dir
        self.logger = logging.getLogger(__name__)
        self._active_processes: dict[str, asyncio.subprocess.Process] = {}
        self._python_cmd: list[str] = []  # resolved by _setup_environment
        self._setup_environment()

    def _setup_environment(self) -> None:
        """Initialize the execution environment.

        Resolves ``temp_dir`` to an absolute path, creates it if needed, and
        validates that the configured Python version is available.

        Raises:
            RuntimeError: If no working Python executable matches
                ``config.python_version``.
        """
        # Convert temp_dir to absolute path to ensure it's created in the right location
        self.config.temp_dir = os.path.abspath(self.config.temp_dir)
        os.makedirs(self.config.temp_dir, exist_ok=True)
        # Validate python availability and resolve the executable
        if not self._check_python_version():
            target = (
                self.config.python_executable
                or f"Python {self.config.python_version}"
            )
            raise RuntimeError(f"{target} not available")

    def _resolve_python_executable(self) -> list[str] | None:
        """
        Find a working Python executable for the configured version.

        Versioned candidates are always tried first and accepted as-is because
        they target the exact version by construction:
        - Unix/macOS: ``python3.12`` (versioned binary)
        - Windows:    ``py -3.12``   (Windows Python Launcher with version flag)

        Generic fallbacks (``python3``, ``python``) are also tried but are only
        accepted when their ``--version`` output actually matches the configured
        version, so the runner never silently runs the wrong Python.

        Returns:
            list[str]: Command prefix to invoke Python (e.g. ``["python3.12"]``
                       or ``["py", "-3.12"]``), or ``None`` if no candidate works.
        """
        import subprocess
        import sys

        # An explicit interpreter path overrides version-based PATH resolution.
        explicit = self.config.python_executable
        if explicit:
            try:
                result = subprocess.run(
                    [explicit, "--version"],
                    capture_output=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    return [explicit]
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass
            return None

        version = self.config.python_version  # e.g. "3.12"

        if sys.platform == "win32":
            candidates: list[tuple[bool, list[str]]] = [
                (True,  ["py", f"-{version}"]),  # Python Launcher – version-pinned
                (False, ["python"]),              # bare interpreter – verify version
                (False, ["python3"]),             # rare Windows setups – verify version
            ]
        else:
            candidates = [
                (True,  [f"python{version}"]),    # e.g. python3.10 – version-pinned
                (False, ["python3"]),              # verify version before accepting
                (False, ["python"]),               # verify version before accepting
            ]

        for is_versioned, candidate in candidates:
            try:
                result = subprocess.run(
                    candidate + ["--version"],
                    capture_output=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    continue
                if not is_versioned:
                    raw = (result.stdout or result.stderr).decode("utf-8", errors="replace").strip()
                    if not raw.startswith(f"Python {version}"):
                        continue
                return candidate
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

        return None

    def _check_python_version(self) -> bool:
        """
        Check if the configured Python version is available and cache the
        resolved executable in ``self._python_cmd``.

        Returns:
            True when a matching Python executable was found and cached;
            False otherwise.
        """
        resolved = self._resolve_python_executable()
        if resolved is not None:
            self._python_cmd = resolved
            return True
        return False

    def _ensure_venv(self) -> None:
        """Create a managed venv under ``temp_dir`` and use it for installs/execs.

        No-op when the caller pinned ``python_executable`` (the verifier path
        depends on running under ``sys.executable``). Idempotent — reuses an
        existing venv at the same path. After this call ``self._python_cmd``
        points at the venv's interpreter.
        """
        import subprocess
        if self.config.python_executable:
            return
        venv_dir = Path(self.config.temp_dir) / "mimosa_venv"
        bin_dir = "Scripts" if sys.platform == "win32" else "bin"
        venv_python = venv_dir / bin_dir / ("python.exe" if sys.platform == "win32" else "python")
        if not venv_python.exists():
            subprocess.run(
                [*self._python_cmd, "-m", "venv", str(venv_dir)],
                check=True, capture_output=True, timeout=120,
            )
            self.logger.info(f"Created venv at {venv_dir}")
        self._python_cmd = [str(venv_python)]

    async def ensure_pip(self) -> None:
        """Ensure pip is installed and up-to-date."""
        import subprocess

        try:
            # Try to ensure pip is available using ensurepip module
            # This works for Python installations that include ensurepip
            result = subprocess.run(
                [*self._python_cmd, "-m", "pip", "--version"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                # pip is already available
                return

            # Try to bootstrap pip using ensurepip
            subprocess.run(
                [*self._python_cmd, "-m", "ensurepip", "--upgrade"],
                check=True,
                capture_output=True,
                timeout=30,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            # Log warning but don't fail - pip might already be available via other means
            self.logger.warning(f"Could not ensure pip via ensurepip: {e}. Assuming pip is available.")

    async def install_dependencies(
        self, requirements: list[str] | None = None
    ) -> ExecutionResult:
        """Install dependencies asynchronously.

        Args:
            requirements: Explicit list of pip requirement specifiers. When
                omitted, the runner falls back to ``config.requirements_file``.

        Returns:
            The :class:`ExecutionResult` of the underlying ``pip install``
            command, or a COMPLETED result with zero output when there is
            nothing to install.

        Raises:
            FileNotFoundError: If a requirements file fallback is configured
                but does not exist on disk.
        """

        if not requirements and not self.config.requirements_file:
            return ExecutionResult(ExecutionStatus.COMPLETED, 0, "", "", 0.0)

        self._ensure_venv()
        await self.ensure_pip()

        cmd = [*self._python_cmd, "-m", "pip", "install"]

        if requirements:
            cmd.extend(requirements)
        elif self.config.requirements_file:
            if not self.config.requirements_file.exists():
                raise FileNotFoundError(
                    f"Requirements file not found: {self.config.requirements_file}"
                )
            cmd.extend(["-r", str(self.config.requirements_file)])
        return await self._run_command(cmd)

    async def execute(
        self,
        code: str,
        execution_id: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> ExecutionResult:
        """Execute workflow code with full async support and monitoring.

        Args:
            code: Python source code to write to a temporary script and run.
            execution_id: Optional identifier used to track and cancel the
                process. A human-readable id is generated when omitted.
            progress_callback: Optional callable invoked once per stdout line
                as the script runs.

        Returns:
            The :class:`ExecutionResult` of the subprocess invocation.
        """

        # Generate human-readable execution ID: exec_MMDD_HHMMSS_shortid
        execution_id = (
            execution_id or f"exec_{time.strftime('%m%d_%H%M%S')}_{id(code) % 10000}"
        )

        # Use absolute path for script to ensure it's accessible regardless of execution_dir
        script_path = os.path.abspath(os.path.join(self.config.temp_dir, f"{execution_id}.py"))
        with open(script_path, "w") as f:
            f.write(code)
        cmd = [*self._python_cmd, script_path]
        return await self._run_command(cmd, execution_id, progress_callback)

    @staticmethod
    def _build_color_env() -> dict[str, str]:
        """Build an environment dict that encourages color output.

        Copies the host environment and adds variables commonly checked by
        CLI tools and Python libraries (rich, click, tqdm, pytest, …) to
        force colored output even when stdout is not a real TTY.

        Returns:
            A copy of ``os.environ`` augmented with color-forcing variables.
        """
        env = dict(os.environ)
        env.setdefault("TERM", "xterm-256color")
        env["FORCE_COLOR"] = "1"          # chalk / supports-color (Node & Python)
        env["PY_COLORS"] = "1"            # pytest, tox, …
        env["CLICOLOR_FORCE"] = "1"       # BSD / GNU convention
        env["PYTHONUNBUFFERED"] = "1"      # disable Python output buffering
        return env

    def _pty_available(self) -> bool:
        """Return True when pseudo-terminal support can be used."""
        return sys.platform != "win32" and getattr(self.config, "use_pty", True)

    async def _run_command(
        self,
        cmd: list[str],
        execution_id: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> ExecutionResult:
        """Core async command execution with monitoring.

        On platforms that support PTYs (Linux / macOS) the subprocess stdout
        is connected to a pseudo-terminal so that child processes see
        ``isatty(1) == True`` and emit ANSI colour codes.  Stderr is still
        captured via a regular pipe.

        Args:
            cmd: Argv list for :func:`asyncio.create_subprocess_exec`.
            execution_id: Optional id used to register the process so it can
                be cancelled mid-flight.
            progress_callback: Optional callable invoked with each stdout
                line as it is produced.

        Returns:
            An :class:`ExecutionResult` describing the outcome (including
            COMPLETED, FAILED, TIMEOUT, or runner-level failure).
        """

        start_time = asyncio.get_event_loop().time()
        master_fd = slave_fd = -1

        try:
            env = self._build_color_env()

            if self._pty_available():
                # --- PTY path: child stdout goes through a pseudo-terminal ---
                master_fd, slave_fd = pty.openpty()

                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=slave_fd,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=self.execution_dir,
                )
                # Parent no longer needs the slave side; the child inherited it.
                os.close(slave_fd)
                slave_fd = -1

                if execution_id:
                    self._active_processes[execution_id] = process

                stdout_data, stderr_data = await asyncio.wait_for(
                    self._stream_output_pty(process, master_fd, progress_callback),
                    timeout=self.config.timeout,
                )
            else:
                # --- Pipe fallback (Windows or if PTY unavailable) ---
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=1024 * 1024,
                    env=env,
                    cwd=self.execution_dir,
                )

                if execution_id:
                    self._active_processes[execution_id] = process

                stdout_data, stderr_data = await asyncio.wait_for(
                    self._stream_output_pipe(process, progress_callback),
                    timeout=self.config.timeout,
                )

            await process.wait()
            execution_time = asyncio.get_event_loop().time() - start_time

            status = (
                ExecutionStatus.COMPLETED
                if process.returncode == 0
                else ExecutionStatus.FAILED
            )

            return ExecutionResult(
                status=status,
                return_code=process.returncode,
                stdout=stdout_data,
                stderr=stderr_data,
                execution_time=execution_time,
            )

        except asyncio.TimeoutError:
            if execution_id and execution_id in self._active_processes:
                await self._kill_process(execution_id)
            return ExecutionResult(
                ExecutionStatus.TIMEOUT, -1, "", "Execution timed out", 0.0
            )

        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            return ExecutionResult(ExecutionStatus.FAILED, -1, "", str(e), 0.0)

        finally:
            if slave_fd >= 0:
                os.close(slave_fd)
            if master_fd >= 0:
                os.close(master_fd)
            if execution_id and execution_id in self._active_processes:
                del self._active_processes[execution_id]

    # ------------------------------------------------------------------
    # Output streaming helpers
    # ------------------------------------------------------------------

    async def _stream_output_pty(
        self,
        process: asyncio.subprocess.Process,
        master_fd: int,
        progress_callback: Callable[[str], None] | None = None,
    ) -> tuple[str, str]:
        """Stream stdout from a PTY master fd and stderr from a pipe.

        The PTY preserves ANSI escape sequences (colours, bold, …) because
        the child process sees a real terminal on its stdout.

        Args:
            process: The running subprocess whose stderr pipe will be drained.
            master_fd: Master end of the pseudo-terminal used as stdout.
            progress_callback: Optional callable invoked with each stdout
                line decoded from the PTY.

        Returns:
            A tuple ``(stdout, stderr)`` with the full captured output.
        """
        loop = asyncio.get_event_loop()
        stdout_chunks: list[str] = []
        stderr_lines: list[str] = []
        stdout_done = asyncio.Event()

        # Make the master fd non-blocking so we can use add_reader.
        flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        def _on_master_readable() -> None:
            """Called by the event loop when data is available on the PTY."""
            try:
                data = os.read(master_fd, 65536)
                if not data:
                    loop.remove_reader(master_fd)
                    stdout_done.set()
                    return
                text = data.decode("utf-8", errors="replace")
                stdout_chunks.append(text)
                if progress_callback:
                    for line in text.splitlines():
                        progress_callback(line)
            except OSError:
                # EIO is expected when the slave side is closed (child exited).
                loop.remove_reader(master_fd)
                stdout_done.set()

        loop.add_reader(master_fd, _on_master_readable)

        async def _read_stderr() -> None:
            async for raw_line in process.stderr:
                stderr_lines.append(raw_line.decode("utf-8", errors="replace"))

        # Wait for both stdout (PTY) and stderr (pipe) to finish.
        await asyncio.gather(stdout_done.wait(), _read_stderr())

        return "".join(stdout_chunks), "".join(stderr_lines)

    async def _stream_output_pipe(
        self,
        process: asyncio.subprocess.Process,
        progress_callback: Callable[[str], None] | None = None,
    ) -> tuple[str, str]:
        """Fallback: stream stdout/stderr when both are plain pipes.

        Args:
            process: The running subprocess whose stdout and stderr pipes
                will be drained concurrently.
            progress_callback: Optional callable invoked once per stdout
                line (with the trailing newline stripped).

        Returns:
            A tuple ``(stdout, stderr)`` with the full captured output.
        """
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        async def read_stdout() -> None:
            async for line in process.stdout:
                line_str = line.decode("utf-8", errors="replace")
                stdout_lines.append(line_str)
                if progress_callback:
                    progress_callback(line_str.rstrip())

        async def read_stderr() -> None:
            async for line in process.stderr:
                stderr_lines.append(line.decode("utf-8", errors="replace"))

        await asyncio.gather(read_stdout(), read_stderr())

        return "".join(stdout_lines), "".join(stderr_lines)

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution.

        Args:
            execution_id: Identifier returned by (or supplied to)
                :meth:`execute`.

        Returns:
            True when the process was found and termination was attempted;
            False when no such execution is registered.
        """
        if execution_id not in self._active_processes:
            return False

        return await self._kill_process(execution_id)

    async def _kill_process(self, execution_id: str) -> bool:
        """Forcefully terminate a process.

        Sends ``SIGTERM`` first and waits up to five seconds; if the process
        is still alive it is escalated to ``SIGKILL``.

        Args:
            execution_id: Identifier of the registered process.

        Returns:
            True when termination was attempted; False when no such process
            is registered.
        """
        process = self._active_processes.get(execution_id)
        if not process:
            return False

        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

        return True

    async def get_active_executions(self) -> list[str]:
        """Get list of currently running executions.

        Returns:
            List of execution ids currently registered with the runner.
        """
        return list(self._active_processes.keys())

    async def cleanup(self) -> None:
        """Clean up all resources and running processes.

        Iterates over every registered execution and forcefully terminates it.
        """
        for execution_id in list(self._active_processes.keys()):
            await self._kill_process(execution_id)


async def main() -> None:
    """Example usage of the WorkflowRunner."""
    config = RuntimeConfig(python_version="3.12", timeout=60, max_memory_mb=256)
    runner = WorkflowRunner(config)
    await runner.install_dependencies(["requests", "numpy"])
    code = """
import sys

# Demonstrate that color is preserved through the PTY
print("\\033[32m✔ Hello from the workflow runner (green)\\033[0m")
print("\\033[1;34mBold blue text\\033[0m")
print(f"stdout is a TTY: {sys.stdout.isatty()}")
"""

    def progress_handler(line: str) -> None:
        print(f"[PROGRESS] {line}")

    result = await runner.execute(code, progress_callback=progress_handler)
    print(f"Output: {result.stdout}")
    print(f"Execution time: {result.execution_time:.2f}s")
    await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())