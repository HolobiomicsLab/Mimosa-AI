"""Sandboxed async execution engine for generated workflows (Layer 3).

This file owns subprocess execution of LLM-generated workflow scripts: it
resolves a Python interpreter, prepares a managed venv, and runs each script
with timeouts, output capture, and cancellation. ``WorkflowOrchestrator``
uses it to run every crafted workflow, and the verifier evaluators reuse it
with a pinned ``sys.executable`` interpreter for scorer scripts. It runs on
this host whenever a workflow is executed or re-executed.
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
    """Lifecycle state of one workflow execution.

    ``WorkflowRunner`` constructs these states and attaches exactly one to every
    :class:`ExecutionResult` it returns. The orchestrator and the verifier
    evaluators compare a result's state against COMPLETED, FAILED, TIMEOUT, and
    CANCELLED to decide the next step.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Outcome of one subprocess execution managed by ``WorkflowRunner``.

    ``WorkflowRunner`` constructs it when a command finishes, times out, or
    fails. Orchestrator and verifier code read ``status``, ``stdout``, and
    ``stderr`` to decide whether a workflow ran cleanly. ``return_code`` is -1
    when the runner itself failed and no real exit code exists.

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

    The orchestrator builds it from the campaign config file; verifier
    evaluators build it with ``python_executable=sys.executable`` and
    ``use_pty=False``. Fields stay read-only after construction;
    ``__post_init__`` fills in the only default.

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
        temp_dir: Directory used to materialize generated scripts. Defaults
            to ``"./tmp"`` when not provided.
        requirements_file: Optional path to a pip requirements file used as a
            fallback when no explicit dependencies are passed.
        use_pty: When False, the runner skips the PTY/color path and uses
            plain pipes. Useful for short, non-interactive scripts (e.g.
            verifier checks) where ANSI color propagation and TTY emulation
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
        """Fill in the default temp directory.

        The dataclass machinery calls it once right after ``RuntimeConfig(...)``.
        It sets ``temp_dir`` to ``"./tmp"`` when the caller left the field ``None``.

        Returns:
            None. The instance is updated in place.
        """
        if self.temp_dir is None:
            self.temp_dir = "./tmp"


class WorkflowRunner:
    """Async subprocess engine that runs Python workflow scripts.

    ``WorkflowOrchestrator`` and the verifier evaluators construct one runner
    per campaign or scoring session and drive it through :meth:`execute`,
    :meth:`install_dependencies`, and :meth:`cleanup`. Key invariant: every
    spawned process is registered under its execution id until it exits, so
    cancellation and cleanup can always reach it.
    """

    def __init__(self, config: RuntimeConfig | None = None, execution_dir: str = '.') -> None:
        """Prepare the runner and resolve the Python interpreter.

        The orchestrator and the verifier evaluators call it once when they build
        their runner; both pass a :class:`RuntimeConfig` built from config-file
        values plus a working directory. It stores the settings and resolves the
        interpreter at once, so a missing Python fails at construction time.

        Args:
            config: Runtime limits and interpreter choice, built from the campaign
                config file. When ``None``, a default :class:`RuntimeConfig` is
                constructed.
            execution_dir: Working directory used when spawning subprocesses. The
                orchestrator passes the sandbox workspace; verifiers pass the
                workspace under test.

        Raises:
            RuntimeError: If the configured Python interpreter cannot be found on
                this host.
        """
        self.config = config or RuntimeConfig()
        self.execution_dir = execution_dir
        self.logger = logging.getLogger(__name__)
        self._active_processes: dict[str, asyncio.subprocess.Process] = {}
        self._python_cmd: list[str] = []  # resolved by _setup_environment
        self._setup_environment()

    def _setup_environment(self) -> None:
        """Prepare the temp directory and validate the interpreter.

        ``__init__`` calls it once during runner construction. It converts
        ``config.temp_dir`` to an absolute path, creates the directory, and checks
        that a usable Python executable exists.

        Raises:
            RuntimeError: If no working Python executable matches the configured
                version or pinned interpreter path.
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
        """Find a working Python executable for the configured version.

        ``_check_python_version`` calls it during runner setup, before any command
        runs. It probes candidate commands on this host with short ``--version``
        calls. Versioned candidates are accepted as-is because they target the
        exact version by construction:

        - Unix/macOS: ``python3.12`` (versioned binary)
        - Windows:    ``py -3.12``   (Windows Python Launcher with version flag)

        Generic fallbacks (``python3``, ``python``) are also tried but accepted
        only when their ``--version`` output names the configured version, so the
        runner never silently runs the wrong Python.

        Returns:
            list[str] | None: Command prefix that invokes Python (for example
            ``["python3.12"]`` or ``["py", "-3.12"]``), or ``None`` when no
            candidate works on this host.
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
        """Resolve the interpreter and cache it on the runner.

        ``_setup_environment`` calls it once during construction. It probes the
        host for a matching Python executable and stores the working command
        prefix in ``self._python_cmd``.

        Returns:
            bool: True when a matching Python executable was found and cached;
            False otherwise.
        """
        resolved = self._resolve_python_executable()
        if resolved is not None:
            self._python_cmd = resolved
            return True
        return False

    def _ensure_venv(self) -> None:
        """Create a managed venv under ``temp_dir`` and route installs through it.

        ``install_dependencies`` calls it before the first ``pip install``. It is
        a no-op when the caller pinned ``python_executable``: the verifier path
        must run under ``sys.executable`` itself. It is idempotent; an existing
        venv at the same path is reused. After this call ``self._python_cmd``
        points at the venv interpreter.
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
        """Make ``pip`` usable inside the resolved interpreter.

        ``install_dependencies`` awaits it before every install. It checks
        ``python -m pip --version`` and, when that fails, bootstraps pip with
        ``python -m ensurepip --upgrade``.

        Returns:
            None. Failures are logged as warnings and never raised; the caller
            sees a broken pip through the install result instead.
        """
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
        """Install pip packages into the managed environment.

        The orchestrator awaits it once per campaign with the dependency list
        declared by the workflow; verifier runs usually install nothing. When
        ``requirements`` is empty it falls back to ``config.requirements_file``;
        when both are empty there is nothing to install.

        Args:
            requirements: Pip requirement specifiers (for example
                ``["requests", "numpy"]``) taken from the caller's config, or
                ``None`` to use the configured requirements file.

        Returns:
            ExecutionResult: Outcome of the ``pip install`` command, or a
            COMPLETED result with empty output when there is nothing to install.

        Raises:
            FileNotFoundError: If the requirements-file fallback is configured
                but the file does not exist on disk.
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
        """Run generated workflow code as a tracked subprocess.

        The orchestrator awaits it for every crafted workflow, passing the
        LLM-generated script and a progress callback for live log lines; verifier
        evaluators await it for scorer and re-run scripts. It writes the code to
        ``<execution_id>.py`` under ``temp_dir`` and runs it with the resolved
        interpreter.

        Args:
            code: Python source to execute. It comes from the workflow factory or
                from the verifier's scorer script.
            execution_id: Optional caller-chosen identifier used for tracking and
                cancellation. A timestamp-based id is generated when ``None``.
            progress_callback: Optional callable invoked once per stdout line as
                the script runs; the orchestrator passes a live-log handler.

        Returns:
            ExecutionResult: Final status, exit code, captured stdout and stderr,
            and wall-clock time of the run.
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
        """Build an environment dict that forces colored child output.

        ``_run_command`` calls it before every spawn so child processes emit ANSI
        colors even when their stdout is a pipe. It copies ``os.environ`` and adds
        the variables that rich, click, tqdm, and pytest check.

        Returns:
            dict[str, str]: Copy of the host environment plus color-forcing
            variables and ``PYTHONUNBUFFERED=1``.
        """
        env = dict(os.environ)
        env.setdefault("TERM", "xterm-256color")
        env["FORCE_COLOR"] = "1"          # chalk / supports-color (Node & Python)
        env["PY_COLORS"] = "1"            # pytest, tox, …
        env["CLICOLOR_FORCE"] = "1"       # BSD / GNU convention
        env["PYTHONUNBUFFERED"] = "1"      # disable Python output buffering
        return env

    def _pty_available(self) -> bool:
        """Report whether the PTY stdout path may be used.

        ``_run_command`` calls it to choose between the PTY path and the plain
        pipe fallback. The PTY needs a non-Windows platform and ``use_pty`` set
        to True; verifier configs disable it.

        Returns:
            bool: True when the pseudo-terminal path is allowed on this host and
            in the config; False otherwise.
        """
        return sys.platform != "win32" and getattr(self.config, "use_pty", True)

    async def _run_command(
        self,
        cmd: list[str],
        execution_id: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> ExecutionResult:
        """Spawn one command, stream its output, and collect the result.

        ``execute`` and ``install_dependencies`` await it; it is the single spawn
        point for every subprocess. On Linux/macOS it connects child stdout to a
        pseudo-terminal so the child sees ``isatty`` and prints colors, while
        stderr stays on a regular pipe. Otherwise (or when the PTY is disabled)
        both streams use plain pipes. The command is stopped when it exceeds
        ``config.timeout``.

        Args:
            cmd: Argv list passed to ``asyncio.create_subprocess_exec``, built by
                the caller from the resolved interpreter and the target script.
            execution_id: Optional identifier that registers the process for
                cancellation; omitted for dependency installs.
            progress_callback: Optional callable invoked once per stdout line.

        Returns:
            ExecutionResult: COMPLETED with the real exit code on success, FAILED
            with the exit code or error text on failure, TIMEOUT with ``-1`` and
            a fixed message on timeout, and FAILED with ``-1`` for runner-level
            errors.
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
        """Drain child output from a PTY master fd and a stderr pipe.

        ``_run_command`` awaits it on the PTY path. The PTY preserves ANSI escape
        sequences because the child sees a real terminal on stdout; stderr
        arrives through a regular pipe.

        Args:
            process: Running subprocess whose stderr pipe is drained.
            master_fd: File descriptor of the PTY master end, used as the child
                stdout side.
            progress_callback: Optional callable invoked with each decoded stdout
                line.

        Returns:
            tuple[str, str]: ``(stdout, stderr)`` with the full captured text of
            both streams.
        """
        loop = asyncio.get_event_loop()
        stdout_chunks: list[str] = []
        stderr_lines: list[str] = []
        stdout_done = asyncio.Event()

        # Make the master fd non-blocking so we can use add_reader.
        flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        def _on_master_readable() -> None:
            """Read one ready chunk from the PTY and record it.

            The asyncio event loop calls it whenever the master fd is readable. It
            appends the decoded text to the captured stdout and forwards each line to
            the progress callback; EOF closes the reader and marks stdout done.

            Returns:
                None. OSError is swallowed because it is expected when the child
                exits and closes the slave side.
            """
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
            """Drain the child stderr pipe into the captured lines.

            ``_stream_output_pty`` schedules it next to the PTY reader. It decodes
            each stderr chunk as UTF-8 and appends it.

            Returns:
                None. Captured text accumulates in the enclosing list.
            """
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
        """Drain child stdout and stderr when both are plain pipes.

        ``_run_command`` awaits it on non-PTY platforms and whenever PTY use is
        disabled. It reads both pipes concurrently and forwards each stdout line
        to the progress callback with the trailing newline stripped.

        Args:
            process: Running subprocess whose stdout and stderr pipes are drained
                concurrently.
            progress_callback: Optional callable invoked once per stdout line.

        Returns:
            tuple[str, str]: ``(stdout, stderr)`` with the full captured text of
            both streams.
        """
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        async def read_stdout() -> None:
            """Collect the child stdout pipe line by line.

            ``_stream_output_pipe`` schedules it next to the stderr reader. It
            decodes each line and forwards it to the progress callback.

            Returns:
                None. Captured lines accumulate in the enclosing list.
            """
            async for line in process.stdout:
                line_str = line.decode("utf-8", errors="replace")
                stdout_lines.append(line_str)
                if progress_callback:
                    progress_callback(line_str.rstrip())

        async def read_stderr() -> None:
            """Collect the child stderr pipe line by line.

            ``_stream_output_pipe`` schedules it next to the stdout reader. It
            decodes each line as UTF-8 and appends it.

            Returns:
                None. Captured lines accumulate in the enclosing list.
            """
            async for line in process.stderr:
                stderr_lines.append(line.decode("utf-8", errors="replace"))

        await asyncio.gather(read_stdout(), read_stderr())

        return "".join(stdout_lines), "".join(stderr_lines)

    async def cancel_execution(self, execution_id: str) -> bool:
        """Terminate one running execution chosen by the caller.

        Orchestrator and UI code call it to stop a workflow early. It resolves
        the id to a registered process and kills it; unknown ids do nothing.

        Args:
            execution_id: Identifier returned by (or supplied to)
                :meth:`execute`.

        Returns:
            bool: True when the process was found and termination was attempted;
            False when no such execution is registered.
        """
        if execution_id not in self._active_processes:
            return False

        return await self._kill_process(execution_id)

    async def _kill_process(self, execution_id: str) -> bool:
        """Force one registered process to stop, escalating if needed.

        ``cancel_execution`` and ``cleanup`` call it, and ``_run_command`` also
        calls it when a command exceeds its timeout. It sends SIGTERM, waits up
        to five seconds, then sends SIGKILL if the process is still alive.

        Args:
            execution_id: Identifier of the registered process.

        Returns:
            bool: True when termination was attempted; False when no such
            process is registered.
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
        """List the executions currently registered on this runner.

        Monitoring code and tests call it to see what is still running. It copies
        the keys of the active-process registry at call time.

        Returns:
            list[str]: Execution ids of all registered processes; empty when
            nothing is running.
        """
        return list(self._active_processes.keys())

    async def cleanup(self) -> None:
        """Stop every registered process so the runner can be discarded.

        The orchestrator awaits it at campaign shutdown and in its exit handlers,
        and the demo ``main`` awaits it at the end. It terminates each registered
        execution in turn.

        Returns:
            None.
        """
        for execution_id in list(self._active_processes.keys()):
            await self._kill_process(execution_id)


async def main() -> None:
    """Demonstrate the runner end to end as a smoke check.

    Developers run this file directly. It installs two demo packages, runs a
    small color-printing script with a progress handler, prints the captured
    output and run time, and cleans up.

    Returns:
        None. Progress lines and the run summary are printed to stdout.
    """
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
        """Print one progress line from the demo run.

        ``main`` passes it to ``execute`` as the progress callback, and
        ``execute`` calls it with each stdout line.

        Args:
            line: One stdout line produced by the demo script.

        Returns:
            None. The line is printed with a ``[PROGRESS]`` prefix.
        """
        print(f"[PROGRESS] {line}")

    result = await runner.execute(code, progress_callback=progress_handler)
    print(f"Output: {result.stdout}")
    print(f"Execution time: {result.execution_time:.2f}s")
    await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())