"""
This module provides an asynchronous workflow execution engine for Python code.
"""

import asyncio
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    return_code: int
    stdout: str
    stderr: str
    execution_time: float
    resource_usage: dict[str, Any] | None = None


@dataclass
class RuntimeConfig:
    python_version: str = "3.10"
    timeout: int = 1800
    max_memory_mb: int = 1024
    max_cpu_percent: int = 100
    temp_dir: Path | None = None
    requirements_file: Path | None = "requirements.txt"

    def __post_init__(self):
        if self.temp_dir is None:
            self.temp_dir = "./tmp"


class WorkflowRunner:
    """Async workflow execution engine for python code."""

    def __init__(self, config: RuntimeConfig = None, execution_dir = '.'):
        self.config = config or RuntimeConfig()
        self.execution_dir = execution_dir
        self.logger = logging.getLogger(__name__)
        self._active_processes: dict[str, asyncio.subprocess.Process] = {}
        self._setup_environment()

    def _setup_environment(self) -> None:
        """Initialize the execution environment."""
        # Convert temp_dir to absolute path to ensure it's created in the right location
        self.config.temp_dir = os.path.abspath(self.config.temp_dir)
        os.makedirs(self.config.temp_dir, exist_ok=True)
        # Validate python version availability
        if not self._check_python_version():
            raise RuntimeError(f"Python {self.config.python_version} not available")

    def _check_python_version(self) -> bool:
        """Check if the specified Python version is available."""
        import subprocess

        try:
            result = subprocess.run(
                [f"python{self.config.python_version}", "--version"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    async def ensure_pip(self) -> None:
        """Ensure pip is installed and up-to-date."""
        import subprocess

        try:
            # Try to ensure pip is available using ensurepip module
            # This works for Python installations that include ensurepip
            result = subprocess.run(
                [f"python{self.config.python_version}", "-m", "pip", "--version"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                # pip is already available
                return
            
            # Try to bootstrap pip using ensurepip
            subprocess.run(
                [f"python{self.config.python_version}", "-m", "ensurepip", "--upgrade"],
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
        """Install dependencies asynchronously."""

        if not requirements and not self.config.requirements_file:
            return ExecutionResult(ExecutionStatus.COMPLETED, 0, "", "", 0.0)

        await self.ensure_pip()

        cmd = [f"python{self.config.python_version}", "-m", "pip", "install"]

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
        """Execute workflow code with full async support and monitoring."""

        # Generate human-readable execution ID: exec_MMDD_HHMMSS_shortid
        execution_id = (
            execution_id or f"exec_{time.strftime('%m%d_%H%M%S')}_{id(code) % 10000}"
        )

        # Use absolute path for script to ensure it's accessible regardless of execution_dir
        script_path = os.path.abspath(os.path.join(self.config.temp_dir, f"{execution_id}.py"))
        with open(script_path, "w") as f:
            f.write(code)
        print(f"Executing script: {script_path}")
        cmd = [f"python{self.config.python_version}", script_path]
        return await self._run_command(cmd, execution_id, progress_callback)

    async def _run_command(
        self,
        cmd: list[str],
        execution_id: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> ExecutionResult:
        """Core async command execution with monitoring."""

        start_time = asyncio.get_event_loop().time()

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024 * 1024,  # 1MB buffer limit
                env=dict(os.environ),  # Pass host environment variables
                cwd=self.execution_dir,  # Set working directory for execution
            )

            if execution_id:
                self._active_processes[execution_id] = process

            stdout_data, stderr_data = await asyncio.wait_for(
                self._stream_output(process, progress_callback),
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
            if execution_id and execution_id in self._active_processes:
                del self._active_processes[execution_id]

    async def _stream_output(
        self,
        process: asyncio.subprocess.Process,
        progress_callback: Callable[[str], None] | None = None,
    ) -> tuple[str, str]:
        """Stream process output with real-time callbacks."""

        stdout_lines = []
        stderr_lines = []

        async def read_stdout():
            async for line in process.stdout:
                line_str = line.decode("utf-8", errors="replace")
                stdout_lines.append(line_str)
                if progress_callback:
                    progress_callback(line_str.rstrip())

        async def read_stderr():
            async for line in process.stderr:
                stderr_lines.append(line.decode("utf-8", errors="replace"))

        await asyncio.gather(read_stdout(), read_stderr())

        return "".join(stdout_lines), "".join(stderr_lines)

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution."""
        if execution_id not in self._active_processes:
            return False

        return await self._kill_process(execution_id)

    async def _kill_process(self, execution_id: str) -> bool:
        """Forcefully terminate a process."""
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
        """Get list of currently running executions."""
        return list(self._active_processes.keys())

    async def cleanup(self) -> None:
        """Clean up all resources and running processes."""
        for execution_id in list(self._active_processes.keys()):
            await self._kill_process(execution_id)
        # import shutil

        # if os.path.exists(self.config.temp_dir):
        #    shutil.rmtree(self.config.temp_dir, ignore_errors=True)


async def main():
    """Example usage of the WorkflowRunner."""
    config = RuntimeConfig(python_version="3.10", timeout=60, max_memory_mb=256)
    runner = WorkflowRunner(config)
    await runner.install_dependencies(["requests", "numpy"])
    code = """
print("Hello from the workflow runner!")
"""

    def progress_handler(line: str):
        print(f"[PROGRESS] {line}")

    result = await runner.execute(code, progress_callback=progress_handler)
    print(f"Output: {result.stdout}")
    print(f"Execution time: {result.execution_time:.2f}s")
    await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
