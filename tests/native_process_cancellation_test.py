"""Process deadlines stop work, including detached owned descendants."""

import asyncio
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

from sources.core.workflow_runner import ExecutionStatus, RuntimeConfig, WorkflowRunner


def child_program(tmp_path):
    pidfile = tmp_path / "child.pid"
    code = f"import os,time; open({str(pidfile)!r},'w').write(str(os.getpid())); time.sleep(40)"
    return f"import subprocess,sys,time\nsubprocess.Popen([sys.executable,'-c',{code!r}], start_new_session=True)\ntime.sleep(40)\n", pidfile


def living(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.parametrize("mode", ["timeout", "cancel", "task_cancel", "callback_error"])
@pytest.mark.parametrize("use_pty", [False, True])
def test_runner_stops_detached_child_before_return(tmp_path, mode, use_pty):
    code, pidfile = child_program(tmp_path)
    config = RuntimeConfig(python_executable=sys.executable, timeout=0.6 if mode == "timeout" else 10,
                           temp_dir=tmp_path, requirements_file=None, use_pty=use_pty)
    runner = WorkflowRunner(config, str(tmp_path))
    async def run():
        callback = None
        if mode == "callback_error":
            def callback(line):
                raise RuntimeError("controlled output callback failure")
        action = code + "print('ready',flush=True)\n"
        if mode == "callback_error":
            action = code.replace('time.sleep(40)\n', "time.sleep(0.3)\nprint('ready',flush=True)\ntime.sleep(40)\n")
        task = asyncio.create_task(runner.execute(action, execution_id="bounded", progress_callback=callback))
        if mode in {"cancel", "task_cancel"}:
            deadline = time.monotonic() + 5
            while not pidfile.exists() and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            assert pidfile.exists()
            if mode == "cancel":
                assert await runner.cancel_execution("bounded")
            else:
                task.cancel()
        try:
            return await task
        except asyncio.CancelledError:
            return None
    pid = None
    try:
        result = asyncio.run(run())
        assert pidfile.exists()
        pid = int(pidfile.read_text())
        assert not living(pid), f"owned child {pid} survived {mode}"
        if mode == "timeout":
            assert result.status == ExecutionStatus.TIMEOUT
        elif mode == "cancel":
            assert result.status == ExecutionStatus.CANCELLED
        elif mode == "callback_error":
            assert result.status == ExecutionStatus.FAILED
            assert "controlled output callback failure" in result.stderr
    finally:
        if pidfile.exists():
            pid = int(pidfile.read_text())
        if pid and living(pid):
            psutil.Process(pid).kill()


def test_native_agent_watchdog_exits_and_kills_tool_child(tmp_path):
    code, pidfile = child_program(tmp_path)
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / "watchdog.py"
    script.write_text(f'''import sys
sys.path.insert(0, {str(root)!r})
from sources.core.native_agent import run_native_agent
class Agent:
    def run(self, instructions):
        exec({code!r})
run_native_agent(Agent(), 'fixture', dict(timeout_seconds=0.6, memory_path={str(tmp_path)!r}))
''')
    try:
        result = subprocess.run([sys.executable, str(script)], capture_output=True, timeout=15)
        assert result.returncode == 124, result.stderr.decode()
        assert pidfile.exists()
        assert not living(int(pidfile.read_text()))
    finally:
        if pidfile.exists() and living(int(pidfile.read_text())):
            psutil.Process(int(pidfile.read_text())).kill()


def test_concurrent_cancel_uses_tracked_reparented_child(tmp_path):
    code, pidfile = child_program(tmp_path)
    code = code.replace("time.sleep(40)\n", "time.sleep(0.3)\n")
    runner = WorkflowRunner(RuntimeConfig(python_executable=sys.executable, timeout=5,
        temp_dir=tmp_path, requirements_file=None, use_pty=False), str(tmp_path))
    async def run():
        task = asyncio.create_task(runner.execute(code, execution_id="reparent"))
        for _ in range(100):
            if pidfile.exists():
                break
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.4)
        results = await asyncio.gather(runner.cancel_execution("reparent"), runner.cancel_execution("reparent"))
        assert all(results)
        assert not living(int(pidfile.read_text()))
        result = await task
        assert result.status == ExecutionStatus.CANCELLED
        assert await runner.get_active_executions() == []
    try:
        asyncio.run(run())
    finally:
        if pidfile.exists() and living(int(pidfile.read_text())):
            psutil.Process(int(pidfile.read_text())).kill()


def test_runner_keeps_fast_success_and_cleans_registry(tmp_path):
    runner = WorkflowRunner(RuntimeConfig(python_executable=sys.executable,
        temp_dir=tmp_path, requirements_file=None, use_pty=False), str(tmp_path))
    async def run():
        result = await runner.execute("print('ready')")
        assert result.status == ExecutionStatus.COMPLETED
        assert result.stdout.strip() == "ready"
        assert await runner.get_active_executions() == []
    asyncio.run(run())


def test_duplicate_execution_id_is_rejected_before_another_launch(tmp_path):
    runner = WorkflowRunner(RuntimeConfig(python_executable=sys.executable,
        temp_dir=tmp_path, requirements_file=None, use_pty=False), str(tmp_path))
    async def run():
        first = asyncio.create_task(runner.execute("import time; time.sleep(0.2)", execution_id="same"))
        await asyncio.sleep(0)
        try:
            with pytest.raises(ValueError, match="already active"):
                await runner.execute("raise AssertionError('must never run')", execution_id="same")
        finally:
            await first
            await runner.cleanup()
    asyncio.run(run())
