"""Cleanup uncertainty and repeated cancellation must remain explicit."""

import asyncio
from pathlib import Path
import sys

import pytest

from container_runner_test import make_runtime
from sources.core import container_runner


def test_host_cleanup_uncertainty_cannot_be_reported_verified(tmp_path, monkeypatch):
    original = container_runner._finish_process
    calls = 0

    async def uncertain(process, tree, watcher):
        nonlocal calls
        await original(process, tree, watcher)
        calls += 1
        if calls == 1:
            raise RuntimeError('controlled cleanup verification failure')

    monkeypatch.setattr(container_runner, '_finish_process', uncertain)
    result = asyncio.run(container_runner.ContainerWorkflowRunner(make_runtime(tmp_path)).execute())
    assert result.status == 'cleanup_failed'
    assert not result.cleanup_verified


def test_cancel_does_not_hide_failed_container_cleanup(tmp_path):
    config = make_runtime(tmp_path, 'import time;time.sleep(60)')
    path = Path(config.docker_executable)
    path.write_text(path.read_text().replace("elif sys.argv[1] in ('rm', 'ps'):\n    pass", "elif sys.argv[1] in ('rm', 'ps'):\n    raise SystemExit(1)"))

    async def run():
        task = asyncio.create_task(container_runner.ContainerWorkflowRunner(config).execute())
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(RuntimeError, match='cleanup'):
            await task

    asyncio.run(run())


def test_repeated_cancel_during_spawn_does_not_lose_owned_process(monkeypatch):
    original = asyncio.create_subprocess_exec
    children = []

    async def delayed_spawn(*args, **kwargs):
        process = await original(*args, **kwargs)
        children.append(process)
        await asyncio.sleep(0.15)
        return process

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', delayed_spawn)

    async def run():
        async def child():
            async with container_runner._owned_process([sys.executable, '-c', 'import time;time.sleep(60)']):
                await asyncio.sleep(60)
        task = asyncio.create_task(child())
        try:
            for _ in range(100):
                if children:
                    break
                await asyncio.sleep(0.01)
            assert children
            task.cancel()
            await asyncio.sleep(0.02)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert all(process.returncode is not None for process in children)
        finally:
            for process in children:
                if process.returncode is None:
                    process.kill()
                    await process.wait()

    asyncio.run(run())
