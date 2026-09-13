"""A host cleanup failure must not erase an outstanding cancellation."""

import asyncio

import pytest

from container_runner_test import make_runtime
from sources.core import container_runner


def test_cancel_and_host_cleanup_failure_raise_cleanup_error(tmp_path, monkeypatch):
    original = container_runner._finish_process
    calls = 0

    async def uncertain(process, tree, watcher):
        nonlocal calls
        await original(process, tree, watcher)
        calls += 1
        if calls == 1:
            raise RuntimeError('controlled host cleanup uncertainty')

    monkeypatch.setattr(container_runner, '_finish_process', uncertain)
    runner = container_runner.ContainerWorkflowRunner(make_runtime(tmp_path, 'import time;time.sleep(60)'))

    async def run():
        task = asyncio.create_task(runner.execute())
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(RuntimeError, match='cleanup'):
            await task

    asyncio.run(run())
