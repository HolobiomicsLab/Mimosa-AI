"""Retain bounded cleanup evidence without changing failure or cancellation gates."""

import asyncio
from pathlib import Path

import pytest

from container_runner_test import make_runtime
from sources.core import container_runner


@pytest.mark.parametrize('command,code,output,expected', [
    ('ps', 7, 'daemon unavailable', False),
    ('ps', 0, 'retained-container', False),
    ('rm', 1, 'already removed', True),
])
def test_control_diagnostics_preserve_failure_and_absence_semantics(
    tmp_path, command, code, output, expected
):
    config = make_runtime(tmp_path)
    docker = Path(config.docker_executable)
    docker.write_text(docker.read_text().replace(
        "elif sys.argv[1] in ('rm', 'ps'):\n    pass",
        f"elif sys.argv[1] == {command!r}:\n    print({output!r})\n    raise SystemExit({code})\nelif sys.argv[1] in ('rm', 'ps'):\n    pass",
    ))
    result = asyncio.run(container_runner.ContainerWorkflowRunner(config).execute())
    assert result.cleanup_verified is expected
    assert result.status == ('completed' if expected else 'cleanup_failed')
    phase = 'container_remove' if command == 'rm' else 'container_inspect'
    record = next(row for row in result.cleanup_diagnostics if row['phase'] == phase)
    assert record['return_code'] == code
    assert record['stdout'].strip() == output


def test_cleanup_launch_exception_retains_phase_and_reason(tmp_path, monkeypatch):
    async def fail_control(process, payload, maximum):
        raise OSError('controlled Docker transport failure')

    # A plain workflow does not use _exchange until Docker cleanup begins.
    monkeypatch.setattr(container_runner, '_exchange', fail_control)
    result = asyncio.run(container_runner.ContainerWorkflowRunner(make_runtime(tmp_path)).execute())
    assert result.status == 'cleanup_failed'
    assert not result.cleanup_verified
    record = result.cleanup_diagnostics[-1]
    assert record['phase'] == 'container_remove'
    assert record['error_type'] == 'OSError'
    assert record['error'] == 'controlled Docker transport failure'


def test_host_cleanup_error_retains_underlying_reason(tmp_path, monkeypatch):
    original = container_runner._finish_process
    calls = 0

    async def uncertain(process, tree, watcher):
        nonlocal calls
        await original(process, tree, watcher)
        calls += 1
        if calls == 1:
            raise RuntimeError('controlled owned-descendant verification failure')

    monkeypatch.setattr(container_runner, '_finish_process', uncertain)
    result = asyncio.run(container_runner.ContainerWorkflowRunner(make_runtime(tmp_path)).execute())
    assert result.status == 'cleanup_failed'
    assert not result.cleanup_verified
    record = next(row for row in result.cleanup_diagnostics if row['phase'] == 'host_processes')
    assert record['error_type'] == 'ProcessCleanupError'
    assert record['cause_type'] == 'RuntimeError'
    assert record['cause'] == 'controlled owned-descendant verification failure'


def test_control_output_is_bounded(tmp_path):
    config = make_runtime(tmp_path)
    docker = Path(config.docker_executable)
    docker.write_text(docker.read_text().replace(
        "elif sys.argv[1] in ('rm', 'ps'):\n    pass",
        "elif sys.argv[1] == 'rm':\n    print('x' * 10000)\n    print('y' * 10000, file=sys.stderr)\nelif sys.argv[1] == 'ps':\n    pass",
    ))
    result = asyncio.run(container_runner.ContainerWorkflowRunner(config).execute())
    assert result.cleanup_verified
    record = next(row for row in result.cleanup_diagnostics if row['phase'] == 'container_remove')
    assert len(record['stdout']) == 4096
    assert len(record['stderr']) == 4096
    assert record['stdout_truncated'] and record['stderr_truncated']
