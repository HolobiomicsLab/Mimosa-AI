"""Only staged regular bytes may enter the container or host module lookup."""

import asyncio
import os
from pathlib import Path
import socket

import pytest

from container_runner_test import make_runtime, request_code
from sources.core.container_runner import ContainerWorkflowRunner


@pytest.mark.parametrize('kind', ['symlink', 'hardlink', 'fifo', 'socket'])
def test_nonregular_or_aliased_staged_inputs_are_refused(tmp_path, kind, monkeypatch):
    config = make_runtime(tmp_path)
    target = config.public_dir / 'exposed'
    private = tmp_path / 'private/value'
    private.write_text('not public')
    endpoint = None
    try:
        if kind == 'symlink':
            target.symlink_to(private)
        elif kind == 'hardlink':
            os.link(private, target)
        elif kind == 'fifo':
            os.mkfifo(target)
        else:
            endpoint = socket.socket(socket.AF_UNIX)
            monkeypatch.chdir(config.public_dir)
            endpoint.bind(target.name)
        with pytest.raises(ValueError, match='regular'):
            ContainerWorkflowRunner(config)
    finally:
        if endpoint:
            endpoint.close()


def test_declared_private_evidence_cannot_be_mounted(tmp_path):
    config = make_runtime(tmp_path)
    config.private_paths = (config.public_dir / 'reference.json',)
    with pytest.raises(ValueError, match='private'):
        ContainerWorkflowRunner(config)


def test_cwd_cannot_shadow_host_completion_worker(tmp_path, monkeypatch):
    config = make_runtime(tmp_path, request_code())
    shadow = config.public_dir / 'sources'
    shadow.mkdir()
    marker = tmp_path / 'shadow-imported'
    (shadow / '__init__.py').write_text(f'open({str(marker)!r},"w").write("executed")')
    monkeypatch.chdir(config.public_dir)
    result = asyncio.run(ContainerWorkflowRunner(config).execute())
    assert result.status == 'completed', result
    assert not marker.exists()


def test_host_worker_does_not_borrow_unrelated_secret_environment(tmp_path, monkeypatch):
    config = make_runtime(tmp_path, request_code())
    bridge = Path(config.settings['bridge_path'])
    code = bridge.read_text().replace('def complete(request):', 'import os\ndef complete(request):\n    assert "ORACLE_PRIVATE_SENTINEL" not in os.environ')
    bridge.write_text(code)
    import hashlib
    config.settings['bridge_sha256'] = hashlib.sha256(bridge.read_bytes()).hexdigest()
    monkeypatch.setenv('ORACLE_PRIVATE_SENTINEL', 'host-only-fixture')
    result = asyncio.run(ContainerWorkflowRunner(config).execute())
    assert result.status == 'completed', result
