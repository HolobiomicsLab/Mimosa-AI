"""Opt-in container supervision owns deadlines and never trusts solver routing."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import psutil
import pytest

from sources.core.container_runner import ContainerRuntime, ContainerWorkflowRunner

IMAGE = 'sha256:' + 'a' * 64
FRAME = 'MIMOSA_COMPLETION_V1 '


def make_runtime(tmp_path, code='print("ok")', bridge_body=None, **changes):
    public, work, private = (tmp_path / name for name in ('public', 'work', 'private'))
    for path in (public, work, private):
        path.mkdir()
    (public / 'workflow.py').write_text(code)
    bridge = private / 'bridge.py'
    bridge.write_text(bridge_body or '''def complete(request):
    return dict(protocol_version=1, status='completed', text='result',
        requested_model=request['model'], actual_model=None, backend='codex_cli',
        auth_mode='subscription', usage_kind='chatgpt_subscription', usage=None,
        model_identity={'requested':{'model':request['model'],'source':'request'},
                        'configured':{'model':request['model'],'source':'explicit_cli_argument'}, 'reported':None},
        cost_usd=None,cost_kind='unavailable',cli_version='offline-fixture',diagnostic_count=0,error=None)
''')
    docker = private / 'docker'
    docker.write_text(f'''#!{sys.executable}
import os,sys
if sys.argv[1] == 'run':
    os.execv(sys.executable, [sys.executable, '-u', {str(public / 'workflow.py')!r}])
elif sys.argv[1] in ('rm', 'ps'):
    pass
else:
    raise SystemExit(2)
''')
    docker.chmod(0o755)
    settings = dict(bridge_path=str(bridge), bridge_sha256=hashlib.sha256(bridge.read_bytes()).hexdigest(),
                    ledger_path=str(private / 'ledger.jsonl'), max_calls=1,
                    total_timeout_seconds=30, call_timeout_seconds=10, reasoning_effort='high')
    values = dict(image=IMAGE, public_dir=public, work_dir=work, timeout_seconds=5,
                  model_id='gpt-5.6-luna', settings=settings, docker_executable=str(docker))
    values.update(changes)
    return ContainerRuntime(**values)


def request_code(**extra):
    payload = {'id':'a'*32,'messages':[{'role':'user','content':'Measure public input.'}], **extra}
    return f'import sys\nprint({FRAME + json.dumps(payload)!r},flush=True)\nprint(sys.stdin.readline(),flush=True)\n'


def alive(pidfile):
    if not pidfile.exists():
        return False
    try:
        return psutil.Process(int(pidfile.read_text())).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.parametrize('changes', [
    {'image':'mutable:latest'}, {'timeout_seconds':0}, {'timeout_seconds':float('nan')},
    {'timeout_seconds':True}, {'memory_mb':0}, {'cpus':float('inf')}, {'pids_limit':0},
])
def test_invalid_runtime_is_refused_before_dispatch(tmp_path, changes):
    with pytest.raises(ValueError):
        ContainerWorkflowRunner(make_runtime(tmp_path, **changes))


@pytest.mark.parametrize('exposed', ['bridge_path', 'ledger_path'])
def test_private_completion_paths_cannot_be_mounted(tmp_path, exposed):
    config = make_runtime(tmp_path)
    config.settings[exposed] = str(config.public_dir / 'private.json')
    with pytest.raises(ValueError, match='private'):
        ContainerWorkflowRunner(config)


def test_dirty_output_refused_before_dispatch(tmp_path):
    config = make_runtime(tmp_path)
    (config.work_dir / 'answer').write_text('prior output')
    with pytest.raises(ValueError, match='empty'):
        ContainerWorkflowRunner(config)


def test_container_argv_has_only_public_mounts_and_fixed_policy(tmp_path):
    config = make_runtime(tmp_path)
    argv = ContainerWorkflowRunner(config).container_argv('owned-container')
    for value in ('--network=none', '--read-only', '--cap-drop=ALL',
                  '--security-opt=no-new-privileges', '--pull=never', '--user=1000:1000',
                  '--entrypoint=python3', '-i'):
        assert value in argv
    assert str(tmp_path / 'private') not in ' '.join(argv[1:])
    mounts = [argv[i+1] for i,v in enumerate(argv) if v == '--mount']
    assert len(mounts) == 2 and mounts[0].endswith('dst=/data,readonly')
    assert argv[-3:] == [IMAGE, '-u', '/data/workflow.py']


def test_real_worker_preserves_model_receipt_and_private_budget(tmp_path):
    config = make_runtime(tmp_path, request_code())
    result = asyncio.run(ContainerWorkflowRunner(config).execute())
    assert result.status == 'completed', result
    assert result.return_code == 0 and result.cleanup_verified
    response = json.loads(result.stdout)
    assert response['id'] == 'a'*32
    assert response['result']['text'] == 'result'
    assert response['result']['model_identity']['reported'] is None
    assert 'bridge_path' not in result.stdout
    ledger = Path(config.settings['ledger_path'])
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [row['event'] for row in records] == ['policy', 'reservation', 'completion']


@pytest.mark.parametrize('extra', [dict(model='unapproved'),dict(backend='claude_cli'),dict(api_key_env='SECRET'),dict(path='/private')])
def test_solver_routing_override_refused_without_host_worker(tmp_path, extra):
    config = make_runtime(tmp_path, request_code(**extra))
    result = asyncio.run(ContainerWorkflowRunner(config).execute())
    assert result.status == 'failed' and result.cleanup_verified
    assert not Path(config.settings['ledger_path']).exists()


def test_host_budget_rejects_second_call_even_if_container_resets_local_state(tmp_path):
    code = request_code() + request_code()
    config = make_runtime(tmp_path, code)
    result = asyncio.run(ContainerWorkflowRunner(config).execute())
    assert result.status == 'failed' and result.cleanup_verified
    records = [json.loads(line) for line in Path(config.settings['ledger_path']).read_text().splitlines()]
    assert sum(row['event']=='reservation' for row in records) == 1


@pytest.mark.parametrize('mode', ['timeout', 'cancel'])
def test_blocked_host_completion_and_detached_child_are_stopped(tmp_path, mode):
    pidfile = tmp_path / 'worker-child.pid'
    child = f'import os,time;open({str(pidfile)!r},"w").write(str(os.getpid()));time.sleep(60)'
    bridge = f'''import subprocess,sys,time
def complete(request):
    subprocess.Popen([sys.executable,'-c',{child!r}], start_new_session=True)
    time.sleep(60)
'''
    config = make_runtime(tmp_path, request_code(), bridge, timeout_seconds=1.5 if mode=='timeout' else 15)
    async def run():
        task = asyncio.create_task(ContainerWorkflowRunner(config).execute())
        if mode == 'cancel':
            for _ in range(200):
                if pidfile.exists():
                    break
                await asyncio.sleep(0.01)
            assert pidfile.exists()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return None
        return await task
    started = time.monotonic()
    try:
        result = asyncio.run(run())
        assert pidfile.exists()
        assert not alive(pidfile), 'host completion descendant survived'
        assert time.monotonic() - started < 8
        if result:
            assert result.status == 'timeout' and result.cleanup_verified
        records = [json.loads(line) for line in Path(config.settings['ledger_path']).read_text().splitlines()]
        assert sum(row['event']=='reservation' for row in records) == 1
        assert not any(row['event']=='completion' for row in records)
    finally:
        if alive(pidfile):
            psutil.Process(int(pidfile.read_text())).kill()


def test_nonreading_container_cannot_block_response_deadline(tmp_path):
    code = request_code().split('print(sys.stdin')[0] + 'import time;time.sleep(60)\n'
    config = make_runtime(tmp_path, code, timeout_seconds=1.5)
    path = Path(config.settings['bridge_path'])
    original = path.read_text()
    path.write_text(original.replace("text='result'", "text='x'*400000"))
    config.settings['bridge_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = asyncio.run(ContainerWorkflowRunner(config).execute())
    assert result.status == 'timeout' and result.cleanup_verified


@pytest.mark.parametrize('code', [
    "print('MIMOSA_COMPLETION_V1 '+'x'*1048576,flush=True)",
    "print('MIMOSA_COMPLETION_V1 {}',end='',flush=True)",
    "print('x'*5000000,flush=True)",
])
def test_invalid_or_unbounded_output_is_terminal(tmp_path, code):
    config = make_runtime(tmp_path, code)
    result = asyncio.run(ContainerWorkflowRunner(config).execute())
    assert result.status == 'failed' and result.cleanup_verified
    assert not Path(config.settings['ledger_path']).exists()


def test_cleanup_failure_is_never_reported_verified(tmp_path):
    config = make_runtime(tmp_path)
    docker = Path(config.docker_executable)
    docker.write_text(docker.read_text().replace("elif sys.argv[1] in ('rm', 'ps'):\n    pass", "elif sys.argv[1] in ('rm', 'ps'):\n    raise SystemExit(1)"))
    result = asyncio.run(ContainerWorkflowRunner(config).execute())
    assert result.status == 'cleanup_failed' and not result.cleanup_verified


def test_reusing_runner_is_refused_even_after_failure(tmp_path):
    runner = ContainerWorkflowRunner(make_runtime(tmp_path, 'raise RuntimeError("fixture")'))
    assert asyncio.run(runner.execute()).status == 'failed'
    with pytest.raises(RuntimeError, match='used'):
        asyncio.run(runner.execute())
