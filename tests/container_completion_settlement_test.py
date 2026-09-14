"""A bridge deadline must leave bounded time for its durable terminal receipt."""

import asyncio
import json
from pathlib import Path
import time

import pytest

from container_runner_test import make_runtime, request_code
from sources.core.container_runner import ContainerWorkflowRunner
from sources.core.harness_model import HarnessCompletionModel, HarnessBudgetError


@pytest.mark.parametrize('outcome', ['timeout', 'raises'])
def test_bridge_deadline_can_settle_one_terminal_receipt(tmp_path, outcome):
    bridge = '''import time
def complete(request):
    time.sleep(request['timeout_seconds'] + 0.2)
'''
    if outcome == 'raises':
        bridge += "    raise RuntimeError('offline bridge deadline')\n"
    else:
        bridge += '''    return dict(protocol_version=1, status='timeout', text='',
        requested_model=request['model'], actual_model=None, backend='codex_cli',
        auth_mode='subscription', usage_kind='chatgpt_subscription', usage=None,
        model_identity={'requested':{'model':request['model'],'source':'request'},
                        'configured':{'model':request['model'],'source':'explicit_cli_argument'}, 'reported':None},
        cost_usd=None,cost_kind='unavailable',cli_version='offline-fixture',diagnostic_count=0,error='offline timeout')
'''
    config = make_runtime(tmp_path, request_code(), bridge, timeout_seconds=12)
    config.settings.update(call_timeout_seconds=1, max_calls=2)
    result = asyncio.run(ContainerWorkflowRunner(config).execute())
    rows = [json.loads(line) for line in Path(config.settings['ledger_path']).read_text().splitlines()]
    assert [row['event'] for row in rows] == ['policy', 'reservation', 'completion']
    assert rows[-1]['status'] == ('unknown' if outcome == 'raises' else 'timeout')
    assert rows[1]['timeout_seconds'] == 1
    assert rows[-1]['usage'] is None
    assert result.status == 'failed' and result.cleanup_verified
    before = Path(config.settings['ledger_path']).read_bytes()
    model = HarnessCompletionModel(config.model_id, config.settings)
    with pytest.raises(HarnessBudgetError):
        model.generate([{'role': 'user', 'content': 'Another synthetic request.'}])
    assert Path(config.settings['ledger_path']).read_bytes() == before


@pytest.mark.parametrize('workflow_timeout', [0.5, 15])
def test_worker_settlement_remains_bounded_by_both_deadlines(tmp_path, workflow_timeout):
    bridge = "import time\ndef complete(request):\n    time.sleep(60)\n"
    config = make_runtime(tmp_path, request_code(), bridge, timeout_seconds=workflow_timeout)
    config.settings['call_timeout_seconds'] = 1
    started = time.monotonic()
    result = asyncio.run(ContainerWorkflowRunner(config).execute())
    elapsed = time.monotonic() - started
    assert result.status == 'timeout' and result.cleanup_verified
    assert elapsed < (3 if workflow_timeout == 0.5 else 10)
