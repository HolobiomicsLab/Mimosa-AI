#!/usr/bin/env python3
"""A stderr line longer than asyncio's 64 KiB line limit must not kill a run.

2026-10-15 campaign: generation 20261015_130529_db010f99 ended with
"[RUNTIME_ERROR] Workflow execution failed: Separator is not found, and chunk
exceed the limit" after 16 minutes of agent work, because the runner read the
child's stderr line by line. Both the PTY and the pipe path are checked.
"""

import asyncio
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.core.workflow_runner import (  # noqa: E402
    ExecutionStatus,
    RuntimeConfig,
    WorkflowRunner,
)

LINE_CHARS = 200_000
SCRIPT = (
    "import sys\n"
    f"sys.stderr.write('é' * {LINE_CHARS})\n"
    "sys.stderr.flush()\n"
    "print('done')\n"
)


@pytest.mark.parametrize("use_pty", [True, False])
def test_long_stderr_line_completes_with_full_text(tmp_path, use_pty):
    config = RuntimeConfig(python_executable=sys.executable, timeout=60,
                           temp_dir=tmp_path, use_pty=use_pty)
    runner = WorkflowRunner(config, execution_dir=str(tmp_path))
    result = asyncio.run(runner.execute(SCRIPT, execution_id="long_stderr"))
    assert result.status == ExecutionStatus.COMPLETED, result.stderr[:300]
    assert result.stderr == "é" * LINE_CHARS  # multi-byte chars split across chunks
    assert "done" in result.stdout
