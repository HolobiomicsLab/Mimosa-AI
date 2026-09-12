"""Core components remain usable without unrelated CLI/MCP initialization."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BLOCK_UNRELATED = '''
import importlib.abc
import sys

class RejectUnrelated(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("sources.cli", "fastmcp", "sources.core.orchestrator")):
            raise ImportError("unrelated dependency blocked: " + fullname)

sys.meta_path.insert(0, RejectUnrelated())
'''


def run_fresh(source):
    """Observe imports in a fresh process, without cached packages or API calls."""
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", LITELLM_LOCAL_MODEL_COST_MAP="True")
    result = subprocess.run([sys.executable, "-c", BLOCK_UNRELATED + source],
                            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module", ["harness_model", "native_agent", "harness_budget", "declared_outputs"])
def test_independent_submodules_do_not_load_optional_stacks(module):
    run_fresh(f'''
import importlib
importlib.import_module("sources.core.{module}")
assert not any(name.startswith(("sources.cli", "fastmcp", "sources.core.orchestrator")) for name in sys.modules)
''')


def test_existing_public_exports_preserve_identity_and_discovery():
    run_fresh('''
import sources.core as core
from sources.core import TaskStatus, WorkflowInfo, SelectionStrategy
from sources.core.schema import TaskStatus as original_status
from sources.core.workflow_info import WorkflowInfo as original_info
from sources.core.selection import SelectionStrategy as original_strategy
assert (TaskStatus, WorkflowInfo, SelectionStrategy) == (original_status, original_info, original_strategy)
assert core.TaskStatus is TaskStatus
assert core.__dict__["TaskStatus"] is TaskStatus
assert set(core.__all__).issubset(dir(core))
assert not hasattr(core, "unrecognized_public_export")
assert "sources.core.orchestrator" not in sys.modules
''')


def test_requested_component_still_reports_its_dependency_failure():
    run_fresh('''
import sources.core as core
try:
    core.WorkflowOrchestrator
except ImportError as error:
    assert str(error) == "unrelated dependency blocked: sources.core.orchestrator"
else:
    raise AssertionError("Requested component dependency error was hidden")
assert "WorkflowOrchestrator" not in core.__dict__
''')


def test_cli_printer_exports_do_not_initialize_interactive_commands():
    run_fresh('''
sys.meta_path.pop(0)
class RejectInteractive(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ("sources.cli.onboard_cli", "sources.cli.evaluation_cli", "sources.cli.memory_chat_cli"):
            raise ImportError("interactive command blocked: " + fullname)
sys.meta_path.insert(0, RejectInteractive())
import sources.cli as cli
from sources.cli import print_ok, CYAN
from sources.cli.pretty_print import print_ok as original_print, CYAN as original_color
assert print_ok is original_print
assert CYAN == original_color
assert set(cli.__all__).issubset(dir(cli))
assert not hasattr(cli, "unrecognized_public_export")
assert "sources.core.tools_manager" not in sys.modules
''')
