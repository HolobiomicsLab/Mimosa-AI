"""A failing plan step must be attributable.

The step-execution handler wrapped the cause as
`Exception(f"Critical error in step execution: {e}")`. `from e` preserves the
chain for a Python caller, but the operator only ever sees the formatted
string — so a bare `TypeError: unhashable type: 'slice'` arrived with no file,
no line, and no exception type. Observed on a real run that had already
produced its deliverable and written its ASTRA capsule, which made the failure
look like a total loss when it wasn't.

Planner also had no logger at all, so the first attempt at this fix would have
raised AttributeError from inside the error handler.
"""

import logging

from sources.core.planner import Planner


class _Cfg:
    workspace_dir = "/tmp/x"
    planner_llm_model = "anthropic/claude-haiku-4-5"
    reasoning_effort = "medium"
    max_tokens = 8192
    pushover_token = None
    pushover_user = None
    memory_dir = "/tmp/x"


def test_planner_has_a_logger():
    """The error handler logs through it; without one it raises inside except."""
    assert isinstance(logging.getLogger("sources.core.planner"), logging.Logger)
    src = open("sources/core/planner.py").read()
    assert "self.logger = logging.getLogger(__name__)" in src
    assert "import logging" in src


def test_step_error_names_the_exception_type():
    """`unhashable type: 'slice'` alone is unattributable; the type narrows it."""
    src = open("sources/core/planner.py").read()
    assert 'f"❌ Critical error in step execution: {type(e).__name__}: {e}"' in src


def test_step_error_logs_the_traceback():
    src = open("sources/core/planner.py").read()
    assert "self.logger.exception(" in src


def test_original_exception_is_still_chained():
    """`from e` must survive — it is what a Python caller unwraps.

    Anchored on the ``raise`` itself, not on the message text: the phrase also
    appears in prose elsewhere in the module, and matching the first occurrence
    made this test read the wrong block.
    """
    src = open("sources/core/planner.py").read()
    handler = src[src.index('raise Exception(\n                        f"❌ Critical error in step execution'):]
    assert "from e" in handler[:400]
