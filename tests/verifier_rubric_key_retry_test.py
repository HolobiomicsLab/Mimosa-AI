"""The rubric cache must survive a planner retry of the same task.

``VerifierEvaluator.evaluate`` documents the guarantee: "the first run freezes
the full ranked claim list, every subsequent run reuses it verbatim". The cache
key is ``sha256(goal)``. But the goal handed to the verifier is the
*knowledge-wrapped* goal, and the retry loop prepends the previous attempt's
answer to it before re-running the same step — so the key changes on every
retry, the cache misses, and each attempt is scored against a freshly written,
differently worded rubric.

Observed on a real run (p_iimn task_001, three attempts of the step
``task_and_resource_discovery``): ``original_task_<uuid>.txt`` was byte-identical
across all three attempts (2657 bytes, md5 02b411d9…), while
``goal_<uuid>.txt`` grew 2657 -> 6505 -> 11353 bytes. The three attempts drew
21, 24 and 20 claims with disjoint ids and scored 0.000 / 0.649 / 0.557 — three
numbers on three different scales, which the engine then compared with ``max``.

``WorkflowInfo`` already keeps the unwrapped task precisely for this reason
("The original unwrapped task for similarity matching"), and both
``record_lineage`` and the variation prompts already prefer
``original_task or goal``. The verifier's cache key is the one place that
did not.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(_REPO_ROOT))

# Same import-order dance as verifier_claim_cache_test: this resolves the
# verifier <-> cli circular import for alphabetically-later tests.
from sources.core.failure_fingerprint import compute_failure_fingerprint  # noqa: E402, F401
from sources.core.workflow_info import WorkflowInfo  # noqa: E402
from sources.evaluators.verifier import VerifierEvaluator  # noqa: E402


_TASK = (
    "Document the inputs required to reproduce the IIMN spectral library for "
    "MSV000080492 and MSV000083472, and write /workspace/reproduction_iimn.md."
)

_WRAPPER = (
    "From previous tasks you learned:\n"
    "* From task 'task_and_resource_discovery':\n"
    "\t - {'status': 'SUCCESS', 'approach': 'HTTP HEAD probes', 'errors': "
    "['MassIVE MSV000080492: HTTP 403']}\n\n"
    "Now, use this knowledge to complete the following task:\n"
)


def _attempt_goals() -> list[str]:
    """The same step, as the retry loop re-presents it on attempts 1, 2 and 3."""
    return [_TASK, _WRAPPER + _TASK, _WRAPPER + _WRAPPER + _TASK]


def _write_workflow(root: Path, uuid: str, goal: str, original_task: str) -> WorkflowInfo:
    folder = root / uuid
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"goal_{uuid}.txt").write_text(goal)
    (folder / f"original_task_{uuid}.txt").write_text(original_task)
    wf = WorkflowInfo(uuid, folder)
    wf._goal = goal  # bypass state_result.json; the goal property reads it
    return wf


# ---------- the defect, stated as a property of the key --------------------


def test_wrapped_goal_gives_a_different_key_on_every_retry():
    """Characterises why the cache missed: the wrapper is inside the hashed text."""
    keys = {VerifierEvaluator._task_cache_key(g) for g in _attempt_goals()}
    assert len(keys) == 3, "three attempts of one task must not share a goal string"


def test_unwrapped_task_gives_one_key_across_retries():
    """The fix's premise: the unwrapped task is invariant under the retry loop."""
    keys = {VerifierEvaluator._task_cache_key(_TASK) for _ in _attempt_goals()}
    assert len(keys) == 1


# ---------- the fix, at the level the verifier actually uses ---------------


def test_workflow_info_recovers_one_task_from_three_wrapped_goals(tmp_path: Path):
    """``original_task`` is what the verifier should key on: same across attempts."""
    root = tmp_path / "workflows"
    tasks = [
        _write_workflow(root, f"uuid_{i}", goal, _TASK).original_task
        for i, goal in enumerate(_attempt_goals())
    ]
    assert set(tasks) == {_TASK}
    assert len({VerifierEvaluator._task_cache_key(t) for t in tasks}) == 1


def test_original_task_falls_back_to_unwrapping_the_goal(tmp_path: Path):
    """No ``original_task_<uuid>.txt`` on disk: unwrap the marker from the goal.

    factory.save_workflow_files skips the file on a write error, so the
    fallback path is reachable in production, not just in tests.
    """
    root = tmp_path / "workflows"
    folder = root / "no_file"
    folder.mkdir(parents=True)
    (folder / "goal_no_file.txt").write_text(_WRAPPER + _TASK)
    wf = WorkflowInfo("no_file", folder)
    wf._goal = _WRAPPER + _TASK
    assert wf.original_task.strip() == _TASK


def test_extract_claims_accepts_a_cache_key_distinct_from_the_prompt_text():
    """The extractor must key on one string while prompting the model with another.

    Keying on the unwrapped task while still showing the claim writer the full
    wrapped goal is the whole point: prior-attempt context stays in the prompt,
    the rubric stays fixed.
    """
    import inspect

    sig = inspect.signature(VerifierEvaluator._extract_claims)
    assert "cache_key_text" in sig.parameters, (
        "_extract_claims must take the cache-key text separately from `goal`"
    )
    assert sig.parameters["cache_key_text"].default in (None, ""), (
        "cache_key_text must default to falsy so existing callers keep hashing `goal`"
    )


def test_evaluate_keys_the_rubric_on_the_unwrapped_task():
    """Source guard: `evaluate` passes original_task, not the wrapped goal."""
    src = Path("sources/evaluators/verifier.py").read_text()
    call = src[src.index("claims = self._extract_claims("):]
    head = call[:400]
    assert "cache_key_text=" in head
    assert "original_task" in head, "the key must come from the unwrapped task"
