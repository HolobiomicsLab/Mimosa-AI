"""live._classify: file changes become semantic events, scoped per root."""

from __future__ import annotations

from app.live import _classify

MEM = "/data/memory"
RUN = "20260822_183340_88cdd1fe"


def _etype(path: str) -> str | None:
    ev = _classify(path, MEM)
    return ev["type"] if ev else None


def test_workflow_dir_events_keep_their_types():
    assert _etype(f"/data/workflows/{RUN}/run_metrics.json") == "iteration_complete"
    assert _etype(f"/data/workflows/{RUN}/state_result.json") == "execution_complete"
    assert _etype(f"/data/workflows/{RUN}/workflow_genotype_{RUN}.py") == "workflow_crafted"
    assert _etype(f"/data/workflows/{RUN}/textual_gradient.txt") == "gradient_updated"
    assert _etype(f"/data/workflows/{RUN}/evaluation.txt") == "evaluation_updated"


def test_capsule_and_eval_layers_emit_decision_events():
    ev = _classify(f"/data/capsules/{RUN}/astra.yaml", MEM)
    assert ev == {"type": "astra_updated", "run_id": RUN, "filename": "astra.yaml"}
    ev = _classify("/data/evals/p_iimn/task_001/eval_astra.yaml", MEM)
    assert ev is not None and ev["type"] == "evaluation_capsule_updated"
    assert ev["run_id"] is None  # the eval tree carries no run id in its path


def test_memory_steps_and_calls_are_scoped_to_the_memory_root():
    assert _etype(f"{MEM}/{RUN}/task_data_analyst.json") == "step_appended"
    assert _etype(f"{MEM}/{RUN}/workflow_creator.json") == "llm_call_logged"
    # the same shapes OUTSIDE memory_dir are not call events
    assert _etype(f"/data/workflows/{RUN}/rubric_cache_1.json") is None
    assert _etype(f"/data/workflows/{RUN}/lineage_{RUN}.json") is None


def test_run_id_extraction_handles_single_agent_prefix():
    ev = _classify(f"{MEM}/single_agent_{RUN}/task_main.json", MEM)
    assert ev is not None and ev["run_id"] == f"single_agent_{RUN}"
