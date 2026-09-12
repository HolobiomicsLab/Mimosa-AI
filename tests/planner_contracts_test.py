"""Planner integration for opt-in canonical artifact contracts."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from config import Config
from sources.core.artifact_contracts import (
    ArtifactValidationError,
    ContractValidationError,
    load_artifact_contract,
)
from sources.core.evolution_engine import EvolutionEngine
from sources.core.planner import DependencyError, Planner, UserInterventionRequired
from sources.core.schema import IndividualRun, Task, TaskStatus


def _write_contract(tmp_path: Path, *, supplied: bool = True) -> Path:
    supplied_payload = b'{"rate":8.0}'
    document = {
        "schema": "mimosa-artifact-contract/v1",
        "artifacts": [
            {
                "id": "intermediate",
                "path": "input/intermediate.json",
                "schema": {
                    "type": "object",
                    "properties": {"rate": {"type": "number"}},
                    "required": ["rate"],
                    "additionalProperties": False,
                },
                "units": {"/rate": "Hz"},
            },
            {
                "id": "result",
                "path": "output/result.json",
                "schema": {
                    "type": "object",
                    "properties": {"doubled": {"type": "number"}},
                    "required": ["doubled"],
                    "additionalProperties": False,
                },
                "units": {"/doubled": "Hz"},
            },
        ],
        "steps": [
            {
                "name": "produce_intermediate",
                "task": "Produce the intermediate from an unavailable upstream source.",
                "complexity": "high",
                "inputs": [],
                "outputs": ["intermediate"],
            },
            {
                "name": "double_rate",
                "task": "Read the supplied rate and write twice its value.",
                "complexity": "low",
                "inputs": ["intermediate"],
                "outputs": ["result"],
            },
        ],
        "targets": ["result"],
        "supplied": (
            [{"artifact": "intermediate", "sha256": hashlib.sha256(supplied_payload).hexdigest()}]
            if supplied
            else []
        ),
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    if supplied:
        artifact = tmp_path / "input" / "intermediate.json"
        artifact.parent.mkdir()
        artifact.write_bytes(supplied_payload)
    return path


def _plan(contract):
    return contract.project_plan(
        {
            "contract_digest": contract.digest,
            "steps": [{"name": name} for name in contract.required_step_names],
        },
        "Double a small measurement.",
    )


def _planner(tmp_path: Path, contract):
    planner = object.__new__(Planner)
    planner.workspace_path = str(tmp_path)
    planner.task_history = []
    planner.current_plan = _plan(contract)
    planner._active_contract = contract
    planner._contract_tasks = {}
    planner._workspace_files_before_step = []
    planner.tts = None
    planner.logger = logging.getLogger("planner_contracts_test")
    planner.notifier = SimpleNamespace(send_message=lambda *args, **kwargs: None)
    planner.config = SimpleNamespace(temp_dir=str(tmp_path / "tmp"), workflow_dir=str(tmp_path))
    planner._record_declared_outputs = lambda *args, **kwargs: None
    return planner


def _successful_run(reward: float = 0.9):
    return IndividualRun(
        goal="g",
        prompt="p",
        reward=reward,
        cost=1.5,
        state_result={"success": [True]},
        answers=[{"status": "SUCCESS"}],
        current_uuid="run-1",
    )


def test_config_contract_path_survives_round_trip(tmp_path: Path):
    path = _write_contract(tmp_path).resolve()
    original = Config()
    original.planner_contract_path = str(path)
    restored = Config()
    restored.from_json(original.jsonify())
    assert restored.planner_contract_path == str(path)


def test_configured_empty_contract_path_does_not_fall_back_to_legacy():
    planner = object.__new__(Planner)
    planner.config = SimpleNamespace(planner_contract_path="")

    with pytest.raises(ContractValidationError, match="absolute"):
        planner._resolve_artifact_contract()


def test_legacy_plan_is_visibly_unchecked():
    planner = object.__new__(Planner)
    plan = planner._parse_and_validate_plan(
        {"steps": [{"name": "legacy", "task": "do work"}]}, "legacy goal"
    )
    assert plan.contract_status == "legacy_unchecked"
    assert plan.steps[0].contract_status == "legacy_unchecked"


def test_direct_run_revalidates_seal_before_any_work(tmp_path: Path):
    import asyncio

    contract = load_artifact_contract(_write_contract(tmp_path))
    planner = _planner(tmp_path, contract)
    step = planner.current_plan.steps[0]
    step.task = "mutated after plan admission"
    planner.evolve_runs = AsyncMock(side_effect=AssertionError("must not execute"))

    with pytest.raises(ContractValidationError, match="changed after admission"):
        asyncio.run(planner.run_attempts({}, 1, step, True))
    planner.evolve_runs.assert_not_awaited()


def test_contract_output_must_validate_on_final_attempt(tmp_path: Path):
    import asyncio

    contract = load_artifact_contract(_write_contract(tmp_path))
    planner = _planner(tmp_path, contract)
    planner.evolve_runs = AsyncMock(return_value=[_successful_run()])

    with patch("sources.core.planner.time.sleep", return_value=None):
        step = asyncio.run(
            planner.run_attempts({}, 1, planner.current_plan.steps[0], True)
        )

    assert step.status is TaskStatus.FAILED
    assert planner.task_history[-1].status is TaskStatus.FAILED
    assert "output/result.json" in step.missing_outputs


def test_valid_contract_run_records_hashes_without_producer_credit(tmp_path: Path):
    import asyncio

    contract = load_artifact_contract(_write_contract(tmp_path))
    planner = _planner(tmp_path, contract)

    async def execute(*args, **kwargs):
        output = tmp_path / "output" / "result.json"
        output.parent.mkdir()
        output.write_text('{"doubled":16.0}', encoding="utf-8")
        return [_successful_run()]

    planner.evolve_runs = execute
    with patch("sources.core.planner.time.sleep", return_value=None):
        step = asyncio.run(
            planner.run_attempts({}, 1, planner.current_plan.steps[0], True)
        )

    assert step.status is TaskStatus.COMPLETED
    assert [task.name for task in planner.task_history] == ["double_rate"]
    task = planner.task_history[0]
    assert task.contract_status == "validated"
    assert task.input_artifact_sha256["intermediate"]
    assert task.output_artifact_sha256["result"]
    assert task.supplied_artifact_ids == ["intermediate"]


def test_direct_downstream_run_requires_completed_producer(tmp_path: Path):
    import asyncio

    contract = load_artifact_contract(_write_contract(tmp_path, supplied=False))
    planner = _planner(tmp_path, contract)
    intermediate = tmp_path / "input" / "intermediate.json"
    intermediate.parent.mkdir()
    intermediate.write_text('{"rate":8.0}', encoding="utf-8")
    planner.evolve_runs = AsyncMock(side_effect=AssertionError("must not execute"))

    with pytest.raises(DependencyError, match="produce_intermediate"):
        asyncio.run(planner.run_attempts({}, 1, planner.current_plan.steps[1], True))
    planner.evolve_runs.assert_not_awaited()


def test_downstream_input_hash_is_bound_to_producer_receipt(tmp_path: Path):
    contract = load_artifact_contract(_write_contract(tmp_path, supplied=False))
    planner = _planner(tmp_path, contract)
    intermediate = tmp_path / "input" / "intermediate.json"
    intermediate.parent.mkdir()
    original = b'{"rate":8.0}'
    intermediate.write_bytes(original)
    planner._contract_tasks["produce_intermediate"] = Task(
        name="produce_intermediate",
        description="canonical producer",
        status=TaskStatus.COMPLETED,
        output_artifact_sha256={
            "intermediate": hashlib.sha256(original).hexdigest()
        },
    )
    intermediate.write_text('{"rate":9.0}', encoding="utf-8")

    can_execute, missing = planner._can_execute_step(planner.current_plan.steps[1])

    assert can_execute is False
    assert missing == ["produce_intermediate[artifact_changed:intermediate]"]


def test_preexisting_contract_output_cannot_earn_production_credit(tmp_path: Path):
    import asyncio

    contract = load_artifact_contract(_write_contract(tmp_path))
    planner = _planner(tmp_path, contract)
    output = tmp_path / "output" / "result.json"
    output.parent.mkdir()
    output.write_text('{"doubled":16.0}', encoding="utf-8")
    planner.evolve_runs = AsyncMock(side_effect=AssertionError("must not execute"))

    with pytest.raises(ArtifactValidationError, match="before the first attempt"):
        asyncio.run(planner.run_attempts({}, 1, planner.current_plan.steps[0], True))
    planner.evolve_runs.assert_not_awaited()


def test_retry_noop_cannot_reuse_a_prior_attempt_output(tmp_path: Path):
    import asyncio

    contract = load_artifact_contract(_write_contract(tmp_path))
    planner = _planner(tmp_path, contract)
    calls = 0

    async def execute(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            output = tmp_path / "output" / "result.json"
            output.parent.mkdir()
            output.write_text('{"doubled":16.0}', encoding="utf-8")
            return [_successful_run(reward=0.1)]
        return [_successful_run()]

    planner.evolve_runs = execute
    with patch("sources.core.planner.time.sleep", return_value=None):
        step = asyncio.run(
            planner.run_attempts({}, 2, planner.current_plan.steps[0], True)
        )

    assert calls == 2
    assert step.status is TaskStatus.FAILED
    assert all(task.status is TaskStatus.FAILED for task in planner.task_history)
    assert planner.task_history[0].output_artifact_sha256["result"]


def test_retry_can_rewrite_identical_deterministic_output(tmp_path: Path):
    import asyncio

    contract = load_artifact_contract(_write_contract(tmp_path))
    planner = _planner(tmp_path, contract)
    calls = 0

    async def execute(*args, **kwargs):
        nonlocal calls
        calls += 1
        output = tmp_path / "output" / "result.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text('{"doubled":16.0}', encoding="utf-8")
        return [_successful_run(reward=0.1 if calls == 1 else 0.9)]

    planner.evolve_runs = execute
    with patch("sources.core.planner.time.sleep", return_value=None):
        step = asyncio.run(
            planner.run_attempts({}, 2, planner.current_plan.steps[0], True)
        )

    assert calls == 2
    assert step.status is TaskStatus.COMPLETED
    assert planner.task_history[0].status is TaskStatus.FAILED
    assert planner.task_history[1].status is TaskStatus.COMPLETED
    assert (
        planner.task_history[0].output_artifact_sha256
        == planner.task_history[1].output_artifact_sha256
    )


def test_low_score_contract_attempt_records_failed_task(tmp_path: Path):
    import asyncio

    contract = load_artifact_contract(_write_contract(tmp_path))
    planner = _planner(tmp_path, contract)
    planner.evolve_runs = AsyncMock(return_value=[_successful_run(reward=0.1)])

    asyncio.run(planner.run_attempts({}, 1, planner.current_plan.steps[0], True))

    assert planner.task_history[-1].status is TaskStatus.FAILED


def test_first_step_gate_is_not_skipped_in_contract_mode(tmp_path: Path):
    import asyncio

    contract_path = _write_contract(tmp_path)
    contract = load_artifact_contract(contract_path)
    planner = _planner(tmp_path, contract)
    planner.config.planner_contract_path = str(contract_path.resolve())
    planner.config.workspace_dir = str(tmp_path)
    planner.use_visualization = False
    planner.visualizer = None
    planner._generate_plan_with_human_validation = lambda goal, contract=None: _plan(contract)
    planner._can_execute_step = lambda step: (False, ["forced-first-step-block"])
    planner.run_attempts = AsyncMock(side_effect=AssertionError("must not execute"))
    planner.request_user_exit = lambda message: (_ for _ in ()).throw(
        UserInterventionRequired(message)
    )

    with pytest.raises(ValueError, match="forced-first-step-block"):
        asyncio.run(planner.start_planner("goal", max_task_retry=1))
    planner.run_attempts.assert_not_awaited()


def test_contract_start_returns_only_current_run_tasks(tmp_path: Path):
    import asyncio

    contract_path = _write_contract(tmp_path)
    contract = load_artifact_contract(contract_path)
    planner = _planner(tmp_path, contract)
    planner.task_history = [Task(name="old", description="prior run")]
    planner.config.planner_contract_path = str(contract_path.resolve())
    planner.config.workspace_dir = str(tmp_path)
    planner.use_visualization = False
    planner.visualizer = None
    planner._generate_plan_with_human_validation = lambda goal, contract=None: _plan(contract)
    planner._init_visualization = lambda plan: None
    planner._update_visualization = lambda total_cost: None
    planner._cleanup_visualization = lambda: None

    async def execute(attempt_counts, max_attempts, step, judge):
        step.status = TaskStatus.COMPLETED
        planner.task_history.append(
            Task(name=step.name, description=step.task, status=TaskStatus.COMPLETED)
        )
        return step

    planner.run_attempts = execute
    tasks = asyncio.run(planner.start_planner("current goal", max_task_retry=1))

    assert [task.name for task in tasks] == ["double_rate"]


def test_contract_prompt_disables_plan_response_cache(tmp_path: Path):
    contract_path = _write_contract(tmp_path)
    contract = load_artifact_contract(contract_path)
    planner = object.__new__(Planner)
    planner.config = SimpleNamespace(
        memory_dir=str(tmp_path),
        perspicacite_agent_grounding_enabled=False,
        planner_contract_path=str(contract_path.resolve()),
    )
    planner.config_llm = object()
    planner.notifier = SimpleNamespace(send_message=lambda *args, **kwargs: None)
    observed = {}

    class FakeProvider:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, prompt, use_cache=True):
            observed.update(prompt=prompt, use_cache=use_cache)
            return json.dumps(
                {"contract_digest": contract.digest, "steps": [{"name": "double_rate"}]}
            )

    with patch("sources.core.planner.LLMProvider", FakeProvider):
        plan = planner.make_plan("legacy system", "goal", max_retries=1)

    assert plan.contract_status == "validated"
    assert observed["use_cache"] is False
    assert contract.digest in observed["prompt"]
    assert "required_inputs" not in observed["prompt"]


def test_planner_disables_both_workflow_reuse_layers_explicitly():
    import asyncio

    planner = object.__new__(Planner)
    planner.wf_selector = SimpleNamespace(
        select_best_workflows=lambda *args, **kwargs: pytest.fail("result cache read")
    )
    observed = {}

    class FakeEvolution:
        async def start_workflow_evolution(self, **kwargs):
            observed.update(kwargs)
            return [_successful_run()]

    planner.evolve = FakeEvolution()
    runs = asyncio.run(
        planner.evolve_runs(
            "canonical task",
            True,
            cached_wf_allow=False,
            reuse_workflows=False,
        )
    )

    assert runs
    assert observed["reuse_workflows"] is False
    assert observed["template_uuid"] is None


def test_evolution_reuse_false_skips_selector_and_rejects_template():
    import asyncio

    engine = object.__new__(EvolutionEngine)
    with pytest.raises(ValueError, match="template_uuid=None"):
        asyncio.run(
            engine.start_workflow_evolution(
                "goal", template_uuid="existing", reuse_workflows=False
            )
        )

    engine.config = SimpleNamespace(max_learning_evolve_iterations=1)
    engine.selection = SimpleNamespace(_archive=[object()])
    engine.variation = SimpleNamespace(
        score_history=[], textual_gradient_history=[], agent_count_history=[]
    )
    engine.workspace_mgr = SimpleNamespace(
        begin_session=Mock(), restore_best=Mock()
    )
    engine.viz_utils = SimpleNamespace(create_rewards_curve_plot=Mock())
    engine.get_genotype_instructions = Mock(return_value="fresh seed")
    engine.select_parent_workflow = Mock(
        side_effect=AssertionError("selector must not run")
    )
    engine.evolve_generation = AsyncMock(return_value=[])

    asyncio.run(
        engine.start_workflow_evolution(
            "goal", template_uuid=None, reuse_workflows=False
        )
    )

    engine.select_parent_workflow.assert_not_called()
    assert engine.evolve_generation.await_args.kwargs["reuse_workflows"] is False
