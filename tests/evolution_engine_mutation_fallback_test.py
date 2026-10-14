#!/usr/bin/env python3
"""Mutation fallback when parent selection finds no parent.

The real ``evolve_generation`` runs with a real ``VariationEngine``; only
the orchestrator, the evaluator and the directive LLM are stubbed. With
no selectable parent, the next mutation must change the generation that
just ran, so its code, verifier gradient and run log describe one run.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolution_engine_stop_test import _make_engine  # noqa: E402
from variation_engine_test import _FakeConfig, _FakeProvider  # noqa: E402

import sources.core.variation_engine as variation_engine_module  # noqa: E402
from sources.core.schema import IndividualRun  # noqa: E402
from sources.core.variation_engine import VariationEngine  # noqa: E402


class _GenotypeOrchestrator:
    """Writes each genotype to disk like the real orchestrator does."""

    def __init__(self, wf_dir: Path, fail: bool = False):
        self.wf_dir = wf_dir
        self.fail = fail
        self.uuids: list[str] = []

    async def orchestrate_workflow(self, **kw):
        n = len(self.uuids) + 1
        uuid = f"20261014_000000_{2000 + n:08d}"
        self.uuids.append(uuid)
        if self.fail:
            return f"workflow failed in generation {n}", uuid, "", False
        code = f"print('generation {n} code')"
        folder = self.wf_dir / uuid
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"workflow_genotype_{uuid}.py").write_text(code, encoding="utf-8")
        return f"stdout of generation {n}", uuid, code, True


def _run_three_generations(tmp_path: Path, monkeypatch, fail: bool):
    """Drive seed → no-parent mutation → cap; return runs and orchestrator."""
    monkeypatch.setattr(variation_engine_module, "LLMProvider", _FakeProvider)
    orch = _GenotypeOrchestrator(tmp_path / "workflows", fail=fail)
    engine, _, _ = _make_engine(tmp_path, orch)
    engine.variation = VariationEngine(_FakeConfig())
    evaluate = engine._evaluate_and_calculate_cost

    async def _evaluate_with_gradient(executed, judge, uuid, *rest):
        """Write a real-shaped state result and the verifier gradient."""
        result = await evaluate(executed, judge, uuid, *rest)
        folder = orch.wf_dir / uuid
        state_file = folder / "state_result.json"
        state = json.loads(state_file.read_text(encoding="utf-8"))
        state["step_name"] = ["solver"]
        state_file.write_text(json.dumps(state), encoding="utf-8")
        n = orch.uuids.index(uuid) + 1
        (folder / "textual_gradient.txt").write_text(
            f"NO_RUN_SH in generation {n}", encoding="utf-8"
        )
        return result

    engine._evaluate_and_calculate_cost = _evaluate_with_gradient
    runs = asyncio.run(
        engine.evolve_generation(
            [IndividualRun(goal="g", prompt="p", max_depth=3)],
            rewards_history=[],
            assertion_history=[],
            enable_evolution=True,
        )
    )
    return runs, orch


def _lineage(orch: _GenotypeOrchestrator, uuid: str) -> dict:
    """Read the lineage sidecar of one generation."""
    path = orch.wf_dir / uuid / f"lineage_{uuid}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_no_parent_mutates_the_last_generation(tmp_path: Path, monkeypatch):
    runs, orch = _run_three_generations(tmp_path, monkeypatch, fail=False)
    child = orch.uuids[1]
    assert (runs[2].evolution_kind, runs[2].parent_uuids) == ("mutation", [child])
    lineage = _lineage(orch, orch.uuids[2])
    assert (lineage["evolution_kind"], lineage["parents"]) == ("mutation", [child])
    # The directive LLM saw the child's own gradient, not another run's.
    directive_input = _FakeProvider.last.prompt
    assert "<diagnosis>\nNO RUN SH in generation 2\n</diagnosis>" in directive_input
    assert "generation 1" not in directive_input
    assert "print('generation 2 code')" in runs[2].prompt


def test_failed_child_without_genotype_stays_parentless(tmp_path: Path, monkeypatch):
    runs, orch = _run_three_generations(tmp_path, monkeypatch, fail=True)
    assert (runs[2].evolution_kind, runs[2].parent_uuids) == ("seed", [])
    lineage = _lineage(orch, orch.uuids[2])
    assert (lineage["evolution_kind"], lineage["parents"]) == ("seed", [])
    # The failed run's log is not evidence for code it did not produce.
    assert "workflow failed in generation 2" not in _FakeProvider.last.prompt


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
