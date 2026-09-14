"""Tests for VariationEngine observer diagnostics, search-state block and
parent-centered mutation budget."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import sources.core.variation_engine as variation_engine_module
from sources.core.variation_engine import VariationEngine


class _FakeConfig:
    """Minimal stand-in for the real Config, just enough to satisfy
    VariationEngine.setup_llm without touching any network/provider code."""

    workflow_llm_model = "openai/gpt-4o-mini"
    reasoning_effort = "low"
    max_tokens = 8192

    def openrouter_provider_for(self, model):
        return None

    def openrouter_quantizations_for(self, model):
        return None


def _make_engine():
    return VariationEngine(_FakeConfig())


class _FakeProvider:
    """Offline stand-in for LLMProvider: captures prompts, returns canned text."""

    last = None

    def __init__(self, system_msg=None, config=None, **kwargs):
        self.system_msg = system_msg or ""
        self.config = config
        self.prompt = None
        _FakeProvider.last = self

    def __call__(self, prompt):
        self.prompt = prompt
        return (
            "Directive: make the solver prompt more domain-specific. "
            "Kind: prompt tweak. Magnitude: small tweak."
        )


class _FakeWorkflowInfo:
    """Duck-typed stand-in for WorkflowInfo used by mutation_prompt."""

    def __init__(self, score=0.7, gradient="claim verification failed", state=None):
        self.overall_score = score
        self.abstracted_textual_gradient = gradient
        self.state_result = state if state is not None else {
            "step_name": ["solver", "verifier"],
            "answers": ["did x", "did y"],
        }


# ── _iters_since_improvement ──────────────────────────────────────────────


def test_iters_since_improvement_empty_history_is_zero():
    ve = _make_engine()
    assert ve._iters_since_improvement() == 0


def test_iters_since_improvement_no_scored_entries_is_zero():
    ve = _make_engine()
    for _ in range(4):
        ve.record_offspring_gradient("crash", is_failure=True)
    ve.record_offspring_gradient("no scores attached")  # child_score = None
    assert ve._iters_since_improvement() == 0


def test_iters_since_improvement_counts_consecutive_non_improvers():
    ve = _make_engine()
    ve.record_offspring_gradient("up",   child_score=0.30, best_before=0.20)
    ve.record_offspring_gradient("flat", child_score=0.30, best_before=0.30)
    ve.record_offspring_gradient("flat", child_score=0.30, best_before=0.30)
    ve.record_offspring_gradient("flat", child_score=0.30, best_before=0.30)
    assert ve._iters_since_improvement() == 3


def test_iters_since_improvement_resets_after_strict_improvement():
    ve = _make_engine()
    for _ in range(4):
        ve.record_offspring_gradient("flat", child_score=0.5, best_before=0.5)
    ve.record_offspring_gradient("up", child_score=0.6, best_before=0.5)
    assert ve._iters_since_improvement() == 0


def test_iters_since_improvement_skips_failures_and_none_entries():
    """Failures and unscored entries are transparent: they neither break nor
    extend the streak. The streak walks through them as if they weren't there."""
    ve = _make_engine()
    ve.record_offspring_gradient("flat", child_score=0.5, best_before=0.5)
    ve.record_offspring_gradient("crash", is_failure=True)
    ve.record_offspring_gradient("nul",   child_score=None, best_before=0.5)
    ve.record_offspring_gradient("flat", child_score=0.5, best_before=0.5)
    assert ve._iters_since_improvement() == 2


def test_iters_since_improvement_tie_is_not_an_improvement():
    """``c == b`` is a tie, not a strict improvement: the streak must NOT
    reset on equality."""
    ve = _make_engine()
    ve.record_offspring_gradient("flat", child_score=0.5, best_before=0.5)
    ve.record_offspring_gradient("flat", child_score=0.5, best_before=0.5)
    assert ve._iters_since_improvement() == 2


# ── _compute_success_rate (kept; unchanged contract) ──────────────────────


def test_compute_success_rate_none_when_no_scored_history():
    ve = _make_engine()
    ve.record_offspring_gradient("only a gradient, no score")
    assert ve._compute_success_rate() is None


def test_compute_success_rate_counts_strict_improvements():
    ve = _make_engine()
    # 2 improvements out of 4 ⇒ success_rate = 0.5
    ve.record_offspring_gradient("a", child_score=0.30, best_before=0.20)
    ve.record_offspring_gradient("b", child_score=0.30, best_before=0.30)  # tie ≠ improvement
    ve.record_offspring_gradient("c", child_score=0.40, best_before=0.30)
    ve.record_offspring_gradient("d", child_score=0.40, best_before=0.40)  # tie ≠ improvement
    assert ve._compute_success_rate() == 0.5


def test_compute_success_rate_excludes_failures():
    ve = _make_engine()
    ve.record_offspring_gradient("ok", child_score=0.50, best_before=0.40)
    ve.record_offspring_gradient(
        "crash", is_failure=True, child_score=0.99, best_before=0.40
    )
    ve.record_offspring_gradient("ok", child_score=0.55, best_before=0.50)
    assert ve._compute_success_rate() == 1.0


# ── Search-state block (deterministic observer) ───────────────────────────


def test_search_state_block_reports_observer_fields_only():
    ve = _make_engine()
    for _ in range(5):
        ve.record_offspring_gradient("stuck", child_score=0.92, best_before=0.92)
    block = ve._search_state_block(parent_score=0.92, iteration_count=4, max_iterations=10)

    assert "parent_score: 0.92" in block
    assert "iteration: 5 of 10" in block
    assert "iterations_since_improvement: 5" in block
    assert "success_rate_last_5: 0.00" in block
    assert "recent_scores_oldest_first: 0.92" in block

    state = ve.last_variation_state
    assert state["iters_since_improvement"] == 5
    assert abs(state["plateau"] - 5 / 6) < 1e-9
    assert state["success_rate"] == 0.0
    assert state["parent_score"] == 0.92
    # Controller fields are gone for good.
    assert "effective_boldness" not in state
    assert "respeciation_gate_open" not in state


def test_search_state_trajectory_lists_last_five_scored_children():
    ve = _make_engine()
    for c in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
        ve.record_offspring_gradient("g", child_score=c, best_before=0.05)
    block = ve._search_state_block(parent_score=0.7, iteration_count=7, max_iterations=10)
    assert "recent_scores_oldest_first: 0.30, 0.40, 0.50, 0.60, 0.70" in block


def test_search_state_block_contains_no_diagnosis_text():
    ve = _make_engine()
    ve.record_offspring_gradient("DATA_LEAKAGE: rubric claim 3 failed", child_score=0.4, best_before=0.4)
    block = ve._search_state_block(parent_score=0.4, iteration_count=1, max_iterations=10)
    assert "DATA_LEAKAGE" not in block
    assert "rubric" not in block


# ── Directive prompt (offline, mocked provider) ────────────────────────────


def test_directive_prompt_has_search_state_and_no_boldness(monkeypatch):
    monkeypatch.setattr(variation_engine_module, "LLMProvider", _FakeProvider)
    ve = _make_engine()
    search_state = ve._search_state_block(parent_score=0.5, iteration_count=2, max_iterations=10)
    ve.llm_think_mutation_directive(
        agent_answers="agent solver: did x",
        textual_gradient_block="claim verification failed",
        search_state=search_state,
        goal="solve the task",
    )
    captured = _FakeProvider.last
    assert "<search_state>" in captured.prompt
    assert "parent_score: 0.50" in captured.prompt
    assert "success_rate_last_5" in captured.prompt
    for banned in ("boldness", "<boldness>", "band", "EXPLOITATION", "RE-SPECIATION"):
        assert banned.lower() not in captured.prompt.lower()
        assert banned.lower() not in captured.system_msg.lower()


def test_directive_flow_embedded_in_mutation_prompt(monkeypatch):
    monkeypatch.setattr(variation_engine_module, "LLMProvider", _FakeProvider)
    ve = _make_engine()
    prompt = ve.mutation_prompt(
        goal="solve the task",
        wf_info=_FakeWorkflowInfo(),
        genotype="# parent code",
        run_stderr="",
        iteration_count=1,
        max_iterations=10,
    )
    assert "<directive>" in prompt
    assert "prompt tweak" in prompt  # canned directive text flows through
    # Hard guardrails stay.
    assert "Do not add or remove more than 1 agent at a time" in prompt
    assert "keep 90% of the previous workflow prompts and code unchanged" in prompt
    assert "Do not change the workflow's overall topology" in prompt


def test_mutation_prompt_has_no_scope_band_text(monkeypatch):
    monkeypatch.setattr(variation_engine_module, "LLMProvider", _FakeProvider)
    ve = _make_engine()
    prompt = ve.mutation_prompt(
        goal="solve the task",
        wf_info=_FakeWorkflowInfo(),
        genotype="# parent code",
        run_stderr="",
        iteration_count=1,
        max_iterations=10,
    )
    for banned in (
        "Mutation scope:",
        "Boldness",
        "boldness",
        "EXPLOITATION",
        "ALIGNMENT",
        "ADAPTATION",
        "EXPLORATION",
        "RE-SPECIATION",
    ):
        assert banned.lower() not in prompt.lower(), banned


# ── Parent-centered mutation agent budget ──────────────────────────────────


def test_mutation_agent_budget_is_parent_centered(monkeypatch):
    monkeypatch.setattr(variation_engine_module, "LLMProvider", _FakeProvider)
    ve = _make_engine()
    state = {"step_name": ["solver", "verifier", "repair"], "answers": ["a", "b", "c"]}
    for _ in range(25):
        ve.mutation_prompt(
            goal="g", wf_info=_FakeWorkflowInfo(state=state), genotype="code",
            run_stderr="", iteration_count=1, max_iterations=10,
        )
        budget = ve.last_variation_state["agent_budget"]
        assert 2 <= budget <= 4  # parent has 3 agents → within ±1
        assert 1 <= budget <= ve.max_possible_agents


def test_mutation_agent_budget_counts_distinct_agents_only(monkeypatch):
    """Retry loops repeat step names; distinct agents are what counts."""
    monkeypatch.setattr(variation_engine_module, "LLMProvider", _FakeProvider)
    ve = _make_engine()
    state = {"step_name": ["solver", "verifier", "solver", "verifier"],
             "answers": ["a", "b", "c", "d"]}
    ve.mutation_prompt(
        goal="g", wf_info=_FakeWorkflowInfo(state=state), genotype="code",
        run_stderr="", iteration_count=1, max_iterations=10,
    )
    assert 1 <= ve.last_variation_state["agent_budget"] <= 3  # 2 distinct agents


def test_mutation_agent_budget_caps_at_max_possible_agents(monkeypatch):
    monkeypatch.setattr(variation_engine_module, "LLMProvider", _FakeProvider)
    ve = _make_engine()
    state = {"step_name": [f"agent{i}" for i in range(9)], "answers": ["a"] * 9}
    for _ in range(10):
        ve.mutation_prompt(
            goal="g", wf_info=_FakeWorkflowInfo(state=state), genotype="code",
            run_stderr="", iteration_count=1, max_iterations=10,
        )
        assert 6 <= ve.last_variation_state["agent_budget"] <= 7


def test_mutation_agent_budget_falls_back_to_history(monkeypatch):
    """Without step names the last sampled budget anchors the window."""
    monkeypatch.setattr(variation_engine_module, "LLMProvider", _FakeProvider)
    ve = _make_engine()
    state = {"answers": "raw string, no step_name"}
    ve.mutation_prompt(
        goal="g", wf_info=_FakeWorkflowInfo(state=state), genotype="code",
        run_stderr="", iteration_count=1, max_iterations=10,
    )
    assert 1 <= ve.last_variation_state["agent_budget"] <= 2  # no history → 1


def test_seed_prompt_budget_stays_in_one_to_four():
    import re

    ve = _make_engine()
    for _ in range(25):
        prompt = ve.seed_genome_prompt("g")
        m = re.search(r"maximum (\d+) agents", prompt)
        assert m and 1 <= int(m.group(1)) <= 4


# ── Controller-removal guard (grep-based) ──────────────────────────────────


def test_no_rechenberg_controller_left_in_sources():
    """Guard: no source file may reference the removed controller."""
    repo_root = Path(__file__).parent.parent
    skip_parts = {"__pycache__", "workflows", "memory", "cache"}
    offenders = []
    for py in (repo_root / "sources").rglob("*.py"):
        if skip_parts & set(py.parts):
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        for banned in ("_get_prompt_step_size", "effective_boldness",
                       "respeciation_gate_open", "_RESPECIATION_"):
            if banned in text:
                offenders.append(f"{py}: {banned}")
    assert not offenders, offenders


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✓ {name}")
    print("All variation_engine_test passed.")
