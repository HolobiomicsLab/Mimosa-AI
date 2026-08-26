"""Unit tests for the post-run ASTRA exporter.

The LLM call inside :func:`extract_decisions` is monkey-patched so tests are
fast, deterministic, and offline. End-to-end behavior is verified by
asserting the schema of the emitted YAML files against the ASTRA spec
(see https://astra-spec.org/latest/specification/#decisions).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sources.transparency.decision_extractor import (
    _MALFORMED,
    Decision,
    ExtractionResult,
    Option,
    _parse_response,
    extract_decisions,
)
from sources.transparency.memory_trace import (
    extract_code,
    extract_output_text,
    reconstruct_recipe,
)
from sources.transparency.trace_compaction import (
    _MECHANICAL_CODE_PATTERNS,
    compact_step,
    compact_trace,
    load_trace,
)
from sources.transparency.yaml_writer import (
    build_analysis,
    build_universe,
    write_export,
)

_METHODOLOGICAL_STEP = {
    "step_number": 1,
    "model_output_message": {
        "content": "Variance is unequal so I use Welch's t-test instead of Student's."
    },
    "code_action": "from scipy import stats\nresult = stats.ttest_ind(a, b, equal_var=False)",
    "observations": "Ttest_indResult(statistic=2.31, pvalue=0.022)",
    "model_input_messages": [{"role": "system", "content": "x" * 50_000}],
}

_MECHANICAL_STEP = {
    "step_number": 2,
    "model_output_message": {"content": "List the workspace."},
    "code_action": "import os\nos.listdir('.')",
    "observations": "['a.csv', 'b.csv']",
}


def test_compact_step_strips_input_messages() -> None:
    compact = compact_step(_METHODOLOGICAL_STEP, index=0)
    assert "model_input_messages" not in compact
    assert "Welch" in compact["reasoning"]
    assert "ttest_ind" in compact["code"]


def test_compact_step_carries_per_step_model() -> None:
    step = dict(_METHODOLOGICAL_STEP, model="openrouter/qwen/qwen3.7-plus")
    assert compact_step(step, index=0)["model"] == "openrouter/qwen/qwen3.7-plus"


def test_compact_step_defaults_model_empty_for_legacy_traces() -> None:
    # Traces saved before save_memories stamped step["model"] lack the key.
    assert compact_step(_METHODOLOGICAL_STEP, index=0)["model"] == ""


def test_prefilter_drops_mechanical_steps() -> None:
    kept = compact_trace([_METHODOLOGICAL_STEP, _MECHANICAL_STEP])
    assert len(kept) == 1
    assert "ttest_ind" in kept[0]["code"]


def test_extract_code_prefers_code_action() -> None:
    step = {"code_action": "model.fit(X, y)", "action_output": "0.91"}
    assert extract_code(step) == "model.fit(X, y)"


def test_extract_code_falls_back_to_tool_calls() -> None:
    step = {"tool_calls": [{"function": {"name": "python_interpreter",
                                         "arguments": "df.dropna()"}}]}
    assert extract_code(step) == "df.dropna()"


def test_extract_code_ignores_action_output_as_code() -> None:
    # action_output is the step's RESULT, never the code — must not leak in.
    step = {"action_output": "Accuracy: 0.91", "model_output": "no code here"}
    assert extract_code(step) == ""


def test_extract_output_text_reads_message_content() -> None:
    step = {"model_output_message": {"content": "reasoning here"}}
    assert extract_output_text(step) == "reasoning here"


def test_reconstruct_recipe_orders_real_code(tmp_path: Path) -> None:
    (tmp_path / "task_single_agent.json").write_text(json.dumps([
        {"step_number": 1, "code_action": "X = load()"},
        {"step_number": 2, "code_action": "model.fit(X)"},
    ]))
    recipe = reconstruct_recipe(tmp_path)
    assert "X = load()" in recipe and "model.fit(X)" in recipe
    assert recipe.index("X = load()") < recipe.index("model.fit(X)")
    assert "step 1 · single_agent" in recipe


def test_reconstruct_recipe_annotates_step_model_when_available(tmp_path: Path) -> None:
    (tmp_path / "task_single_agent.json").write_text(json.dumps([
        {"step_number": 1, "code_action": "X = load()",
         "model": "openrouter/qwen/qwen3.7-plus"},
        {"step_number": 2, "code_action": "model.fit(X)"},
    ]))
    recipe = reconstruct_recipe(tmp_path)
    assert "step 1 · single_agent · openrouter/qwen/qwen3.7-plus" in recipe
    assert "step 2 · single_agent ──" in recipe  # no model → no annotation


def test_reconstruct_recipe_empty_when_no_code(tmp_path: Path) -> None:
    (tmp_path / "task_single_agent.json").write_text(json.dumps([
        {"step_number": 1, "model_output": "just thinking, no code"},
    ]))
    assert reconstruct_recipe(tmp_path) == ""


def test_mechanical_patterns_match_common_io() -> None:
    samples = [
        "import os",
        "from pandas import read_csv",
        "pd.read_csv('x.csv')",
        "plt.savefig('out.png')",
        "print('hi')",
    ]
    for line in samples:
        assert any(p.search(line) for p in _MECHANICAL_CODE_PATTERNS), line


def test_load_trace_handles_missing_dir(tmp_path: Path) -> None:
    assert load_trace(tmp_path / "does_not_exist") == []


def test_load_trace_skips_unreadable_files(tmp_path: Path) -> None:
    (tmp_path / "task_a.json").write_text(json.dumps([_METHODOLOGICAL_STEP]))
    (tmp_path / "task_b.json").write_text("{not json}")
    steps = load_trace(tmp_path)
    assert len(steps) == 1


def test_parse_response_accepts_valid_json() -> None:
    raw = (
        '{"id": "fit_method", "label": "Fit", "rationale": "Outliers matter.",'
        ' "option_id": "ols", "option_label": "OLS", "option_description": "min L2"}'
    )
    decision = _parse_response(raw, source_step=4)
    assert decision is not None
    assert decision.id == "fit_method"
    assert decision.source_step == 4


def test_parse_response_rejects_invalid_id() -> None:
    raw = (
        '{"id": "Fit-Method", "label": "x", "rationale": "x",'
        ' "option_id": "ols", "option_label": "x", "option_description": "x"}'
    )
    assert _parse_response(raw, 0) is _MALFORMED
    # $ would match before a trailing newline; ids must be newline-free.
    trailing_newline = (
        '{"id": "fit_method\\n", "label": "x", "rationale": "x",'
        ' "option_id": "ols", "option_label": "x", "option_description": "x"}'
    )
    assert _parse_response(trailing_newline, 0) is _MALFORMED


def test_parse_response_strips_markdown_fences() -> None:
    raw = (
        "```json\n"
        '{"id": "fit_method", "label": "Fit", "rationale": "r",'
        ' "option_id": "ols", "option_label": "OLS", "option_description": "d"}\n'
        "```"
    )
    assert isinstance(_parse_response(raw, 1), Decision)


def test_parse_response_distinguishes_null_from_garbage() -> None:
    # A legitimate "no decision" must never be conflated with unusable output.
    assert _parse_response("null", 0) is None
    assert _parse_response("```\nnull\n```", 0) is None
    assert _parse_response("", 0) is _MALFORMED
    assert _parse_response("The agent clearly chose OLS here because", 0) is _MALFORMED
    assert _parse_response('{"id": "fit_method", "label": "Fit', 0) is _MALFORMED
    # litellm yields None content for empty/reasoning-only completions.
    assert _parse_response(None, 0) is _MALFORMED


def test_parse_response_never_guesses_chosen_among_multiple_options() -> None:
    # Guessing could record a rejected alternative as the realised decision.
    body = (
        '{{"id": "fit_method", "label": "Fit", "rationale": "r",{chosen}'
        ' "options": [{{"id": "robust", "label": "Robust", "description": "d"}},'
        ' {{"id": "ols", "label": "OLS", "description": "d"}}]}}'
    )
    missing = body.format(chosen="")
    typoed = body.format(chosen=' "chosen_option_id": "olz",')
    assert _parse_response(missing, 0) is _MALFORMED
    assert _parse_response(typoed, 0) is _MALFORMED


def test_parse_response_rejects_promotion_when_chosen_option_was_dropped() -> None:
    # The model DID name its chosen option, but option validation drops it
    # (missing description / bad id). The surviving rejected alternative must
    # not slip through the single-option resolution as "unambiguous".
    missing_description = (
        '{"id": "fit_method", "label": "Fit", "rationale": "r",'
        ' "chosen_option_id": "robust",'
        ' "options": [{"id": "robust", "label": "Robust"},'
        ' {"id": "ols", "label": "OLS", "description": "d"}]}'
    )
    bad_id_pattern = (
        '{"id": "fit_method", "label": "Fit", "rationale": "r",'
        ' "chosen_option_id": "Robust",'
        ' "options": [{"id": "Robust", "label": "Robust", "description": "d"},'
        ' {"id": "ols", "label": "OLS", "description": "d"}]}'
    )
    assert _parse_response(missing_description, 0) is _MALFORMED
    assert _parse_response(bad_id_pattern, 0) is _MALFORMED


def test_parse_response_rejects_duplicate_id_collapse_as_single_option() -> None:
    # Two different alternatives sharing an id dedupe to one surviving option;
    # that survivor must not be resolved as "the only option listed".
    raw = (
        '{"id": "fit_method", "label": "Fit", "rationale": "r",'
        ' "options": [{"id": "m", "label": "Method A", "description": "d"},'
        ' {"id": "m", "label": "Method B", "description": "d"}]}'
    )
    assert _parse_response(raw, 0) is _MALFORMED


def test_parse_response_resolves_single_option_without_chosen_id() -> None:
    # One option is unambiguous — the schema requires the chosen one listed.
    raw = (
        '{"id": "fit_method", "label": "Fit", "rationale": "r",'
        ' "options": [{"id": "ols", "label": "OLS", "description": "d"}]}'
    )
    decision = _parse_response(raw, 0)
    assert isinstance(decision, Decision)
    assert decision.chosen_option_id == "ols"


def test_extract_decisions_dedupes_by_id(monkeypatch, tmp_path: Path) -> None:
    # Skip cleanly when the LLM stack can't import in this environment —
    # the test exercises the dedup logic, not the network call.
    pytest.importorskip("litellm")
    fake_payload = (
        '{"id": "fit_method", "label": "Fit", "rationale": "r",'
        ' "option_id": "ols", "option_label": "OLS", "option_description": "d"}'
    )

    class _FakeProvider:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, prompt, use_cache=True):
            return fake_payload

    import sources.core.llm_provider as llm_mod
    monkeypatch.setattr(llm_mod, "LLMProvider", _FakeProvider)
    steps = [
        {"index": 0, "reasoning": "a", "code": "c", "observation": "o"},
        {"index": 1, "reasoning": "a", "code": "c", "observation": "o"},
    ]
    result = extract_decisions(steps, "goal", tmp_path, llm_config=None)
    assert len(result.decisions) == 1
    assert result.decisions[0].id == "fit_method"
    assert result.decisions[0].source_step == 0
    # Dedup must not drop provenance: BOTH contributing steps are kept.
    assert result.decisions[0].source_steps == (0, 1)
    assert (result.steps_total, result.crashed, result.malformed) == (2, 0, 0)


def test_decision_source_steps_defaults_to_the_scalar_step() -> None:
    # Existing constructors pass only source_step; source_steps must always
    # be populated so no reader has to special-case single-step decisions.
    decision = _decision("fit_method", "ols")
    assert decision.source_steps == (decision.source_step,)


def test_merge_decisions_collects_steps_ordered_and_deduped() -> None:
    from dataclasses import replace

    from sources.transparency.decision_extractor import _merge_decisions

    first = _decision("fit_method", "ols")            # source_step 0
    later = replace(_decision("fit_method", "ols"),
                    source_step=7, source_steps=(7,))
    merged = _merge_decisions(first, later)
    assert merged.source_step == 0          # scalar accessor: first occurrence
    assert merged.source_steps == (0, 7)
    remerged = _merge_decisions(merged, later)  # same step again: no dup
    assert remerged.source_steps == (0, 7)


def test_merge_decisions_keeps_steps_even_when_options_add_nothing() -> None:
    # The old early return ("no new options -> keep existing") silently
    # dropped the later step from provenance; that is the severed join.
    from dataclasses import replace

    from sources.transparency.decision_extractor import _merge_decisions

    first = _decision("fit_method", "ols")
    later = replace(first, source_step=4, source_steps=(4,))
    assert _merge_decisions(first, later).source_steps == (0, 4)


def test_extract_decisions_attaches_step_model(monkeypatch, tmp_path: Path) -> None:
    # The decision's model is trace provenance from the step, never LLM output.
    pytest.importorskip("litellm")
    fake_payload = (
        '{"id": "fit_method", "label": "Fit", "rationale": "r",'
        ' "option_id": "ols", "option_label": "OLS", "option_description": "d"}'
    )

    class _FakeProvider:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, prompt, use_cache=True):
            return fake_payload

    import sources.core.llm_provider as llm_mod
    monkeypatch.setattr(llm_mod, "LLMProvider", _FakeProvider)
    steps = [{"index": 0, "reasoning": "a", "code": "c", "observation": "o",
              "model": "openrouter/qwen/qwen3.7-plus"}]
    result = extract_decisions(steps, "goal", tmp_path, llm_config=None)
    assert result.decisions[0].model == "openrouter/qwen/qwen3.7-plus"


def test_extract_decisions_empty_steps_returns_zeroed_result(tmp_path: Path) -> None:
    # Fully-prefiltered traces reach extract_decisions with []; the early
    # return must keep the ExtractionResult shape (export() reads .decisions).
    assert extract_decisions([], "goal", tmp_path) == ExtractionResult((), 0, 0, 0)


def test_warn_on_failures_keeps_misconfig_hint_on_total_loss(caplog) -> None:
    # A judge misrouted to a prose-answering model loses every call to the
    # malformed bucket — that total loss deserves the misconfiguration hint
    # just as much as the all-crashed case; partial losses do not.
    from sources.transparency.decision_extractor import _warn_on_failures

    with caplog.at_level(logging.WARNING):
        _warn_on_failures(0, 3, 3)
    assert "misconfigured" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _warn_on_failures(1, 2, 3)
    assert "misconfigured" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _warn_on_failures(1, 1, 3)
    assert "misconfigured" not in caplog.text
    assert "malformed" in caplog.text


def test_extract_decisions_counts_crashes_and_malformed_output(
    monkeypatch, tmp_path: Path
) -> None:
    # Step 0 crashes, step 1 returns prose, step 2 says null, step 3 succeeds:
    # the counters must separate all three failure-ish outcomes from "null".
    pytest.importorskip("litellm")
    fake_payload = (
        '{"id": "fit_method", "label": "Fit", "rationale": "r",'
        ' "option_id": "ols", "option_label": "OLS", "option_description": "d"}'
    )
    responses = {
        "astra_decision_step_1": "Sure! Here is my analysis of this step:",
        "astra_decision_step_2": "null",
        "astra_decision_step_3": fake_payload,
    }

    class _FlakyProvider:
        def __init__(self, *args, **kwargs):
            self.agent_name = kwargs.get("agent_name", "")

        def __call__(self, prompt, use_cache=True):
            if self.agent_name not in responses:
                raise RuntimeError("provider exploded")
            return responses[self.agent_name]

    import sources.core.llm_provider as llm_mod
    monkeypatch.setattr(llm_mod, "LLMProvider", _FlakyProvider)
    steps = [{"index": i, "reasoning": "a", "code": "c", "observation": "o"}
             for i in range(4)]
    result = extract_decisions(steps, "goal", tmp_path, llm_config=None)
    assert (result.steps_total, result.crashed, result.malformed) == (4, 1, 1)
    assert [d.id for d in result.decisions] == ["fit_method"]


def _decision(decision_id: str, chosen: str, alternatives: tuple[str, ...] = ()) -> Decision:
    """A decision whose options are the chosen one followed by any alternatives."""
    option_ids = (chosen, *alternatives)
    return Decision(
        id=decision_id,
        label=decision_id.replace("_", " ").title(),
        rationale="r",
        chosen_option_id=chosen,
        options=tuple(
            Option(id=option_id, label=option_id.upper(), description="d")
            for option_id in option_ids
        ),
        source_step=0,
    )


def test_build_analysis_emits_required_astra_fields() -> None:
    decisions = [_decision("fit_method", "ols")]
    analysis = build_analysis("Predict X.", "abc", ["report.md"], decisions)
    for key in ("version", "name", "inputs", "outputs", "decisions"):
        assert key in analysis, f"missing required ASTRA field: {key}"
    assert analysis["decisions"]["fit_method"]["options"]["ols"]["label"] == "OLS"


def test_analysis_records_rejected_alternatives_beside_the_chosen_option() -> None:
    """The capsule exists to show what was chosen over what, and which one won."""
    decisions = [_decision("fit_method", "ols", alternatives=("ridge", "lasso"))]
    analysis = build_analysis("g", "abc", ["r.md"], decisions)

    entry = analysis["decisions"]["fit_method"]
    assert set(entry["options"]) == {"ols", "ridge", "lasso"}
    assert entry["default"] == "ols"


def test_analysis_records_decision_model_as_tag() -> None:
    # Since 0.0.12 the step's model travels as a "model:<id>" tag; the legacy
    # per-decision "model" key is never written on new capsules (readers keep
    # accepting it on the 35 pre-0.0.12 capsules on disk).
    from dataclasses import replace

    decision = replace(_decision("fit_method", "ols"),
                       model="openrouter/qwen/qwen3.7-plus")
    entry = build_analysis("g", "abc", ["r.md"], [decision])["decisions"]["fit_method"]
    assert "model:openrouter/qwen/qwen3.7-plus" in entry["tags"]
    assert "model" not in entry


def test_analysis_omits_model_tag_for_legacy_traces() -> None:
    entry = build_analysis(
        "g", "abc", ["r.md"], [_decision("fit_method", "ols")]
    )["decisions"]["fit_method"]
    assert "model" not in entry
    assert not [t for t in entry["tags"] if t.startswith("model:")]


def test_analysis_tags_every_contributing_trace_step() -> None:
    # The decision -> trace-step join: one "trace_step:<N>" tag per
    # contributing step, in trace order (evidence files are
    # sources/memory/<uuid>/astra_decision_step_<N>.json).
    from dataclasses import replace

    decision = replace(_decision("fit_method", "ols"), source_steps=(3, 9, 17))
    entry = build_analysis("g", "abc", ["r.md"], [decision])["decisions"]["fit_method"]
    assert [t for t in entry["tags"] if t.startswith("trace_step:")] == [
        "trace_step:3", "trace_step:9", "trace_step:17",
    ]


def test_analysis_version_marks_the_new_writer_generation() -> None:
    analysis = build_analysis("g", "abc", ["r.md"], [_decision("fit_method", "ols")])
    assert analysis["version"] == "0.0.12"


def test_analysis_ports_use_data_type() -> None:
    # ASTRA port type is "data" — the old writer's "text"/"file" were
    # writer-local vocabulary.
    analysis = build_analysis("g", "abc", ["r.md"], [_decision("fit_method", "ols")])
    assert analysis["inputs"][0]["type"] == "data"
    assert all(o["type"] == "data" for o in analysis["outputs"])


def test_outputs_carry_honest_empty_attribution() -> None:
    # The pre-0.0.12 writer stamped EVERY decision id on EVERY output —
    # attribution-shaped noise. New capsules say "untracked" instead.
    decisions = [_decision("fit_method", "ols"), _decision("metric", "r2")]
    analysis = build_analysis("g", "abc", ["a.csv", "b.csv"], decisions)
    assert all(o["decisions"] == [] for o in analysis["outputs"])
    assert "mimosa:output_attribution=untracked" in analysis["tags"]


def test_analysis_tags_distinguish_extractor_empty_from_pre_extractor() -> None:
    # decisions:{} used to be indistinguishable from a capsule written before
    # the extractor existed; the honest-empty tag closes that gap.
    empty = build_analysis("g", "abc", ["r.md"], [])
    assert "mimosa:decisions=none (extractor produced no decisions)" in empty["tags"]
    populated = build_analysis("g", "abc", ["r.md"], [_decision("fit_method", "ols")])
    assert not [t for t in populated["tags"] if t.startswith("mimosa:decisions=none")]


def test_analysis_records_extraction_health_block() -> None:
    # A capsule from a degraded extraction must say so itself.
    extraction = ExtractionResult(decisions=(), steps_total=7, crashed=2, malformed=3)
    analysis = build_analysis("g", "abc", ["r.md"], [], extraction=extraction)
    assert analysis["extraction"] == {
        "steps_considered": 7,
        "decisions_recorded": 0,
        "llm_call_failures": 2,
        "malformed_responses": 3,
    }


def test_analysis_omits_extraction_block_when_not_provided() -> None:
    analysis = build_analysis("g", "abc", ["r.md"], [])
    assert "extraction" not in analysis


def test_recipe_command_threads_into_every_output() -> None:
    analysis = build_analysis("g", "abc", ["a.csv", "b.csv"], [], "python recipe.py")
    commands = [o["recipe"]["command"] for o in analysis["outputs"]]
    assert commands == ["python recipe.py", "python recipe.py"]


def test_recipe_command_defaults_to_pointer_when_no_code() -> None:
    analysis = build_analysis("g", "abc", ["a.csv"], [])
    assert "sources/memory" in analysis["outputs"][0]["recipe"]["command"]


def test_universe_pins_every_decision_to_its_chosen_option() -> None:
    decisions = [_decision("fit_method", "ols"), _decision("metric", "r2")]
    universe = build_universe(decisions, "abc")
    assert universe["decisions"] == {"fit_method": "ols", "metric": "r2"}
    assert universe["id"] == "best"


def test_universe_pins_the_chosen_option_even_when_it_is_not_listed_first() -> None:
    """The realised option is read from chosen_option_id, never from option order."""
    decision = Decision(
        id="fit_method", label="Fit", rationale="r",
        chosen_option_id="lasso",
        options=(
            Option(id="ols", label="OLS", description="d"),
            Option(id="lasso", label="Lasso", description="d"),
        ),
        source_step=0,
    )
    universe = build_universe([decision], "abc")
    assert universe["decisions"] == {"fit_method": "lasso"}


def test_write_export_writes_both_files(tmp_path: Path) -> None:
    decisions = [_decision("fit_method", "ols")]
    analysis = build_analysis("g", "abc", ["x.csv"], decisions)
    universe = build_universe(decisions, "abc")
    path = write_export(tmp_path, analysis, universe)
    assert path == tmp_path / "astra.yaml"
    loaded = yaml.safe_load(path.read_text())
    assert loaded["decisions"]["fit_method"]["default"] == "ols"
    universe_loaded = yaml.safe_load((tmp_path / "universes" / "best.yaml").read_text())
    assert universe_loaded["decisions"]["fit_method"] == "ols"


def test_resolve_artefacts_dir_prefers_tmp_snapshot(tmp_path: Path) -> None:
    # Both a /tmp snapshot and the live workspace exist; the snapshot wins.
    from sources.transparency.astra_exporter import AstraExporter

    snapshot = tmp_path / "mimosa_run_abcdef123456_run-xyz"
    snapshot.mkdir()
    (snapshot / "model.pkl").write_text("snap")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "model.pkl").write_text("live")

    exporter = AstraExporter(SimpleNamespace())
    exporter._SNAPSHOT_ROOT = tmp_path  # divert glob away from real /tmp
    assert exporter._resolve_artefacts_dir("run-xyz", workspace) == snapshot


def test_resolve_artefacts_dir_falls_back_to_workspace(tmp_path: Path) -> None:
    # No snapshot for this uuid → fall back to workspace_dir.
    from sources.transparency.astra_exporter import AstraExporter

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    exporter = AstraExporter(SimpleNamespace())
    exporter._SNAPSHOT_ROOT = tmp_path
    assert exporter._resolve_artefacts_dir("orphan-uuid", workspace) == workspace


def test_list_workspace_files_drops_junk_and_dotfiles(tmp_path: Path) -> None:
    from sources.transparency.astra_exporter import AstraExporter

    for name in ("model.pkl", "report.md", ".DS_Store", ".hidden", "recipe.py"):
        (tmp_path / name).write_text("x")
    exporter = AstraExporter(SimpleNamespace())
    assert exporter._list_workspace_files(tmp_path) == ["model.pkl", "report.md"]


def test_write_recipe_creates_script_and_returns_command(tmp_path: Path) -> None:
    from sources.transparency.astra_exporter import AstraExporter

    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "task_single_agent.json").write_text(json.dumps([
        {"step_number": 1, "code_action": "result = train(model)"},
    ]))
    capsule = tmp_path / "capsule"
    exporter = AstraExporter(SimpleNamespace())
    command = exporter._write_recipe(capsule, memory)
    assert command == "python recipe.py"
    assert "result = train(model)" in (capsule / "recipe.py").read_text()


def test_write_recipe_falls_back_when_no_code(tmp_path: Path) -> None:
    from sources.transparency.astra_exporter import AstraExporter

    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "task_single_agent.json").write_text(json.dumps([
        {"step_number": 1, "model_output": "no code"},
    ]))
    capsule = tmp_path / "capsule"
    exporter = AstraExporter(SimpleNamespace())
    command = exporter._write_recipe(capsule, memory)
    assert "sources/memory" in command
    assert not (capsule / "recipe.py").exists()


def test_engine_gate_skips_export_when_flag_off(monkeypatch) -> None:
    # The engine's _export_astra must early-return when config.export_astra
    # is False; AstraExporter must NEVER be imported in that path.
    pytest.importorskip("litellm")
    from sources.core import evolution_engine as engine_mod

    sentinel = {"called": False}

    def _boom(*_args, **_kwargs):
        sentinel["called"] = True
        raise AssertionError("AstraExporter should not be constructed when gate is off")

    monkeypatch.setattr(
        "sources.transparency.AstraExporter", _boom, raising=False,
    )
    engine = engine_mod.EvolutionEngine.__new__(engine_mod.EvolutionEngine)
    engine.config = SimpleNamespace(export_astra=False)
    engine.logger = __import__("logging").getLogger("test")
    engine._export_astra("any-uuid", "any goal")
    assert sentinel["called"] is False


def test_default_config_has_astra_export_disabled() -> None:
    # Safety property: ASTRA export must be opt-in. A fresh Config has it off.
    pytest.importorskip("litellm")
    from config import Config
    config = Config()
    assert config.export_astra is False


def test_export_writes_extraction_health_block_and_warns(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    # End-to-end through export(): a degraded extraction must land in the
    # written astra.yaml AND be announced on stdout — deleting either the
    # extraction= kwarg or the print_warn block must fail this test.
    pytest.importorskip("litellm")
    from sources.transparency import astra_exporter as exporter_mod

    memory = tmp_path / "memory" / "run-x"
    memory.mkdir(parents=True)
    (memory / "task_single_agent.json").write_text(json.dumps([
        {"step_number": 1, "code_action": "result = stats.ttest_ind(a, b)"},
    ]))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "report.md").write_text("x")

    degraded = ExtractionResult(
        decisions=(_decision("fit_method", "ols"),),
        steps_total=3, crashed=1, malformed=1,
    )
    monkeypatch.setattr(exporter_mod, "extract_decisions", lambda *a, **k: degraded)
    monkeypatch.setattr(
        exporter_mod.AstraExporter, "_build_llm_config", lambda self: None
    )
    config = SimpleNamespace(
        memory_dir=str(tmp_path / "memory"),
        workspace_dir=str(workspace),
        runs_capsule_dir=str(tmp_path / "capsule"),
    )
    exporter = exporter_mod.AstraExporter(config)
    exporter._SNAPSHOT_ROOT = tmp_path  # no snapshots here → workspace fallback

    path = exporter.export("run-x", "the goal")

    loaded = yaml.safe_load(path.read_text())
    assert loaded["extraction"] == {
        "steps_considered": 3,
        "decisions_recorded": 1,
        "llm_call_failures": 1,
        "malformed_responses": 1,
    }
    assert loaded["decisions"]["fit_method"]["default"] == "ols"
    assert "Extraction degraded" in capsys.readouterr().out


def test_write_outputs_manifest_digests_every_output(tmp_path: Path) -> None:
    from sources.transparency.astra_exporter import AstraExporter

    artefacts = tmp_path / "artefacts"
    artefacts.mkdir()
    (artefacts / "report.md").write_bytes(b"hello")
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    exporter = AstraExporter(SimpleNamespace())
    path = exporter._write_outputs_manifest(capsule, artefacts, ["report.md"])
    manifest = json.loads(path.read_text())
    assert path.name == "outputs_manifest.json" and path.parent == capsule
    entry = manifest["report_md"]  # keyed by the SAME slug astra.yaml uses
    assert entry["path"] == "report.md"
    assert entry["bytes"] == 5
    import hashlib
    assert entry["sha256"] == hashlib.sha256(b"hello").hexdigest()


def test_outputs_manifest_is_fail_soft_per_file(tmp_path: Path) -> None:
    # One unreadable artefact yields an error entry (no absolute path in it);
    # the readable one is still digested — never a crash, never silence.
    from sources.transparency.astra_exporter import AstraExporter

    artefacts = tmp_path / "artefacts"
    artefacts.mkdir()
    (artefacts / "good.csv").write_bytes(b"a,b\n")
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    exporter = AstraExporter(SimpleNamespace())
    path = exporter._write_outputs_manifest(
        capsule, artefacts, ["good.csv", "vanished.pkl"]
    )
    manifest = json.loads(path.read_text())
    assert manifest["good_csv"]["bytes"] == 4
    bad = manifest["vanished_pkl"]
    assert bad["path"] == "vanished.pkl"
    assert bad["error"].startswith("unreadable (")
    assert "sha256" not in bad
    assert str(tmp_path) not in json.dumps(manifest)


def test_outputs_manifest_empty_snapshot_writes_empty_object(tmp_path: Path) -> None:
    from sources.transparency.astra_exporter import AstraExporter

    exporter = AstraExporter(SimpleNamespace())
    path = exporter._write_outputs_manifest(tmp_path, tmp_path, [])
    assert json.loads(path.read_text()) == {}


def test_colliding_filenames_get_distinct_manifest_entries_with_correct_digests(
    tmp_path: Path,
) -> None:
    # "a-b.csv" and "a b.csv" slugify identically; a shared key would silently
    # attribute one file's digest to the other. Both surfaces (astra.yaml
    # output ids, manifest keys) must de-collide the same way.
    import hashlib
    from sources.transparency.astra_exporter import AstraExporter

    artefacts = tmp_path / "artefacts"
    artefacts.mkdir()
    (artefacts / "a-b.csv").write_bytes(b"dash")
    (artefacts / "a b.csv").write_bytes(b"space")
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    files = ["a b.csv", "a-b.csv"]  # exporter order (sorted): space < dash
    exporter = AstraExporter(SimpleNamespace())
    manifest = json.loads(
        exporter._write_outputs_manifest(capsule, artefacts, files).read_text()
    )
    assert len(manifest) == 2
    assert manifest["a_b_csv"]["path"] == "a b.csv"
    assert manifest["a_b_csv"]["sha256"] == hashlib.sha256(b"space").hexdigest()
    assert manifest["a_b_csv_1"]["path"] == "a-b.csv"
    assert manifest["a_b_csv_1"]["sha256"] == hashlib.sha256(b"dash").hexdigest()


def test_manifest_keys_equal_astra_output_ids_even_on_collisions(
    tmp_path: Path,
) -> None:
    from sources.transparency.astra_exporter import AstraExporter

    artefacts = tmp_path / "artefacts"
    artefacts.mkdir()
    files = ["a b.csv", "a-b.csv", "report.md"]
    for name in files:
        (artefacts / name).write_bytes(b"x")
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    manifest = json.loads(
        AstraExporter(SimpleNamespace())
        ._write_outputs_manifest(capsule, artefacts, files)
        .read_text()
    )
    analysis = build_analysis("goal", "uuid-1", files, [])
    assert [o["id"] for o in analysis["outputs"]] == list(manifest.keys())


def test_empty_snapshot_keeps_outputs_and_manifest_in_lockstep() -> None:
    # No synthetic "(none captured)" output: the analysis carries an empty
    # outputs list plus an honest-empty tag, matching the manifest's {}.
    analysis = build_analysis("goal", "uuid-1", [], [])
    assert analysis["outputs"] == []
    assert "mimosa:outputs=none (no artefacts captured)" in analysis["tags"]


def test_nonempty_snapshot_carries_no_outputs_none_tag() -> None:
    analysis = build_analysis("goal", "uuid-1", ["report.md"], [])
    assert not any(t.startswith("mimosa:outputs=none") for t in analysis["tags"])


def test_export_writes_manifest_and_environment_blocks(
    monkeypatch, tmp_path: Path
) -> None:
    # End-to-end through export(): the capsule gains outputs_manifest.json and
    # astra.yaml gains the environment block (with the run's grounding stats
    # joined from run_metrics.json and labelled self-declared).
    pytest.importorskip("litellm")
    from sources.transparency import astra_exporter as exporter_mod

    memory = tmp_path / "memory" / "run-x"
    memory.mkdir(parents=True)
    (memory / "task_single_agent.json").write_text(json.dumps([
        {"step_number": 1, "code_action": "result = stats.ttest_ind(a, b)"},
    ]))
    (memory / "verifier_call.json").write_text(json.dumps({"temperature": 0.2}))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "report.md").write_bytes(b"content")
    workflows = tmp_path / "workflows" / "run-x"
    workflows.mkdir(parents=True)
    (workflows / "run_metrics.json").write_text(json.dumps(
        {"grounding": {"attempts": 2, "hit_rate": 0.5}}
    ))

    extraction = ExtractionResult(
        decisions=(_decision("fit_method", "ols"),), steps_total=1,
        crashed=0, malformed=0,
    )
    monkeypatch.setattr(exporter_mod, "extract_decisions", lambda *a, **k: extraction)
    monkeypatch.setattr(
        exporter_mod.AstraExporter, "_build_llm_config", lambda self: None
    )
    config = SimpleNamespace(
        memory_dir=str(tmp_path / "memory"),
        workspace_dir=str(workspace),
        runs_capsule_dir=str(tmp_path / "capsule"),
        workflow_dir=str(tmp_path / "workflows"),
        smolagent_model_id="openrouter/deepseek/deepseek-v4-flash",
    )
    exporter = exporter_mod.AstraExporter(config)
    exporter._SNAPSHOT_ROOT = tmp_path  # no snapshots here → workspace fallback

    path = exporter.export("run-x", "the goal")

    manifest = json.loads((path.parent / "outputs_manifest.json").read_text())
    assert manifest["report_md"]["bytes"] == 7
    loaded = yaml.safe_load(path.read_text())
    environment = loaded["environment"]
    assert environment["runner_env"].startswith("partial")
    assert environment["grounding"] == {
        "attempts": 2, "hit_rate": 0.5, "declared_by": "subject",
    }
    assert environment["temperature"] == {"min": 0.2, "max": 0.2, "n_calls": 1}
    assert environment["model_roles"]["smolagent_model_id"].endswith("v4-flash")
    assert environment["config_digest"].startswith("sha256:")
    # Public repo: the exported analysis must never carry local abs paths.
    assert str(tmp_path) not in path.read_text()


def test_exporter_skips_when_memory_dir_missing(tmp_path: Path) -> None:
    # export() prints via sources.cli.pretty_print which transitively imports
    # the project's LLM stack; skip when that stack is unbuildable locally.
    pytest.importorskip("litellm")
    from sources.transparency import AstraExporter
    config = SimpleNamespace(
        memory_dir=str(tmp_path / "memory"),
        workspace_dir=str(tmp_path / "workspace"),
        runs_capsule_dir=str(tmp_path / "runs_capsule"),
        judge_model="anthropic/claude-sonnet-4-5",
        openrouter_provider_for=lambda _m: None,
    )
    (tmp_path / "workspace").mkdir()
    result = AstraExporter(config).export("nonexistent-uuid", "any goal")
    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
