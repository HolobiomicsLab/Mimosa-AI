"""Unit tests for the post-run ASTRA exporter.

The LLM call inside :func:`extract_decisions` is monkey-patched so tests are
fast, deterministic, and offline. End-to-end behavior is verified by
asserting the schema of the emitted YAML files against the ASTRA spec
(see https://astra-spec.org/latest/specification/#decisions).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sources.transparency.decision_extractor import (
    Decision,
    _parse_response,
    extract_decisions,
)
from sources.transparency.trace_compaction import (
    _MECHANICAL_CODE_PATTERNS,
    compact_step,
    compact_trace,
    is_methodological_candidate,
    load_trace,
)
from sources.transparency.yaml_writer import (
    build_analysis,
    build_universe,
    write_export,
)


_METHODOLOGICAL_STEP = {
    "model_output_message": {
        "content": "Variance is unequal so I use Welch's t-test instead of Student's."
    },
    "action_output": "from scipy import stats\nresult = stats.ttest_ind(a, b, equal_var=False)",
    "observations": "Ttest_indResult(statistic=2.31, pvalue=0.022)",
    "model_input_messages": [{"role": "system", "content": "x" * 50_000}],
}

_MECHANICAL_STEP = {
    "model_output_message": {"content": "List the workspace."},
    "action_output": "import os\nos.listdir('.')",
    "observations": "['a.csv', 'b.csv']",
}


def test_compact_step_strips_input_messages() -> None:
    compact = compact_step(_METHODOLOGICAL_STEP, index=0)
    assert "model_input_messages" not in compact
    assert "Welch" in compact["reasoning"]
    assert "ttest_ind" in compact["code"]


def test_prefilter_drops_mechanical_steps() -> None:
    kept = compact_trace([_METHODOLOGICAL_STEP, _MECHANICAL_STEP])
    assert len(kept) == 1
    assert "ttest_ind" in kept[0]["code"]


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
    assert _parse_response(raw, 0) is None


def test_parse_response_strips_markdown_fences() -> None:
    raw = (
        "```json\n"
        '{"id": "fit_method", "label": "Fit", "rationale": "r",'
        ' "option_id": "ols", "option_label": "OLS", "option_description": "d"}\n'
        "```"
    )
    assert _parse_response(raw, 1) is not None


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
    decisions = extract_decisions(steps, "goal", tmp_path, llm_config=None)
    assert len(decisions) == 1
    assert decisions[0].id == "fit_method"
    assert decisions[0].source_step == 0


def test_build_analysis_emits_required_astra_fields() -> None:
    decisions = [
        Decision(
            id="fit_method", label="Fit", rationale="r",
            option_id="ols", option_label="OLS", option_description="d",
            source_step=2,
        ),
    ]
    analysis = build_analysis("Predict X.", "abc", ["report.md"], decisions)
    for key in ("version", "name", "inputs", "outputs", "decisions"):
        assert key in analysis, f"missing required ASTRA field: {key}"
    assert analysis["decisions"]["fit_method"]["options"]["ols"]["label"] == "OLS"


def test_universe_pins_every_decision_to_its_chosen_option() -> None:
    decisions = [
        Decision(
            id="fit_method", label="Fit", rationale="r",
            option_id="ols", option_label="OLS", option_description="d",
            source_step=0,
        ),
        Decision(
            id="metric", label="Metric", rationale="r",
            option_id="r2", option_label="R2", option_description="d",
            source_step=1,
        ),
    ]
    universe = build_universe(decisions, "abc")
    assert universe["decisions"] == {"fit_method": "ols", "metric": "r2"}
    assert universe["id"] == "best"


def test_write_export_writes_both_files(tmp_path: Path) -> None:
    decisions = [
        Decision(
            id="fit_method", label="Fit", rationale="r",
            option_id="ols", option_label="OLS", option_description="d",
            source_step=0,
        ),
    ]
    analysis = build_analysis("g", "abc", ["x.csv"], decisions)
    universe = build_universe(decisions, "abc")
    path = write_export(tmp_path, analysis, universe)
    assert path == tmp_path / "astra.yaml"
    loaded = yaml.safe_load(path.read_text())
    assert loaded["decisions"]["fit_method"]["default"] == "ols"
    universe_loaded = yaml.safe_load((tmp_path / "universes" / "best.yaml").read_text())
    assert universe_loaded["decisions"]["fit_method"] == "ols"


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


def test_exporter_skips_when_memory_dir_missing(tmp_path: Path) -> None:
    # export() prints via sources.cli.pretty_print which transitively imports
    # the project's LLM stack; skip when that stack is unbuildable locally.
    pytest.importorskip("litellm")
    from sources.transparency import AstraExporter
    config = SimpleNamespace(
        memory_dir=str(tmp_path / "memory"),
        workspace_dir=str(tmp_path / "workspace"),
        judge_model="anthropic/claude-sonnet-4-5",
        openrouter_provider_for=lambda _m: None,
    )
    (tmp_path / "workspace").mkdir()
    result = AstraExporter(config).export("nonexistent-uuid", "any goal")
    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
