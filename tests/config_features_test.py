"""Tests for the cost/fairness config features: model tiers, the cheap
judge-extraction fallback, per-side Perspicacite KB selection, and the verifier
cost knobs. Pure config behaviour — no network or LLM calls."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import Config


# ── model capability tiers ────────────────────────────────────────────────

def test_resolve_model_alias_and_passthrough():
    c = Config()
    assert c.resolve_model("heavy") == c.model_tiers["heavy"]
    assert c.resolve_model("light") == c.model_tiers["light"]
    # a concrete id is returned unchanged
    assert c.resolve_model("openrouter/foo/bar") == "openrouter/foo/bar"
    # None passes through (e.g. unset judge_extraction_model)
    assert c.resolve_model(None) is None


def test_defaults_are_concrete_so_tiers_are_inert():
    c = Config()
    # default role values are concrete ids, not tier aliases -> unchanged
    assert "/" in c.planner_llm_model
    assert c.resolve_model(c.planner_llm_model) == c.planner_llm_model
    # cheap extraction tier defaults to None (reuse judge_model -> no change)
    assert c.judge_extraction_model is None


def test_from_json_resolves_role_aliases():
    c = Config()
    c.from_json({
        "model_tiers": {"heavy": "prov/big", "light": "prov/small"},
        "planner_llm_model": "heavy",
        "workflow_llm_model": "heavy",
        "judge_model": "heavy",
        "judge_extraction_model": "light",
        "smolagent_model_id": "light",
        "capsule_namer_model": "light",
    })
    assert c.planner_llm_model == "prov/big"
    assert c.workflow_llm_model == "prov/big"
    assert c.judge_model == "prov/big"
    assert c.judge_extraction_model == "prov/small"
    assert c.smolagent_model_id == "prov/small"
    assert c.capsule_namer_model == "prov/small"


# ── per-side Perspicacite KB + verifier cost knobs ────────────────────────

def test_grounding_and_verifier_defaults_reproduce_current_behaviour():
    c = Config()
    # agent grounding on, no KB pinned (web search), as before
    assert c.perspicacite_agent_grounding_enabled is True
    assert c.perspicacite_agent_kb_name is None
    assert c.perspicacite_verifier_kb_name is None
    # verifier cost knobs unset -> verifier keeps its own library defaults
    assert c.verifier_max_claims is None
    assert c.verifier_use_grounding is None


def test_grounding_and_verifier_knobs_roundtrip(tmp_path):
    c = Config()
    c.perspicacite_agent_grounding_enabled = False
    c.perspicacite_agent_kb_name = "asb-brief-haffner"
    c.perspicacite_verifier_kb_name = "asb-paper-haffner"
    c.verifier_max_claims = 40
    c.verifier_use_grounding = True
    p = tmp_path / "cfg.json"
    c.dump(str(p))
    c2 = Config()
    c2.load(str(p))
    assert c2.perspicacite_agent_grounding_enabled is False
    assert c2.perspicacite_agent_kb_name == "asb-brief-haffner"
    assert c2.perspicacite_verifier_kb_name == "asb-paper-haffner"
    assert c2.verifier_max_claims == 40
    assert c2.verifier_use_grounding is True


if __name__ == "__main__":
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            if "tmp_path" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"  ✓ {name}")
