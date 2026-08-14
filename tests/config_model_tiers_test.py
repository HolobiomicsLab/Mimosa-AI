"""Tests for model capability tiers and alias resolution on config roles.

Pure config behaviour - no network or LLM calls.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config


def test_resolve_model_alias_and_passthrough():
    c = Config()
    assert c.resolve_model("heavy") == c.model_tiers["heavy"]
    assert c.resolve_model("light") == c.model_tiers["light"]
    # a concrete id is returned unchanged
    assert c.resolve_model("openrouter/foo/bar") == "openrouter/foo/bar"
    # None passes through (e.g. an unset optional role)
    assert c.resolve_model(None) is None


def test_defaults_are_concrete_so_tiers_are_inert():
    c = Config()
    # default role values are concrete ids, not tier aliases -> unchanged
    assert "/" in c.planner_llm_model
    assert c.resolve_model(c.planner_llm_model) == c.planner_llm_model


def test_from_json_resolves_role_aliases():
    c = Config()
    c.from_json({
        "model_tiers": {"heavy": "prov/big", "light": "prov/small"},
        "planner_llm_model": "heavy",
        "workflow_llm_model": "heavy",
        "judge_model": "heavy",
        "vision_judge_model": "light",
        "smolagent_model_id": "light",
        "capsule_namer_model": "light",
    })
    assert c.planner_llm_model == "prov/big"
    assert c.workflow_llm_model == "prov/big"
    assert c.judge_model == "prov/big"
    assert c.vision_judge_model == "prov/small"
    assert c.smolagent_model_id == "prov/small"
    assert c.capsule_namer_model == "prov/small"


def test_from_json_without_tiers_leaves_concrete_ids_untouched():
    c = Config()
    c.from_json({"planner_llm_model": "prov/explicit"})
    assert c.planner_llm_model == "prov/explicit"


def test_model_tiers_survive_a_jsonify_from_json_roundtrip():
    c = Config()
    c.from_json({"model_tiers": {"heavy": "prov/big", "light": "prov/small"}})
    restored = Config()
    restored.from_json(c.jsonify())
    assert restored.model_tiers == {"heavy": "prov/big", "light": "prov/small"}
