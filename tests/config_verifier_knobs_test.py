"""Tests for the config-driven verifier cost knobs.

Pure config behaviour - no network or LLM calls.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config


def test_defaults_are_none_so_verifier_keeps_its_own_defaults():
    c = Config()
    assert c.verifier_max_claims is None
    assert c.verifier_use_grounding is None


def test_knobs_load_from_json():
    c = Config()
    c.from_json({"verifier_max_claims": 40, "verifier_use_grounding": False})
    assert c.verifier_max_claims == 40
    assert c.verifier_use_grounding is False


def test_zero_max_claims_is_preserved_and_not_treated_as_unset():
    c = Config()
    c.from_json({"verifier_max_claims": 0})
    assert c.verifier_max_claims == 0


def test_knobs_survive_a_jsonify_from_json_roundtrip():
    c = Config()
    c.from_json({"verifier_max_claims": 40, "verifier_use_grounding": False})
    restored = Config()
    restored.from_json(c.jsonify())
    assert restored.verifier_max_claims == 40
    assert restored.verifier_use_grounding is False


def test_unset_knobs_are_omitted_from_the_verifier_kwargs():
    """The construction site must not pass None through to the verifier."""
    c = Config()
    kwargs = {"workspace_dir": "/tmp/ws"}
    if getattr(c, "verifier_max_claims", None) is not None:
        kwargs["max_claims"] = c.verifier_max_claims
    if getattr(c, "verifier_use_grounding", None) is not None:
        kwargs["use_grounding"] = c.verifier_use_grounding
    assert "max_claims" not in kwargs
    assert "use_grounding" not in kwargs
