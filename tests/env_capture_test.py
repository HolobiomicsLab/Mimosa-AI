"""Unit tests for the ASTRA environment capture (sources/transparency/env_capture).

Everything here is stdlib-offline: no LLM stack, no network. The module's
contract is honest degradation — every field either carries a real value or
an "absent (<reason>)" string — and public-repo hygiene: no key material and
no absolute local paths in any emitted value.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sources.transparency.env_capture import (
    _config_digest,
    _drop_secret_keys,
    _grounding_block,
    _temperature_aggregate,
    capture_environment,
)
from sources.utils.git_info import get_git_info


def _config(**overrides) -> SimpleNamespace:
    base = {
        "smolagent_model_id": "openrouter/deepseek/deepseek-v4-flash",
        "judge_model": "openrouter/deepseek/deepseek-v4-flash",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_capture_emits_every_contract_field(tmp_path: Path) -> None:
    env = capture_environment(_config(), tmp_path, None)
    assert set(env) == {
        "git", "python_version", "platform", "model_roles", "temperature",
        "config_digest", "grounding", "runner_env",
    }


def test_git_block_has_commit_branch_dirty_shape() -> None:
    assert set(get_git_info()) == {"commit", "branch", "dirty"}


def test_git_block_degrades_to_none_outside_a_repo(tmp_path: Path) -> None:
    info = get_git_info(tmp_path)
    assert info == {"commit": None, "branch": None, "dirty": None}


def test_model_roles_are_verbatim_including_list_valued_ids(tmp_path: Path) -> None:
    # smolagent_model_id may legitimately be a list of candidates; the roles
    # are copied verbatim, never resolved or flattened.
    config = _config(
        smolagent_model_id=["openrouter/a/b", "openrouter/c/d"],
        planner_llm_model="openrouter/deepseek/deepseek-v4-pro",
    )
    roles = capture_environment(config, tmp_path, None)["model_roles"]
    assert roles["smolagent_model_id"] == ["openrouter/a/b", "openrouter/c/d"]
    assert roles["planner_llm_model"] == "openrouter/deepseek/deepseek-v4-pro"
    assert roles["judge_model"] == "openrouter/deepseek/deepseek-v4-flash"


def test_model_roles_honest_empty_when_config_declares_none(tmp_path: Path) -> None:
    env = capture_environment(SimpleNamespace(), tmp_path, None)
    assert env["model_roles"] == "absent (config declares no model roles)"


def test_temperature_aggregates_both_memory_shapes(tmp_path: Path) -> None:
    # Dict-shaped LLM cache files AND list-shaped task traces both count.
    (tmp_path / "verifier_x.json").write_text(json.dumps({"temperature": 0.2}))
    (tmp_path / "task_agent.json").write_text(json.dumps(
        [{"temperature": 1.0}, {"step_number": 2}]
    ))
    aggregate = _temperature_aggregate(tmp_path)
    assert aggregate == {"min": 0.2, "max": 1.0, "n_calls": 2}


def test_temperature_excludes_the_extractors_own_calls(tmp_path: Path) -> None:
    # astra_decision_step_*.json are the capture instrument (temperature 0.0),
    # not the subject run — folding them in would fake determinism.
    (tmp_path / "astra_decision_step_3.json").write_text(
        json.dumps({"temperature": 0.0})
    )
    (tmp_path / "verifier_x.json").write_text(json.dumps({"temperature": 0.7}))
    assert _temperature_aggregate(tmp_path) == {"min": 0.7, "max": 0.7, "n_calls": 1}


def test_temperature_ignores_booleans_and_unreadable_files(tmp_path: Path) -> None:
    (tmp_path / "weird.json").write_text(json.dumps({"temperature": True}))
    (tmp_path / "broken.json").write_text("{not json")
    assert _temperature_aggregate(tmp_path) == (
        "absent (no temperature recorded in memory JSONs)"
    )


def test_temperature_absent_when_memory_dir_missing(tmp_path: Path) -> None:
    assert _temperature_aggregate(tmp_path / "nope") == (
        "absent (memory dir unavailable)"
    )


def test_config_digest_never_embeds_the_config(tmp_path: Path) -> None:
    config = _config(workspace_dir="/Users/someone/private/workspace")
    env = capture_environment(config, tmp_path, None)
    assert env["config_digest"].startswith("sha256:")
    assert "/Users/someone" not in json.dumps(env)


def test_secret_deny_list_drops_key_material_before_hashing(tmp_path: Path) -> None:
    # THE public-repo guard: a config carrying secrets must hash identically
    # to the same config without them (they are dropped, not hashed), and the
    # secret value must never appear anywhere in the emitted block.
    secret = "sk-SHOULD-NEVER-LEAK-1234"
    with_secrets = _config(
        openrouter_api_key=secret,
        hf_token=secret,
        db_password=secret,
        nested={"provider": {"Secret_Value": secret}, "kept": "yes"},
    )
    without_secrets = _config(nested={"provider": {}, "kept": "yes"})
    assert _config_digest(with_secrets) == _config_digest(without_secrets)
    env = capture_environment(with_secrets, tmp_path, None)
    assert secret not in json.dumps(env)


def test_secret_deny_list_matches_substrings_case_insensitively() -> None:
    scrubbed = _drop_secret_keys({
        "OPENROUTER_API_KEY": "x",
        "MyToken": "x",
        "passwords": ["x"],
        "judge_model": "kept",
        "options": [{"api_base": "x", "label": "kept"}],
    })
    assert scrubbed == {"judge_model": "kept", "options": [{"label": "kept"}]}


def test_config_digest_changes_when_a_benign_key_changes() -> None:
    # The digest must still discriminate real config changes.
    assert _config_digest(_config()) != _config_digest(
        _config(judge_model="openrouter/other/model")
    )


def test_grounding_copied_verbatim_and_labelled_self_declared(tmp_path: Path) -> None:
    metrics = tmp_path / "run_metrics.json"
    grounding = {"attempts": 4, "hit_rate": 0.75, "kb_name": None, "mode": "agent"}
    metrics.write_text(json.dumps({"grounding": grounding}))
    block = _grounding_block(metrics)
    assert block == {**grounding, "declared_by": "subject"}


def test_grounding_absence_always_carries_a_reason(tmp_path: Path) -> None:
    assert _grounding_block(None) == "absent (workflow dir not configured)"
    assert _grounding_block(tmp_path / "missing.json") == (
        "absent (run_metrics.json unavailable)"
    )
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert _grounding_block(broken) == "absent (run_metrics.json unreadable)"
    no_block = tmp_path / "no_block.json"
    no_block.write_text(json.dumps({"uuid": "x"}))
    assert _grounding_block(no_block) == (
        "absent (run_metrics.json has no grounding block)"
    )


def test_runner_env_names_the_uncaptured_sandbox(tmp_path: Path) -> None:
    # The science runs in a separate python3.12 venv (ensure_env.py) that this
    # capture does NOT cover; the block must say so, not imply completeness.
    env = capture_environment(_config(), tmp_path, None)
    assert env["runner_env"] == (
        "partial (orchestrator only; sandbox runner venv uncaptured)"
    )


def test_capture_on_the_real_config_leaks_no_local_paths(tmp_path: Path) -> None:
    # Real Config.jsonify() carries absolute dirs (workspace_dir, memory_dir);
    # none of them may survive into the exported block of this PUBLIC repo.
    pytest.importorskip("litellm")
    from config import Config

    env = capture_environment(Config(), tmp_path / "missing", None)
    emitted = json.dumps(env)
    assert env["config_digest"].startswith("sha256:")
    assert "/Users/" not in emitted and "/home/" not in emitted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
