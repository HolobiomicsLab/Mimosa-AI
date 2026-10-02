"""Tests for config_store: config patching, key upserts, readiness views."""

from __future__ import annotations

import json
import stat

import pytest

from app import config_store
from app.settings import get_settings


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Point the settings singleton at throwaway config/env files."""
    monkeypatch.setenv("MIMOSA_CONFIG", str(tmp_path / "config.json"))
    get_settings.cache_clear()
    settings = get_settings()
    settings.env_files = [tmp_path / "project.env", tmp_path / "xdg.env"]
    yield tmp_path
    get_settings.cache_clear()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_update_creates_file_and_seeds_discovery(isolated):
    applied = config_store.update_config({"workspace_dir": "/tmp/ws", "max_tokens": "4096"})
    assert applied == {"workspace_dir": "/tmp/ws", "max_tokens": 4096}
    data = read_json(isolated / "config.json")
    assert data["discovery_addresses"] == [config_store.DEFAULT_DISCOVERY]
    assert data["max_tokens"] == 4096


def test_update_preserves_foreign_keys(isolated):
    (isolated / "config.json").write_text(
        json.dumps({"runner_requirements": ["numpy"], "discovery_addresses": [{"ip": "1.2.3.4"}]})
    )
    config_store.update_config({"judge_model": "anthropic/claude-opus-4-8"})
    data = read_json(isolated / "config.json")
    assert data["runner_requirements"] == ["numpy"]
    assert data["discovery_addresses"] == [{"ip": "1.2.3.4"}]
    assert data["judge_model"] == "anthropic/claude-opus-4-8"


def test_update_seeds_discovery_when_absent_from_existing_config(isolated):
    # A non-empty config lacking discovery_addresses must gain the default,
    # else Mimosa's from_json resets it to [] and MCP discovery breaks.
    (isolated / "config.json").write_text(json.dumps({"runner_requirements": ["numpy"]}))
    config_store.update_config({"judge_model": "x/y"})
    data = read_json(isolated / "config.json")
    assert data["discovery_addresses"] == [config_store.DEFAULT_DISCOVERY]


def test_update_respects_explicit_empty_discovery(isolated):
    (isolated / "config.json").write_text(json.dumps({"discovery_addresses": []}))
    config_store.update_config({"judge_model": "x/y"})
    assert read_json(isolated / "config.json")["discovery_addresses"] == []


@pytest.mark.parametrize(
    "patch",
    [
        {"max_tokens": "not-a-number"},
        {"export_astra": "yes"},
        {"max_tokens": True},
        {"learned_score_threshold": False},
        {"max_tokens": 4096.5},          # non-integral float for an int field
        {"workspace_dir": ["/a"]},       # structured value for a str field
        {"workspace_dir": 123},          # number for a str field
    ],
)
def test_update_rejects_bad_types(isolated, patch):
    with pytest.raises(ValueError):
        config_store.update_config(patch)


def test_update_accepts_integral_float_for_int(isolated):
    applied = config_store.update_config({"max_tokens": 4096.0})
    assert applied == {"max_tokens": 4096}


def test_update_ignores_unknown_and_none(isolated):
    applied = config_store.update_config({"rm_rf": "/", "workspace_dir": None})
    assert applied == {}
    assert "rm_rf" not in read_json(isolated / "config.json")


def test_save_key_prefers_project_env_when_defined_there(isolated):
    project = isolated / "project.env"
    project.write_text("OPENAI_API_KEY=old\nOTHER=1\n")
    saved = config_store.save_key("OPENAI_API_KEY", "sk-new")
    assert saved["saved_to"] == str(project)
    lines = project.read_text().splitlines()
    assert lines.count("OPENAI_API_KEY=sk-new") == 1
    assert "OTHER=1" in lines


def test_save_key_defaults_to_xdg_env(isolated):
    saved = config_store.save_key("ANTHROPIC_API_KEY", " sk-ant ")
    xdg = isolated / "xdg.env"
    assert saved["saved_to"] == str(xdg)
    assert xdg.read_text() == "ANTHROPIC_API_KEY=sk-ant\n"
    assert stat.S_IMODE(xdg.stat().st_mode) == 0o600


@pytest.mark.parametrize("name,value", [("NOT_A_KNOWN_KEY", "x"), ("HF_TOKEN", ""), ("HF_TOKEN", "a\nb")])
def test_save_key_rejects_bad_input(isolated, name, value):
    with pytest.raises(ValueError):
        config_store.save_key(name, value)


def test_editable_view_reports_workspace_missing(isolated):
    config_store.update_config({"workspace_dir": str(isolated / "nope")})
    view = config_store.editable_view()
    assert view["_workspace_exists"] is False
    assert view["_config_exists"] is True
    assert view["capsule_namer_model"] is None
