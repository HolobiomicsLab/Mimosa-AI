#!/usr/bin/env python3
"""Tests for onboarding CLI helper functions."""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.cli.onboard_cli import (
    _RECOMMENDED_MODELS,
    _list_subdirectories,
    _parse_indices,
    _recommended_presets,
    _upsert_env_file,
)

_ALL_KEY_VARS = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "MISTRAL_API_KEY",
    "HF_TOKEN",
    "OPENROUTER_API_KEY",
]


def test_recommended_presets_follow_available_keys(monkeypatch):
    for var in _ALL_KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")

    orchestration = _recommended_presets("orchestration")
    assert [m for _, m in orchestration] == [
        "anthropic/claude-opus-4-8",
        "mistral/mistral-medium-3-5",
    ]
    agents = _recommended_presets("agent")
    assert [m for _, m in agents] == [
        "anthropic/claude-sonnet-5",
        "mistral/mistral-small-2603",
    ]
    judges = _recommended_presets("judge")
    assert [m for _, m in judges] == [
        "anthropic/claude-sonnet-5",
        "mistral/mistral-medium-3-5",
    ]


def test_recommended_presets_empty_for_unmapped_keys(monkeypatch):
    for var in _ALL_KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert _recommended_presets("orchestration") == []


def test_recommendation_table_covers_all_roles():
    for env_key, models in _RECOMMENDED_MODELS.items():
        assert set(models) == {"orchestration", "agent", "judge"}, env_key


def test_parse_indices_singles_and_ranges():
    assert _parse_indices("1,3", 6) == ({1, 3}, False)
    assert _parse_indices("2-4", 6) == ({2, 3, 4}, False)
    assert _parse_indices(" 1, 5 ", 6) == ({1, 5}, False)


def test_parse_indices_out_of_range_and_garbage():
    assert _parse_indices("0,7", 6) == (set(), False)
    assert _parse_indices("abc", 6) == (set(), True)
    assert _parse_indices("my-folder", 6) == (set(), True)
    assert _parse_indices("", 6) == (set(), False)


def test_list_subdirectories(tmp_path):
    (tmp_path / "b_dir").mkdir()
    (tmp_path / "a_dir").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "file.txt").write_text("x")
    assert _list_subdirectories(str(tmp_path)) == ["a_dir", "b_dir"]
    assert _list_subdirectories(str(tmp_path / "missing")) == []


def test_upsert_env_file_creates_and_updates(tmp_path):
    env_file = tmp_path / "cfg" / ".env"
    _upsert_env_file(env_file, {"A_KEY": "one"})
    assert env_file.read_text() == "A_KEY=one\n"

    _upsert_env_file(env_file, {"A_KEY": "two", "B_KEY": "b"})
    content = env_file.read_text().splitlines()
    assert "A_KEY=two" in content
    assert "B_KEY=b" in content
    assert "A_KEY=one" not in content
    assert oct(env_file.stat().st_mode & 0o777) == "0o600"


if __name__ == "__main__":
    test_parse_indices_singles_and_ranges()
    test_parse_indices_out_of_range_and_garbage()
    print("✅ onboard CLI helper smoke checks passed")
