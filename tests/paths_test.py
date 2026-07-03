#!/usr/bin/env python3
"""Tests for sources/utils/paths.py and Config path resolution."""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from sources.utils import paths


def test_repo_checkout_detected():
    assert paths.is_repo_checkout(), "tests run from a checkout"
    assert (paths.PACKAGE_ROOT / "pyproject.toml").is_file()


def test_resource_path_points_at_shipped_files():
    for relative in (
        "sources/prompts/planner_reproduction.md",
        "sources/modules/state_schema.py",
        "sources/modules/smolagent_factory.py",
    ):
        resolved = paths.resource_path(relative)
        assert os.path.isabs(resolved)
        assert os.path.isfile(resolved), f"missing resource: {resolved}"


def test_xdg_env_overrides(monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", "/custom/config")
    monkeypatch.setenv("XDG_DATA_HOME", "/custom/data")
    monkeypatch.setenv("XDG_CACHE_HOME", "/custom/cache")
    assert str(paths.config_dir()) == "/custom/config/mimosa"
    assert str(paths.data_dir()) == "/custom/data/mimosa"
    assert str(paths.cache_dir()) == "/custom/cache/mimosa"
    assert str(paths.user_config_file()) == "/custom/config/mimosa/config.json"
    assert str(paths.user_env_file()) == "/custom/config/mimosa/.env"


def test_state_dir_repo_mode_keeps_checkout_layout():
    assert paths.default_memory_dir() == str(paths.PACKAGE_ROOT / "sources/memory")
    assert paths.default_workflow_dir() == str(paths.PACKAGE_ROOT / "sources/workflows")
    assert paths.default_runs_capsule_dir() == str(paths.PACKAGE_ROOT / "runs_capsule")
    assert paths.default_tmp_dir() == str(paths.PACKAGE_ROOT / "tmp")


def test_state_dir_installed_mode_uses_data_dir(monkeypatch):
    monkeypatch.setattr(paths, "is_repo_checkout", lambda: False)
    monkeypatch.setenv("XDG_DATA_HOME", "/custom/data")
    monkeypatch.setenv("XDG_CACHE_HOME", "/custom/cache")
    assert paths.default_memory_dir() == "/custom/data/mimosa/memory"
    assert paths.default_workflow_dir() == "/custom/data/mimosa/workflows"
    assert paths.default_tmp_dir() == "/custom/cache/mimosa/tmp"
    assert paths.pricing_cache_file() == "/custom/cache/mimosa/openrouter_pricing.json"


def test_config_defaults_are_absolute():
    config = Config()
    for value in (
        config.prompt_planner,
        config.prompt_workflow_creator,
        config.prompt_smolagent,
        config.schema_code_path,
        config.smolagent_factory_code_path,
        config.workflow_dir,
        config.memory_dir,
        config.runs_capsule_dir,
        config.runner_temp_dir,
    ):
        assert os.path.isabs(value), f"expected absolute path, got: {value}"


def test_from_json_reanchors_legacy_relative_paths():
    config = Config()
    config.from_json(
        {
            "prompt_planner": "sources/prompts/workflow_v10.md",
            "workflow_dir": "sources/workflows",
            "memory_dir": "sources/memory",
            "runs_capsule_dir": "runs_capsule/",
            "runner_temp_dir": "./tmp",
        }
    )
    assert config.prompt_planner == paths.resource_path("sources/prompts/workflow_v10.md")
    assert config.workflow_dir == str(paths.PACKAGE_ROOT / "sources/workflows")
    assert config.memory_dir == str(paths.PACKAGE_ROOT / "sources/memory")
    assert config.runs_capsule_dir == str(paths.PACKAGE_ROOT / "runs_capsule")
    assert config.runner_temp_dir == str(paths.PACKAGE_ROOT / "tmp")


def test_from_json_reanchors_tmp_to_cache_when_installed(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "is_repo_checkout", lambda: False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    config = Config()
    config.from_json({"runner_temp_dir": "./tmp", "memory_dir": "sources/memory"})
    assert config.runner_temp_dir == str(tmp_path / "cache/mimosa/tmp")
    assert config.memory_dir == str(tmp_path / "data/mimosa/memory")


def test_from_json_keeps_absolute_paths(tmp_path):
    config = Config()
    config.from_json({"workflow_dir": str(tmp_path / "wf")})
    assert config.workflow_dir == str(tmp_path / "wf")


def test_dump_creates_parent_directories(tmp_path):
    config = Config()
    target = tmp_path / "nested" / "dir" / "config.json"
    config.dump(str(target))
    assert target.is_file()


if __name__ == "__main__":
    test_repo_checkout_detected()
    test_resource_path_points_at_shipped_files()
    test_state_dir_repo_mode_keeps_checkout_layout()
    test_config_defaults_are_absolute()
    test_from_json_reanchors_legacy_relative_paths()
    print("✅ paths smoke checks passed")
