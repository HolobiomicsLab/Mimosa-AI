#!/usr/bin/env python3
"""Tests for onboarding CLI helper functions."""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.cli.onboard_cli import (
    _list_subdirectories,
    _parse_indices,
    _upsert_env_file,
)


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
