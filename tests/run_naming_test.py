#!/usr/bin/env python3
"""
Tests for deterministic run and capsule naming.

Two fallbacks produced a different name on every process: `abs(hash(goal))` for
the capsule name (salted by PYTHONHASHSEED) and `id(row)` for the workspace name
(a memory address). Reruns of the same task therefore never reused their own
workspace, capsule or run notes.
"""

import hashlib
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.benchmark_evaluation.csv_mode import CsvEvaluationMode
from sources.utils.transfer_toolomics import LocalTransfer

extract_name = CsvEvaluationMode._extract_workspace_name_from_row


def _transfer():
    config = SimpleNamespace(capsule_namer_model="deepseek/deepseek-chat")
    return LocalTransfer(config, workspace_path="/tmp/workspace")


def test_capsule_fallback_is_a_stable_digest_of_the_goal():
    """With no task_token and no usable LLM reply, the name must be stable.

    `abs(hash(goal))` is salted per process, so the archive layout changed on
    every run despite the docstring promising a deterministic fallback.
    """
    print("\n🧪 Testing capsule name fallback determinism...")

    goal = "Reproduce the untargeted metabolomics annotation experiment"
    expected = "capsule_" + hashlib.sha256(goal.encode("utf-8")).hexdigest()[:12]

    transfer = _transfer()
    with patch(
        "sources.utils.transfer_toolomics.LLMProvider",
        side_effect=RuntimeError("no LLM in tests"),
    ):
        name = transfer.create_capsule_name(goal)

    assert name == expected, f"Expected {expected}, got {name}"
    print("✅ Capsule fallback is a stable digest")


def test_capsule_fallback_still_prefers_the_task_token():
    print("\n🧪 Testing capsule fallback with a task token...")

    transfer = _transfer()
    with patch(
        "sources.utils.transfer_toolomics.LLMProvider",
        side_effect=RuntimeError("no LLM in tests"),
    ):
        name = transfer.create_capsule_name("some goal", task_token="task_007")

    assert name == "capsule_task_007", f"Got {name}"
    print("✅ Task token still wins")


def test_workspace_name_prefers_gold_program_name():
    """Existing ScienceAgentBench behaviour must not change."""
    print("\n🧪 Testing gold_program_name is still preferred...")

    assert extract_name({"gold_program_name": "clintox_nn.py"}) == "clintox_nn"
    assert extract_name({"instance_id": "inst-9"}) == "inst_9"
    print("✅ Unchanged for ScienceAgentBench rows")


def test_workspace_name_uses_task_id_then_title():
    """ASB rows carry TaskID and Title but neither SAB column."""
    print("\n🧪 Testing TaskID and Title fallbacks...")

    assert extract_name({"TaskID": "chal-3", "Title": "A paper"}) == "chal_3"
    assert extract_name({"Title": "A paper title"}) == "A_paper_title"
    print("✅ TaskID preferred over Title")


def test_workspace_name_is_deterministic_without_any_identifier():
    """A row with no identifying column must still name deterministically."""
    print("\n🧪 Testing digest fallback for unidentified rows...")

    row = {"Prompt": "do the thing", "Difficulty": "hard"}
    first = extract_name(row)
    second = extract_name(dict(row))

    assert first == second, f"Not deterministic: {first} != {second}"
    assert first.startswith("task_"), f"Unexpected name {first}"
    assert "0x" not in first, "Name still contains a memory address"
    print(f"✅ Stable name for an unidentified row: {first}")


def test_workspace_name_is_bounded_in_length():
    """A Title-derived name must stay path-friendly."""
    print("\n🧪 Testing name length bound...")

    name = extract_name({"Title": "x" * 500})

    assert len(name) <= 80, f"Name is {len(name)} chars"
    print("✅ Name capped at 80 characters")


def run_all_tests():
    """Run all run-naming tests."""
    print("Starting run naming tests...\n")

    try:
        test_capsule_fallback_is_a_stable_digest_of_the_goal()
        test_capsule_fallback_still_prefers_the_task_token()
        test_workspace_name_prefers_gold_program_name()
        test_workspace_name_uses_task_id_then_title()
        test_workspace_name_is_deterministic_without_any_identifier()
        test_workspace_name_is_bounded_in_length()

        print("\n🎉 All run naming tests passed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
