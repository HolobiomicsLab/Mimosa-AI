#!/usr/bin/env python3
"""
Tests for deterministic workspace naming in CSV evaluation mode.

`_extract_workspace_name_from_row` fell back to `id(row)` — the memory address
of a throwaway dict — whenever a row lacked both `gold_program_name` and
`instance_id`. Any CSV outside ScienceAgentBench therefore named its workspace
and run notes differently on every process, so reruns of the same task could
not be compared against their predecessor.
"""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.benchmark_evaluation.csv_mode import CsvEvaluationMode

extract_name = CsvEvaluationMode._extract_workspace_name_from_row


def test_workspace_name_prefers_gold_program_name():
    """Existing ScienceAgentBench behaviour must not change."""
    print("\n🧪 Testing gold_program_name is still preferred...")

    assert extract_name({"gold_program_name": "clintox_nn.py"}) == "clintox_nn"
    assert extract_name({"instance_id": "inst-9"}) == "inst_9"
    print("✅ Unchanged for ScienceAgentBench rows")


def test_workspace_name_uses_task_id_then_title():
    """Rows from other benchmarks carry TaskID and Title but neither SAB column."""
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
    """Run all workspace naming tests."""
    print("Starting workspace naming tests...\n")

    try:
        test_workspace_name_prefers_gold_program_name()
        test_workspace_name_uses_task_id_then_title()
        test_workspace_name_is_deterministic_without_any_identifier()
        test_workspace_name_is_bounded_in_length()

        print("\n🎉 All workspace naming tests passed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
