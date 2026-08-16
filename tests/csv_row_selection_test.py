#!/usr/bin/env python3
"""
Tests for CSV row selection in batch evaluation mode.

`csv_runs_limit` was compared against the absolute CSV index rather than the
number of rows actually evaluated, so combining it with `--start_row` truncated
the run or selected nothing at all.
"""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.benchmark_evaluation.csv_mode import rows_to_evaluate

ROWS = [{"n": i} for i in range(10)]


def test_limit_counts_rows_from_start_row():
    """`--start_row 5 --csv_runs_limit 1` must evaluate exactly row 5.

    Comparing the limit to the absolute index made this select nothing, and the
    run reported "No tasks to process" with no explanation.
    """
    print("\n🧪 Testing limit is counted from start_row...")

    selected = rows_to_evaluate(ROWS, start_row=5, runs_limit=1)

    assert len(selected) == 1, f"Expected 1 row, got {len(selected)}"
    assert selected[0][0] == 5, f"Expected row index 5, got {selected[0][0]}"
    assert selected[0][1] == {"n": 5}
    print("✅ One row evaluated, starting at the requested offset")


def test_limit_without_offset_is_unchanged():
    """With start_row=0 the behaviour must match the previous implementation."""
    print("\n🧪 Testing limit without an offset...")

    selected = rows_to_evaluate(ROWS, start_row=0, runs_limit=3)

    assert [i for i, _ in selected] == [0, 1, 2], f"Got {[i for i, _ in selected]}"
    print("✅ Unchanged when no rows are skipped")


def test_limit_larger_than_remaining_rows_takes_the_rest():
    print("\n🧪 Testing limit beyond the end of the file...")

    selected = rows_to_evaluate(ROWS, start_row=8, runs_limit=100)

    assert [i for i, _ in selected] == [8, 9], f"Got {[i for i, _ in selected]}"
    print("✅ Takes every remaining row")


def test_start_row_beyond_the_file_selects_nothing():
    print("\n🧪 Testing start_row past the end...")

    assert rows_to_evaluate(ROWS, start_row=99, runs_limit=5) == []
    print("✅ Empty selection")


def test_zero_limit_selects_nothing():
    print("\n🧪 Testing a zero limit...")

    assert rows_to_evaluate(ROWS, start_row=0, runs_limit=0) == []
    print("✅ Empty selection")


def test_on_skip_reports_every_skipped_index():
    """Callers keep their per-row progress output."""
    print("\n🧪 Testing the on_skip callback...")

    skipped = []
    rows_to_evaluate(ROWS, start_row=3, runs_limit=2, on_skip=skipped.append)

    assert skipped == [0, 1, 2], f"Got {skipped}"
    print("✅ Skipped indices reported in order")


def run_all_tests():
    """Run all CSV row selection tests."""
    print("Starting CSV row selection tests...\n")

    try:
        test_limit_counts_rows_from_start_row()
        test_limit_without_offset_is_unchanged()
        test_limit_larger_than_remaining_rows_takes_the_rest()
        test_start_row_beyond_the_file_selects_nothing()
        test_zero_limit_selects_nothing()
        test_on_skip_reports_every_skipped_index()

        print("\n🎉 All CSV row selection tests passed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
