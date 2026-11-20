"""Whole-file column facts of task input tables, for reconnaissance and the guard.

Reconnaissance sees the first lines and a bounded preview of each file. Facts
that need the whole file stay hidden there: how many rows a table has, how
many distinct values a column holds, and a test label column that holds one
placeholder value in every row (dkpes Signal-inhibition = -999, ClinTox
FDA_APPROVED = 1, both found in run_1 2026-10-20). This module reads each
input table once with the csv module and reports two facts:

- ``input_profile``: rows, and per column the distinct count, the most common
  values and the numeric range, bounded for a prompt;
- ``one_value_columns``: columns with one value in every row of one table
  while the same column name varies in another input table.

No model call and no regex. The facts are evidence only: the criteria guard
decides what they mean for the goal.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

# Delimited text tables; BBBC002 keeps its counts in .txt files.
TABLE_SUFFIXES = (".csv", ".tsv", ".txt")
# Larger tables are not scanned.
MAX_TABLE_BYTES = 64 * 1024 * 1024
# A --task workspace can hold any number of tables; goal-named go first.
MAX_SCANNED_TABLES = 64
MIN_TABLE_ROWS = 2
# A .txt file needs this many fields to count as a table, not prose.
MIN_TEXT_TABLE_FIELDS = 2
TEXT_SUFFIX = ".txt"
DELIMITERS = ("\t", ";", ",")
# More rows off the header width than this share: prose, not a table.
MAX_RAGGED_SHARE = 0.5
# Distinct values counted exactly per column; more marks it saturated.
MAX_TRACKED_VALUES = 64
MAX_ONE_VALUE_TABLES = 8
MAX_ONE_VALUE_COLUMNS = 8
MAX_VARIED_TABLES = 4
MAX_PROFILE_TABLES = 8
MAX_PROFILE_COLUMNS = 24
TOP_VALUES = 3
VALUE_CHARACTERS = 40
# Shorter names match inside any goal word, so they must be whole tokens.
MIN_NAMED_CHARACTERS = 3


class Column:
    """Distinct values and numeric range of one table column.

    ``_tally`` makes one per header name and adds every cell of the column.
    The distinct values are exact up to ``MAX_TRACKED_VALUES``; the numeric
    range always covers every cell.
    """

    def __init__(self) -> None:
        """Start an empty column with no range and no saturation."""
        self.counts: Counter[str] = Counter()
        self.saturated = False
        self.numeric = True
        self.low: float | None = None
        self.high: float | None = None

    def add(self, cell: str) -> None:
        """Count one cell and widen the numeric range.

        Args:
            cell (str): Cell text without surrounding spaces; "" is a missing cell.

        Returns:
            None.
        """
        self._widen(cell)
        if cell in self.counts or len(self.counts) < MAX_TRACKED_VALUES:
            self.counts[cell] += 1
        else:
            self.saturated = True

    def single(self) -> str | None:
        """Return the one value of the column, when it has exactly one.

        Returns:
            str | None: The value ("" when every cell is empty), or None.
        """
        if self.saturated or len(self.counts) != 1:
            return None
        return next(iter(self.counts))

    def varies(self) -> bool:
        """Tell whether the column holds two or more distinct values.

        Returns:
            bool: True when it is saturated or has several values.
        """
        return self.saturated or len(self.counts) > 1

    def describe(self) -> dict:
        """Describe the column for a prompt.

        Returns:
            dict: ``distinct`` (int, or "64+" when saturated), ``top`` (value
            and count pairs, values cut to ``VALUE_CHARACTERS``) and ``range``
            (``[min, max]`` for a numeric column, else None).
        """
        distinct = f"{MAX_TRACKED_VALUES}+" if self.saturated else len(self.counts)
        top = [
            [value[:VALUE_CHARACTERS], count]
            for value, count in self.counts.most_common(TOP_VALUES)
        ]
        numeric = self.numeric and self.low is not None
        return {
            "distinct": distinct,
            "top": top,
            "range": [self.low, self.high] if numeric else None,
        }

    def _widen(self, cell: str) -> None:
        """Widen the numeric range, or mark the column as not numeric.

        Args:
            cell (str): Cell text; empty, NaN and infinite cells are skipped.

        Returns:
            None.
        """
        if not cell or not self.numeric:
            return
        number = _number(cell)
        if number is None:
            self.numeric = False
        elif math.isfinite(number):
            self.low = number if self.low is None else min(self.low, number)
            self.high = number if self.high is None else max(self.high, number)


def scan_tables(root: Path, keys: list[str], goal: str) -> dict[str, dict]:
    """Read each input table once, completely.

    Reconnaissance calls it with the input files from
    ``criteria_guard.input_files``: of the pre-run baseline, or of the first
    snapshot when there is no baseline. Only the first
    ``MAX_SCANNED_TABLES`` tables are read, those the goal names first.

    Args:
        root (Path): Folder that holds the keys.
        keys (list[str]): Root-relative paths; non-tables are skipped.
        goal (str): Task goal text.

    Returns:
        dict[str, dict]: Per readable table, ``rows`` (int) and ``columns``
        (header name to Column).
    """
    tables = [key for key in keys if _is_table(root / key)]
    scans = {
        key: _scan(root / key)
        for key in _goal_tables(tables, goal)[:MAX_SCANNED_TABLES]
    }
    return {key: scan for key, scan in scans.items() if scan is not None}


def one_value_columns(scans: dict[str, dict], goal: str) -> dict[str, dict]:
    """Find columns with one value in one table that vary in another table.

    Reconnaissance calls it once per unfrozen task and shows the result to
    the recon model and the criteria guard as hidden-label evidence.

    Args:
        scans (dict[str, dict]): Tables from ``scan_tables``.
        goal (str): Task goal; tables and columns it names are kept first.

    Returns:
        dict[str, dict]: Table key to ``columns`` (name to its ``value`` and
        the tables it ``varies_in``) and ``omitted_columns`` (int), for at
        most ``MAX_ONE_VALUE_TABLES`` tables.
    """
    varied = _varied_tables(scans)
    singles = {
        key: _singles(key, scans[key], varied)
        for key in _goal_tables(list(scans), goal)
    }
    found = {
        key: _bounded(columns, goal) for key, columns in singles.items() if columns
    }
    return dict(list(found.items())[:MAX_ONE_VALUE_TABLES])


def input_profile(scans: dict[str, dict], goal: str) -> dict:
    """Describe every input table from its complete content.

    Reconnaissance calls it once per unfrozen task. Tables and columns the
    goal names come first, so the bounds keep the inputs the goal names.

    Args:
        scans (dict[str, dict]): Tables from ``scan_tables``.
        goal (str): Task goal text.

    Returns:
        dict: ``tables`` (key to ``rows``, ``column_count`` and per-column
        ``columns`` descriptions) and ``omitted_tables`` (int).
    """
    keys = _goal_tables(list(scans), goal)[:MAX_PROFILE_TABLES]
    tables = {}
    for key in keys:
        columns = scans[key]["columns"]
        shown = _goal_first(list(columns), goal)[:MAX_PROFILE_COLUMNS]
        tables[key] = {
            "rows": scans[key]["rows"],
            "column_count": len(columns),
            "columns": {name: columns[name].describe() for name in shown},
        }
    return {"tables": tables, "omitted_tables": len(scans) - len(keys)}


def _singles(key: str, scan: dict, varied: dict[str, set[str]]) -> dict[str, dict]:
    """List the one-value columns of one table that vary in another table.

    Called by ``one_value_columns`` for each table.

    Args:
        key (str): Table key of ``scan``.
        scan (dict): One table from ``scan_tables``.
        varied (dict[str, set[str]]): Column name to the tables where it varies.

    Returns:
        dict[str, dict]: Column name to its ``value`` and ``varies_in``.
    """
    if scan["rows"] < MIN_TABLE_ROWS:
        return {}
    found = {}
    for name, column in scan["columns"].items():
        value, elsewhere = column.single(), sorted(varied.get(name, set()) - {key})
        if value is not None and elsewhere:
            found[name] = {
                "value": value[:VALUE_CHARACTERS],
                "varies_in": elsewhere[:MAX_VARIED_TABLES],
            }
    return found


def _bounded(columns: dict[str, dict], goal: str) -> dict:
    """Keep at most ``MAX_ONE_VALUE_COLUMNS`` findings, goal-named first.

    Called by ``one_value_columns``. FACTORS has over a thousand zero
    descriptor columns; the cap must not hide the goal's label.

    Args:
        columns (dict[str, dict]): Findings of one table from ``_singles``.
        goal (str): Task goal text.

    Returns:
        dict: ``columns`` (the kept findings) and ``omitted_columns`` (int).
    """
    kept = _goal_first(list(columns), goal)[:MAX_ONE_VALUE_COLUMNS]
    return {
        "columns": {name: columns[name] for name in kept},
        "omitted_columns": len(columns) - len(kept),
    }


def _scan(path: Path) -> dict | None:
    """Read one table file in a single pass.

    Called by ``scan_tables`` for each selected table.

    Args:
        path (Path): Table file accepted by ``_is_table``.

    Returns:
        dict | None: ``rows`` and ``columns``, or None for an unreadable or
        non-UTF-8 file, a table without a header, or a .txt file that is
        prose (one field, or no row under its first line, such as a note).
    """
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            first = stream.readline()
            stream.seek(0)
            scan = _tally(
                csv.reader(stream, delimiter=max(DELIMITERS, key=first.count))
            )
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    is_prose = (
        path.suffix.lower() == TEXT_SUFFIX
        and scan
        and (len(scan["columns"]) < MIN_TEXT_TABLE_FIELDS or not scan["rows"])
    )
    return None if is_prose else scan


def _tally(rows: Iterator[list[str]]) -> dict | None:
    """Count every cell of a table under its header row.

    Called by ``_scan``. A first row of numbers only is data, not a header
    (DAVIS affinity matrices), so that table is skipped. A file where most
    rows do not match the header width is prose, not a table.

    Args:
        rows (Iterator[list[str]]): ``csv.reader`` over the file, header row first.

    Returns:
        dict | None: ``rows`` and ``columns``, or None.
    """
    header = [name.strip() for name in next(rows, [])]
    if not any(header) or all(_number(name) is not None for name in header):
        return None
    columns = {name: Column() for name in header}
    count = ragged = 0
    for row in (row for row in rows if row):
        count += 1
        ragged += len(row) != len(header)
        for name, cell in zip(header, row, strict=False):
            columns[name].add(cell.strip())
    return (
        None
        if ragged > count * MAX_RAGGED_SHARE
        else {"rows": count, "columns": columns}
    )


def _varied_tables(scans: dict[str, dict]) -> dict[str, set[str]]:
    """Map each column name to the tables where it varies.

    Called by ``one_value_columns`` once for all tables.

    Args:
        scans (dict[str, dict]): Tables from ``scan_tables``.

    Returns:
        dict[str, set[str]]: Column name to table keys.
    """
    varied: dict[str, set[str]] = {}
    for key, scan in scans.items():
        for name, column in scan["columns"].items():
            if column.varies():
                varied.setdefault(name, set()).add(key)
    return varied


def _goal_tables(keys: list[str], goal: str) -> list[str]:
    """Order table keys so the files the goal names come first.

    Called by ``scan_tables`` and both public facts, so their table bounds
    keep the same tables.

    Args:
        keys (list[str]): Table keys.
        goal (str): Task goal text.

    Returns:
        list[str]: Sorted keys; goal-named file names first.
    """
    words = _goal_words(goal)
    return sorted(sorted(keys), key=lambda key: not _is_named(Path(key).name, words))


def _goal_first(names: list[str], goal: str) -> list[str]:
    """Order column names so the names the goal mentions come first.

    Called by ``_bounded`` and ``input_profile`` before they cut a list.

    Args:
        names (list[str]): Column names in file order.
        goal (str): Task goal text.

    Returns:
        list[str]: The same names; the sort keeps file order otherwise.
    """
    words = _goal_words(goal)
    return sorted(names, key=lambda name: not _is_named(name, words))


def _goal_words(goal: str) -> tuple[str, set[str]]:
    """Prepare the goal once for many name checks.

    Called by the two ranking helpers; FACTORS ranks over a thousand names.

    Args:
        goal (str): Task goal text.

    Returns:
        tuple[str, set[str]]: The lowered goal with whitespace collapsed,
        and its alphanumeric tokens (underscore kept).
    """
    text = " ".join(goal.lower().split())
    return text, set(
        "".join(char if char.isalnum() or char == "_" else " " for char in text).split()
    )


def _is_named(name: str, words: tuple[str, set[str]]) -> bool:
    """Tell whether the goal mentions a column or file name.

    Called by the two ranking helpers; a ranking hint only, never a filter.

    Args:
        name (str): Column name or file name.
        words (tuple[str, set[str]]): Goal text and tokens from ``_goal_words``.

    Returns:
        bool: A name shorter than ``MIN_NAMED_CHARACTERS`` (the TDC label
        "Y") must be a whole goal token; a longer one may be a substring.
    """
    lowered = " ".join(name.lower().split())
    text, tokens = words
    return lowered in tokens if len(lowered) < MIN_NAMED_CHARACTERS else lowered in text


def _is_table(path: Path) -> bool:
    """Tell whether a path is a regular table file of acceptable size.

    Called by ``scan_tables`` before it reads a file.

    Args:
        path (Path): Candidate path.

    Returns:
        bool: True for a table suffix, a regular file (not a link) and at
        most ``MAX_TABLE_BYTES``.
    """
    return (
        path.suffix.lower() in TABLE_SUFFIXES
        and not path.is_symlink()
        and path.is_file()
        and path.stat().st_size <= MAX_TABLE_BYTES
    )


def _number(text: str) -> float | None:
    """Parse a cell or header as a number.

    Called by ``_tally`` (numeric header check) and ``Column._widen``.

    Args:
        text (str): Cell text.

    Returns:
        float | None: The number, or None when float() rejects the text.
    """
    try:
        return float(text)
    except ValueError:
        return None


if __name__ == "__main__":
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "train.csv").write_text(
            "id,Signal\n1,0.2\n2,0.9\n3,0.4\n", encoding="utf-8"
        )
        (root / "test.csv").write_text("id,Signal\n4,-999\n5,-999\n", encoding="utf-8")
        found = scan_tables(root, ["test.csv", "train.csv"], "Predict Signal")
        facts = {
            "one_value_columns": one_value_columns(found, "Predict Signal"),
            "input_profile": input_profile(found, "Predict Signal"),
        }
        print(json.dumps(facts, indent=1))
