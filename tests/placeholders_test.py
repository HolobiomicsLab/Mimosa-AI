"""Whole-file column facts that reconnaissance and the criteria guard read."""

from pathlib import Path

import pytest

from sources.evaluators.policy_bt_verifier.placeholders import (
    MAX_ONE_VALUE_COLUMNS,
    MAX_PROFILE_TABLES,
    MAX_TRACKED_VALUES,
    input_profile,
    one_value_columns,
    scan_tables,
    table_keys,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _facts(root: Path, goal: str = "") -> tuple[dict, dict]:
    scans = scan_tables(root, table_keys(root))
    return one_value_columns(scans, goal), input_profile(scans, goal)


def test_hidden_test_label_is_reported_with_the_table_where_it_varies(tmp_path):
    """dkpes: Signal-inhibition is -999 in every test row and varies in train."""
    _write(
        tmp_path / "dkpes" / "dkpes_train.csv",
        "index,Signal-inhibition\n1,0.2\n2,0.9\n",
    )
    _write(
        tmp_path / "dkpes" / "dkpes_test.csv",
        "index,Signal-inhibition\n3,-999\n4,-999\n5,-999\n",
    )
    found, _ = _facts(tmp_path)
    assert found == {
        "dkpes/dkpes_test.csv": {
            "columns": {
                "Signal-inhibition": {
                    "value": "-999",
                    "varies_in": ["dkpes/dkpes_train.csv"],
                }
            },
            "omitted_columns": 0,
        }
    }


def test_column_constant_in_every_table_is_no_placeholder(tmp_path):
    _write(tmp_path / "train.csv", "id,source\n1,lab\n2,lab\n")
    _write(tmp_path / "test.csv", "id,source\n3,lab\n4,lab\n")
    assert _facts(tmp_path)[0] == {}


def test_headerless_numeric_matrix_is_not_read_as_a_header(tmp_path):
    """DAVIS affinity_val.csv: a numeric first row made column '1100' a false finding."""
    _write(tmp_path / "affinity_train.csv", "120.0,1100.0\n5.0,7.0\n9.0,3.0\n")
    _write(
        tmp_path / "affinity_val.csv",
        "120.0,1100.0\n10000.0,10000.0\n10000.0,10000.0\n",
    )
    found, profile = _facts(tmp_path)
    assert found == {} and profile["tables"] == {}


@pytest.mark.parametrize("label", ["Activity", "Y"])
def test_cap_keeps_goal_named_columns_and_counts_the_rest(tmp_path, label):
    """FACTORS: thousands of zero descriptors must not hide the goal's label.

    "Y" (the TDC label name) is too short for a substring match.
    """
    names = [f"D_{index:05d}" for index in range(MAX_ONE_VALUE_COLUMNS + 3)] + [label]
    rows = {"train.csv": ["1", "2"], "test.csv": ["0", "0"]}
    for table, values in rows.items():
        lines = [",".join(names)] + [",".join([value] * len(names)) for value in values]
        _write(tmp_path / table, "\n".join(lines))
    found = _facts(tmp_path, f"Predict {label} for test.csv")[0]["test.csv"]
    assert label in found["columns"] and len(found["columns"]) == MAX_ONE_VALUE_COLUMNS
    assert found["omitted_columns"] == len(names) - MAX_ONE_VALUE_COLUMNS


def test_profile_counts_the_whole_file_beyond_any_preview(tmp_path):
    """The last row decides the distinct count, the range and the row total."""
    rows = [f"{index},A,{index}" for index in range(5000)] + ["5000,B,-7"]
    _write(tmp_path / "test.csv", "id,label,score\n" + "\n".join(rows) + "\n")
    columns = _facts(tmp_path)[1]["tables"]["test.csv"]
    assert columns["rows"] == 5001
    assert columns["columns"]["label"] == {
        "distinct": 2,
        "top": [["A", 5000], ["B", 1]],
        "range": None,
    }
    assert columns["columns"]["score"]["distinct"] == f"{MAX_TRACKED_VALUES}+"
    assert columns["columns"]["score"]["range"] == [-7.0, 4999.0]


def test_profile_keeps_the_goal_named_table_when_a_folder_has_too_many(tmp_path):
    for index in range(MAX_PROFILE_TABLES + 2):
        _write(tmp_path / f"aux_{index:02d}.csv", "a,b\n1,2\n")
    _write(tmp_path / "zz_test.csv", "id,Y\n1,0\n")
    profile = _facts(tmp_path, "Predict Y for zz_test.csv")[1]
    assert "zz_test.csv" in profile["tables"] and profile["omitted_tables"] == 3


def test_semicolon_table_and_single_column_csv_are_profiled_but_prose_is_not(tmp_path):
    """nvc uses ';'; dili test.csv has one column, so its label is visibly absent."""
    _write(tmp_path / "nvc.csv", "Model;Rule;Num persons\nA;r1;3\nB;r2;4\n")
    _write(tmp_path / "test.csv", "standardised_smiles\nCCO\nCCN\n")
    _write(tmp_path / "dem_source.txt", "NASADEM Merged DEM Global\n")
    tables = _facts(tmp_path)[1]["tables"]
    assert sorted(tables) == ["nvc.csv", "test.csv"]
    assert list(tables["nvc.csv"]["columns"]) == ["Model", "Rule", "Num persons"]
    assert tables["test.csv"]["column_count"] == 1


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
