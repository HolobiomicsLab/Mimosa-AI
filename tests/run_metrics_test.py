"""Tests for sources.utils.run_metrics."""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from sources.utils.run_metrics import append_jsonl, write_run_metrics


def test_write_run_metrics_creates_file(tmp_path):
    folder = tmp_path / "abc"
    path = write_run_metrics(folder, {"uuid": "abc", "score": 0.5})
    assert path == folder / "run_metrics.json"
    assert json.loads(path.read_text()) == {"uuid": "abc", "score": 0.5}


def test_write_run_metrics_overwrites(tmp_path):
    folder = tmp_path / "abc"
    write_run_metrics(folder, {"v": 1})
    write_run_metrics(folder, {"v": 2})
    assert json.loads((folder / "run_metrics.json").read_text()) == {"v": 2}


def test_write_run_metrics_creates_parents(tmp_path):
    folder = tmp_path / "deep" / "nested" / "abc"
    write_run_metrics(folder, {"v": 1})
    assert (folder / "run_metrics.json").exists()


def test_write_run_metrics_coerces_non_json(tmp_path):
    folder = tmp_path / "abc"
    write_run_metrics(folder, {"path": Path("/tmp/x")})
    data = json.loads((folder / "run_metrics.json").read_text())
    assert isinstance(data["path"], str)


def test_append_jsonl_writes_one_line(tmp_path):
    p = tmp_path / "log.jsonl"
    append_jsonl(p, {"a": 1})
    assert p.read_text() == '{"a": 1}\n'


def test_append_jsonl_appends_multiple(tmp_path):
    p = tmp_path / "log.jsonl"
    append_jsonl(p, {"a": 1})
    append_jsonl(p, {"a": 2})
    lines = [json.loads(line) for line in p.read_text().splitlines()]
    assert lines == [{"a": 1}, {"a": 2}]


def test_append_jsonl_creates_parents(tmp_path):
    p = tmp_path / "a" / "b" / "log.jsonl"
    append_jsonl(p, {"a": 1})
    assert p.exists()


def test_append_jsonl_coerces_non_json(tmp_path):
    p = tmp_path / "log.jsonl"
    append_jsonl(p, {"path": Path("/tmp/x")})
    entry = json.loads(p.read_text())
    assert isinstance(entry["path"], str)
