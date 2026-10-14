"""Behavior checks for the restricted policy language and its isolated worker."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sources.evaluators.policy_bt_verifier.policy import (
    SOURCE_BYTES,
    AdaptationError,
    ArtifactReader,
    Interpreter,
    PolicyError,
    run_pair,
    screen_source,
)

WORKER_TIMEOUT = 30.0
HANG_TIMEOUT = 1.0
CRITERIA = [{"id": "score", "stage": 1, "description": "Report the score.", "direction": "higher"}]
UNKNOWN_CRITERIA = CRITERIA + [{"id": "runtime", "stage": 2,
                                "description": "Report the runtime.", "direction": "lower"}]

ADAPTER = '''
def normalize():
    data = read_json("result.json")
    return {"values": {"score": data["score"]},
            "evidence": [{"criterion": "score", "path": "result.json",
                          "value": data["score"], "detail": "score field"}]}
'''
CORE = '''
def tagged(side, normalized):
    entries = []
    for item in normalized["evidence"]:
        entry = dict(item)
        entry["side"] = side
        entries.append(entry)
    return entries

def compare(a, b):
    x = a["values"]["score"]
    y = b["values"]["score"]
    if x == y:
        return {"winner": "tie", "stage": 0, "feedback_a": f"Score {x:.3f} ties.",
                "feedback_b": f"Score {y:.3f} ties.", "evidence": []}
    winner = "a" if x > y else "b"
    return {"winner": winner, "stage": 1,
            "feedback_a": f"Score {x:.3f} against {y:.3f}.",
            "feedback_b": f"Score {y:.3f} against {x:.3f}.",
            "evidence": tagged("a", a) + tagged("b", b)}
'''
ALWAYS_A = '''
def compare(a, b):
    evidence = []
    for item in a["evidence"]:
        entry = dict(item)
        entry["side"] = "a"
        evidence.append(entry)
    return {"winner": "a", "stage": 1, "feedback_a": "Wins.", "feedback_b": "Loses.",
            "evidence": evidence}
'''
HANGING = '''
def compare(a, b):
    total = 0
    for step in range(10 ** 12):
        total += step
    return {"winner": "tie", "stage": 0, "feedback_a": "x", "feedback_b": "x", "evidence": []}
'''
FAILING_ADAPTER = '''
def normalize():
    raise ValueError("score file has no usable field")
'''


def _pair(root: Path, score_a: float, score_b: float) -> Path:
    """Create a real pair root for worker tests.

    Args: root (Path): pytest tmp dir; score_a, score_b (float): values written to a/ and b/.
    Returns: Path: Pair root holding a/result.json and b/result.json.
    """
    for side, score in (("a", score_a), ("b", score_b)):
        (root / side).mkdir(parents=True)
        (root / side / "result.json").write_text(json.dumps({"score": score}))
    return root


def _payload(root: Path, core: str = CORE, adapter: str = ADAPTER, criteria: list | None = None) -> dict:
    """Build the worker request contract.

    Args: root (Path): pair root; core, adapter (str): policy sources; criteria (list|None): frozen criteria.
    Returns: dict: Payload accepted by run_pair.
    """
    return {"root": str(root), "core_source": core, "criteria": criteria or CRITERIA,
            "adapters": {"a": adapter, "b": adapter}}


def _evaluate(body: str):
    """Run one interpreted function with no artifact capabilities.

    Args: body (str): source of a zero-argument function named f.
    Returns: Value returned by the interpreted f().
    """
    return Interpreter(body).invoke("f", [])


@pytest.mark.parametrize("source", [
    "import os",
    "from os import path",
    "while True:\n    pass",
    "try:\n    x = 1\nexcept ValueError:\n    pass",
    "f = lambda: 1",
    "class Box:\n    pass",
    "x = y.__class__",
    "x = _hidden",
    "def _private():\n    pass",
    "@cache\ndef f():\n    pass",
    "def f():\n    return f'{x.__class__}'",
])
def test_screen_rejects_forbidden_syntax(source):
    with pytest.raises(PolicyError):
        screen_source(source)


def test_screen_rejects_oversized_source():
    line = "x = 1\n"
    with pytest.raises(PolicyError):
        screen_source(line * (SOURCE_BYTES // len(line) + 1))


@pytest.mark.parametrize("call", ["open('x')", "getattr(1, 'real')", "eval('1')", "exec('1')",
                                  "compile('1', 'x', 'eval')", "'{}'.format(1)", "[1].pop()"])
def test_interpreter_blocks_host_access(call):
    with pytest.raises(PolicyError):
        _evaluate(f"def f():\n    return {call}")


def test_interpreter_blocks_dunder_import():
    with pytest.raises(PolicyError):
        Interpreter("def f():\n    return __import__('os')")


@pytest.mark.parametrize(("expression", "expected"), [
    ("f\"{1.23456:.2f} {'a'!r}\"", "1.23 'a'"),
    ("f\"{'x':>{3}}|{7:03d}\"", "  x|007"),
    ("f\"{'é'!a} {5!s:>2}\"", "'\\xe9'  5"),
    ("f\"{{literal}} {len([1, 2])}\"", "{literal} 2"),
])
def test_fstrings_follow_python_semantics(expression, expected):
    assert _evaluate(f"def f():\n    return {expression}") == expected


def test_nested_functions_see_enclosing_scope_like_python():
    """Live regression: a nested helper failed with 'Undefined name: stages'."""
    source = (
        "LIMIT = 3\n"
        "def compare(a, b):\n"
        "    stages = {'x': 1}\n"
        "    def decide(key):\n"
        "        return stages[key] + LIMIT\n"
        "    return decide('x')\n"
        "def fact(n):\n"
        "    return 1 if n <= 1 else n * fact(n - 1)\n"
    )
    interpreter = Interpreter(source)
    assert interpreter.invoke("compare", [None, None]) == 4
    assert interpreter.invoke("fact", [5]) == 120


def test_reader_confines_paths_and_records_reads(tmp_path):
    root = tmp_path / "side"
    root.mkdir()
    (root / "data.csv").write_text("id,score\n1,0.5\n")
    (tmp_path / "secret.txt").write_text("outside")
    (root / "escape.txt").symlink_to(tmp_path / "secret.txt")
    reader = ArtifactReader(root)
    for name in (str(tmp_path / "secret.txt"), "../secret.txt", "escape.txt"):
        with pytest.raises(ValueError):
            reader.read_text(name)
    assert reader.read_csv("data.csv") == [{"id": "1", "score": "0.5"}]
    assert reader.exists("missing.json") is False
    assert reader.observed == {"data.csv", "missing.json"}
    assert reader.list_files() == ["data.csv"]


def test_worker_picks_higher_score_in_both_orders(tmp_path):
    forward = run_pair(_payload(_pair(tmp_path / "f", 0.9, 0.5)), WORKER_TIMEOUT)
    backward = run_pair(_payload(_pair(tmp_path / "b", 0.5, 0.9)), WORKER_TIMEOUT)
    assert (forward["winner"], backward["winner"]) == ("a", "b")
    assert forward["feedback_a"] == "Score 0.900 against 0.500."
    assert {entry["side"] for entry in forward["evidence"]} == {"a", "b"}


def test_worker_rejects_order_sensitive_comparator(tmp_path):
    with pytest.raises(PolicyError, match="symmetry") as caught:
        run_pair(_payload(_pair(tmp_path, 0.9, 0.5), core=ALWAYS_A), WORKER_TIMEOUT)
    assert not isinstance(caught.value, AdaptationError)


BLIND_TIE = '''
def compare(a, b):
    x = a.get("score")
    y = b.get("score")
    return {"winner": "tie", "stage": 0, "feedback_a": "No measured difference.",
            "feedback_b": "No measured difference.", "evidence": []}
'''


ALL_NULL_ADAPTER = ADAPTER.replace(
    '{"score": data["score"]}',
    '{"score": None if data.get("score") is None else None}',
).replace(
    '"evidence": [{"criterion": "score", "path": "result.json",\n                          "value": data["score"], "detail": "score field"}]',
    '"evidence": [{"criterion": "score", "path": "result.json",\n                          "value": None, "detail": "score field"}]',
)


def test_worker_rejects_adapter_that_measures_nothing(tmp_path):
    """Live regression: an all-null side silently tied pairs instead of repairing."""
    with pytest.raises(AdaptationError, match="measured nothing"):
        run_pair(_payload(_pair(tmp_path, 0.9, 0.5), adapter=ALL_NULL_ADAPTER), WORKER_TIMEOUT)


def test_worker_rejects_tie_that_ignores_a_measured_difference(tmp_path):
    """Live regression: a comparator reading the wrong keys tied -9 vs -8."""
    with pytest.raises(PolicyError, match="differs"):
        run_pair(_payload(_pair(tmp_path, 0.9, 0.5), core=BLIND_TIE), WORKER_TIMEOUT)


def test_worker_accepts_tie_on_equal_measurements(tmp_path):
    verdict = run_pair(_payload(_pair(tmp_path, 0.7, 0.7), core=BLIND_TIE), WORKER_TIMEOUT)
    assert verdict["winner"] == "tie"


def test_worker_reports_adapter_failure_as_adaptation(tmp_path):
    with pytest.raises(AdaptationError):
        run_pair(_payload(_pair(tmp_path, 0.9, 0.5), adapter=FAILING_ADAPTER), WORKER_TIMEOUT)


def test_worker_kills_nonterminating_comparator(tmp_path):
    with pytest.raises(PolicyError, match="timed out"):
        run_pair(_payload(_pair(tmp_path, 0.9, 0.5), core=HANGING), HANG_TIMEOUT)


def test_worker_accepts_unknown_value_without_evidence(tmp_path):
    adapter = ADAPTER.replace('{"score": data["score"]}', '{"score": data["score"], "runtime": None}')
    verdict = run_pair(_payload(_pair(tmp_path, 0.9, 0.5), adapter=adapter, criteria=UNKNOWN_CRITERIA),
                       WORKER_TIMEOUT)
    assert verdict["winner"] == "a"
    assert verdict["normalized_a"]["values"]["runtime"] is None


def test_worker_rejects_value_without_evidence(tmp_path):
    adapter = ADAPTER.replace('{"score": data["score"]}', '{"score": data["score"], "runtime": 3.0}')
    with pytest.raises(AdaptationError, match="no artifact evidence"):
        run_pair(_payload(_pair(tmp_path, 0.9, 0.5), adapter=adapter, criteria=UNKNOWN_CRITERIA),
                 WORKER_TIMEOUT)
