"""Tolerant LLM-JSON parsing.

The defects here are not hypothetical. Running ASB capsules against
``stealth/ox-alpha``, the planner failed all three of its attempts with
``Failed to extract valid JSON from LLM response`` / ``Invalid control
character at …`` and the whole task was abandoned. The bare-inner-quote case
was reproduced directly against the same model.

The contract: valid JSON is never altered, repair is attempted only after a
strict parse fails, and when nothing parses the ORIGINAL decode error surfaces
so callers still see the true defect.
"""

import json

import pytest

from sources.utils.llm_json import (
    loads_llm_json,
    repair_json_strings,
    strip_json_fence,
    strip_trailing_commas,
)


# ---------------------------------------------------------------------------
# Valid JSON must survive untouched
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    {"a": 1},
    {"steps": [{"name": "x", "complexity": "medium"}]},
    {"text": "already \\\"escaped\\\" quotes"},
    {"text": "line1\nline2"},          # properly escaped by json.dumps
    {"nested": {"deep": [1, 2, {"k": None}]}},
    [],
    {},
])
def test_valid_json_round_trips_unchanged(payload):
    encoded = json.dumps(payload)
    assert loads_llm_json(encoded) == payload
    # The repair pass must be a no-op on already-valid JSON.
    assert repair_json_strings(encoded) == encoded


# ---------------------------------------------------------------------------
# The defects actually observed
# ---------------------------------------------------------------------------

def test_raw_control_character_inside_a_string_is_recovered():
    """The exact shape that killed the planner: a pasted multi-line span."""
    broken = '{"code": "import pandas as pd\nprint(df.head())"}'
    with pytest.raises(json.JSONDecodeError):
        json.loads(broken)
    assert loads_llm_json(broken) == {
        "code": "import pandas as pd\nprint(df.head())"
    }


def test_bare_inner_quotes_are_escaped():
    broken = '{"code": "df = pd.read_csv("data.csv")"}'
    with pytest.raises(json.JSONDecodeError):
        json.loads(broken)
    assert loads_llm_json(broken) == {"code": 'df = pd.read_csv("data.csv")'}


def test_trailing_prose_after_a_complete_object_is_ignored():
    assert loads_llm_json('{"ok": true}\n\nHope that helps!') == {"ok": True}


def test_tab_and_carriage_return_are_escaped():
    assert loads_llm_json('{"t": "a\tb\r\nc"}') == {"t": "a\tb\r\nc"}


def test_combined_defects_in_one_payload():
    broken = '{"span": "the "obiwarp" method\nwas used"}'
    assert loads_llm_json(broken) == {"span": 'the "obiwarp" method\nwas used'}


# ---------------------------------------------------------------------------
# Fences
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fenced,expected", [
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('```\n{"a": 1}\n```', {"a": 1}),
    ('```JSON\n{"a": 1}\n```', {"a": 1}),
])
def test_fenced_payloads_are_unwrapped(fenced, expected):
    assert loads_llm_json(fenced) == expected


def test_none_content_is_treated_as_empty():
    """A reasoning model that burns its budget returns null content."""
    assert strip_json_fence(None) == ""
    with pytest.raises(json.JSONDecodeError):
        loads_llm_json(None)


# ---------------------------------------------------------------------------
# Failure surfaces the true defect
# ---------------------------------------------------------------------------

def test_unparseable_input_raises_the_original_error():
    with pytest.raises(json.JSONDecodeError) as strict:
        json.loads("not json at all")
    with pytest.raises(json.JSONDecodeError) as tolerant:
        loads_llm_json("not json at all")
    assert tolerant.value.msg == strict.value.msg


def test_repair_never_invents_a_value_for_empty_input():
    with pytest.raises(json.JSONDecodeError):
        loads_llm_json("")


# ---------------------------------------------------------------------------
# Planner integration — the call site that failed
# ---------------------------------------------------------------------------

def test_planner_extracts_a_plan_containing_a_raw_newline():
    from sources.core.planner import Planner

    response = (
        "Here is the plan:\n"
        "```json\n"
        '{"steps": [{"description": "run mzmine\non the mzML files"}]}\n'
        "```\n"
    )
    got = Planner._extract_json_from_code_block(response)
    assert got == {"steps": [{"description": "run mzmine\non the mzML files"}]}


def test_planner_returns_none_when_there_is_no_code_block():
    from sources.core.planner import Planner

    assert Planner._extract_json_from_code_block("no fenced block here") is None


# ---------------------------------------------------------------------------
# Trailing commas
# ---------------------------------------------------------------------------
# JSON forbids the trailing comma that JavaScript and Python allow. It surfaces
# as "Expecting property name enclosed in double quotes", which reads like a
# quoting fault and is not. Observed six times in one failed planner run
# (lane1_t0 / p_iimn) against stealth/ox-alpha, after the control-character
# repair had already landed.


@pytest.mark.parametrize("broken,expected", [
    ('{"a": 1,}', {"a": 1}),
    ('{"a": [1, 2,],}', {"a": [1, 2]}),
    ('{"steps": [{"n": 1,}, {"n": 2,},],}', {"steps": [{"n": 1}, {"n": 2}]}),
    ('[1, 2, 3,]', [1, 2, 3]),
    ('{"a": 1 , }', {"a": 1}),
    ('{"a": {"b": 2,},}', {"a": {"b": 2}}),
])
def test_trailing_commas_are_dropped(broken, expected):
    with pytest.raises(json.JSONDecodeError):
        json.loads(broken)
    assert loads_llm_json(broken) == expected


def test_commas_inside_strings_are_preserved():
    payload = {"note": "one, two, and three,"}
    assert loads_llm_json(json.dumps(payload)) == payload
    assert strip_trailing_commas(json.dumps(payload)) == json.dumps(payload)


def test_a_comma_before_a_brace_inside_a_string_is_not_dropped():
    payload = {"code": "d = {'a': 1,}"}
    assert loads_llm_json(json.dumps(payload)) == payload


def test_trailing_comma_and_raw_newline_together():
    """The repairs compose — a real response can carry both."""
    broken = '{"code": "import os\nprint(1)",}'
    assert loads_llm_json(broken) == {"code": "import os\nprint(1)"}


def test_valid_json_is_untouched_by_the_comma_pass():
    encoded = json.dumps({"a": [1, 2], "b": {"c": 3}})
    assert strip_trailing_commas(encoded) == encoded


# ---------------------------------------------------------------------------
# Unfenced responses
# ---------------------------------------------------------------------------
# A model told to answer in JSON frequently just answers in JSON. The planner's
# extractor only scanned for ```json fences and returned None otherwise, which
# the caller turns into "Failed to extract valid JSON from LLM response".
# Observed against stealth/ox-alpha: a valid 7558-character plan object,
# unfenced, discarded — twice, on lane1_t1 and lane2_t0.

def test_planner_accepts_a_bare_json_response():
    from sources.core.planner import Planner

    response = '{"goal": "reproduce the run", "steps": [{"description": "step one"}]}'
    got = Planner._extract_json_from_code_block(response)
    assert got["goal"] == "reproduce the run"
    assert got["steps"] == [{"description": "step one"}]


def test_planner_accepts_json_after_a_leading_sentence():
    from sources.core.planner import Planner

    response = 'Here is the plan:\n{"goal": "g", "steps": []}'
    assert Planner._extract_json_from_code_block(response) == {"goal": "g", "steps": []}


def test_planner_accepts_a_bare_response_with_a_trailing_comma():
    from sources.core.planner import Planner

    response = '{"goal": "g", "steps": [{"description": "s"},],}'
    assert Planner._extract_json_from_code_block(response) == {
        "goal": "g", "steps": [{"description": "s"}]
    }


def test_planner_still_prefers_the_fenced_block_when_present():
    from sources.core.planner import Planner

    response = 'Ignore this {"decoy": true}\n```json\n{"goal": "real"}\n```'
    assert Planner._extract_json_from_code_block(response) == {"goal": "real"}


def test_planner_returns_none_for_prose_with_no_json():
    from sources.core.planner import Planner

    assert Planner._extract_json_from_code_block("I could not produce a plan.") is None


def test_planner_returns_none_for_an_empty_response():
    from sources.core.planner import Planner

    assert Planner._extract_json_from_code_block("") is None
