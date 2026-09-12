"""Contract admission and artifact validation stay structural and fail closed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import sources.core.artifact_contracts as artifact_contracts
from sources.core.artifact_contracts import (
    ArtifactValidationError,
    ContractValidationError,
    load_artifact_contract,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _document(supplied_mid: bytes | None = None) -> dict:
    supplied = (
        [{"artifact": "calibrated", "sha256": _sha(supplied_mid)}]
        if supplied_mid is not None
        else [{"artifact": "measurements", "sha256": _sha(b'{"values":[2,4]}')}]
    )
    return {
        "schema": "mimosa-artifact-contract/v1",
        "artifacts": [
            {
                "id": "measurements",
                "path": "inputs/measurements.json",
                "schema": {
                    "type": "object",
                    "properties": {
                        "values": {"type": "array", "items": {"type": "integer"}}
                    },
                    "required": ["values"],
                    "additionalProperties": False,
                },
                "units": {"/values": "count"},
            },
            {
                "id": "calibrated",
                "path": "work/calibrated.json",
                "schema": {
                    "type": "object",
                    "properties": {"rate": {"type": "number"}},
                    "required": ["rate"],
                    "additionalProperties": False,
                },
                "units": {"/rate": "Hz"},
            },
            {
                "id": "summary",
                "path": "results/summary.json",
                "schema": {
                    "$defs": {"finiteNumber": {"type": "number"}},
                    "type": "object",
                    "properties": {"normalized": {"$ref": "#/$defs/finiteNumber"}},
                    "required": ["normalized"],
                    "additionalProperties": False,
                },
                "units": {"/normalized": "1"},
            },
        ],
        "steps": [
            {
                "name": "calibrate",
                "task": "Convert the supplied measurements into a calibrated rate.",
                "complexity": "low",
                "inputs": ["measurements"],
                "outputs": ["calibrated"],
            },
            {
                "name": "summarize",
                "task": "Normalize the calibrated rate and write the summary.",
                "complexity": "medium",
                "inputs": ["calibrated"],
                "outputs": ["summary"],
            },
        ],
        "targets": ["summary"],
        "supplied": supplied,
    }


def _load(tmp_path: Path, document: dict):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return load_artifact_contract(path)


def test_supplied_intermediate_cuts_producer_from_required_closure(tmp_path: Path):
    payload = b'{"rate":8.0}'
    contract = _load(tmp_path, _document(payload))

    assert contract.required_step_names == ("summarize",)
    plan = contract.project_plan(
        {"contract_digest": contract.digest, "steps": [{"name": "summarize"}]},
        "Summarize a small sensor series.",
    )

    assert [step.name for step in plan.steps] == ["summarize"]
    assert plan.steps[0].required_inputs == ["work/calibrated.json"]
    assert plan.steps[0].expected_outputs == ["results/summary.json"]
    assert plan.steps[0].input_artifact_ids == ["calibrated"]
    assert plan.steps[0].depends_on == []


@pytest.mark.parametrize(
    "response",
    [
        {"contract_digest": "0" * 64, "steps": [{"name": "summarize"}]},
        {"contract_digest": "DIGEST", "steps": []},
        {"contract_digest": "DIGEST", "steps": [{"name": "unknown"}]},
        {"contract_digest": "DIGEST", "steps": [{"name": "summarize"}, {"name": "summarize"}]},
        {
            "contract_digest": "DIGEST",
            "steps": [{"name": "summarize", "task": "override trusted task"}],
        },
    ],
)
def test_model_projection_cannot_change_or_omit_authoritative_contract(
    tmp_path: Path, response: dict
):
    contract = _load(tmp_path, _document(b'{"rate":8.0}'))
    candidate = json.loads(json.dumps(response).replace("DIGEST", contract.digest))
    with pytest.raises(ContractValidationError):
        contract.project_plan(candidate, "goal")


def test_post_admission_mutation_breaks_seal(tmp_path: Path):
    contract = _load(tmp_path, _document(b'{"rate":8.0}'))
    plan = contract.project_plan(
        {"contract_digest": contract.digest, "steps": [{"name": "summarize"}]},
        "goal",
    )
    plan.steps[0].task = "silently changed after admission"

    with pytest.raises(ContractValidationError, match="changed after admission"):
        contract.revalidate_plan(plan)


def test_loaded_contract_detects_mutated_schema_and_dependencies(tmp_path: Path):
    contract = _load(tmp_path, _document(None))
    contract.artifacts["calibrated"].schema["additionalProperties"] = True
    with pytest.raises(ContractValidationError, match="changed after loading"):
        contract.planner_projection()

    contract = _load(tmp_path, _document(None))
    contract.dependencies["summarize"] = ()
    with pytest.raises(ContractValidationError, match="changed after loading"):
        contract.planner_projection()


def test_schema_local_ref_validates_and_supplied_hash_is_preserved(tmp_path: Path):
    supplied = b'{"rate":8.0}'
    contract = _load(tmp_path, _document(supplied))
    path = tmp_path / "work" / "calibrated.json"
    path.parent.mkdir()
    path.write_bytes(supplied)

    hashes = contract.validate_inputs(contract.required_steps[0], tmp_path)

    assert hashes == {"calibrated": _sha(supplied)}


def test_mutated_supplied_artifact_fails_before_execution(tmp_path: Path):
    contract = _load(tmp_path, _document(b'{"rate":8.0}'))
    path = tmp_path / "work" / "calibrated.json"
    path.parent.mkdir()
    path.write_text('{"rate":9.0}', encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="sha256"):
        contract.validate_inputs(contract.required_steps[0], tmp_path)


def test_output_path_is_exact_and_schema_is_enforced(tmp_path: Path):
    contract = _load(tmp_path, _document(b'{"rate":8.0}'))
    wrong = tmp_path / "results" / "summary-copy.json"
    wrong.parent.mkdir()
    wrong.write_text('{"normalized":1.0}', encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="results/summary.json"):
        contract.validate_outputs(contract.required_steps[0], tmp_path)

    exact = tmp_path / "results" / "summary.json"
    exact.write_text('{"normalized":"not-a-number"}', encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="schema"):
        contract.validate_outputs(contract.required_steps[0], tmp_path)


@pytest.mark.parametrize("bad_path", ["../escape.json", "/absolute.json", "a\\b.json"])
def test_contract_rejects_noncanonical_workspace_paths(tmp_path: Path, bad_path: str):
    document = _document(b'{"rate":8.0}')
    document["artifacts"][0]["path"] = bad_path
    with pytest.raises(ContractValidationError, match="path"):
        _load(tmp_path, document)


def test_remote_and_recursive_schema_references_are_rejected(tmp_path: Path):
    for schema in (
        {"$ref": "https://example.invalid/schema.json"},
        {"$defs": {"loop": {"$ref": "#/$defs/loop"}}, "$ref": "#/$defs/loop"},
    ):
        document = _document(b'{"rate":8.0}')
        document["artifacts"][2]["schema"] = schema
        with pytest.raises(ContractValidationError, match="schema"):
            _load(tmp_path, document)


def test_schema_property_names_that_look_like_keywords_are_data(tmp_path: Path):
    document = _document(b'{"rate":8.0}')
    document["artifacts"][2]["schema"] = {
        "type": "object",
        "properties": {
            "$id": {"type": "string"},
            "$ref": {"const": "literal data, not a schema reference"},
        },
        "required": ["$id", "$ref"],
        "additionalProperties": False,
    }
    contract = _load(tmp_path, document)
    assert contract.artifacts["summary"].schema["properties"]["$id"] == {
        "type": "string"
    }


def test_percent_encoded_document_local_reference_is_supported(tmp_path: Path):
    document = _document(b'{"rate":8.0}')
    document["artifacts"][2]["schema"] = {
        "$defs": {"finite number": {"type": "number"}},
        "type": "object",
        "properties": {"normalized": {"$ref": "#/$defs/finite%20number"}},
        "required": ["normalized"],
        "additionalProperties": False,
    }
    contract = _load(tmp_path, document)
    assert contract.artifacts["summary"].schema["$defs"]["finite number"] == {
        "type": "number"
    }
    output = tmp_path / "results" / "summary.json"
    output.parent.mkdir()
    output.write_text('{"normalized":1.0}', encoding="utf-8")
    assert contract.validate_outputs(contract.required_steps[0], tmp_path)["summary"]


def test_supplied_targets_and_required_producer_overlaps_are_rejected(tmp_path: Path):
    supplied_target = _document(b'{"rate":8.0}')
    supplied_target["supplied"].append(
        {"artifact": "summary", "sha256": _sha(b'{"normalized":1.0}')}
    )
    with pytest.raises(ContractValidationError, match="Target artifact"):
        _load(tmp_path, supplied_target)

    overlap = _document(None)
    overlap["artifacts"].append(
        {
            "id": "audit",
            "path": "inputs/audit.json",
            "schema": {"type": "object"},
            "units": {},
        }
    )
    overlap["steps"][0]["outputs"].append("audit")
    overlap["supplied"].append(
        {"artifact": "audit", "sha256": _sha(b"{}")}
    )
    with pytest.raises(ContractValidationError, match="overwrite supplied"):
        _load(tmp_path, overlap)


def test_cycles_and_missing_producers_are_rejected(tmp_path: Path):
    cyclic = _document(None)
    cyclic["supplied"] = []
    cyclic["steps"][0]["inputs"] = ["summary"]
    with pytest.raises(ContractValidationError, match="cycle"):
        _load(tmp_path, cyclic)

    gap = _document(b'{"rate":8.0}')
    gap["supplied"] = []
    with pytest.raises(ContractValidationError, match="not supplied"):
        _load(tmp_path, gap)


def test_duplicate_json_keys_and_nonfinite_artifacts_are_rejected(tmp_path: Path):
    contract_path = tmp_path / "contract.json"
    contract_path.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
    with pytest.raises(ContractValidationError, match="Duplicate"):
        load_artifact_contract(contract_path)

    contract = _load(tmp_path, _document(b'{"rate":NaN}'))
    artifact = tmp_path / "work" / "calibrated.json"
    artifact.parent.mkdir()
    artifact.write_text('{"rate":NaN}', encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="finite"):
        contract.validate_inputs(contract.required_steps[0], tmp_path)

    duplicate = b'{"rate":8.0,"rate":9.0}'
    contract = _load(tmp_path, _document(duplicate))
    artifact.write_bytes(duplicate)
    with pytest.raises(ArtifactValidationError, match="Duplicate"):
        contract.validate_inputs(contract.required_steps[0], tmp_path)


def test_artifact_read_enforces_actual_byte_limit(tmp_path: Path, monkeypatch):
    data = b'{"rate":8.0}'
    contract = _load(tmp_path, _document(data))
    artifact = tmp_path / "work" / "calibrated.json"
    artifact.parent.mkdir()
    artifact.write_bytes(data)
    monkeypatch.setattr(artifact_contracts, "MAX_ARTIFACT_BYTES", len(data) - 1)

    with pytest.raises(ArtifactValidationError, match="size limit"):
        contract.validate_inputs(contract.required_steps[0], tmp_path)
