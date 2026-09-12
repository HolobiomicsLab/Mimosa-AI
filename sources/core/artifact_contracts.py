"""Trusted artifact contracts for opt-in planner execution.

The contract is authoritative.  A planner model may order the required step
names, but it cannot author executable text, paths, schemas, units, dependency
edges, or supplied-artifact provenance.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any
from urllib.parse import unquote

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .schema import Plan, PlanStep, TaskComplexity


CONTRACT_SCHEMA = "mimosa-artifact-contract/v1"
CONTRACT_STATUS = "validated"
MAX_CONTRACT_BYTES = 1_000_000
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_SCHEMA_DEPTH = 64
MAX_SCHEMA_NODES = 4096

_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CONTRACT_KEYS = {"schema", "artifacts", "steps", "targets", "supplied"}
_ARTIFACT_KEYS = {"id", "path", "schema", "units"}
_STEP_KEYS = {"name", "task", "complexity", "inputs", "outputs"}
_SUPPLIED_KEYS = {"artifact", "sha256"}
_MODEL_KEYS = {"contract_digest", "steps"}


class ContractValidationError(ValueError):
    """The trusted contract or model projection is structurally invalid."""


class ArtifactValidationError(ValueError):
    """A workspace artifact does not match its canonical binding."""


@dataclass(frozen=True)
class ArtifactSpec:
    """Canonical path, JSON Schema, and declared unit metadata for one artifact."""

    id: str
    path: str
    schema: dict[str, Any]
    units: dict[str, str]


@dataclass(frozen=True)
class ContractStep:
    """Canonical executable instruction and artifact bindings for one step."""

    name: str
    task: str
    complexity: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return parsed


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _strict_json(data: bytes, *, context: str, contract: bool) -> Any:
    try:
        text = data.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_unique_pairs,
            parse_float=_finite_float,
            parse_constant=_reject_constant,
        )
    except ContractValidationError as exc:
        if contract:
            raise
        raise ArtifactValidationError(str(exc)) from exc
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        error = ContractValidationError if contract else ArtifactValidationError
        message = str(exc)
        if "Nonfinite" in message:
            message = f"finite JSON required: {message}"
        raise error(f"Invalid {context}: {message}") from exc


def _exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ContractValidationError(
            f"{context} fields differ (missing={missing}, extra={extra})"
        )


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ContractValidationError(f"Invalid {context}: {value!r}")
    return value


def _string_list(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ContractValidationError(f"{context} must be a list")
    result = tuple(_identifier(item, context) for item in value)
    if len(result) != len(set(result)):
        raise ContractValidationError(f"{context} contains duplicates")
    return result


def _workspace_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractValidationError(f"Invalid artifact path: {value!r}")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != value
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ContractValidationError(f"Invalid artifact path: {value!r}")
    return value


def _count_schema_nodes(
    value: Any, depth: int = 0, counter: list[int] | None = None
) -> None:
    """Bound the complete JSON value without interpreting data as schemas."""
    counter = counter if counter is not None else [0]
    counter[0] += 1
    if counter[0] > MAX_SCHEMA_NODES or depth > MAX_SCHEMA_DEPTH:
        raise ContractValidationError("Artifact schema exceeds finite size/depth limits")
    if isinstance(value, dict):
        for child in value.values():
            _count_schema_nodes(child, depth + 1, counter)
    elif isinstance(value, list):
        for child in value:
            _count_schema_nodes(child, depth + 1, counter)


_SINGLE_SCHEMA_KEYWORDS = {
    "additionalItems",
    "additionalProperties",
    "contains",
    "contentSchema",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
}
_SCHEMA_ARRAY_KEYWORDS = {"allOf", "anyOf", "oneOf", "prefixItems"}
_SCHEMA_MAP_KEYWORDS = {
    "$defs",
    "definitions",
    "dependentSchemas",
    "patternProperties",
    "properties",
}


def _schema_locations(value: Any):
    """Yield only actual schema objects, never property names or example data."""
    if isinstance(value, bool):
        return
    if not isinstance(value, dict):
        return
    yield value
    for keyword in _SINGLE_SCHEMA_KEYWORDS:
        child = value.get(keyword)
        if isinstance(child, (dict, bool)):
            yield from _schema_locations(child)
    for keyword in _SCHEMA_ARRAY_KEYWORDS:
        children = value.get(keyword)
        if isinstance(children, list):
            for child in children:
                yield from _schema_locations(child)
    for keyword in _SCHEMA_MAP_KEYWORDS:
        children = value.get(keyword)
        if isinstance(children, dict):
            for child in children.values():
                yield from _schema_locations(child)


def _resolve_pointer(schema: Any, ref: str) -> Any:
    if not ref.startswith("#/"):
        raise ContractValidationError("Artifact schema $ref must be document-local")
    try:
        pointer = unquote(ref[1:], errors="strict")
    except UnicodeDecodeError as exc:
        raise ContractValidationError(
            f"Artifact schema has invalid UTF-8 in local $ref: {ref}"
        ) from exc
    current = schema
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        try:
            if isinstance(current, dict):
                current = current[part]
            elif isinstance(current, list):
                current = current[int(part)]
            else:
                raise KeyError(part)
        except (KeyError, IndexError, ValueError) as exc:
            raise ContractValidationError(
                f"Artifact schema has unresolved local $ref: {ref}"
            ) from exc
    return current


def _nested_refs(value: Any):
    for location in _schema_locations(value):
        ref = location.get("$ref")
        if ref is not None:
            yield ref


def _validate_local_refs(schema: dict[str, Any]) -> None:
    visiting: set[str] = set()
    completed: set[str] = set()

    def visit(ref: Any) -> None:
        if not isinstance(ref, str) or not ref.startswith("#/"):
            raise ContractValidationError("Artifact schema $ref must be document-local")
        if ref in visiting:
            raise ContractValidationError("Artifact schema contains a recursive local $ref")
        if ref in completed:
            return
        visiting.add(ref)
        target = _resolve_pointer(schema, ref)
        for nested in _nested_refs(target):
            visit(nested)
        visiting.remove(ref)
        completed.add(ref)

    for ref in _nested_refs(schema):
        visit(ref)


def _validate_schema(value: Any, artifact_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractValidationError(f"Artifact {artifact_id!r} schema must be an object")
    schema = deepcopy(value)
    try:
        _count_schema_nodes(schema)
        locations = list(_schema_locations(schema))
    except RecursionError as exc:
        raise ContractValidationError("Artifact schema exceeds finite depth limits") from exc
    forbidden = {"$id", "$anchor", "$dynamicAnchor", "$dynamicRef", "$recursiveRef"}
    for location in locations:
        present = forbidden & location.keys()
        if present:
            keyword = sorted(present)[0]
            raise ContractValidationError(
                f"Artifact {artifact_id!r} schema uses unsupported keyword {keyword}"
            )
    _validate_local_refs(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ContractValidationError(
            f"Artifact {artifact_id!r} schema is invalid: {exc.message}"
        ) from exc
    return schema


def _find_cycle(dependencies: dict[str, set[str]]) -> None:
    visited: set[str] = set()
    active: set[str] = set()

    def visit(name: str) -> None:
        if name in active:
            raise ContractValidationError(f"Contract step graph contains a cycle at {name!r}")
        if name in visited:
            return
        active.add(name)
        for dependency in dependencies[name]:
            visit(dependency)
        active.remove(name)
        visited.add(name)

    for name in dependencies:
        visit(name)


def _parse_units(value: Any, artifact_id: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ContractValidationError(f"Artifact {artifact_id!r} units must be an object")
    for pointer, unit in value.items():
        if (
            not isinstance(pointer, str)
            or not pointer.startswith("/")
            or not isinstance(unit, str)
            or not unit.strip()
        ):
            raise ContractValidationError(
                f"Artifact {artifact_id!r} has invalid unit metadata"
            )
    return dict(value)


def _parse_artifacts(value: Any) -> dict[str, ArtifactSpec]:
    if not isinstance(value, list) or not value:
        raise ContractValidationError("Contract artifacts must be a nonempty list")
    artifacts: dict[str, ArtifactSpec] = {}
    paths: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ContractValidationError(f"Artifact {index} must be an object")
        _exact_keys(raw, _ARTIFACT_KEYS, f"Artifact {index}")
        artifact_id = _identifier(raw["id"], f"artifact {index} id")
        if artifact_id in artifacts:
            raise ContractValidationError(f"Duplicate artifact id: {artifact_id}")
        path = _workspace_path(raw["path"])
        if path in paths:
            raise ContractValidationError(f"Duplicate artifact path: {path}")
        artifacts[artifact_id] = ArtifactSpec(
            id=artifact_id,
            path=path,
            schema=_validate_schema(raw["schema"], artifact_id),
            units=_parse_units(raw["units"], artifact_id),
        )
        paths.add(path)
    return artifacts


def _parse_steps(
    value: Any, artifacts: dict[str, ArtifactSpec]
) -> tuple[dict[str, ContractStep], dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ContractValidationError("Contract steps must be a nonempty list")
    steps: dict[str, ContractStep] = {}
    producers: dict[str, str] = {}
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ContractValidationError(f"Step {index} must be an object")
        _exact_keys(raw, _STEP_KEYS, f"Step {index}")
        name = _identifier(raw["name"], f"step {index} name")
        if name in steps:
            raise ContractValidationError(f"Duplicate step name: {name}")
        task = raw["task"]
        if not isinstance(task, str) or not task.strip():
            raise ContractValidationError(f"Step {name!r} task must be nonempty")
        complexity = raw["complexity"]
        if complexity not in {item.value for item in TaskComplexity}:
            raise ContractValidationError(f"Step {name!r} has invalid complexity")
        inputs = _string_list(raw["inputs"], f"step {name!r} inputs")
        outputs = _string_list(raw["outputs"], f"step {name!r} outputs")
        if not outputs:
            raise ContractValidationError(f"Step {name!r} must produce an artifact")
        unknown = (set(inputs) | set(outputs)) - artifacts.keys()
        if unknown:
            raise ContractValidationError(
                f"Step {name!r} references unknown artifacts: {sorted(unknown)}"
            )
        for artifact_id in outputs:
            if artifact_id in producers:
                raise ContractValidationError(
                    f"Artifact {artifact_id!r} has duplicate producers: "
                    f"{producers[artifact_id]}, {name}"
                )
            producers[artifact_id] = name
        steps[name] = ContractStep(name, task, complexity, inputs, outputs)
    return steps, producers


def _parse_supplied(
    value: Any, artifacts: dict[str, ArtifactSpec]
) -> dict[str, str]:
    if not isinstance(value, list):
        raise ContractValidationError("Contract supplied must be a list")
    supplied: dict[str, str] = {}
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ContractValidationError(f"Supplied artifact {index} must be an object")
        _exact_keys(raw, _SUPPLIED_KEYS, f"Supplied artifact {index}")
        artifact_id = _identifier(raw["artifact"], "supplied artifact id")
        if artifact_id not in artifacts:
            raise ContractValidationError(f"Unknown supplied artifact: {artifact_id}")
        if artifact_id in supplied:
            raise ContractValidationError(f"Duplicate supplied artifact: {artifact_id}")
        digest = raw["sha256"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ContractValidationError(
                f"Supplied artifact {artifact_id!r} has invalid sha256"
            )
        supplied[artifact_id] = digest
    return supplied


def _derive_execution_graph(
    artifacts: dict[str, ArtifactSpec],
    steps: dict[str, ContractStep],
    producers: dict[str, str],
    supplied: dict[str, str],
    raw_targets: Any,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, tuple[str, ...]]]:
    targets = _string_list(raw_targets, "contract targets")
    if not targets:
        raise ContractValidationError("Contract targets must be nonempty")
    unknown_targets = set(targets) - artifacts.keys()
    if unknown_targets:
        raise ContractValidationError(f"Unknown target artifacts: {sorted(unknown_targets)}")
    full_dependencies = {
        step.name: {
            producers[item] for item in step.inputs if item in producers
        }
        for step in steps.values()
    }
    _find_cycle(full_dependencies)
    required = _required_closure(targets, steps, producers, supplied)
    order = tuple(steps)
    order_index = {name: index for index, name in enumerate(order)}
    dependencies: dict[str, tuple[str, ...]] = {}
    for name in required:
        overlap = set(steps[name].outputs) & supplied.keys()
        if overlap:
            raise ContractValidationError(
                f"Required step {name!r} would overwrite supplied artifacts: {sorted(overlap)}"
            )
        deps = {
            producers[item]
            for item in steps[name].inputs
            if item not in supplied and item in producers
        }
        if not deps <= required:
            raise ContractValidationError(f"Required closure gap at step {name!r}")
        dependencies[name] = tuple(sorted(deps, key=order_index.__getitem__))
    required_names = tuple(name for name in order if name in required)
    return targets, required_names, dependencies


def _required_closure(
    targets: tuple[str, ...],
    steps: dict[str, ContractStep],
    producers: dict[str, str],
    supplied: dict[str, str],
) -> set[str]:
    required: set[str] = set()
    visiting: set[str] = set()

    def require_artifact(artifact_id: str) -> None:
        if artifact_id in supplied:
            return
        if artifact_id in visiting:
            raise ContractValidationError(
                f"Contract artifact graph contains a cycle at {artifact_id!r}"
            )
        producer = producers.get(artifact_id)
        if producer is None:
            raise ContractValidationError(
                f"Required artifact {artifact_id!r} is not supplied and has no producer"
            )
        if producer in required:
            return
        visiting.add(artifact_id)
        for input_id in steps[producer].inputs:
            require_artifact(input_id)
        visiting.remove(artifact_id)
        required.add(producer)

    for target in targets:
        if target in supplied:
            raise ContractValidationError(
                f"Target artifact {target!r} cannot be supplied; targets require execution credit"
            )
        require_artifact(target)
    if not required:
        raise ContractValidationError("Contract has no executable required steps")
    return required


class ArtifactContract:
    """Parsed, sealed v1 contract with derived execution closure."""

    def __init__(self, document: dict[str, Any]):
        _exact_keys(document, _CONTRACT_KEYS, "Contract")
        if document["schema"] != CONTRACT_SCHEMA:
            raise ContractValidationError(
                f"Unsupported contract schema: {document['schema']!r}"
            )
        artifacts = _parse_artifacts(document["artifacts"])
        steps, producers = _parse_steps(document["steps"], artifacts)
        supplied = _parse_supplied(document["supplied"], artifacts)
        targets, required_names, dependencies = _derive_execution_graph(
            artifacts, steps, producers, supplied, document["targets"]
        )
        try:
            canonical = json.dumps(
                document, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ContractValidationError("Contract must contain finite JSON data") from exc
        self.digest = hashlib.sha256(canonical).hexdigest()
        self.artifacts = artifacts
        self.steps = steps
        self.targets = targets
        self.supplied = supplied
        self.required_step_names = required_names
        self.dependencies = dependencies
        self.producer_by_artifact = producers
        self._runtime_seal = self._state_digest()

    def _state_digest(self) -> str:
        state = {
            "digest": self.digest,
            "artifacts": [
                {
                    "id": item.id,
                    "path": item.path,
                    "schema": item.schema,
                    "units": item.units,
                }
                for item in self.artifacts.values()
            ],
            "steps": [
                {
                    "name": item.name,
                    "task": item.task,
                    "complexity": item.complexity,
                    "inputs": item.inputs,
                    "outputs": item.outputs,
                }
                for item in self.steps.values()
            ],
            "targets": self.targets,
            "supplied": self.supplied,
            "required_step_names": self.required_step_names,
            "dependencies": self.dependencies,
            "producer_by_artifact": self.producer_by_artifact,
        }
        try:
            encoded = json.dumps(
                state, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ContractValidationError("Artifact contract changed after loading") from exc
        return hashlib.sha256(encoded).hexdigest()

    def _check_seal(self) -> None:
        if self._state_digest() != self._runtime_seal:
            raise ContractValidationError("Artifact contract changed after loading")

    @property
    def required_steps(self) -> tuple[ContractStep, ...]:
        self._check_seal()
        return tuple(self.steps[name] for name in self.required_step_names)

    def planner_projection(self) -> dict[str, Any]:
        """Return trusted ordering context, without contents, hashes, or output data."""
        self._check_seal()
        return {
            "contract_digest": self.digest,
            "targets": list(self.targets),
            "supplied_artifacts": sorted(self.supplied),
            "required_steps": [
                {
                    "name": step.name,
                    "task": step.task,
                    "complexity": step.complexity,
                    "inputs": list(step.inputs),
                    "outputs": list(step.outputs),
                    "depends_on": list(self.dependencies[step.name]),
                }
                for step in self.required_steps
            ],
        }

    def execution_projection(self, step: ContractStep | PlanStep) -> dict[str, Any]:
        """Return the exact artifact interface supplied to one workflow run."""
        spec = self._canonical_step(step)

        def artifact(artifact_id: str) -> dict[str, Any]:
            item = self.artifacts[artifact_id]
            return {
                "id": item.id,
                "path": item.path,
                "schema": deepcopy(item.schema),
                "units": dict(item.units),
                "supplied": artifact_id in self.supplied,
            }

        return {
            "contract_digest": self.digest,
            "step": spec.name,
            "inputs": [artifact(item) for item in spec.inputs],
            "outputs": [artifact(item) for item in spec.outputs],
        }

    def project_plan(self, response: Any, goal: str) -> Plan:
        """Validate the model's names/order and hydrate canonical PlanSteps."""
        self._check_seal()
        if not isinstance(response, dict):
            raise ContractValidationError("Contract plan response must be an object")
        _exact_keys(response, _MODEL_KEYS, "Contract plan response")
        if response["contract_digest"] != self.digest:
            raise ContractValidationError("Contract plan response changed contract digest")
        raw_steps = response["steps"]
        if not isinstance(raw_steps, list):
            raise ContractValidationError("Contract plan response steps must be a list")
        names: list[str] = []
        for index, raw in enumerate(raw_steps):
            if not isinstance(raw, dict):
                raise ContractValidationError(f"Contract plan step {index} must be an object")
            _exact_keys(raw, {"name"}, f"Contract plan step {index}")
            names.append(_identifier(raw["name"], f"contract plan step {index} name"))
        required = set(self.required_step_names)
        if len(names) != len(set(names)):
            raise ContractValidationError("Contract plan response contains duplicate steps")
        if set(names) != required:
            raise ContractValidationError(
                "Contract plan response must contain the exact required step closure"
            )
        seen: set[str] = set()
        for name in names:
            missing = set(self.dependencies[name]) - seen
            if missing:
                raise ContractValidationError(
                    f"Contract plan step {name!r} is not topologically ordered; missing {sorted(missing)}"
                )
            seen.add(name)

        steps = []
        for name in names:
            spec = self.steps[name]
            steps.append(
                PlanStep(
                    name=name,
                    goal_context=goal,
                    task=spec.task,
                    cost=0.0,
                    score=0.0,
                    depends_on=list(self.dependencies[name]),
                    required_inputs=[self.artifacts[item].path for item in spec.inputs],
                    expected_outputs=[self.artifacts[item].path for item in spec.outputs],
                    complexity=spec.complexity,
                    contract_status=CONTRACT_STATUS,
                    contract_digest=self.digest,
                    input_artifact_ids=list(spec.inputs),
                    output_artifact_ids=list(spec.outputs),
                )
            )
        plan = Plan(
            goal=goal,
            steps=steps,
            contract_status=CONTRACT_STATUS,
            contract_digest=self.digest,
        )
        self.revalidate_plan(plan)
        return plan

    def revalidate_plan(self, plan: Plan) -> None:
        """Reject any mutation of an admitted plan's authoritative fields."""
        self._check_seal()
        if (
            not isinstance(plan, Plan)
            or plan.contract_status != CONTRACT_STATUS
            or plan.contract_digest != self.digest
        ):
            raise ContractValidationError("Validated plan has an invalid contract seal")
        names = [step.name for step in plan.steps]
        if set(names) != set(self.required_step_names) or len(names) != len(set(names)):
            raise ContractValidationError("Validated plan changed after admission: step closure")
        seen: set[str] = set()
        for step in plan.steps:
            spec = self.steps.get(step.name)
            if spec is None:
                raise ContractValidationError("Validated plan changed after admission: unknown step")
            expected = {
                "task": spec.task,
                "complexity": spec.complexity,
                "depends_on": list(self.dependencies[step.name]),
                "required_inputs": [self.artifacts[item].path for item in spec.inputs],
                "expected_outputs": [self.artifacts[item].path for item in spec.outputs],
                "contract_status": CONTRACT_STATUS,
                "contract_digest": self.digest,
                "input_artifact_ids": list(spec.inputs),
                "output_artifact_ids": list(spec.outputs),
            }
            for field, value in expected.items():
                if getattr(step, field, None) != value:
                    raise ContractValidationError(
                        f"Validated plan changed after admission: {step.name}.{field}"
                    )
            if step.goal_context != plan.goal:
                raise ContractValidationError(
                    f"Validated plan changed after admission: {step.name}.goal_context"
                )
            missing = set(self.dependencies[step.name]) - seen
            if missing:
                raise ContractValidationError(
                    f"Validated plan changed after admission: order for {step.name}"
                )
            seen.add(step.name)

    def _canonical_step(self, step: ContractStep | PlanStep) -> ContractStep:
        self._check_seal()
        name = getattr(step, "name", None)
        spec = self.steps.get(name)
        if spec is None or name not in self.required_step_names:
            raise ContractValidationError(f"Step {name!r} is outside required contract closure")
        if isinstance(step, PlanStep):
            expected = self.projected_step_fields(spec)
            for field, value in expected.items():
                if getattr(step, field, None) != value:
                    raise ContractValidationError(
                        f"Validated plan changed after admission: {name}.{field}"
                    )
        return spec

    def projected_step_fields(self, spec: ContractStep) -> dict[str, Any]:
        self._check_seal()
        return {
            "task": spec.task,
            "complexity": spec.complexity,
            "depends_on": list(self.dependencies[spec.name]),
            "required_inputs": [self.artifacts[item].path for item in spec.inputs],
            "expected_outputs": [self.artifacts[item].path for item in spec.outputs],
            "contract_status": CONTRACT_STATUS,
            "contract_digest": self.digest,
            "input_artifact_ids": list(spec.inputs),
            "output_artifact_ids": list(spec.outputs),
        }

    def validate_inputs(
        self, step: ContractStep | PlanStep, workspace: str | Path
    ) -> dict[str, str]:
        spec = self._canonical_step(step)
        return {
            artifact_id: self._validate_artifact(
                artifact_id,
                workspace,
                expected_sha256=self.supplied.get(artifact_id),
            )
            for artifact_id in spec.inputs
        }

    def validate_outputs(
        self, step: ContractStep | PlanStep, workspace: str | Path
    ) -> dict[str, str]:
        spec = self._canonical_step(step)
        return {
            artifact_id: self._validate_artifact(artifact_id, workspace)
            for artifact_id in spec.outputs
        }

    def snapshot_outputs(
        self, step: ContractStep | PlanStep, workspace: str | Path
    ) -> dict[str, str | None]:
        """Hash exact output paths without treating existing content as produced."""
        spec = self._canonical_step(step)
        result: dict[str, str | None] = {}
        for artifact_id in spec.outputs:
            data = self._read_artifact(self.artifacts[artifact_id], workspace, allow_missing=True)
            result[artifact_id] = hashlib.sha256(data).hexdigest() if data is not None else None
        return result

    def remove_outputs_for_retry(
        self, step: ContractStep | PlanStep, workspace: str | Path
    ) -> None:
        """Remove only this canonical step's exact output files before a retry."""
        spec = self._canonical_step(step)
        root = Path(workspace)
        if not root.is_absolute():
            raise ArtifactValidationError("Workspace path must be absolute")
        try:
            resolved_root = root.resolve(strict=True)
        except OSError as exc:
            raise ArtifactValidationError(f"Workspace is unavailable: {root}") from exc
        for artifact_id in spec.outputs:
            relative = PurePosixPath(self.artifacts[artifact_id].path)
            candidate = root.joinpath(*relative.parts)
            current = root
            for part in relative.parts[:-1]:
                current = current / part
                if current.is_symlink():
                    raise ArtifactValidationError(
                        f"Artifact {relative} uses a symbolic-link parent"
                    )
            if candidate.is_symlink():
                candidate.unlink()
                continue
            if not candidate.exists():
                continue
            if not candidate.is_file() or not candidate.resolve().is_relative_to(resolved_root):
                raise ArtifactValidationError(
                    f"Artifact path is not a confined file: {relative}"
                )
            candidate.unlink()

    def _validate_artifact(
        self,
        artifact_id: str,
        workspace: str | Path,
        expected_sha256: str | None = None,
    ) -> str:
        self._check_seal()
        spec = self.artifacts[artifact_id]
        data = self._read_artifact(spec, workspace)
        assert data is not None
        digest = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ArtifactValidationError(
                f"Artifact {spec.path} sha256 differs from supplied provenance"
            )
        instance = _strict_json(data, context=f"artifact {spec.path}", contract=False)
        error = next(Draft202012Validator(spec.schema).iter_errors(instance), None)
        if error is not None:
            raise ArtifactValidationError(
                f"Artifact {spec.path} schema validation failed: {error.message}"
            )
        return digest

    @staticmethod
    def _read_artifact(
        spec: ArtifactSpec,
        workspace: str | Path,
        *,
        allow_missing: bool = False,
    ) -> bytes | None:
        root = Path(workspace)
        if not root.is_absolute():
            raise ArtifactValidationError("Workspace path must be absolute")
        try:
            resolved_root = root.resolve(strict=True)
        except OSError as exc:
            raise ArtifactValidationError(f"Workspace is unavailable: {root}") from exc
        candidate = root.joinpath(*PurePosixPath(spec.path).parts)
        current = root
        for part in PurePosixPath(spec.path).parts:
            current = current / part
            if current.is_symlink():
                raise ArtifactValidationError(
                    f"Artifact {spec.path} uses a symbolic-link path"
                )
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            if allow_missing and not candidate.exists():
                return None
            raise ArtifactValidationError(f"Artifact path is missing: {spec.path}") from exc
        if not resolved.is_relative_to(resolved_root) or not candidate.is_file():
            raise ArtifactValidationError(f"Artifact path is not a confined file: {spec.path}")
        try:
            with candidate.open("rb") as handle:
                data = handle.read(MAX_ARTIFACT_BYTES + 1)
        except OSError as exc:
            raise ArtifactValidationError(f"Artifact is unreadable: {spec.path}") from exc
        if len(data) > MAX_ARTIFACT_BYTES:
            raise ArtifactValidationError(f"Artifact exceeds size limit: {spec.path}")
        return data


def load_artifact_contract(path: str | Path) -> ArtifactContract:
    """Load one trusted absolute regular file and return its sealed contract."""
    if not isinstance(path, (str, Path)):
        raise ContractValidationError("planner contract path must be a path string")
    contract_path = Path(path)
    if not contract_path.is_absolute():
        raise ContractValidationError("planner contract path must be absolute")
    if contract_path.is_symlink() or not contract_path.is_file():
        raise ContractValidationError("planner contract path must be a regular non-symlink file")
    try:
        with contract_path.open("rb") as handle:
            data = handle.read(MAX_CONTRACT_BYTES + 1)
        if len(data) > MAX_CONTRACT_BYTES:
            raise ContractValidationError("Planner contract exceeds size limit")
        document = _strict_json(
            data, context="planner contract JSON", contract=True
        )
    except OSError as exc:
        raise ContractValidationError(f"Could not read planner contract: {contract_path}") from exc
    if not isinstance(document, dict):
        raise ContractValidationError("Planner contract must be a JSON object")
    return ArtifactContract(document)
