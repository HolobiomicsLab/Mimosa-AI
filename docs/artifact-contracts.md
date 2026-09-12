# Canonical artifact contracts

Mimosa can bind planner output to a trusted, versioned description of the
steps and JSON artifacts a run is allowed to use. This mode is opt-in. Plans
without a contract continue to run and are marked `legacy_unchecked`.

Set `planner_contract_path` to an absolute path for a trusted, regular,
non-symbolic-link JSON file:

```python
from config import Config
from sources.core.planner import Planner

config = Config()
config.workspace_dir = "/absolute/path/to/workspace"
config.planner_contract_path = "/absolute/path/to/contract.json"

tasks = await Planner(config).start_planner("Summarize the sensor readings")
```

`Planner.make_plan(...)` also honors the configured path when a caller only
needs admission and projection. `start_planner(...)` performs the runtime
artifact checks and records execution provenance.

## Contract document

The first schema version has five exact top-level fields:

```json
{
  "schema": "mimosa-artifact-contract/v1",
  "artifacts": [
    {
      "id": "counts",
      "path": "inputs/counts.json",
      "schema": {
        "type": "object",
        "properties": {
          "values": {
            "type": "array",
            "items": {"type": "integer"}
          }
        },
        "required": ["values"],
        "additionalProperties": false
      },
      "units": {"/values": "count"}
    },
    {
      "id": "rates",
      "path": "work/rates.json",
      "schema": {
        "type": "object",
        "properties": {
          "values": {
            "type": "array",
            "items": {"type": "number"}
          }
        },
        "required": ["values"],
        "additionalProperties": false
      },
      "units": {"/values": "Hz"}
    },
    {
      "id": "summary",
      "path": "outputs/summary.json",
      "schema": {
        "type": "object",
        "properties": {"mean_rate": {"type": "number"}},
        "required": ["mean_rate"],
        "additionalProperties": false
      },
      "units": {"/mean_rate": "Hz"}
    }
  ],
  "steps": [
    {
      "name": "estimate_rates",
      "task": "The counts were acquired over exactly one second. Read them and write the corresponding rates in hertz.",
      "complexity": "low",
      "inputs": ["counts"],
      "outputs": ["rates"]
    },
    {
      "name": "summarize_rates",
      "task": "Read the rates and write their arithmetic mean.",
      "complexity": "low",
      "inputs": ["rates"],
      "outputs": ["summary"]
    }
  ],
  "targets": ["summary"],
  "supplied": [
    {
      "artifact": "counts",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  ]
}
```

Artifact paths are unique relative POSIX paths below `workspace_dir`. An
artifact schema is a JSON Schema Draft 2020-12 object. Document-local `$ref`
values are supported, including percent-encoded fragment names. Remote,
dynamic, recursive, or unresolved references are rejected. Schema size and
depth, contract size, artifact size, strict JSON parsing, and finite JSON
numbers are bounded before validation. Unit entries are declared metadata;
Mimosa does not infer units or verify scientific equivalence.

Each artifact has at most one producer. `targets` determines the exact
producer closure. A supplied artifact cuts that closure and is validated by
its SHA-256 digest, so its omitted producer receives no execution credit. A
target cannot be supplied, and a required step cannot output any supplied
artifact.

## Planner and runtime behavior

The model may return only the contract digest and an ordered list of names:

```json
{
  "contract_digest": "the digest supplied in the prompt",
  "steps": [
    {"name": "estimate_rates"},
    {"name": "summarize_rates"}
  ]
}
```

Mimosa rejects missing, extra, duplicate, unknown, or incorrectly ordered
steps. It projects the canonical task, complexity, dependencies, paths,
schemas, and units from the trusted document. Mutating those bindings after
admission breaks the contract seal. Complexity is recorded metadata in this
slice; it does not automatically select a different model for each step.
Existing role configuration still chooses the models used by Mimosa.

Contract mode disables planner grounding, broad workspace file listing, plan
response cache reads, workflow result reuse, and parent-workflow selection.
Every step checks supplied hashes and completed producer receipts before it
runs. A produced input must still match the SHA-256 digest recorded by its
producer. An output receives credit only when its exact path contains fresh,
strict, schema-valid JSON written during the accepted attempt. Preexisting
outputs are refused, and a retry cannot bank unchanged output left by an
earlier failed attempt. Task receipts retain input and output hashes and mark
supplied artifact IDs separately.

These checks enforce the declared structure and provenance. They do not prove
that a contract is scientifically correct, validate arbitrary task prose, or
provide an operating-system sandbox. Independent scientific verification is
still required. The contract feature also does not make CLI text completion
backends into native tool-execution engines.
