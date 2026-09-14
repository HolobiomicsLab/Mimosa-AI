# Contained native oracle execution

`sources.core.container_runner.ContainerWorkflowRunner` is an opt-in executor
for a complete, already generated Python workflow. It leaves the default
`WorkflowRunner` and CLI behavior unchanged. A completed process is execution
evidence, not a scientifically validated reference or a maturity promotion.

## Information flow

```text
evaluator staging --> /data (read-only) --> generated workflow + CodeAgent
                                            |
                                  text messages + correlation ID
                                            |
host supervisor --> private worker --> HarnessCompletionModel --> pinned bridge
                    private ledger                         --> Codex completion
                                            |
                                  safe response envelope
                                            |
                                     CodeAgent --> /work
```

The container completion shim implements the existing `complete(request)` seam.
Only `id` and `messages` leave it. The host rejects other fields and constructs
model, effort, backend, authentication mode and timeout from its own settings.
`HarnessCompletionModel` validates messages and response identity and reserves
the authoritative private budget before dispatch. The container's native ledger
is advisory: deleting or modifying it cannot increase the host allowance.
Requested and provider-observed model identity remain separate; a missing actual
model stays null. Offline fixture completions are not real provider observations.

The real bridge and authentication are never staged or mounted. Each completion
runs in an owned host process, with a trusted source working directory, safe
module lookup and an environment allowlist: home/PATH/Codex home, locale, temp,
certificate and proxy settings. Unrelated API keys and project environment are
not inherited. The bridge itself is evaluator-trusted code whose digest is
verified by the existing native model. Use the tool-disabled completion bridge;
a digest of an arbitrary Python file does not certify its behavior or context.

## Staging and invocation

Prepare a fresh directory of approved public bytes with `workflow.py`. Copy
`sources/core/container_completion.py` to `completion_bridge.py` there. When
generating the existing native workflow, use container-side settings with:

```python
container_settings = {
    **host_settings,
    "bridge_path": "/data/completion_bridge.py",
    "bridge_sha256": sha256(public_bridge_bytes).hexdigest(),
    "ledger_path": "/work/advisory-ledger.jsonl",
}
```

Only the substituted container settings belong in generated Python. The private
host settings retain the actual bridge path/hash and private ledger. Stage normal
files, not symlinks, hardlinks, FIFOs or sockets; these are refused. The output
directory must be empty, outside the public directory and writable by UID1000.
Declare KB/reference/auth paths in `private_paths` so overlapping mounts are
refused in addition to the always-private bridge and ledger.

```python
from sources.core.container_runner import ContainerRuntime, ContainerWorkflowRunner

runtime = ContainerRuntime(
    image=pinned_local_image_id,  # full sha256:... local image ID
    public_dir=public_directory,
    work_dir=fresh_output_directory,
    model_id=approved_model_id,
    settings=host_settings,
    private_paths=(private_kb_directory, private_reference_directory),
    timeout_seconds=30,
)
result = await ContainerWorkflowRunner(runtime).execute()
```

The image must contain the workflow's runtime dependencies, including a supported
Python, smolagents and psutil for the packaged native adapter. Use image provenance
and a reviewed public-file inventory: the runner cannot recognize an answer in an
otherwise ordinary file or remove secrets baked into an image. The evaluator must
freeze those assets and avoid concurrent writers. It does not copy or certify an
arbitrary working directory. Input/software substitutions remain declared factors.

The trusted Docker executable and daemon must refer to the intended local engine
with the same host path namespace. Docker client context/configuration remains
operator-owned; an image ID is not engine attestation. The fixed invocation
overrides the entrypoint, disables network, drops capabilities, uses nonroot UID,
mounts only public inputs and fresh outputs, and sets CPU/memory/process limits.
The writable bind mount has no disk quota; size-control output ingestion separately.
Do not follow solver-created symlinks or ingest special files as result artifacts.

## CodeAgent expressions versus saved Python scripts

CodeAgent's local executor interprets its code blocks; installing CPython in the
image does not give those blocks complete Python language support. In the tested
smolagents 1.26.0 runtime, `yield` is unsupported and interpreted work has an
operation limit. A small arithmetic smoke test does not establish readiness for
data-processing scripts. Library limits, such as CSV's default field-size cap,
also remain the script author's responsibility.

For this contained executor, an evaluator may authorize `subprocess` in the
CodeAgent imports and document the existing route: write a script under `/work`,
then invoke the container's Python with `subprocess.run`, an argument list, an
explicit timeout and checked exit status. The script runs inside the same Docker
boundary with the same mounts and resource limits. This is not permission to
enable generated subprocesses in an uncontained host executor. Freeze the chosen
capability and instructions identically across comparison arms before dispatch.

Retain script bytes, invocation, exit status and diagnostic output. Keep model
observations compact rather than printing whole data rows. A synthetic generator
and wide-field CSV canary, paired with invalid input that must fail without a
success artifact, checks this execution route. Fixed-response canaries prove
runtime capability only; they do not prove that an oracle will submit a valid
scientific solution. Missing scripts remain incomplete submissions.

## Deadlines, failure and cleanup

Global and per-completion deadlines cover asynchronous reads, writes and the host
completion process. Every terminal path stops owned host processes and removes
the named container. Repeated cancellation during spawn waits until the created
process can be tracked and stopped. Host descendant discovery reuses the existing
process tracker and is intended for trusted bridge/CLI descendants; polling is
not an OS containment guarantee against a hostile host program that double-forks.

`status` is `completed`, `failed`, `timeout` or `cleanup_failed`.
`cleanup_verified` requires both owned host cleanup and verified Docker absence.
Cancellation propagates after cleanup; if cleanup cannot be verified, it raises a
cleanup error instead of hiding that uncertainty behind `CancelledError`.
`container_name` is available on the runner for inspection even when cancelled.
Cleanup has its own finite timeout and may extend beyond the execution deadline.
The current `timeout` status does not distinguish a per-completion deadline from
the outer workflow deadline. Compare frozen limits and available timing receipts;
do not infer that the whole workflow allowance elapsed from this status alone.
Likewise, later Docker absence cannot retroactively establish a failed host
cleanup check.

No automatic retry or host fallback exists. A runner instance is single-use.
An interrupted completion may leave a private ledger reservation without a terminal
receipt; the existing native budget then refuses another dispatch. Do not delete
or rewrite it to resume. Inspect the actual provider/process outcome first.

Stdout/stderr, frames and responses are bounded. Protocol failures terminate the
workflow. The container may fabricate completion requests or logs; those are not
trusted evaluator findings. Always verify result bytes and mandatory ASB criteria
with separate private instruments before assigning a computational maturity level.
