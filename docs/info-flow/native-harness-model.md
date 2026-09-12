# Native CodeAgent through a harness completion model

`native_harness_config` explicitly enables Codex as the text model behind
Mimosa's existing smolagents `CodeAgent`. Mimosa owns the code parser, tools,
observations, memory and final-answer handling. The inner Codex CLI has its own
tools disabled by the existing completion bridge. This mode currently accepts
only `codex-cli/<model>` with ChatGPT subscription authentication. Claude CLI
text roles and explicit API endpoints retain their existing separate support.

Example settings (replace the absolute paths and digest with reviewed values):

```json
{
  "smolagent_model_id": "codex-cli/gpt-5.6-luna",
  "harness_auth_mode": "subscription",
  "orchestrator_choose_model": false,
  "agent_execution_timeout": 840,
  "native_harness_config": {
    "bridge_path": "/absolute/path/harness_completion.py",
    "bridge_sha256": "<64 lowercase SHA-256 characters>",
    "ledger_path": "/absolute/path/new-run/calls.jsonl",
    "max_calls": 4,
    "total_timeout_seconds": 900,
    "call_timeout_seconds": 240,
    "reasoning_effort": "high",
    "max_observed_tokens": null
  }
}
```

Each independent attempt needs a distinct ledger. Agents within a workflow use
the same ledger. A resumed ledger retains consumed calls and its original
deadline; changing the policy or encountering an unfinished reservation stops
dispatch. The ledger stores sanitized append-only reservation and completion
events with file locking and fsync. A request is reserved before transport.
An uncertain, failed or malformed response prevents further calls.

The transcript comes directly from `CodeAgent.generate()` in order. System,
user and assistant roles are preserved; smolagents tool-call/tool-response roles
map to assistant/user with their observation text. Only text blocks are
supported. Native function schemas, images and unsupported generation arguments
are rejected. Stop strings are applied locally to returned text; the complete
provider usage is still recorded. This does not impose a provider-side token
limit, save tokens, or support temperature/logprobs. An optional observed-token
ceiling can stop subsequent calls after measurement; call and time limits are
the enforceable pre-dispatch bounds.

The model passes protocol-v1 messages, explicit model and reasoning effort to
the pinned bridge through the shared `completion_backends` consumer. Requested,
configured and reported identity remain separate. `actual_model` stays null
when the CLI provides no attributable response identity. Requested effort is
not a measurement of effective effort. Unknown usage and cost remain unknown.
The native adapter requires the pinned bridge's sourced `model_identity` and
`chatgpt_subscription` usage kind; older optional-identity text envelopes remain
accepted only by the existing text-call path.

Both generated factories embed the canonical adapter, budget, consumer and
process-lifecycle sources in a content-named package. Generated scripts do not
import the development checkout. `psutil` is declared in the workflow runtime
requirements as well as Mimosa's dependencies. The precheck validates the bridge
digest, executable and ChatGPT login without sending a model request. Run this
check in the actual execution environment; a host check does not authenticate
a different host or cluster.

Native mode skips memory reuse and whole-agent retries. A failed tool action
can still be followed by another native CodeAgent step within the shared call
budget. A failed bridge call poisons model health, including when CodeAgent
tries to synthesize a final answer after failure. Per-agent timeout terminates
the generated workflow process; WorkflowRunner tracks and cleans up owned
descendants on timeout, cancellation, errors and exit. This is operational
process cleanup, not an OS security boundary against hostile code that evades
tracking. Local tool and workspace permissions remain the caller's responsibility.

Per-call `native_completion_<reservation-id>.json` receipts survive the removal
of `ChatMessage.raw` from agent memory. Pricing reads these receipts and skips
API price estimation for native CLI task memories, avoiding double charging.
The durable call ledger retains reservations even when a process is killed.
The numeric cost total remains the known subtotal; its unknown-cost flag must
travel with it.

Multi-agent task prompts contain the configured task, workspace path and prior
workflow-node answers. They no longer demand an unprovided shell tool. Neither
factory supplies hidden reference artifacts. This transport does not guarantee
benchmark blindness, scientific correctness, or gold acceptance; those require
separate exposure controls and artifact verification.
