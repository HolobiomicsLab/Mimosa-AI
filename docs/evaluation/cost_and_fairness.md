# Cost & fairness knobs for benchmark evaluation

Reference for the config-driven cost and leak-free-fairness controls used when
running Mimosa as a benchmark (e.g. ASB capsule reproduction). Every knob below
is **opt-in**: the defaults reproduce prior behaviour, so adding them changes
nothing until a config sets them.

## Cost levers

| Knob (config) | Default | Effect |
|---|---|---|
| `model_tiers` + role aliases | tiers defined, roles use concrete ids | Any `*_model` role may be set to a tier alias (`"heavy"`/`"light"`); resolved to the tier's model at load. Swap the whole fleet's cost profile in two lines. A concrete id or `None` passes through unchanged. |
| `judge_extraction_model` | `None` → reuse `judge_model` | Cheap tier for the verifier's *mechanical* judge calls (claim extraction, dedup, importance rating, file selection, verifier-script generation, package checks). Final claim verdicts and the mutation gradient stay on `judge_model`. The verifier fires 50–100+ judge calls/eval, most mechanical — routing those to a cheap tier is the dominant saving. |
| Prompt caching | on for Claude + OpenRouter routes | The (large, stable) system prompt is marked with an ephemeral `cache_control` breakpoint, so it bills at the cached (~10%) input rate on repeated calls within the 5-min TTL. No-op where the provider does not support caching. Cache read/write tokens are surfaced in the usage log. |
| `verifier_max_claims` | `None` → verifier default (100) | Grade only the top-N most important claims. Lower it for a faster/cheaper run; trades coverage of low-priority claims for speed. |
| `verifier_use_grounding` | `None` → verifier default (on) | Toggle the verifier's literature grounding. |
| `max_retries` (LLMProvider) | 6 | Hard cap on transient-error retries (the loop was previously unbounded). Backoff is unchanged; the call is now guaranteed to terminate. |

## Fairness / leak-free benchmark

A benchmark is only meaningful if the agent cannot see the answers. The answer
set is **the values the rubric grades** (computed results + study-specific
tuning parameters); public data *pointers* (accessions, repo URLs) are not
answers — the agent is given them and graded on using them.

| Knob (config) | Default | Effect |
|---|---|---|
| `perspicacite_agent_grounding_enabled` | `True` | Set `false` to disable agent-side Perspicacite grounding. The default (web search) can surface the source paper, leaking answers even with the browser MCP off. |
| `perspicacite_agent_kb_name` | `None` (web search) | KB the **agent** grounds from. Point at a leak-free brief KB for a controlled grounding arm. |
| `perspicacite_verifier_kb_name` | `None` (web search) | KB the **verifier** grounds from — may be full ground truth, since the verifier is allowed to know the answers. The agent and verifier ground from *different* KBs. |

Grounding conditions form an **ablation**: primary = no agent grounding;
ablation arm = leak-free brief KB; web search is excluded from fair runs;
paper/full-capsule KBs are diagnostics only (verifier-side or leak upper-bound).

### Leak linting

Treat "stripped" as objective: extract the rubric's graded result values and
fail the build if any appears in agent-visible material (prompt, KB docs,
workspace, RAG index, embedding cache). A token-based linter is the right tool
for prose/config; for a numeric data table use a **semantic** check instead
(no annotation/label/differential columns; the staged feature set is the full
set, not the graded subset) — coincidental digits in m/z or abundance values
are not leaks.

### Internet isolation

Disabling the browser MCP is **not** sufficient: the shell and Python tools can
also reach the network. A fair run is either *bounded-fair* (browser + explicit
web tools off, no paper/repo pointers in the prompt) or *airtight*
(pre-staged input data + OS-level network block). The shell MCP is **required**
for the agent to execute workflows, so it cannot simply be removed.

## Sandbox capability

The agent sandbox authorizes `pandas`/`numpy` and now `scipy`/`scikit-learn`
(plus common submodules), so data-analysis reproductions (RandomForest,
PCA/PCoA, distance metrics, clustering) can run in-process without shelling out.

## Fair-run recipe

1. Strip the task prompt of graded result values; lint to zero.
2. Pre-stage allowed input data into the workspace (it persists across
   `WorkspaceManager`'s per-run reset, which backs up the workspace at session
   start). Point the prompt at the local files.
3. Use a config that sets: tier aliases, `judge_extraction_model: "light"`,
   `perspicacite_agent_grounding_enabled: false`, a `verifier_max_claims` cap.
4. Start Toolomics with the browser MCP disabled (shell stays on).
5. Verify: agent bundle lints to zero, browser MCP down, data staged.
