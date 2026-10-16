# Transparency & replay

Mimosa-AI is built for **scientific use**, which means every decision must
be inspectable after the fact. This page lists the tools for stepping
through a finished run.

## `memory_explorer.py` — interactive replay

The headline tool: step through the full execution trace of any
generation.

```bash
uv run memory_explorer.py 20260115_113303_9bb63437
```

(Use the UUID printed in the run output, or any folder name under
`sources/workflows/`.)

You'll see, for each agent step:

- **Thoughts** — what the agent reasoned.
- **Tool calls** — every MCP call, with args.
- **Outputs** — what the tool returned.
- **State delta** — what changed in the shared LangGraph state.

This is the single best tool for understanding *why* a workflow did what
it did.

## `--memory_cli` — ask questions about a run

Sometimes you don't want to scroll through every step — you just want to ask
"*what classifier did the task_builder end up using?*" or "*did any agent hit
an error involving rdkit?*". The memory chat CLI gives you a small RAG-backed
Q&A interface over a single run's memory.

```bash
# Latest run by modification time
uv run main.py --memory_cli

# A specific run UUID
uv run main.py --memory_cli --memory_uuid 20260604_133122_06d6d38f
```

### How it works

1. **Chunking.** Every JSON file under `sources/memory/<uuid>/` is split into
   searchable chunks. A step-trace file (`task_builder.json`,
   `task_data_curator.json`, …) yields one chunk per smolagent step
   (`step_number`, `model_output`, `code_action`, `observations`,
   `action_output`, `error`). A single-call file (`workflow_creator.json`,
   …) yields one chunk for the whole LLM completion.
2. **Embedding.** All chunk summaries are embedded once at startup with
   `sentence-transformers/all-MiniLM-L6-v2` (downloaded automatically on
   first use, ~90 MB).
3. **Retrieval.** Each question goes through two LLM calls on `judge_model`
   (see [Configuration reference](../reference/configuration.md)):
   - **Query rewrite** — the question is compressed into a short
     keyword-focused search query.
   - **Answer** — top-K chunks (default `K=5`) are scored by cosine
     similarity, formatted as context, and fed back to the judge model
     which is instructed to answer **only from the retrieved chunks** and
     cite them by header (e.g. `task_builder.json · step 4`).

### UI

A curses interface in the same family as `memory_explorer.py`:

| Key | Action |
| --- | ------ |
| `a` | Ask a new question (curses pauses, you type at the shell prompt). |
| `↑` / `↓` | Scroll the answer pane. |
| `←` / `→` | Move between previously asked Q&As. |
| `c` / `C` | Scroll the code pane down / up. |
| `r` | Re-load the memory directory and re-embed (useful if a run finishes while the CLI is open). |
| `q` | Quit. |

The screen is split into:

- **Top:** the question, the judge model's answer, and a list of the
  retrieved chunks with their similarity scores.
- **Bottom:** the *executed code* of the top-ranked chunk (the
  `code_action` of that step, or the `python_interpreter` tool-call
  arguments). This is what you usually want to see when the answer
  mentions "the model_builder did X".

### When to reach for it

- You remember a detail from the run but not which agent or step produced
  it.
- A learning loop produced dozens of step-trace files and `memory_explorer.py`
  would be tedious to scroll through.
- You want to confirm whether the agents *actually* used a specific tool,
  library, or model name before drawing a conclusion from the score.

The CLI is read-only — it does not modify any memory file.

## Reading `state_result.json`

The verifier writes summary scores plus the workflow's final state to
`sources/workflows/<uuid>/state_result.json` under
`evaluation.verifier.*` (hybrid verifier fields; see
[Evaluation pipeline](../concepts/evaluation-pipeline.md#gradient-and-artifacts)
for the full list):

| Field | What it tells you |
| ----- | ----------------- |
| `overall_score` | The reward — mean win-rate against every previous generation of the task (first generation: mean claim score). Also surfaced as `reward`. |
| `win_rate` / `n_pairs` / `n_wins` / `n_losses` / `n_ties` | The pairwise record behind the reward. |
| `mean_claim_score` | Mean score over the surviving claims (absolute quality, independent of the pairwise reward). |
| `n_claims` / `n_surviving` / `n_dropped` / `n_scored` / `n_scorer_failures` / `n_replacements` | Claim-ladder bookkeeping: live claims, dead (zero-variance) claims dropped, scorer execution outcome, refinement replacements. |
| `claim_summary` | Per-claim entries: id, category, statement, 0..1 score, `surviving` flag, measured evidence. |
| `abstracted_textual_gradient` | The elimination-point gradient the mutator sees (also written to the `textual_gradient.txt` sidecar; the legacy `abstractec_` typo key is written for one release). |

The full per-claim detail (stage headers, scores, evidence, pairwise
transcript, dead-claim report) lives in
`sources/workflows/<uuid>/evaluation.txt` alongside the JSON.

## Inspecting the genotype

The workflow code itself lives at:

```
sources/workflows/<uuid>/workflow_genotype_<uuid>.py
```

It's plain Python — readable end-to-end, no DSL. Look at it when:

- The textual gradient is vague and you want to see what the
  agents actually do.
- You suspect the verifier missed something.
- You want to lift a successful workflow into another project as a
  starting point.

## Inspecting the lineage

```
sources/workflows/<uuid>/lineage_<uuid>.json
```

Tells you which parent(s) produced this generation and via which operator
(`seed | mutation | crossover`). Combine with the `evolution_prompt_*.md`
in the same folder to see the *exact* prompt that produced the code.

For a graphical view of the whole lineage tree, the best UUID gets an
`evolution_tree.png` rendered automatically.

## Per-iteration logs

Per-iteration console logs (with the `--debug` or `--verbose` flag) and
Pushover notifications give you a real-time view of progress. The Langfuse
dashboard (see [Telemetry](telemetry.md)) gives you span-level traces of
every LLM call.

## What's hidden from the mutator

Important for understanding the audit trail: the mutator sees **only**
the verifier's `abstracted_textual_gradient`. It cannot see:

- The policy-scorer scripts (they stay in the per-task registry).
- The pairwise machinery — win-rate computation, registry score
  vectors, dead-claim/refinement internals.
- The workspace snapshot and logs beyond what the gradient quotes as
  evidence.

This is by design: the gradient is a faithful transcript of measured
facts (the elimination point, both scores, the evidence), so the
mutator is steered by what was actually measured rather than by the
agents' narration — and the scoring machinery itself stays out of the
evolution prompt. See
[Evaluation pipeline](../concepts/evaluation-pipeline.md).

## See also

- [Workspace & audit trail](workspace.md) — full filesystem layout.
- [Iterative learning](learning.md) — what to inspect across generations.
- [Notifications](notifications.md) — real-time progress on your phone.
