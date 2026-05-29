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

## `memory_timelapse.py` — memory growth visualisation

For longer `--learn` runs, you may want a coarser view: how the agent's
memory grew over iterations. Run:

```bash
uv run memory_timelapse.py <uuid>
```

It renders an animated frame-by-frame view of the memory state, useful for
spotting where the agent's understanding shifted.

## Reading `state_result.json`

The verifier writes summary scores plus the workflow's final state to
`sources/workflows/<uuid>/state_result.json` under
`evaluation.verifier.*`. Useful fields:

| Field | What it tells you |
| ----- | ----------------- |
| `overall_score` | Capped (≤ `_HARD_FAIL_CAP`, currently `0.99`) when a hard claim is refuted. |
| `overall_score_uncapped` | Same score pre-cap — used as `reward_uncapped` for QD ranking. |
| `base_mean` | Mean over non-error per-claim scores. |
| `information_bonus` | Saturating thoroughness bonus, `α·(1 − exp(−n_hard_pass / β))`. |
| `hard_fail_capped` | `true` when a hard claim was refuted (cap fired). |
| `n_claims` / `n_pass` / `n_fail` / `n_error` / `n_unsure` / `n_scored` | Per-claim status counts. |
| `n_hard_pass` | Count of `hard` claims that passed (drives `information_bonus`). |
| `cheat_penalty` | Cheat-detector penalty. Currently always `0.0` (detector disabled, pending rewrite). |
| `abstracted_prompt_gradient` | Rubric-blind code-named summary — the only signal the mutator sees. |

The full per-claim detail (status, rationale, stderr tail, recomputed
values) lives in `sources/workflows/<uuid>/evaluation.txt` alongside the
JSON.

## Inspecting the genotype

The workflow code itself lives at:

```
sources/workflows/<uuid>/workflow_genotype_<uuid>.py
```

It's plain Python — readable end-to-end, no DSL. Look at it when:

- The abstracted prompt gradient is vague and you want to see what the
  agents actually do.
- You suspect a cheat the verifier missed.
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

Important for understanding the audit trail: the mutator sees **only** the
`abstracted_prompt_gradient`. It cannot see:

- Numerical scores.
- Per-claim verdicts.
- Which source (A–F) raised any given claim.

This is by design — it stops the loop from learning to game the rubric.
See [Evaluation pipeline](../concepts/evaluation-pipeline.md).

## See also

- [Workspace & audit trail](workspace.md) — full filesystem layout.
- [Iterative learning](learning.md) — what to inspect across generations.
- [Notifications](notifications.md) — real-time progress on your phone.
