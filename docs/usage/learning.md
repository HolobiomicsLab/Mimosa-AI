# Iterative learning

Run any task with `--learn` and Mimosa retries it across generations,
keeping only improvements. This page covers the practical side; for the
mechanics, see [Evolution engine](../concepts/evolution-engine.md).

## Why learn first

Single-shot multi-agent workflows are noisy: even strong LLMs produce
workflows that need refinement. Iterative learning lets Mimosa explore the
workflow space, archive what worked, and recombine.

In practice:

- **Cold tasks** (no related runs on disk) benefit most — the first
  generation is often weak; iteration finds the strong region.
- **Warm tasks** (similar runs already on disk) start from a stronger
  baseline thanks to the disk-similarity scan.
- **Familiar tasks** (a previous run hit the threshold) can reuse a saved
  workflow without iteration — see [Workspace & audit trail](workspace.md).

## How to enable it

Add `--learn` to a task or goal mode invocation:

```bash
uv run main.py --task "Train a multitask model on the Clintox dataset…" \
              --learn \
              --config my_config.json
```

Or, for batch evaluations:

```bash
uv run main.py --science_agent_bench --csv_runs_limit 20 --learn
```

Without `--learn`, the engine still runs the verifier but never iterates
— you get a one-shot result.

## Termination

Mimosa stops the moment either condition is met:

- `overall_score >= learned_score_threshold` (config field, default `0.9`).
- `iteration ≥ max_learning_evolve_iterations` (default `20`).

The "best" workflow at termination — the one with the highest
`reward_uncapped` in the archive — has its workspace snapshot restored as
the run's final state.

!!! tip "Tuning the threshold"
    - Setting the threshold too high (e.g. `1.0`) means the loop never
      stops early; you'll always hit `max_learning_evolve_iterations`.
    - Setting it too low (e.g. `0.7`) means the loop stops after one lucky
      generation that *happens* to fool the judge.
    - The default `0.9` is a reasonable balance — adjust based on observed
      score distributions for your task family.

## What evolves between generations

| Effective boldness | Mutation scope                                                              |
| ------------------ | --------------------------------------------------------------------------- |
| < 0.35             | `EXPLOITATION` — point mutation: minor phrasing / prompt-adjective tweaks   |
| < 0.50             | `ALIGNMENT` — interface optimization: refine handoff prompts, IO contracts  |
| < 0.65             | `ADAPTATION` — component overhaul: rewrite lagging agent prompts, swap tools |
| < 0.90             | `EXPLORATION` — macro structural mutation: add/merge agents, change routing |
| ≥ 0.90             | `RE-SPECIATION` — clean-slate redesign of the multi-agent architecture      |

Effective boldness is computed from two signals: a plateau counter
(`iters_since_improvement / 6`) over recent scored offspring, and the
Rechenberg 1/5 success rate of the last 5 scored offspring (below `0.20`
the search escalates, above it boldness damps; at `≥ 0.80` it collapses
regardless of plateau). Only in the last 5 % of the score range does
the parent's absolute score re-enter, as a near-finish damper, so
near-winners aren't gambled away one generation before early-stop. The
agent budget grows with boldness up to a hard ceiling of `7`.

By default `~10 %` of generations do **crossover** instead of mutation
(`crossover_rate = 0.1`) — two parents combined, best-parent-first, with
the offspring hard-capped at the highest parent agent count.

## What you'll see on disk

After a `--learn` run, each generation has its own folder under
`sources/workflows/<uuid>/`. The "best" UUID is logged to the console at
the end. Across the run you'll also get:

- `sources/workflows/<best_uuid>/reward_progress.png` — score-over-iteration curve.
- `sources/workflows/<best_uuid>/evolution_tree.png` — lineage tree (each
  node a workflow, edges showing mutation/crossover).
- `runs_capsule/<capsule_name>/` — archived snapshot of the best run.

![Evolution tree example](../images/evolution_tree.png){ width="60%" }

## Mid-run interruption

`SIGINT` (Ctrl-C) and `SIGTERM` are caught by `setup_signal_handlers()` in
`main.py`. Mimosa cancels in-flight async tasks and exits cleanly. The
session archive lives only in memory, so an interrupted `--learn` run
discards uncommitted progress — but every completed generation is already
on disk under `sources/workflows/<uuid>/` and can be inspected.

## See also

- [Evolution engine](../concepts/evolution-engine.md) — mechanics of QD selection.
- [Evaluation pipeline](../concepts/evaluation-pipeline.md) — what the scores mean.
- [Workspace & audit trail](workspace.md) — where each generation's artefacts go.
- [Transparency & replay](transparency.md) — step through any generation interactively.
