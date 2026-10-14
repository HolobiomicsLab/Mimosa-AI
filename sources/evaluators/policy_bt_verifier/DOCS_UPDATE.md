# Docs update instructions — Policy-BT verifier (2026-10-13)

The code for the new Policy-BT verifier is in place. These docs still describe
the old verifier and must be updated by a docs pass. Already done:
`docs/concepts/evaluation-pipeline.md` has a new "Policy-based Bradley–Terry
verifier (default)" section. Check it against the facts below.

## Facts to document

1. **Default channel.** `verifier_kind = "policy_bt"` is now the default in
   both `config.py` and `config_default.json`. Hybrid and legacy stay
   available.
2. **Reward.** A regularized Bradley–Terry fit over all pair verdicts.
   After each evaluation, every generation of the task session gets a new
   current reward. `arrival_reward` keeps the reward at arrival time. The only
   allowed `policy_bt_reward` value is `"bradley_terry"`; `"win_rate"` raises
   an error.
3. **Reproduction contract.** Each generation needs a `.sh` script that
   installs requirements and regenerates every output. A missing or invalid
   script gives reward `0` and a gradient that names the missing script.
   This applies even when re-execution is off.
4. **Re-execution.** `policy_bt_reexecute` (default `true`, CLI
   `--policy_bt_reexecute / --no-policy_bt_reexecute`). The script runs with
   bash on a copy, in a fresh venv first on PATH with no API keys (so plain
   `pip`/`python` resolve to that venv). Old outputs move aside first, so
   stale outputs cannot pass. This is NOT an OS sandbox. Run logs are copied
   to `.policy_bt_reexec/` inside the snapshot so adapters can read
   stdout-only metrics. A failed run's stderr tail goes into the gradient.
5. **Pipeline.** snapshot copy → reproduction gate → LLM reconnaissance
   (complete inventory, bounded previews, explicit byte-page inspection; a
   malformed answer is fed back to the model, not fatal) → frozen task
   criteria (each with direction `higher`/`lower`/`satisfy`) + one
   comparison policy → per-schema adapters (new schemas add adapters;
   criteria never change) → pair compare in a restricted interpreter
   subprocess, both orders + self-tie + "a known difference must decide"
   checks → visual tie-break with the vision model when both sides have
   relevant images → Bradley–Terry fit → gradient.
6. **Instrument errors.** If the verifier itself fails (bad LLM answer after
   all repair rounds, adapter/policy error, venv creation failure, missing
   goal), the generation is NOT scored. `state_result.json` gets
   `skipped_reason = "verifier_instrument_error"`; the engine treats it as a
   failed generation and keeps it out of the QD archive. A stale cached
   policy (older `PROMPT_VERSION`) is rebuilt, never fatal.
7. **Operator console.** Each evaluation prints a themed report
   (`sources/evaluators/policy_bt_verifier/reporting.py`): reproduction
   outcome, the frozen criteria, previews of the generated adapter and
   comparison-policy code, the files each pair compared, the Bradley-Terry
   standings with bars and win/loss/tie records, and the gradient preview.
   Colors auto-disable when stdout is not a terminal or NO_COLOR is set; the
   JSON-lines sink ``verifier_policy_bt.log`` is unchanged.
8. **Selection.** The QD archive, parent draws and the final capsule use
   the current ratings. For policy_bt, the capsule is the argmax of current
   ratings, not the stop controller's champion.
9. **Artifacts.** Per generation: `evaluation.txt`,
   `textual_gradient.txt`, `policy_bt_evidence.json`,
   `verifier_policy_bt.log`. Per task session:
   `<workflow_dir>/.policy_bt/<sha256(goal, session)>/tournament.json` plus
   `snapshots/<uuid>/`.
10. **Cost.** Verifier LLM calls are saved as
    `memory/<uuid>/verifier_policy_bt_<channel>_<n>.json` and counted in the
    run cost.
11. **Removed knobs.** `policy_bt_refiner`, `policy_bt_max_pairs`,
    `policy_bt_venv_reuse`, `policy_bt_visual_tiebreak_code`,
    `policy_bt_visual_max_pairs_per_gen`, `policy_bt_verbose`,
    `policy_bt_recon_refresh_gap`.

## Pages to update

- `AGENTS.md`: the "Verifier artifact formats" note and the
  `policy_bt` description (old digest/recon/metrics modules no longer
  exist; reward is a fitted BT, not win-rate).
- `docs/DEVELOPER_GUIDE.md`: evaluation section.
- `docs/concepts/architecture.md`: layer 4 row.
- `sources/evaluators/policy_bt_verifier/DESIGN.md` §0 status line
  ("WORK IN PROGRESS" and "Current refactor DELETED …") is now out of date.

## Known limits to state plainly

- Adapters run in a small Python subset (no imports beyond json, math,
  struct, csv parsing). Formats that need numpy, pandas, or binary
  scientific readers stay "unknown" (null), never a fabricated loss.
- Workflow prompts (`sources/prompts/workflow_v*.md`) do not yet ask agents
  to leave a `.sh` reproduction script. Until they do, generations without
  one score 0. Decide whether to add that requirement to the prompts.
- Quality depends on the judge model writing a correct comparison policy.
  The verifier now rejects asymmetric policies and ties that ignore a
  measured difference, but it cannot detect a policy that picks the wrong
  criterion direction if the recon model states it wrongly.
