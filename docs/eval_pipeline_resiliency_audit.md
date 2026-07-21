# Evaluation Pipeline — Resiliency & Error-Handling Audit

**Date:** 2026-07-21
**Scope:** `sources/cli/evaluation_cli.py` → `sources/benchmark_evaluation/` (`csv_mode.py`, `capsule_evaluator.py`, `execution_sandbox.py`, `codebert_scorer.py`, `science_agent_bench.py`) + direct dependencies (`sources/utils/transfer_toolomics.py`, `sources/cli/theme.py`). `sources/core/` excluded from the audit (one cross-boundary claim was verified there and withdrawn — see Appendix).
**Method:** full manual read of ~4 300 lines, followed by an adversarial review pass; every finding below was re-verified against the code after the critique, and corrections are incorporated.

> **Context:** this audit supports the V2 evaluation campaign in `todo_eval.md`
> (3 modes + ablation tables ≈ 36 full-benchmark runs, multi-seed, queued via
> `--evaluation_cli`). Priority is driven by risk to those results: metric
> corruption > artifact corruption > wasted wall-clock > hygiene.

---

## Findings by priority

### P0 — Must fix before the campaign (corrupts results or artifacts)

---

#### C3 · Cross-run state contamination between queued CLI runs

**Status:** confirmed, strengthened (collisions are *guaranteed* for same-goal seeds, trigger from the first batch).

`EvaluationCLI._run_single_eval` (`evaluation_cli.py:1066`) isolates `workflow_dir`, `memory_dir`, `runner_temp_dir` per queued run — but **not**:

1. **`runs_capsule_dir`** — capsule names are LLM-generated from the goal
   (`transfer_toolomics.create_capsule_name`). Two queued runs on the same task
   (the multi-seed plan) produce colliding capsule dirs; the
   `abs(hash(goal))` fallback is *bitwise-identical* for same-goal seeds within
   one process, so collision is guaranteed, not probabilistic. Capsules and
   their `evaluation_results.json` overwrite each other; `CapsuleEvaluator`
   can grade the wrong capsule.
2. **`run_notes/`** — `CsvEvaluationMode._load_previous_run_notes`
   (`csv_mode.py:113`) globs the shared directory; run #2 finds run #1's notes
   (same model) and offers to "restore" its stats → double-counted metrics
   across seeds. Same `{capsule_name}.json` filename collision.
3. **stdin prompts at launch** — every queued run calls
   `_prompt_with_default("Enter starting row")` (`csv_mode.py:31-54`) when it
   *starts*; concurrent runs race on stdin executor threads.

Aggravator: `_INITIAL_CONCURRENCY = 2` (`evaluation_cli.py:79`) — the first
batch already runs two evals concurrently, so no adaptive scale-up is needed
to trigger all three races. The CLI already de-collides `workspace_dir` per
queued run (`_suggest_workspace`), confirming this is an oversight, not a
design choice.

**Fix plan:**
- In `_run_single_eval`, also set per-run `runs_capsule_dir`
  (`runs_capsule/run_{id}`) and make `CsvEvaluationMode.run_notes_dir`
  per-run (constructor parameter, not hard-coded `Path("run_notes")`).
- Resolve start-row and cache-restore decisions during the *configuration*
  phase (store on `EvalRunSpec`), not at launch; pass them into
  `start_evaluation` so no prompt happens after the queue launches.
- Belt-and-braces: suffix capsule names with a run/task token
  (e.g. `f"{name}_{task_id}"`) in `transfer_workspace_files_to_capsule`.

---

#### C4 · Shared class-level venv across queued runs + pip failures misattributed to the agent

**Status:** confirmed; mechanism corrected (drift is an *accumulative union*, not "first-wins").

- `ExecutionSandbox._shared_venv_path` is process-global
  (`execution_sandbox.py:106-109`), reset only at `atexit`. The evaluation CLI
  runs all queued configs in **one process**, so run #2 inherits run #1's venv
  including its per-task installs (deepchem→dgl, …). `_setup_environment`
  re-installs each instance's base packages every time (`:263`) and pip never
  uninstalls → package set drifts as a growing union across ablation
  conditions.
- `_install_packages` treats a non-zero pip return code as **warning-only**
  (`:284-285`). A broken dependency install proceeds silently → task fails
  VER → recorded as `success_level='Failed'` (`csv_mode.py:503`) — an *agent*
  failure — never reaching the `EvalInfraError` exclusion path. This directly
  skews SR downward, defeating the exclusion mechanism.

**Fix plan:**
- On non-zero pip return code for *required* packages (base + pipreqs-derived),
  raise `EvalInfraError` (task excluded) — keep warning-only behavior only for
  `best_effort=True` special-case installs.
- Key the shared venv by a hash of `(base_packages, constraints)` so differing
  environments get separate venvs; at minimum, log the venv's package
  fingerprint (`pip freeze` hash) into run notes for drift forensics.
- Guard per-task installs into the shared venv with a lock independent of
  event-loop serialization (today safety is an accidental side-effect of C1).

---

#### C1 · Blocking calls inside async tasks freeze the entire event loop

**Status:** confirmed, strengthened (no `to_thread`/`run_in_executor` anywhere on the chain; the only executor use is the stdin prompt at `csv_mode.py:53`).

`_process_single_task` is `async`, but everything heavy beneath it is
synchronous:

| Call site | Blocking call | Bound |
|---|---|---|
| `execution_sandbox.py:208` (in `__init__`!) | venv creation | 600 s timeout |
| `execution_sandbox.py:283, 323, 343` | pip installs | 900 s timeout |
| `execution_sandbox.py:429` | VER execution | 300 s timeout |
| `execution_sandbox.py:692` | SR eval script | 180 s timeout (fixed) |
| `codebert_scorer.py:49-83` | CodeBERT load + torch inference | seconds–minutes |
| `transfer_toolomics.py:33` | capsule-namer LLM call (sync retry loop, up to 500 s backoff — `llm_provider.py:452-456`) | unbounded |
| `csv_mode.py:429` | `time.sleep(0.5)` | sequential path only |

Effects: (a) task-level concurrency evaporates during these phases; (b) other
in-flight tasks freeze, including LLM streaming and timeout bookkeeping;
(c) in the CLI queue, **one event loop is shared by all queued runs**, so
these calls stall sibling runs even at per-run concurrency 1 — C1 and C3
compound.

**Fix plan (lands together with M2 — see note):**
- Wrap sandbox construction, `run_generated_code`, `run_eval_script`, CBS
  scoring, and capsule transfer in `asyncio.to_thread(...)`.
- Replace `time.sleep` with `await asyncio.sleep`.
- **Ordering constraint:** module-level CodeBERT caching (M2) must land in the
  same change — `to_thread` un-suppresses the concurrent-load spike that C1
  currently hides.

---

#### M4 · Empty capsule name writes into the capsule root

**Status:** confirmed, with an easier trigger than originally reported.

`transfer_toolomics.create_capsule_name` (`:26-46`): a whitespace-only LLM
reply (e.g. `"\n"`) passes the `if not raw_output` guard, then sanitizes to
`""` → `create_capsule_folder("")` → path = `runs_capsule/` itself → the
workspace is copied into the **capsule root**, contaminating every other
capsule. (Second failure mode of the function behind the known None-guard
issue.)

**Fix plan:** after sanitization, fall back to `f"capsule_{task_token}"` when
the name is empty; assert non-empty in `create_capsule_folder`.

---

### P1 — Fix with the campaign (wall-clock, resource leaks)

---

#### C5 · Unbounded memory retention in `execution_history`

**Status:** confirmed, plus a third retention site found in review.

Entries keep `'runs': runs` — full `IndividualRun` object graphs — at three
sites: the evaluated path (`csv_mode.py:486/502`), **and the harness-error
path (`:530`, missed in the first pass)**. Entries are heavy:
`workflow_template` holds a full `WorkflowInfo` object (despite a
`str | None` annotation) that memoizes `state_result` and `code`. In
concurrent mode, `asyncio.gather`'s results list additionally retains
everything until the slowest task finishes. The retained `runs` lists are
never read after the current task — pure waste. Over 102 tasks × up to
25 generations this is GB-scale, and `_save_run_notes` re-walks it per task.

**Fix plan:** at capture time, project each run to a plain dict of the scalars
actually persisted (`reward`, `cost`, `current_uuid`) and store that instead;
drop `IndividualRun` references immediately after `_save_run_notes`.

---

#### M1 · Subprocess timeouts orphan grandchildren

**Status:** confirmed (no process-group handling on any of the 8
`subprocess.run` sites; timeout handlers only log).

`subprocess.run(timeout=)` kills only the direct child; generated scripts that
spawn workers (multiprocessing, DataLoader, TF) survive → CPU/RAM leaks
accumulate across 102 tasks. Aggravators: after a timeout,
`sandbox.cleanup()` (in `evaluate_all`'s `finally`) deletes the temp dir out
from under still-running orphans; orphan RAM also pollutes M6's
adaptive-concurrency measurement.

**Fix plan:** spawn with `start_new_session=True`; on timeout,
`os.killpg(proc.pid, SIGKILL)` before reaping (use `Popen` + `communicate`
instead of `subprocess.run`).

---

#### M2 · CodeBERT model reloaded per task

**Status:** dead code + per-call reload confirmed; memory-spike mechanism
corrected (currently suppressed by C1).

`preload_codebert_model` (`codebert_scorer.py:115`) has zero callers
repo-wide; the scoring path loads tokenizer + model fresh on every call.
Because scoring is synchronous on the event-loop thread (C1), loads serialize
today — at most one copy resident — so the present-day cost is a *stall* on
all in-flight tasks, not a spike. CBS is skipped when SR=1
(`capsule_evaluator.py:119-123`), so the tax hits only failed tasks.

**Fix plan:** cache tokenizer + model at module level (lazy singleton).
**Must land with the C1 fix** — `to_thread` makes concurrent loads possible
and turns the hidden spike real.

---

#### C2 · Stagger delay can idle workers for ~1 hour

**Status:** confirmed exactly; reachability narrowed.

`csv_mode.py:619-623`: `stagger_delay = launch_index * task_start_delay`
applies to *all* tasks *before* semaphore acquisition (absolute launch index,
no cap) — task #102 sleeps ~50.5 min even if workers are free. However, the
stagger exists only in the concurrent loop, and
`config.py:81` defaults `max_concurrent_eval_tasks = 1` (the CLI's
`getattr(..., 4)` fallback is dead — the attribute always exists). So it
bites only when concurrency is explicitly enabled.

**Fix plan:** stagger only the first `max_concurrent_tasks` launches
(e.g. `min(launch_index, max_concurrent_tasks - 1) * delay`), or move the
sleep after semaphore acquisition.

---

### P2 — Correctness hygiene / hardening (fix opportunistically)

| # | Finding | Status / notes |
|---|---------|----------------|
| M5 | `theme.ask` (`theme.py:204-212`) catches `EOFError` **and** `KeyboardInterrupt` → silent `sys.exit(0)`: `--evaluation_cli` dies silently when stdin isn't a TTY, and `main.py`'s own "Interrupted. Goodbye!" handler is unreachable from all ~20 wizard prompts. Inconsistent with `csv_mode._prompt_with_default`, which handles non-TTY gracefully. | confirmed, broadened |
| M6 | `_launch_queue` RAM adaptation (`evaluation_cli.py:973-1064`, `:1141-1148`): whole-process RSS (no per-run attribution), sampled after the batch exits. `ram_estimate = max(delta, RSS)` is never ~0 (the `>0` guard *holds* concurrency on zero) — real failure modes: floored at process baseline (overstates → can permanently refuse to scale) while child-subprocess work is invisible to RSS and has exited before the snapshot (understates true peak). | confirmed, direction corrected |
| L1 | `csv_mode.py:719`: `'goal' in dir()` trick in the exception path — initialize `goal = None` before `try`. | confirmed |
| L2 | `csv_runs_limit` is compared against the absolute row index (`csv_mode.py:818, 920`) — with `start_row > 0` you get fewer tasks than the limit suggests (index bound, not count). | confirmed |
| L3 | `_cleanup_isolated_workspace` (`csv_mode.py:562`): `rmtree(ignore_errors=True)` inside a `try/except` that can never fire — partial cleanups fully silent. | confirmed |
| L4 | `capture_output=True` buffers unbounded stdout/stderr; the 100 kB slice happens *after* capture. Redirect to a temp file instead. | confirmed |
| L5 | `_copy_capsule_contents_to_temp` copies the whole capsule twice per task (execution + eval dirs). Disk churn is bounded *per task* (subdirs are `rmtree`'d each call), not accumulating. | confirmed, bounded |
| L6 | `ScienceAgentBenchLoader._csv_data` annotated `list` but init `None`; `load_csv_data` ignores a differing `csv_path` on second call (stale cache). | confirmed |
| L7 | Run-notes JSON writes are non-atomic at three sites (`csv_mode.py:262`, `evaluation_cli._update_notes`, `_save_start_notes` at `evaluation_cli.py:940`) — crash mid-write leaves corrupt JSON. Write tmp + rename. | confirmed |
| L8 | CLI injects `evaluator._evaluation_cli_notes_path` as a private attribute (`evaluation_cli.py:1107`); `csv_mode` reads via `getattr(..., None)` so failure is silent. Pass through the constructor. | confirmed |
| L9 | Venv paths assume POSIX (`bin/python`) — unstated platform requirement. | confirmed |

---

## Withdrawn findings (kept for the record)

- **M3 (sequential-mode cost inflation) — withdrawn.** Each
  `start_workflow_evolution` call builds a fresh `IndividualRun`
  (`cost` defaults to `0.0`, `schema.py:51`) and a local `runs` list
  (`evolution_engine.py:391-404`); the accumulation at `:567` is scoped to
  that per-call list and the engine holds no cross-task cost state.
  `runs[-1].cost` at `csv_mode.py:466` is the correct per-task total in both
  modes. No fix needed.

---

## Fix plan

### Phase 0 — before any campaign run (P0)

1. **C3** — per-run `runs_capsule_dir` + `run_notes_dir`; move start-row /
   cache-restore prompts into the configuration phase; task-token suffix on
   capsule names.
2. **C4** — `EvalInfraError` on required-package pip failure; venv keyed by
   env fingerprint (or at minimum a `pip freeze` hash in run notes); install
   lock.
3. **C1 + M2** *(single PR — see ordering constraint)* — `asyncio.to_thread`
   on sandbox build/VER/SR/CBS/capsule-transfer paths; `asyncio.sleep`;
   module-level CodeBERT singleton in the same change.
4. **M4** — empty-capsule-name guard.

### Phase 1 — during preshots (P1)

5. **C5** — project `runs` to scalar dicts at capture; drop object references
   (all three retention sites).
6. **M1** — `start_new_session` + `killpg` on timeout for all sandbox
   subprocess spawns.
7. **C2** — cap the stagger at the worker count.

### Phase 2 — hardening, opportunistic (P2)

8. **M5** — unify headless/interrupt policy in `theme.ask` (propagate
   `KeyboardInterrupt`; explicit non-TTY default or loud error).
9. **M6** — remove or properly instrument adaptive RAM scaling
   (per-worker `resource.getrusage` peaks, including children).
10. **L1–L9** — small hardening batch (atomic JSON writes, constructor
    injection for notes path, file-redirected subprocess output, etc.).

### Validation

- `python3.12 tests/brute_gold_eval.py --light <subset>` after C4 — gold must
  still score VER=SR=100%, and a deliberately broken dep must now exclude
  rather than fail.
- `python -m pytest tests/test_benchmark_eval_error_handling.py`.
- Two-run queued CLI smoke (same 2 tasks × 2 seeds): assert distinct
  `runs_capsule/run_{1,2}` trees, distinct note files, no cache-restore
  cross-talk.
- RSS profile of one 10-task learning-mode preshot before/after C5.

---

## Appendix — what was already solid

Patterns worth keeping: the `EvalInfraError` → exclusion design cleanly
separates harness faults from agent failures; `_prompt_with_default`
explicitly fixed a prior executor-thread leak; the eval-output parser rejects
stray "1"/"success" heuristics; venv creation is lock-guarded and
version-verified; per-task temp dirs are cleaned in `finally`; CBS fallback
`0.0` is logged distinctly via `CBS_error`.
