# Policy-BT Verifier — Design Summary

related Subject: AI, RSI, self-improvement, verification system for AI agents, goodhart law.

> Reverse-engineered from old GLM 5.3 production code (`sources/evaluators/policy_bt_verifier/`,
> all 11 modules) and the offline research program (`experiments_verifiers/`,
> E14/E15/E18/E19/E19c/E24/E25/E26/E44–E48 harness + results of record).

Avoid reading the experiments verdict or code, they are weak. Previous GLM 5.3 agent mistake was implementing the production code by copying patterns from experiments, leaving briddle pattern, regex, and wrong architectural decisions. Keep your find fresh. Experiment are a distraction. Read after implementation not before.

## 0. Context

Mimosa-AI is an **open-source Python framework for autonomous scientific research**: it writes a **custom multi-agent workflow per task**, runs it in a sandbox, scores what the agents actually produced with deterministic Python policy scorers, and evolves the workflow across generations with a **Quality-Diversity** inspired search to find the optimal workflow for the task.

A verifier scores each run by executing deterministic Python **policy scorers** against the artifacts the agents produced (code, logs, outputs files, figures):  

### WHAT MIMOSA IS

Mimosa-AI writes a custom multi-agent workflow as PLAIN PYTHON SOURCE CODE for each
scientific task, runs it in a sandbox, scores what the agents actually produced, and
evolves the workflow across generations with a Quality-Diversity (QD) search.

- The workflow source is the "genotype"; what it produces in the workspace is the
  "phenotype" (code-as-genotype).
- Selection uses an unstructured QD archive, scored by a single scalarised objective
  qd_score = (1-w)*quality + w*novelty  (w = novelty_weight = 0.25). The lowest
  qd_score member is evicted when the archive is full.
- Novelty is cosine-distance k-NN (k = 15) over a code-embedding behaviour descriptor
  (local all-MiniLM-L6-v2 by default).
- Crossover combines two parents in ~40% of generations. Cold start seeds the search
  from past runs on disk (MiniLM cosine >= 0.8).
- An LLM-written TEXTUAL GRADIENT from the verifier steers the next mutation
  (elimination-point-first: it leads with the earliest stage/claim where this
  generation lost).
- Five layers: Planner (optional, --goal) - ToolManager/Perspicacite (MCP + literature)
  - EvolutionEngine - WorkflowRunner (SmolAgents sandbox) - Evaluator/verifier.

### THE VERIFIER DESIGNS SO FAR

The verifier has three jobs: rank generations (reward), write the gradient
(feedback), and pick the final capsule.

(1) LEGACY CLAIMS VERIFIER (deprecated; sources/evaluators/legacy_verifier/)
    - One LLM invents ~25 per-task claims; each claim self-rates its importance.
    - Reward = importance-weighted mean with a 0.89 hard-fail cap.
    - MEASURED MISALIGNED: reward anti-correlates with benchmark success at the top
      of its ranking (Kendall tau ~= -0.82), and self-reported importances are
      essentially noise (Spearman -0.12 vs learned weights).

(2) HYBRID TEMPORAL-LADDER VERIFIER (previous production default, 2026-09-24 → 2026-10-13; still available via verifier_kind = "hybrid"; sources/evaluators/hybrid_verifier/)
    - Claims extracted from the goal; ONE deterministic Python scorer per claim, run
      symmetrically on both sides (no pairwise comparator scripts).
    - A script -> log -> result temporal ladder: the earliest differing stage decides.
    - An execution gate (crash caps the score).
    - Reward = RAW WIN-RATE vs predecessors. Bradley-Terry was measured WORSE
      (0.643 vs 0.742, paired CI excludes 0) and stays opt-in only.
    - Best offline rankers measured around 0.742-0.759 online on the frozen arena.

(3) GOLD-FEEDBACK EVALUATOR (sources/evaluators/gold_feedback/)
    - Reward derived from benchmark gold. Evaluation-side only; a measurement
      channel, not a shipping channel.

(4) POLICY-BT VERIFIER (default; sources/evaluators/policy_bt_verifier/)
    - Shipped 2026-10-13 as the production channel: verifier_kind = "policy_bt" is the
      default in BOTH config.py and config_default.json. Hybrid and legacy stay available.
    - Pipeline (per generation): snapshot copy → reproduction gate (LLM-assessed .sh
      contract: install requirements + regenerate outputs; missing/invalid → reward 0)
      → optional re-execution with bash in a fresh venv without API keys
      (policy_bt_reexecute, default on; not an OS sandbox; logs to .policy_bt_reexec/)
      → LLM reconnaissance (complete inventory, bounded previews, explicit byte-page
      inspection) → frozen task criteria + per-schema adapters + ONE cached comparison
      policy → pair compare in a restricted interpreter subprocess (both orders +
      self-tie checks) → vision-model tie-break when both sides have relevant images →
      regularized Bradley–Terry fit (refreshes every generation's rating; arrival_reward
      keeps the rating at arrival) → evidence-checked gradient.
    - Modules: evaluator.py (orchestration), artifacts.py (snapshots, inventory),
      reexec.py (reproduction contract + venv rerun), recon.py (LLM schema plan),
      policy.py (restricted-subset interpreter + adapters + policy), visual.py
      (vision tie-break), tournament.py (BT fit), registry.py (atomic per-task-session
      state), results.py (public results + evidence files).
    - The E44-E48 digest/metrics/win-rate design was superseded during implementation:
      digest.py, metrics.py, gradient.py and verdicts.py do not exist (digest duties
      moved to artifacts.py/recon.py, gradient folding to results.py, pair logic to
      policy.py). Section 3 below documents the deleted architecture as historical
      reference only.
    - Removed knobs: policy_bt_refiner, policy_bt_max_pairs, policy_bt_venv_reuse,
      policy_bt_visual_tiebreak_code, policy_bt_visual_max_pairs_per_gen,
      policy_bt_verbose, policy_bt_recon_refresh_gap.


## 1. The idea of the Bradley-Terry code+vision policy-based verifier

**Don't let an LLM judge workspaces — let it write, once per task, for every workspaces produced by evolved workflow a small deterministic Python "policy" (or visual LLM for figures) policy that judges any *pair* of workspaces (Figure pairwise comparaison when available and relevant, first code policy then escaladate to pairwise figures compare when tie to find true winner); then reward each generation by its win-rate against all its predecessors in a bradley-terry tournament. Cumulate looser/winner verdict as a textual gradient used to drive evolution of workflow in Mimosa.**

The LLM contributes *task understanding* (what the goal demands, which metric
names matter); the harness contributes *measurement* (fresh re-execution
facts, pairwise metrics compare, deterministic statistics) and *symmetry* (same policy, same
digest schema, both argument orders).

## 2. How it would work

First, you need to understand where the multi-agent workflow artifacts produced by agent are, where you can find the artifacts.

You can find in the Toolomics workspace folder, which is the current workspace for the current workflow run of Mimosa (so the last iteration that has been running and hasn't yet been moved to /tmp/) you can find the produced artifacts in the workspace folder.

```sh
(mimosa-ai) λ martins-macbook-pro workspace → λ git dev* → pwd
/Users/cnrs/Documents/repository/Toolomics/workspace
(mimosa-ai) λ martins-macbook-pro workspace → λ git dev* → ls
protein_folding.py requirements.txt   run.sh             solver.py
```

Once a workflow evolution is done and the verifier gave a verdict, the current workflow workspace file will be transferred to a backup in /tmp/. Therefore, evolution workspace artifacts are always available in a pattern: mimosa_run_<UUID>

```sh
ls /tmp/
__KMP_REGISTERED_LIB_20264
...
mimosa_run_573da035de56_20261012_172209_a58f0412
...
mimosa_run_6d7a737159d1_20261012_171349_9c9d4cb1
mimosa_run_6e7653935559_generation_failed
mimosa_run_7c0c6f2328fc_20261012_143701_f81ccaee
mimosa_run_7c0c6f2328fc_20261012_143724_f9f45af0
mimosa_run_81806177294d_20261012_143314_4018f38d
mimosa_run_81806177294d_20261012_143356_ff96c636
mimosa_run_81806177294d_20261012_143442_4a0a719f
mimosa_run_81806177294d_20261012_143504_b018208b
mimosa_run_98fcb2ce8595_20261012_171930_d33ac870
mimosa_run_a38e991e008f_20261012_140938_4cf6a84f
mimosa_run_a38e991e008f_20261012_142750_f140e7e6
mimosa_run_ce949db3c7fb_20261012_144312_88d408e6
mimosa_run_ce949db3c7fb_20261012_144356_c5cbb5ff
```

You can see by doing ls in a folder that we have some of the results of a previous run on HP Folding Task. So each folder, each Mimosa folder with a UUID, is the result of what a single evolved multi-agent workflow produced for an attempt.

```sh
ls /tmp/mimosa_run_00e7d3360fce_20261012_144141_5b3932d2
protein_folding.py requirements.txt   result.txt         run.sh
```

You can use these to test live your implementation when you finish.

Now that we understand where the workspace artifacts are saved, we can talk about the implementation steps.

These are my suggestions and we are up for a design debate first.

### Proposed approach

First, we know that we need to make a list of pairwise workflows to compare in the Bradley tree. So we'll create this list of workspace paths to compare, like I guess, like a kind of list with a tuple of pairwise workspace so that they can be used to be run pairwise by the policy.

Then, for every pairwise workflow artifact, we'll move them into a folder in /tmp, but a special folder (let's call it pairwise_exec), whose content is cleared and updated at every call of the verifier, where the policy script will execute.

For each pairwise evaluation, both workflows’ artifact folders are placed in this pairwise_exec shared evaluation directory. The recon will explore and the python policy script (or visual policy) will run from this directory, with direct access to both sets of artifacts.

A reconnaissance phase will first inspect every file, identify artifacts that can meaningfully be compared, and produce compact previews for the LLM python policy writer. This phase will combine deterministic processing with schema interpretation, avoiding brittle matching rules. Repetitive content, such as repeated log entries, will be reduced deterministically so that previews remain concise and easy to parse. It doesn't cherry pick file type or use briddle regex. We don't know in advance what we are facing it could be csv, image or even LC-MS/MS data but the focus is mostly for parsing log files.

The LLM will use these previews to generate a Python comparison policy. The policy will compare the actual artifacts and produce a verdict, along with feedback explaining why one workflow won and the other lost (eg : {'winner_feedback': 'Lost by: Accuracy 0.45 and wrong train-test split ', 'looser_feedback': 'Win by accuracy > 0.77 and correct train-test split'})

The policy will be cached and reused across evaluations. However, evolving workflows may produce different artifact schemas: for example, two CSV files across 2 workspace artefacts may represent the same information using different column names. To accommodate this, the policy will separate schema normalization from comparison logic. When a new schema appears, only the normalization layer will be adapted; the core comparison criteria will remain stable.

This could involve two LLM calls. The first would generate or update adapters that map the observed artifact schemas into a unified representation. The second would write the deterministic comparison policy against that representation. Once created, the comparison policy would be reused, while the adapters would expand to support newly encountered schemas.

Each evaluation will produce a small piece of textual gradient feedback describing the decisive differences between the workflows. Winner and loser feedback will be fed as textual gradient to steer future workflow mutations, prioritizing the earliest stage or claim where a workflow lost.

If the deterministic policy returns a tie, the verifier will check for relevant visual artifacts, such as figures, PNGs, or JPEGs. When comparable visual artifacts are available, a vision-language model (VLM) will perform a pairwise comparison to try to resolve the tie. Otherwise, the result will remain a tie.

The resulting wins, losses, and ties will feed a Bradley–Terry tournament, allowing each generation to be ranked against its predecessors while the accumulated textual feedback guides Mimosa-AI’s workflow evolution.

Reward for every workflow so far will be updated as the bradley-terry evolved as new workflow run and get evaluated, every new evaluation give new reward to all workflows. You need to figure out how to update these reward cleanly for the QD algorithm.

(You should have a look at the previous verifiers design such as legacy_verifier to understand how verifier are implemented. )

## 3. Old Architecture (production port)

One evaluation of one generation = a fixed stage pipeline
(`evaluator.py`; stage line
REEXEC→DIGEST→RECON→POLICY→PAIRS→REWARD→GRADIENT→DONE):

File were removedt that's normal, previous implementation was a mess, the overall idea was sound so we let it as a reference. 

Note that the policy_bt_verifier class and entrypoint (now deleted) is still called by `evaluator.py` you might want to keep the same PolicyBTVerifierEvaluator name.

            elif verifier_kind == "policy_bt":
                # E44-E48 architecture: fresh re-execution + deterministic
                # digest + LLM policy tournament (opt-in; default hybrid).
                # Imported lazily so the default path never loads it.
                from .policy_bt_verifier import PolicyBTVerifierEvaluator

                self.verifier_evaluator = PolicyBTVerifierEvaluator(
                    config, workspace_dir=verifier_workspace_dir
                )

The config would also have to be updated.

```
            ┌──────────────────── verifier-owned sandbox ────────────────────┐
workspace ─▶│ REEXEC (reexec.py): copy → venv → pip install → run run.sh/py,  │
            │ process-group kill on timeout → LEDGER (exit, wall, timeout,    │
            │ error_signature, fresh_artifacts, log_tail, manifest hash)      │
            └──────────────┬─────────────────────────────────────────────────┘
                           ▼
            DIGEST (digest.py): deterministic gold-free projection of the
            sandbox copy + ledger → {exec, contract_files, deliverables,
            tables(+id_order_match), arrays, figures, json_metrics, json_flags,
            inputs, train_used} — sorted, capped, rounded; infra filtered
                           ▼
            RECON (recon.py): ONE LLM call per task (cached in registry):
            goal + digests → TASK CONFIG {primary_metrics (id/aliases/
            direction/role/bounds), tie_policy, coverage_footer}
                           ▼
            METRIC LEDGER (metrics.py): deterministic resolver per generation:
            json_metrics alias hits → LLM text extraction from log_tail →
            lexical fallback → writes primary_values ONTO the digest
                           ▼
            POLICY (policy.py): ONE LLM call per task per recon-version:
            goal + config + digest SCHEMA (never values) → compare(dA,dB)
            source → AST static screen → runtime smoke on REAL digests
            (both orders, self-tie) → repair loop → cached (POLICY_SHA)
                           ▼
            PAIRS (verdicts.py): vs every predecessor — crash rung first
            (ledger facts), else code_pair_verdict: policy runs BOTH orders
            in PATH-only subprocesses (stdin digests, stdout JSON verdict);
            mirrored winner required, else instrument-noise tie; verdict
            cache keyed policy_sha|a|b; per-pair artifact diff logged
                           ▼   (figure-dominant tasks: kimi visual channel
            instead — largest pred PNGs per side, cached)
            SCHEMA-ERROR REFRESH: both-order SchemaError → rebuild recon from
            last-N digests → rewrite policy → re-evaluate flagged pairs
                           ▼
            REWARD (verdicts.py): win-rate vs predecessors (ties ½), arrival-
            moment, history never rewritten; optional BT fit
                           ▼
            GRADIENT (gradient.py): fold of pair feedbacks — measured facts
            with values, metrics-from-ledger first, contract restated in the
            goal's own words → textual_gradient.txt + evaluation artifacts
                           ▼
            REGISTRY (registry.py): per-goal JSON — generations (uuid, digest,
            ledger, reward), verdict/policy/recon caches, session tags
```

## 4. Mistake made in previous implementation attempt (Avoid this time)

During the last attempt, GLM 5.3 AI agent attempted based on the experiment result and code to produce a working, policy based verifier. However, it failed because it overfit the design to the E44-E48 experiment code. There was a lot of brittle patterns in the code, such as hardcode regex, hardcoded patterns. But that was wrong. It was wrong because we want to design a verifier that can work on any computational scientific task. It can be a toxicity classification task, it can be designing a machine learning model, it can be HP folding, it can be protein folding, it can be molecular docking, it can be anything in science. Therefore, we cannot know in advance what kind of file we will have. We cannot know in advance the size of the file, whenever it's a minimum or a maximum. We cannot know in advance which file will be relevant and which will be not. We cannot know in advance what kind of data we will have, whenever it's CSV or not. We cannot know in advance what will be the accuracy metrics. So, all of this cannot be hard coded as regex. There are way too many patterns. Instead we need the design to use LLM call to classify and extract data. We cannot use regex. So we cannot have any hardcode reject pattern.