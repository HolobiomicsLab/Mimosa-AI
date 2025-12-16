# Running PaperBench Locally (Quick Note)

This note captures the minimum you need to run PaperBench with pre-existing submissions (via `PBDirectSubmissionSolver`). For a full walkthrough see `paperbench/README.md`.

## 1. Prerequisites

0. Clone OpenAI frontier-evals repository and enter paperbench folder:

```bash
git clone https://github.com/openai/frontier-evals.git --filter=blob:none
cd frontier-evals/project/paperbench
```

1. Install dependencies:
   ```bash
   uv sync
   ```
2. Hydrate the dataset (Git LFS) and set `PAPERBENCH_DATA_DIR` if it lives outside this repository.
3. Build the required Docker images once:
   ```bash
   bash paperbench/scripts/build-docker-images.sh
   ```
4. Prepare your submissions directory (need to rename the capsule to <paper-id-1> and add submission_try_1 manually):
   ```text
   /path/to/mimosa/runs_capsule/
       <paper-id-1>/
           submission_try_1/
           submission_try_2/
       <paper-id-2>/
           submission_try_1/
   ```
   Each `submission_try_*` folder should contain the files you want graded. Missing paper folders simply count as zero for that paper.

**Paper IDs:**

fre
mechanistic-understanding
bridging-data-gaps
test-time-model-adaptation
all-in-one
sequential-neural-score-estimation
robust-clip
what-will-my-model-forget
pinn
stay-on-topic-with-classifier-free-guidance
rice
sample-specific-masks
adaptive-pruning
sapg
lca-on-the-line
stochastic-interpolants
bbox
lbcs
bam
ftrl

## 2. Choose the split

PaperBench launches one task per paper listed in `paperbench/experiments/splits/<split>.txt`. Make sure the split references only papers you actually have submissions for (e.g., create a custom split containing `all-in-one` if that is your only target).

## 3. Example commands

Run all commands from the repository root.

### Code-Dev (judge checks code structure only)

```bash
uv run python -m paperbench.nano.entrypoint \
    paperbench.paper_split=all-in-one \
    paperbench.solver=paperbench.solvers.direct_submission.solver:PBDirectSubmissionSolver \
    paperbench.solver.submissions_dir=/home/mlegrand/Projects/foo_eval \
    paperbench.judge.scaffold=simple \
    runner.recorder=nanoeval.json_recorder:json_recorder \
    paperbench.judge.code_only=True
```

### Full evaluation (reproduction + grading)

```bash
uv run python -m paperbench.nano.entrypoint \
    paperbench.paper_split=all-in-one \
    paperbench.solver=paperbench.solvers.direct_submission.solver:PBDirectSubmissionSolver \
    paperbench.solver.submissions_dir=/home/mlegrand/Projects/foo_eval \
    paperbench.judge.scaffold=simple \
    runner.recorder=nanoeval.json_recorder:json_recorder
```

## 4. Inspecting results

- Each run creates a directory under `paperbench/runs/<run_group_id>/<paper_id_run_id>/` with logs, grades, and archived submissions.
- `grade.json` contains the final score and judge output.
- `group.log` summarizes the entire launch.

If a run reports "No checkpoint exists", confirm the submissions directory contains a folder named exactly after the paper(s) in your split. If reproduction fails because Python/Jupyter are missing in the base container, rerun after pulling the latest code (the runtime now auto-installs Python when necessary).
