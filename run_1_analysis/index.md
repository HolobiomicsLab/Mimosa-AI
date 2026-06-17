# run_1 — workflow evolution analysis

Post-hoc analysis of `sources/workflows/run_1/`: 97 attempted iterations across 5 ScienceAgentBench-style goals.

Snapshot date: 2026-06-14. Evaluation was still ongoing at snapshot time — dkpes had only 12 productive iterations.

## Two reports

- **[Overview report](REPORT.md)** — the trajectories, per-goal summary, lineage trees, QD landscape and cost curves. Start here.
- **[Deep root-cause investigation](DEEP_REPORT.md)** — 13 subagent diagnostics tracing why evolution regresses from its seeds. Stack-ranked fix priorities at the bottom.

## Headline result

Across 58 productive iterations the run cost **$26.74** and **19.8 wall-clock hours** to drift backwards from a strong cold-start: the final candidate of every single lineage is **worse than its seed**. Best gain over seed is **+0.041** on bulk_modulus — within the **±0.08 noise floor** of the LLM-as-judge verifier.

## Reproducing the analysis

```bash
# Re-extract metrics from sources/workflows/run_1/ artifacts
python3 docs/evaluation/run_1_analysis/data/extract.py
python3 docs/evaluation/run_1_analysis/data/extract_eval.py

# Regenerate figures
python3 docs/evaluation/run_1_analysis/data/visualize.py
python3 docs/evaluation/run_1_analysis/data/visualize_deep.py
```

Scripts read directly from `/home/martin/Projects/CNRS/Mimosa-AI/sources/workflows/run_1/`. Edit the `ROOT` constant at the top of each script to point at a different run.
