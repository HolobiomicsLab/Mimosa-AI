# File layout

The repository at a glance.

```text
mimosa-ai/
├── config.py                              # Configuration class
├── main.py                                # CLI entry point & mode dispatch
├── pyproject.toml                         # Project metadata + dependencies
├── cleanup.sh                             # Delete generation folders + tmp/
├── mkdocs.yml                             # Documentation site config
│
├── sources/
│   ├── core/                              # Engine: evolution, selection, factories
│   │   ├── evolution_engine.py            # Depth-first recursion loop
│   │   ├── stopping.py                    # Early-stop logic for --learn
│   │   ├── selection.py                   # QD archive + admission gate
│   │   ├── variation_engine.py            # Mutation / crossover prompt assembly
│   │   ├── workflow_selection.py          # Parent retrieval (archive / disk)
│   │   ├── genotype_embedding.py          # Code-genotype embedding backend → QD behaviour descriptor
│   │   ├── code_features.py               # genotype_embedding_descriptor shim (QD novelty)
│   │   ├── failure_fingerprint.py         # Legacy verifier fingerprint (deprecated; hybrid writes a neutral placeholder)
│   │   ├── lineage.py                     # Parent → child sidecar records
│   │   ├── orchestrator.py                # Grounding → factory → sandbox
│   │   ├── workflow_factory.py            # Multi-agent workflow synthesis
│   │   ├── single_agent_factory.py        # Single-agent baseline
│   │   ├── factory.py                     # Shared factory primitives
│   │   ├── workflow_runner.py             # Sandboxed Python execution
│   │   ├── workflow_info.py               # Workflow metadata reader
│   │   ├── tools_manager.py               # MCP tool discovery
│   │   ├── llm_provider.py                # Multi-provider LLM abstraction
│   │   ├── planner.py                     # Goal → tasks (Layer 0)
│   │   └── schema.py                      # IndividualRun, Plan, Task, …
│   │
│   ├── evaluators/                        # Verifier backends
│   │   ├── evaluator.py                   # Facade (routes to backends)
│   │   ├── base.py                        # Shared primitives
│   │   ├── grounding.py                   # Perspicacité adapter
│   │   ├── policy_bt_verifier/            # Policy-BT verifier (default): reexec, recon, policy,
│   │   │                                  # visual, tournament, registry, results
│   │   ├── hybrid_verifier/               # Hybrid temporal-ladder verifier (historical): claims, digest,
│   │   │                                  # scorers, registry, aggregation, gradient, layers
│   │   ├── legacy_verifier/               # Legacy multi-source per-claim verifier (deprecated)
│   │   ├── generic_verifier/              # generic.py (legacy 4-criterion judge), bs_detection.py
│   │   ├── scenario/scenario.py           # Rubric-based scoring
│   │   └── gold_feedback/                 # Benchmark-grader feedback (oracle mode, research only)
│   │
│   ├── benchmark_evaluation/              # Batch / benchmark eval
│   │   ├── csv_mode.py                    # Concurrent batch runner
│   │   ├── snapshot_grading.py            # Snapshot grading
│   │   ├── capsule_evaluator.py           # ScienceAgentBench metrics
│   │   ├── codebert_scorer.py             # CodeBERT similarity
│   │   ├── execution_sandbox.py           # Safe code execution for eval
│   │   ├── scenario_loader.py             # Load scenario rubrics
│   │   ├── science_agent_bench.py         # Dataset adapter
│   │   └── eval_workflow_generation.py    # Workflow-quality eval mode
│   │
│   ├── cli/                               # Interactive entrypoints
│   │   ├── onboard_cli.py                 # Zero-arg onboarding wizard
│   │   ├── evaluation_cli.py              # Benchmark launcher
│   │   ├── memory_chat_cli.py             # Memory chat (--memory_cli)
│   │   ├── theme.py                       # Terminal theme
│   │   └── pretty_print.py                # Coloured CLI primitives
│   │
│   ├── transparency/                      # Run traces and exports
│   │   ├── yaml_writer.py
│   │   ├── trace_compaction.py
│   │   ├── memory_trace.py
│   │   ├── decision_extractor.py
│   │   └── astra_exporter.py
│   │
│   ├── extensibility/
│   │   ├── human_mode.py                  # Manual mode (no LLM)
│   │   └── text_to_speech.py              # TTS hook
│   │
│   ├── modules/                           # Pre-fab code injected into workflows
│   │   ├── state_schema.py                # LangGraph state template
│   │   └── smolagent_factory.py           # SmolAgent factory template
│   │
│   ├── prompts/                           # LLM prompt templates
│   │   ├── workflow_v15.md                # Current workflow generator prompt (config.py default)
│   │   ├── workflow_v13_model_select.md   # Workflow prompt variant
│   │   ├── workflow_v12_model_select.md   # Workflow prompt variant
│   │   ├── workflow_v11.md                # Workflow prompt (kept for diffing)
│   │   ├── workflow_v10.md                # Workflow prompt (kept for diffing)
│   │   ├── workflow_v9.md                 # (kept for diffing)
│   │   ├── workflow_v8.md                 # (legacy reference)
│   │   ├── planner_reproduction.md        # Planner — reproduction goal
│   │   ├── planner_paperbench_codedev.md  # Planner — paperbench code-dev
│   │   └── smolagent_sys_prompt.md        # SmolAgent system prompt
│   │
│   ├── cache/
│   │   └── openrouter_pricing.json        # Cached pricing
│   │
│   ├── security/
│   │   └── check_package.py               # Pre-flight package vetting
│   │
│   ├── utils/                             # Cross-cutting helpers
│   │   ├── paths.py                       # Default state directories
│   │   ├── ensure_env.py                  # Environment bootstrap
│   │   ├── archive_projection.py          # Archive projection helpers
│   │   ├── run_metrics.py                 # Run metrics
│   │   ├── pricing.py                     # OpenRouter pricing client
│   │   ├── logging.py                     # Structured logging
│   │   ├── notify.py                      # Pushover notifications
│   │   ├── transfer_toolomics.py          # Workspace ↔ Toolomics
│   │   ├── workspace_management.py        # Snapshot / restore best
│   │   ├── perspicacite_client.py         # Literature grounding client
│   │   ├── planner_visualization.py       # Real-time plan view
│   │   ├── evolution_tree.py              # Lineage tree → PNG
│   │   ├── visualization.py               # Reward/assertion plots
│   │   ├── shared_visualization.py        # Shared plot primitives
│   │   ├── email_reporter.py              # Email run summaries
│   │   ├── openrouter_endpoints.py        # OpenRouter endpoint catalogue
│   │   ├── precheck.py                    # Environment validation
│   │   ├── list_files.py                  # Workspace listing helper
│   │   ├── dataset.py                     # CSV / scenario helpers
│   │   └── mock_data.py                   # Test fixtures
│   │
│   ├── memory/                            # LLM call cache (runtime; default_memory_dir)
│   └── workflows/                         # Per-generation artefacts (runtime)
│       └── <uuid>/
│           ├── workflow_genotype_<uuid>.py
│           ├── state_result.json
│           ├── system_prompt_<uuid>.md
│           ├── goal_<uuid>.txt
│           ├── original_task_<uuid>.txt
│           ├── evolution_prompt_<uuid>.md
│           ├── lineage_<uuid>.json
│           ├── evaluation.txt             # Hybrid / legacy / generic verifier report
│           ├── textual_gradient.txt       # Hybrid verifier gradient
│           ├── policy_bt_evidence.json    # Policy-BT evidence
│           ├── .policy_bt/                # Policy-BT registry
│           ├── reward_progress.png        # only on best UUID
│           └── evolution_tree.png         # only on best UUID
│
├── runs_capsule/                          # Archived run snapshots (runs_capsule_dir)
│   └── <capsule_name>/
│       └── evaluation_results.json
│
├── datasets/                              # Benchmark CSVs + scenario rubrics
│   ├── ScienceAgentBench.csv
│   ├── ScienceAgentBench/                 # Per-task workspaces
│   ├── paper_bench.csv
│   ├── paper_bench_light.csv
│   ├── our_benchmark.csv
│   ├── papers_rejection_watch.csv
│   ├── datascience_papers.csv
│   └── scenarios/                         # Scenario rubrics (*_rubric.json)
│
├── docs/                                  # This documentation site
│   ├── index.md                           # Landing page
│   ├── getting-started/
│   ├── concepts/
│   ├── usage/
│   ├── evaluation/
│   ├── reference/
│   ├── developer/
│   ├── about/
│   ├── DEVELOPER_GUIDE.md                 # Code-level deep dive
│   ├── diagrams/                          # .mermaid sources
│   └── images/                            # Rendered .png/.jpg
│
└── tests/                                 # pytest suite (sample)
    ├── evaluator_test.py
    ├── scenario_rubric_test.py
    ├── judge_test.py
    ├── workflow_evaluator_test.py
    ├── tools_manager_test.py
    ├── pricing_test.py
    ├── memory_read.py
    └── cosine_similarity.py
```

## What's safe to delete

| Path | Safe to delete? | Notes |
| ---- | --------------- | ----- |
| `sources/workflows/<uuid>/` | Yes | Per-generation artefacts. Wipe to start fresh. |
| `runs_capsule/` | Yes | Archive snapshots. `cleanup.sh` does not touch it. |
| `sources/memory/` | Yes | LLM cache (`default_memory_dir`). Wipe to force re-judging. `cleanup.sh` does not touch it. |
| `./tmp/` | Yes | Sandbox scratch. |
| `sources/cache/openrouter_pricing.json` | Yes | Re-fetched on next call. |
| `sources/prompts/` | No | Source. Don't delete. |
| `sources/modules/` | No | Workflow templates. Don't delete. |
| `sources/core/`, `sources/utils/`, etc. | No | Source. Don't delete. |

For a clean reset, prefer `./cleanup.sh` (or `./cleanup.sh <subfolder>`
for a campaign folder; `-n` lists first) over hand-deleting.

## See also

- [Workspace & audit trail](../usage/workspace.md) — what's *inside* each runtime folder.
- [Developer guide](../DEVELOPER_GUIDE.md) — what each source module does.
