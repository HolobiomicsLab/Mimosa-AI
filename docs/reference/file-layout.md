# File layout

The repository at a glance.

```text
mimosa-ai/
├── config.py                              # Configuration class
├── main.py                                # CLI entry point & mode dispatch
├── pyproject.toml                         # Project metadata + dependencies
├── cleanup.sh                             # Reset workflows + capsules
├── memory_explorer.py                     # Interactive trace replay
├── mkdocs.yml                             # Documentation site config
│
├── sources/
│   ├── core/                              # Engine: evolution, selection, factories
│   │   ├── evolution_engine.py            # Depth-first recursion loop
│   │   ├── selection.py                   # QD archive + admission gate
│   │   ├── variation_engine.py            # Mutation / crossover prompt assembly
│   │   ├── workflow_selection.py          # Parent retrieval (archive / disk)
│   │   ├── genotype_embedding.py          # Code-genotype embedding backend → QD behaviour descriptor
│   │   ├── code_features.py               # genotype_embedding_descriptor shim (QD novelty)
│   │   ├── failure_fingerprint.py         # Verifier verdicts → failure fingerprint (persisted diagnostic, 6-D centered)
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
│   │   ├── schema.py                      # IndividualRun, Plan, Task, …
│   │   └── evaluators/                    # Verifier backends
│   │       ├── evaluator.py               # Facade (routes to backends)
│   │       ├── verifier.py                # Default multi-source per-claim verifier
│   │       ├── grounding.py               # Perspicacité adapter
│   │       ├── generic.py                 # Legacy 4-criterion judge
│   │       ├── scenario.py                # Rubric-based scoring
│   │       ├── bs_detection.py            # BullshitDetector penalty
│   │       └── base.py                    # Shared primitives
│   │
│   ├── evaluation/                        # Batch / benchmark eval
│   │   ├── csv_mode.py                    # Concurrent batch runner
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
│   │   └── pretty_print.py                # Coloured CLI primitives
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
│   │   ├── workflow_v10.md                # Current workflow generator prompt
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
│   ├── memory/                            # LLM call cache + memory traces (runtime)
│   └── workflows/                         # Per-generation artefacts (runtime)
│       └── <uuid>/
│           ├── workflow_genotype_<uuid>.py
│           ├── state_result.json
│           ├── evolution_prompt_<uuid>.md
│           ├── lineage_<uuid>.json
│           ├── reward_progress.png        # only on best UUID
│           ├── evolution_tree.png         # only on best UUID
│           └── memory/
│
├── runs_capsule/                          # Archived run snapshots
│   └── <capsule_name>/
│       ├── workflow.py
│       ├── results/
│       ├── logs/
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
│   └── scenarios/                         # Scenario rubrics
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
└── tests/                                 # pytest suite
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
| `sources/workflows/<uuid>/` | ✅ | Per-generation artefacts. Wipe to start fresh. |
| `runs_capsule/` | ✅ | Archive snapshots. Wipe before benchmarks. |
| `sources/memory/` | ✅ | LLM cache + task checklists. Wipe to force re-judging. |
| `./tmp/` | ✅ | Sandbox scratch. |
| `sources/cache/openrouter_pricing.json` | ✅ | Re-fetched on next call. |
| `sources/prompts/` | ❌ | Source. Don't delete. |
| `sources/modules/` | ❌ | Workflow templates. Don't delete. |
| `sources/core/`, `sources/utils/`, etc. | ❌ | Source. Don't delete. |

For a clean reset, prefer `./cleanup.sh` over hand-deleting.

## See also

- [Workspace & audit trail](../usage/workspace.md) — what's *inside* each runtime folder.
- [Developer guide](../DEVELOPER_GUIDE.md) — what each source module does.
