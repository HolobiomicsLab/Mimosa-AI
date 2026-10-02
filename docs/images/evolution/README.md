# Evolution-process figures

Ten publication-quality figures illustrating how Mimosa-AI evolves a
multi-agent workflow for a task. Each is emitted as a 320-dpi `.png` (slides,
GitHub) and a vector `.svg` (print, LaTeX, further editing).

Regenerate all of them:

```bash
python3 make_evolution_figures.py        # writes here, into docs/images/evolution/
```

The script (`make_evolution_figures.py`, repo root) is self-contained
(matplotlib + numpy) and parameter-faithful to the engine defaults documented
in [`concepts/evolution-engine.md`](../../concepts/evolution-engine.md) and
[`concepts/evaluation-pipeline.md`](../../concepts/evaluation-pipeline.md).

| # | File | What it shows | Docs section |
|---|------|---------------|--------------|
| 1 | `fig01_evolution_loop` | The per-generation cycle: synthesize → run → verify → select → vary | Evolution engine · Big picture |
| 2 | `fig02_genotype_phenotype` | Code-as-genotype → workspace-as-phenotype, plus the embedding branch | Evolution engine · what's evolving |
| 3 | `fig03_qd_archive` | The unstructured Quality-Diversity archive scored by `qd = ¾·quality + ¼·novelty` | Selection: Quality-Diversity |
| 4 | `fig04_genotype_embedding` | Genotype embedding (MiniLM 384-d) → cosine k-NN novelty, `k = 15` | Behaviour descriptor |
| 5 | `fig05_rechenberg_boldness` | Effective-boldness function vs. success rate / plateau (Rechenberg 1/5) | Variation · evidence-driven scope |
| 6 | `fig06_scope_ladder` | The five mutation-scope bands, EXPLOITATION → RE-SPECIATION | Variation · scope band table |
| 7 | `fig07_variation_operators` | Mutation (directive-LLM split) vs. crossover (best-parent-first) | Mutation directive · Crossover |
| 8 | `fig08_verifier` | Six claim sources, per-claim Python recompute, rubric-blind firewall | Evaluation pipeline |
| 9 | `fig09_lineage_tree` | A lineage tree: score climbing red→green, mutation/crossover edges | Watching evolution happen |
| 10 | `fig10_reward_progress` | Reward-over-generations envelope coupled to the boldness control law | Run-metrics artifacts |

### Notes on the two illustrative figures

- **fig09** and **fig10** use small hand-authored trajectories to make the
  mechanics legible on one canvas; they are schematics, not logged runs. The
  boldness curve in fig10 is computed from its scripted scores with the
  *actual* `VariationEngine` control law (cold-start ramp, Rechenberg 1/5,
  plateau patience 6, RE-SPECIATION hysteresis clamp 0.89), so the two panels
  stay consistent. Real runs write `reward_progress.png` and `evolution_tree.png`
  per run under `sources/workflows/<uuid>/`.
- **fig03**/**fig04** scatter points are synthetic but obey the engine's
  invariants (additive QD score, spread-promoting novelty, semantic collapse
  of similar genotypes).

Palette and typography follow the Mimosa mermaid theme (`docs/diagrams/`).
