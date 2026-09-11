---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

![Mimosa-AI logo](images/logo_mimosa.png){ width="120" }

# Mimosa-AI 🌼🔬

*Self-evolving AI framework for autonomous scientific research.*

[Get started :material-rocket-launch:](getting-started/installation.md){ .md-button .md-button--primary }
[Read the paper :material-file-document:](https://arxiv.org/abs/2603.28986){ .md-button }
[View on GitHub :material-github:](https://github.com/HolobiomicsLab/Mimosa-AI){ .md-button }

</div>

---

## What is Mimosa-AI?

Mimosa-AI — like the mimosa plant that senses, learns, and adapts — is an open-source
framework for autonomous scientific research. It **synthesizes task-specific
multi-agent workflows on the fly**, refines them through execution feedback, and
gives academics a modular and auditable alternative to closed black-box systems.

It is built around three ideas:

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
### :material-dna: Self-evolving workflows
Workflows are emitted as Python programs and mutated as **full source code**, not
prompts or fixed templates. A Darwinian local search keeps only improvements.
</div>

<div class="feature-card" markdown>
### :material-toolbox: MCP-based tool discovery
Scientific tools live behind the **Model Context Protocol**, auto-discovered on
the local network via [Toolomics](https://github.com/HolobiomicsLab/toolomics).
You add new tools without touching Mimosa's core.
</div>

<div class="feature-card" markdown>
### :material-magnify-scan: Verifier-driven evaluation
A multi-source per-claim verifier writes **deterministic Python programs**
that confirm what the agents claim, against five vantages (literature,
user goal, agent narration, math invariants, statistical fingerprint). Only a coarse *prompt gradient* — which does not
leak the verified claims — is fed back to the mutator.
</div>

<div class="feature-card" markdown>
### :material-archive: Full audit trail
Every run produces a workflow genotype, an LLM call trace, a lineage record,
per-claim scores, and a workspace snapshot — everything needed to reproduce or
contest the result.
</div>

</div>

---

## Demo: autonomous paper reproduction

Mimosa-AI reproduced [Nothias et al. (2018)](https://www.researchgate.net/publication/323525305_Bioactivity-Based_Molecular_Networking_for_the_Discovery_of_Drug_Leads_in_Natural_Product_Bioassay-Guided_Fractionation) end-to-end —
from raw `.mzML` files to molecular network — autonomously, in a single command.

![Reproduced molecular network](images/network.png){ width="80%" }

The reproduced network matches the topology reported in the paper, including
cluster separation and edge weights.

---

## Benchmark snapshot

Evaluated on **ScienceAgentBench** (102 tasks, `task` mode):

| Mode                                    | Success Rate | Code-BLEU | Cost / task |
| --------------------------------------- | ------------ | --------- | ----------- |
| DeepSeek-V3.2 single-agent              | 38.2 %       | 0.898     | $0.05       |
| DeepSeek-V3.2 one-shot multi-agent      | 32.4 %       | 0.794     | $0.38       |
| **DeepSeek-V3.2 iterative-learning**    | **43.1 %**   | **0.921** | **$1.70**   |

Iterative learning improves GPT-4o but yields marginal degradation for Claude
Haiku 4.5 — see the [manuscript](https://arxiv.org/abs/2603.28986) for the
model-dependent behaviour analysis.

---

## Architecture at a glance

The framework is organized into five layers:

1. **Planning** *(optional)* — decomposes a high-level goal into discrete tasks.
2. **Tool discovery** — auto-discovers MCP tools via Toolomics.
3. **Meta-orchestration** — synthesizes a task-specific multi-agent workflow and
   evolves it generation by generation.
4. **Agent execution** — code-generating agents run subtasks in a sandbox.
5. **Judge & evaluation** — multi-source per-claim verifier scores outputs and
   drives the next mutation.

![Mimosa architecture overview](images/mimosa_overall.jpg){ width="90%" }

In benchmark `task` mode the planning layer is bypassed so workflow synthesis
and refinement can be evaluated in isolation.

---

## Where to next

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
### :material-package-down: Install Mimosa
Set up Python, the Toolomics MCP server, and your first API key.
[Installation →](getting-started/installation.md)
</div>

<div class="feature-card" markdown>
### :material-flash: Run your first task
A 5-minute quickstart from a fresh checkout to a finished workflow.
[Quickstart →](getting-started/quickstart.md)
</div>

<div class="feature-card" markdown>
### :material-graph: Understand the engine
How workflow evolution, QD selection, and the multi-source verifier work.
[Concepts →](concepts/index.md)
</div>

<div class="feature-card" markdown>
### :material-test-tube: Run benchmarks
ScienceAgentBench, PaperBench, or your own CSV of tasks.
[Evaluation →](evaluation/index.md)
</div>

<div class="feature-card" markdown>
### :material-code-tags: Extend it
Add MCP tools, swap evaluators, plug in new LLM providers.
[Developer guide →](DEVELOPER_GUIDE.md)
</div>

<div class="feature-card" markdown>
### :material-bug: Something broken?
Common errors, quantization gotchas, and Toolomics discovery issues.
[Troubleshooting →](reference/troubleshooting.md)
</div>

</div>

---

## Citation

If Mimosa-AI is useful for your research, please cite:

```bibtex
@article{legrand2026mimosa,
  title   = {Mimosa Framework: Toward Evolving Multi-Agent Systems for Scientific Research},
  author  = {Legrand, Martin and Jiang, Tao and Feraud, Matthieu and Navet, Benjamin
             and Taghzouti, Yousouf and Gandon, Fabien and Dumont, Elise and Nothias, Louis-F{\'e}lix},
  journal = {arXiv preprint arXiv:2603.28986},
  year    = {2026}
}
```

See [Citation](about/citation.md) for the full reference and BibTeX entry.
