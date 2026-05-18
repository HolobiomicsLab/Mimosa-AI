# Geometric Framework for Multi-Agent Workflow Design

*A synthesis of stratified manifolds, concept attractors, and trajectory straightening — applied to the Mimosa architecture*

---

## 1. The Synthesis: One Phenomenon, Three Lenses

Three recent papers, read in isolation, describe distinct geometric properties of LLM internal representations. Read together, they describe the **same underlying mechanism** measured from three angles.

### The Three Measurements

| Lens | What it measures | Where it lives |
|---|---|---|
| **Stratification** (Li & Sarwate) | Embedding space partitions into discrete semantic patches with varying intrinsic dimensions | Final hidden layer |
| **Attractors** (Chytas & Singh) | Within each patch, prompts contract toward fixed points via Iterated Function Systems | Intermediate layer (~middle depth) |
| **Straightening** (Hosseini & Fedorenko) | Trajectories within an attractor's basin become geometrically linear, enabling prediction by extrapolation | Same intermediate layer |

### The Unified Picture

An LLM processing input executes three coupled operations:

1. **Routes** the input into a stratum — a low-dimensional semantic patch (Li & Sarwate)
2. **Contracts** the representation toward an attractor inside that stratum, via the layer-wise IFS dynamics (Chytas & Singh)
3. **Continues** generation along a straightened trajectory in the basin, extrapolating linearly to predict the next state (Hosseini & Fedorenko)

The "middle layer" convergence across all three papers (~40–50% network depth) is **not coincidence**. It is the layer where routing-and-contracting completes and extrapolation begins. This is the geometric heart of the network.

### Why This Matters Operationally

This synthesis reframes several LLM phenomena that previously seemed independent:

- **Hallucinations** are not random noise — they are coherent continuations along the *wrong* basin's straightened trajectory. The model commits to a basin and rides it, even when wrong.
- **Model heterogeneity** (your Table 1: DeepSeek improves with iteration, Haiku regresses) reflects basin sharpness. Sharp basins → mutations produce clean signal; diffuse basins → mutations produce noise.
- **Decomposition benefit** depends on whether the task crosses attractor boundaries. Tasks within a single basin lose information at agent handoffs; tasks crossing basins benefit from explicit boundary-marking.
- **The "middle layer" of LLMs** is where every meaningful geometric event happens. Diagnostics targeting this layer extract maximum signal.

The three papers, taken together, describe LLMs as **dynamical systems with discrete basin structure and locally linear trajectories within basins**. Multi-agent systems are operations *on top of* this substrate. The substrate's geometry determines what works.

---

## 2. The Manifold Discontinuity Problem

### The Tension

A single agent operating on a long trajectory has continuous representational substrate. Late-stage reasoning has access to the *geometric residue* of early-stage reasoning — the trajectory itself, not just the surface tokens.

Multi-agent decomposition serializes this trajectory through a **token bottleneck**. Agent A produces text. Agent B reads that text and reconstructs its hidden state from scratch. The geometric residue is destroyed.

This is information loss in the strict Shannon sense. The architectural question: when does what's destroyed actually matter?

### What's Destroyed at the Boundary

Three categories, often conflated:

1. **Surface information** — the words themselves. Preserved by writing well-structured outputs. Solved.
2. **Attractor state** — which basin the model was operating in. *Partially* recoverable: agent B's prompt re-establishes an attractor, but it is a fresh contraction, not a continuation.
3. **Trajectory residue** — curvature history, implicit alternatives considered, geometric memory. **Unrecoverable through tokens.** A single agent that processed steps 1–5 is geometrically different from a fresh agent that reads a summary of steps 1–5.

The trajectory-straightening paper makes this concrete: model-generated continuations have *lower* curvature than ground-truth continuations. The model actively pulls toward straightness. Each agent boundary forces a **re-curving** — agent B starts at high curvature (~120°) and must re-straighten. This is the **re-entry tax**.

### Why Single-Agent Wins on Coherent Tasks

Your DeepSeek result (single-agent: 38.2% SR, one-shot multi-agent: 32.4% SR) is exactly this phenomenon. DeepSeek has sharp attractors and strong trajectory straightening. When allowed to ride a single straightened trajectory, it stays on it. Forcing it through agent boundaries:

- Breaks the straightened trajectory at each handoff
- Forces re-contraction into a fresh attractor (potentially the wrong one)
- Loses geometric residue that disambiguates later decisions
- Adds noise from token-level summarization

For models with weaker geometric structure (Haiku, GPT-4o), this matters less — there is less trajectory residue to destroy. Decomposition's coordination benefit dominates the geometric cost. **The crossover point** between "decomposition helps" and "decomposition hurts" is determined by how much trajectory residue the model maintains within a coherent run.

### Wrong Resolutions to Avoid

- **"Make agent boundaries semantically clean."** Surface-level fix. Doesn't address geometric discontinuity.
- **"Pass hidden states between agents."** Architecturally tempting, geometrically broken. Hidden states are stratum- and prompt-dependent; they don't transfer.
- **"Use a shared memory / scratchpad."** Same problem as standard handoff with more text.
- **"Share embeddings via a vector store."** Embeddings are stratum-dependent. They don't preserve meaning across prompt contexts.

### Right Resolutions

#### 2.1 Minimize Boundaries — Don't Decompose What Doesn't Need It

Decomposition is not free. The current meta-orchestrator likely over-decomposes because LLMs love producing structured plans.

**Principled criterion:** decompose at points where the task itself crosses an attractor boundary. A task transitioning between basins (data-prep → ML → statistics) *benefits* from explicit boundaries. A task within a single basin *suffers* from them.

**Operational move:** detect attractor transitions in single-agent execution traces. Use those as the *correct* decomposition points. The single agent tells you where boundaries should be; decompose only there.

#### 2.2 Aggressive Attractor Re-Establishment at Boundaries

When you must cross a boundary, treat the new agent's prompt as an **attractor-injection device**, not just a task description.

Concretely: prompts should begin with **attractor primers** — dense clusters of induced tokens that establish the basin during prefill. Not "you are a docking expert, your task is...", but rather:

```
AutoDock Vina ligand pose RMSD binding affinity scoring function
receptor-ligand interaction docking grid box exhaustiveness...
```

This makes re-contraction *fast and correct*, minimizing trajectory disturbance.

#### 2.3 Preserve Trajectory Information Through Structured Handoffs

You cannot pass the hidden trajectory, but you can pass artifacts that reconstruct its task-relevant projection:

- **Decisions made and alternatives rejected.** Force implicit alternatives into surface form: "I considered X, Y, Z; chose X because..."
- **Confidence per claim.** Geometric residue includes path curvature. Surface tokens don't transmit this. Add structured confidence annotations.
- **Boundary state explicitly.** Agent A reports operating context ("operating in molecular-dynamics mode with implicit-solvent assumptions"). Agent B's prompt incorporates this.

#### 2.4 Larger Agents, Fewer Boundaries

Within an agent's step budget, the model maintains continuous trajectory. Between agents, it breaks. Therefore:

**Agent boundaries should be rare and high-stakes; intra-agent computation should be generous.**

A two-agent workflow with 256 steps each can outperform a six-agent workflow with 64 steps each. This contradicts the "atomic agents, one job per agent" instruction in your meta-orchestrator prompt — that principle is geometrically wrong. The correct principle: **agents should be as large as the underlying attractor**. One agent per coherent basin.

#### 2.5 Population Search Over Topology Granularity

Single-incumbent search mutates prompts and edges but not *granularity* — agent count and responsibility size. This is the most important mutation dimension because it directly trades boundary tax against intra-agent drift. A real population search must explore granularity systematically.

#### 2.6 The Speculative Move: Trajectory-Aware Coupling

Far future. Agent A produces a structured "trajectory summary" — a token-level reconstruction of which hidden states A visited. Agent B's prompt is constructed to *match* those states rather than start fresh. Research, not engineering, but it is the natural endpoint of the geometric framing.

### The Honest Trade-off

The tension does not fully resolve. You trade two failure modes:

- **Single-agent failure:** semantic drift over long trajectories. Even straightened trajectories accumulate small errors. Early decisions cannot be undone.
- **Multi-agent failure:** trajectory destruction at boundaries. Each handoff is a small information catastrophe.

Neither dominates universally. The right architecture **dynamically chooses** per-task and per-stage which failure mode to accept.

The deepest move: **detect attractor transitions in single-agent execution traces, and use those as natural decomposition points.** This changes Mimosa from "evolutionary search over LLM-proposed topologies" to "empirically-grounded decomposition based on observed model dynamics."

---

## 3. Curvature as Selection Pressure

### The Core Idea

Use trajectory discontinuity at agent boundaries as a **fitness penalty** in evolutionary search. Reward workflows whose boundary structure matches the task's underlying attractor structure. Penalize boundaries that destroy geometric continuity without justification.

### What Naive Curvature Measurement Misses

The obvious framing — measure curvature of agent A's output and agent B's intake at the surface — does not work. Hosseini & Fedorenko show input-layer curvature is always ~120° (random initialization). Curvature only becomes informative at the **straightened middle layer**.

The discontinuity is not visible at the surface. It is visible at the middle layer, where trajectory information actually lives.

### Three Practical Signals

#### Signal 1: Straightening Lag

A fresh agent reduces curvature progressively across layers, reaching minimum at the middle layer. This takes "geometric work."

- **Smooth handoff** → agent B reaches minimum curvature early (already on trajectory)
- **Disrupted handoff** → agent B does extra work, reaches minimum later, minimum is shallower

Measurable by probing hidden states across layers. Requires hidden state access.

#### Signal 2: Combined A→B Trajectory Curvature

Treat agent A's final tokens followed by agent B's initial tokens as a single trajectory. Measure curvature at the boundary tokens at the straightened layer.

- **Same basin** → low boundary curvature, smooth continuation
- **Different basins** → sharp angle at handoff, high local curvature

Most direct measurement. Requires hidden state access.

#### Signal 3: Surprisal Spike at Boundary

Hosseini & Fedorenko: curvature correlates with surprisal at middle layers (ρ ≈ 0.2). High surprisal = model finds sequence hard to predict from context.

**This signal requires only logprobs.**

```
D(A, B) = mean_logprob(A_output_tail | B_prompt_context)
```

If A's output looks "normal" to B (low surprisal), they are geometrically aligned. If A's output looks "surprising" to B, B must do basin-jumping work.

### Selection Pressure Formulation

Augment the existing judge score:

```
Fitness(W) = JudgeScore(W) − λ · D_total(W) / num_boundaries
```

Dividing by `num_boundaries` matters — average per-boundary discontinuity, not summed. Otherwise longer workflows are unfairly penalized.

### What This Buys You

#### Mechanistic Explanation of Model Heterogeneity

DeepSeek's sharp basins → boundary discontinuities are *measurably larger* on bad handoffs. Iterative refinement signal is strong. Haiku's diffuse basins → smaller D differences across mutations. Signal is in the noise. **The variance of D across iterations becomes a single-number diagnostic for whether iteration will help on a given model-task pair.**

#### Distinguishing Real vs. Fake Decomposition

A workflow where agents A and B do closely-related work has *low* D — the boundary is artificial. A workflow with *high* D between A and B is doing real basin-switching.

This gives a principled mutation operator: high quality + low D between adjacent agents → **merge them**. Geometrically informed mutation, not LLM-guess.

#### Penalizing Inflated Workflows

Common Mimosa failure mode: meta-orchestrator generates seven agents that all do similar work because LLMs love structure. Six handoffs are pure tax — high D, no benefit. The geometric penalty exposes this immediately.

### The Subtle Issue: Discontinuity Isn't Always Bad

A *meaningful* attractor transition between agents has *high* discontinuity by design. A workflow doing data-prep → ML → statistics *should* have D-spikes at each boundary.

Raw D is not the right signal. What you want is **D conditional on task structure**: discontinuity *beyond what the task demands*.

**Crude implementation:** predict expected D per boundary from task structure. Penalize residual `D_observed − D_expected`.

**Better implementation:** run the task single-agent first. Identify naturally-occurring high-curvature points (actual attractor transitions for this task). Multi-agent boundaries placed at those points are "free." Boundaries elsewhere are taxed.

**The single-agent trajectory's curvature profile is the ground truth for where boundaries should be.**

### API Implementation: Logprobs Are Sufficient

The full curvature measurement requires hidden states (not API-accessible). The **surprisal-based proxy** only needs logprobs and is what should be implemented first regardless.

#### API Availability

| Provider | Logprobs | Notes |
|---|---|---|
| OpenAI | Yes | `logprobs=True`, `top_logprobs` up to 20 |
| DeepSeek | Yes | OpenAI-compatible. **Your strongest model supports this.** |
| Mistral / OpenRouter | Varies | Check per-model |
| Local (vLLM, Ollama, MLX) | Yes | Full access |
| Anthropic | **No** | Requires workaround |

**For your setup: DeepSeek-V3.2 supports the full pipeline.** Where the framework predicts the strongest signal (sharpest basins), you have full logprob access.

#### Concrete Pipeline

For each adjacent pair (A, B) in an executed workflow:

1. Extract A's last 20 tokens of generation
2. Construct B's full prompt context (system + accumulated state + input from A)
3. Call API in score-mode: prompt = B's context, completion = A's tail, request logprobs on completion
4. Compute mean logprob across those 20 tokens — raw discontinuity score

**Normalization** (raw logprobs aren't comparable across prompts):

5. Score B's *own* first 20 generated tokens under its own context — natural baseline
6. `D(A,B) = surprisal(A_tail | B_context) − surprisal(B_natural | B_context)`

Positive D → A's output is more alien to B than B's natural reasoning would be.

#### Cost Analysis

Per workflow: ~2 × num_boundaries small extra calls. For a 5-agent workflow: 4 boundaries × 2 = 8 calls. At DeepSeek pricing: ~$0.001–0.005 each. **Total overhead: $0.01–0.04 per workflow.** Negligible against your $1.7/task iterative-learning cost.

#### Workaround for Claude Agents (No Logprobs)

**Use DeepSeek as scoring probe**, even when Claude executes the agent. The probe model needs only to evaluate whether A's text flows into B's context — semantic question is robust across model families. Single API, full logprob access, your best model.

Alternative: embedding-based discontinuity (worse signal, simpler). Avoid LLM-as-judge for this specific signal — defeats the purpose of a quantitative geometric metric.

### Validation Step Before Deployment

**Critical:** before integrating D into live fitness, validate retrospectively on existing logged traces.

Compute D over your existing ScienceAgentBench runs. Check:

- Does D correlate with judge scores?
- Does D correlate with benchmark Success Rate?
- Does D differ between successful and failed workflows?
- Is D stronger for DeepSeek (sharp basins) than Haiku (diffuse basins)?

If yes to most: framework predicts the result, data confirms. If no: either D is too noisy at the surprisal level, or the framing doesn't translate cleanly to your tasks.

**Cost: zero (re-running scoring on completed traces). Do this first.**

### Implementation Path

| Step | Action | Cost |
|---|---|---|
| 1 | Retrospective D analysis on existing runs | Zero (re-scoring) |
| 2 | If validated: integrate D into fitness as soft penalty | Trivial |
| 3 | Add per-boundary D values to meta-orchestrator improvement prompt | Trivial |
| 4 | A/B comparison: λ=0 (current) vs. λ>0 (geometry-aware) | One eval cycle |

### The Strategic Reframing

A single number — D, the total trajectory discontinuity — does several things:

- Provides geometric explanation for model heterogeneity
- Exposes inflated decomposition
- Guides merge/split mutations principledly
- Connects single-agent and multi-agent modes through a continuous spectrum (D=0 is functionally single-agent, large D is fully decomposed)
- Gives a paper-ready story

**The narrative shift:** Mimosa moves from "evolutionary search over multi-agent prompts" to "empirically-grounded decomposition guided by geometric continuity." Different intellectual position. You are not just searching better — you are searching *for the right thing*.

The fact that this works from logprobs alone — no hidden state access required — is a feature. It means anyone running APIs can reproduce the method.

---

## 4. Implications for the QD Behaviour Descriptor

The quality-diversity loop in [sources/core/selection.py](../sources/core/selection.py)
needs a vector descriptor that tells the archive *what kind* of
workflow a candidate is. Under the geometric framing of §1–§3, that
"kind" decomposes into three independent variables, each tied to a
specific lens above.

### 4.1 The three axes the framing predicts matter

| Axis | What it captures | Lens |
|---|---|---|
| **Granularity** | how many basins the workflow visits | §2.5 — boundary tax vs. intra-agent drift |
| **Basin identity** | which attractor each agent operates in | §1 — stratum + attractor location |
| **Boundary discontinuity D** | how much basin-jumping the workflow forces | §3 — surprisal-based curvature proxy |

A descriptor whose axes do not project onto these variables is
measuring something the framing predicts is *not* what determines
performance. Two workflows with identical descriptor coordinates
under such a descriptor can still occupy distinct basins, and the
archive cannot tell them apart.

### 4.2 What the current descriptor covers

The descriptor implemented in
[sources/core/code_features.py](../sources/core/code_features.py)
parses the workflow AST and returns
`[n_agents, n_edges, n_branches, prompt_chars]`. Mapping to §4.1:

- `n_agents`, `n_edges`, `n_branches` are coarse proxies for
  **granularity**. They quantify how many handoffs the workflow
  imposes, which directly trades against the boundary tax.
- `prompt_chars` is a degenerate stand-in for **basin identity** — it
  conflates "long prompt" with "specifically-anchored prompt." A
  5000-char generic instruction and a 5000-char domain-saturated
  attractor primer (§2.2) land in entirely different basins but
  score identically.
- **D is absent.** Boundary discontinuity is not captured at all.

So the descriptor today covers row 1 of the table credibly, row 2
weakly, and row 3 not at all. This is acceptable as a first
non-collinear-with-fitness descriptor — strictly better than
`[reward, cost, iteration]` — but it is not what the framing says
should ultimately drive selection.

### 4.3 Tiered upgrade path

If the framing proves load-bearing on benchmarks, three incremental
moves bring the descriptor into alignment.

**Tier 1 — Drop `prompt_chars`, add prompt-vocabulary entropy.**
Char count is a poor proxy for prompt specificity. Token-set entropy
over extracted prompt strings approximates "is this prompt
domain-saturated" (high-entropy = many distinct tokens = strong
basin anchor) without an embedding model. ~5 LoC, no new deps.

**Tier 2 — Replace `prompt_chars` with prompt-content embedding.**
Extract the string constants passed to `SmolAgentFactory(...)`,
embed each with MiniLM (already a dependency), pool, and project to
3 dims via a fixed random projection. Captures basin identity
proper. ~30 LoC, deterministic across sessions, no cold-start
problem. Descriptor becomes
`[n_agents, n_edges, n_branches, emb_0, emb_1, emb_2]`.

**Tier 3 — Add boundary discontinuity D.**
Per §3, compute D for each adjacent agent pair via logprob surprisal
using DeepSeek as a probe model. Append `D_mean, D_variance` to the
descriptor. This is the geometrically-principled signal the framing
recommends but requires execution-trace plumbing and logprob API
access. Bigger build; paper-grade story.

### 4.4 What this changes about the admit gate

The admit gate in `_try_admit` is descriptor-agnostic — it
Pareto-checks on `(reward_uncapped, novelty_score)` regardless of
how novelty was computed. So tier-1/tier-2/tier-3 changes to the
descriptor flow through the gate without further modifications. The
choice of descriptor is the load-bearing decision; the gate plumbing
is already there.

### 4.5 What we are *not* claiming

This section does not assert the framing is correct, only that *if*
it is correct, the descriptor axes follow as stated. The framing
remains theory until D-on-existing-traces is validated. The current
implementation deliberately keeps the descriptor structural and
cheap so the loop runs with no new dependencies while the validation
step is pending.

---

## Closing Architectural Position

The three geometry papers describe what is happening *inside* a single forward pass. Mimosa orchestrates forward passes. The architectural sweet spot is not redesigning the agent execution layer — it is **instrumenting Mimosa to measure what is happening inside each forward pass and feed those measurements back into orchestration.**

Three measurements, increasing in implementation cost:

1. **Attractor signature mass** — diagnose individual agent capture (cheap, per-agent)
2. **Boundary discontinuity D** — diagnose multi-agent coherence (cheap, per-boundary, the highest-leverage move)
3. **Stratum-aware archive retrieval** — fix the 0% archive hit rate (medium cost, longer-term)