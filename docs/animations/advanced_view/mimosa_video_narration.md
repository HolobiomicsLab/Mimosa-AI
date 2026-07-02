# Mimosa Framework — Video Narration Script

Timed to `mimosa_framework_video.mp4` (10 scenes, ~8 minutes). On-screen captions
carry the argument if unvoiced; this script is the spoken version, ~140 wpm.

---

**Scene 1 — Title & the phenomenon (~0:00–0:35)**

Give a language model a long, open-ended task — analyze a dataset, plan a synthesis, design an experiment — and you'll see a characteristic failure. The agent starts well. Then it makes one early mistake. And instead of recovering, it elaborates that mistake, fluently and with growing confidence, until the entire trajectory is unsalvageable. We call this terminal reasoning collapse. This video explains why it happens — and what it tells us about how to build agent systems.

**Scene 2 — The folk theory (~0:35–1:20)**

There's a folk theory that seems to explain it instantly. Suppose every token has some small independent probability of derailing the task. Then survival decays exponentially with length — this is LeCun's argument that autoregressive models are structurally doomed. But the argument rests on two hidden assumptions: that errors are uniform over positions, and independent across them. Transformers violate both — and in opposite directions.

**Scene 3 — The two-rate model (~1:20–2:15)**

In natural text, only five to ten percent of tokens genuinely depend on long-range context. These are the key decisions. Perplexity computed on just those tokens predicts long-context performance almost perfectly; ordinary perplexity predicts nothing. So the right model has two error rates, not one: rare, hard decisions — and filler the model gets right for free. And if the number of genuine decisions is bounded, reliability plateaus with length. Long trajectories are not intrinsically doomed. Length is not the enemy. The decision budget is.

**Scene 4 — Three spaces (~2:15–3:05)**

To say what actually fails, keep three spaces distinct. Tokens. Trajectories — complete token sequences. And solutions — what the task actually rewards. A projection, phi, collapses trajectories to solutions: many wordings, one answer. The key object is p-t: the distribution over where the trajectory could still end up, given everything committed so far. Every token reshapes it. Drift is how far accumulated context has pushed this distribution away from good solutions — and it's measurable: truncate, resample, compare.

**Scene 5 — Injection, then lock-in (~3:05–4:10)**

Single agents fail through two coupled mechanisms. Mechanism A: injection. Errors enter at key decision points — forks where the sampled token genuinely selects among futures. Rare, stochastic, identifiable. Mechanism B: lock-in. Generation conditions on its own past output, so the effective objective drifts from solving the task toward staying consistent with what was already said. The same mistake at step fifty and at step five thousand have completely different fates. After capture by a wrong mode, escape probability decays geometrically — recovery needs an external kick the trajectory cannot supply from inside. The root cause: one context serves as both memory and conditioning. Every stored error becomes a self-reinforcing prior — which is why self-critique underdelivers: it inherits the corruption it's supposed to catch.

**Scene 6 — The geometry beneath (~4:10–5:20)**

Why are some steps decisions and most steps transcription? Recent work on representation geometry gives the mechanism. First: the model's semantic space is an archipelago — discrete low-dimensional strata, not a uniform fog. Second: within a stratum, the layers act as contractions — paraphrases collapse to concept attractors. Third: trained models straighten trajectories, predicting by linear extrapolation. Put together: contraction transverse to the trajectory, expansion along it. The system itinerates between basins — and a hallucination is an unplanned basin exit followed by confident contraction into the wrong basin. The key claim: key decisions ARE basin routing events. And they're loud — entropy spikes you can measure from ordinary API logprobs.

**Scene 7 — What a boundary really is (~5:20–6:30)**

Now, multi-agent systems. Strip away the diagrams and every handoff is the same operation: agent A writes something down; agent B starts fresh from that write-up. A lossy jump operator. What it buys: coherence pressure resets. Frozen decisions get a scheduled external kick. And a fresh verifier has independent errors — independence, not intelligence, is what makes verification work. What it costs: A's in-context-learned structure cannot cross the token bottleneck — and re-establishing it is a phase transition: below a critical handoff mass, B stays on pretrained priors entirely. A cliff, not a slope. A's revisable beliefs arrive in B as hard premises — errors don't reset, they harden. And B's re-contraction is itself a new key decision that can misroute. Most analyses count only the benefits.

**Scene 8 — The placement principle (~6:30–7:25)**

The two lists share a currency, and it yields one principle. Run the task single-agent, instrumented. The trajectory rides one basin, crosses a ridge — and the crossing is loud in the logprobs. A boundary placed at that transition adds no new decision — the routing was happening anyway — and destroys the least structure. A boundary placed mid-basin destroys live structure, freezes mid-thought beliefs, and adds a decision the task never required. Cut where the dynamics already want to jump. Reinforcement learning found the same principle from the other side, decades earlier: useful options begin and end at bottleneck states.

**Scene 9 — What evolution actually learns (~7:25–8:20)**

Basin structure is model-specific and invisible before the first rollout — so no hand-designed workflow places cuts correctly, except by luck. The map from topology to performance is black-box, noisy, and expensive: exactly where population search earns its keep. And watch what selection does: it keeps the boundaries that align with real transitions. Evolution here is not prompt optimization at scale — it is system identification of the model's basin structure, from rollouts alone. Start minimal and complexify: add a boundary only when fitness pays for its cost. The evolved archive is a learned prior over decompositions — a map of the model's geometry, bought with compute.

**Scene 10 — Falsifiability & close (~8:20–9:00)**

The framework stands or falls on five predictions — crossover, placement, coupling, threshold, verifiers — and four of them are answerable from already-logged traces, today. If the placement prediction fails, the geometry demotes to a heuristic, and the two-rate accounting stands on its own. One sentence to keep: a workflow is a controlled-itinerancy schedule.
