# Mimosa — The Dynamical-Systems View · Video Narration Script

Timed to `mimosa_dynamical_systems_3d.mp4` (9 scenes, ~5:00). Captions carry the
argument unvoiced; this is the spoken version, ~140 wpm.

---

**Scene 1 — Title & state flow (0:00–0:30)**

Forget the transcript for a moment. An agent is a state, evolving in a very high-dimensional space. The tokens you read are its shadow — a one-dimensional projection of something moving up here. And the moment you see it that way, you inherit an entire language with theorems in it: dynamics.

**Scene 2 — State, map, initial condition (0:30–1:05)**

The setup is minimal. One update rule, F — the frozen weights — applied over and over to a state. The prompt is not an instruction. It is an initial condition. Same F, different starting point: a different orbit, a different fate. Everything else in this video follows from that one reframing.

**Scene 3 — The landscape (1:05–1:50)**

The state space is not flat. It is a landscape of basins — code here, prose there, math over there. A prompt drops a state onto this landscape, and routing is just falling. Two nearby initial conditions land in the same well — that is why paraphrases behave alike: transverse contraction. But start near a ridge, and a tiny nudge picks a different basin entirely. The ridges are the key decisions. Sparse. And sensitive.

**Scene 4 — Inside a basin: lock-in and escape (1:50–2:30)**

Now suppose the state falls into the wrong well — one bad key decision. Here is the cruel part: generation conditions on its own output, so the occupied well deepens as the agent keeps talking. That is lock-in, drawn as geometry. Perturbations climb the wall and fall back; escape probability decays like rho to the t. The output stays coherent, fluent — and trapped. Terminal collapse is not error pile-up. It is capture.

**Scene 5 — Itinerancy: the uncontrolled jump (2:30–3:05)**

Basins are not forever. Along the flow the dynamics are expansive: small wobbles grow while the walls hold — until they don't. Watch: an unplanned exit over the rim, then confident contraction into the neighboring well. That is a hallucination, in one picture. Left alone, the system itinerates — it *will* jump between basins. The only question is whether anyone chooses where.

**Scene 6 — A workflow is a chain of dynamical systems (3:05–4:00)**

So here is what a multi-agent system actually is. Agent A and agent B are two flows, each with its own landscape. A runs: routes, contracts, accumulates a whole trajectory of state. Then the handoff — and watch carefully. The full high-dimensional state collapses onto a line. Tokens: a one-dimensional projection. The trajectory is gone; its residue cannot ride the rail. On the far side, B is re-injected as a fresh initial condition. It must re-contract from scratch — and can misroute doing it. But look what B gets in exchange: a fresh landscape. No deepened well. No inherited coherence pressure. Independent errors. This is a hybrid dynamical system — continuous flows punctuated by discrete, lossy jumps — and every multi-agent framework, whatever its diagram, is exactly this object. Design is choosing the jumps.

**Scene 7 — Where to cut the chain (4:00–4:35)**

Run the task as one flow and watch where it wants to go: it rides well one, then crosses the saddle — the transition the task itself demands. At the saddle, the state is between basins: maximum entropy, minimum committed structure. A jump placed there costs nothing the system wasn't already paying, and resets everything worth resetting. A jump mid-well shatters live state and forces a re-route the task never asked for. Cut where the dynamics already want to jump.

**Scene 8 — Evolution: searching for the saddle, blind (4:35–5:00)**

One catch: the landscape is invisible — model-specific, and unseen before the first run. So evolution samples it. Place a jump, run the chain, score the outcome: a black-box query of the geometry. Selection keeps the cuts that score, and they pile up at the saddle. This is not prompt tuning. It is system identification of an invisible landscape, from rollouts alone.

**Scene 9 — Close**

One agent: a flow. Failure: capture by the wrong basin. A workflow: flows chained by lossy jumps. Design: placing jumps at saddles. Evolution: identifying the landscape. A workflow is a controlled-itinerancy schedule.
