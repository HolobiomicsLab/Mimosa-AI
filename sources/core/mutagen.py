
import random
import time

from .workflow_info import WorkflowInfo

from sources.cli.pretty_print import (
    print_ok, print_warn, print_err, print_info,
    CYAN, GREEN, YELLOW, RED, DIM, RESET, BOLD,
)

"""
Mutagen: self-contained perturbation sampler for LLM-guided neuroevolution.
"""

import random
import time


class Perturbation:
    """
    A fully composed, prompt-ready perturbation sampled across all axes.
    Immutable value object — VariationEngine reads it.
    """

    def __init__(self, combo: dict[str, tuple[str, str]]):
        self._combo = combo

    @property
    def signature(self) -> dict[str, str]:
        """Hashable identity of this combo (axis → name, not description)."""
        return {k: v[0] for k, v in self._combo.items()}

    def format_block(self) -> str:
        """
        Render the structural/analytical axes as a prompt section.
        """
        sections = [
            ("THINKING LENS",      self._combo["lens"]),
            ("STRUCTURAL PRESSURE", self._combo["structure"]),
            ("CONSTRAINT",         self._combo["constraint"]),
            ("DECOMPOSITION ANGLE", self._combo["decomposition"]),
        ]
        return "\n".join(
            f"**{title}** [{name}]: {desc}"
            for title, (name, desc) in sections
        )

    def get_voice_framing(self) -> tuple[str, str]:
        """
        Return (opening_frame, closing_frame) that set the LLM's tone for
        the entire prompt.
        """
        name, desc = self._combo["voice"]
        opening = (
            f"**YOUR APPROACH THIS ROUND** [{name}]: {desc}\n"
            f"Adopt this mindset for your entire analysis below. "
            f"Let it shape what you notice and what you change.\n"
        )
        closing = f"Remember: stay in [{name}] mode. {desc.split('.')[0]}."
        return opening, closing

    @property
    def voice_name(self) -> str:
        return self._combo["voice"][0]


class Mutagen:
    """
    Perturbation sampler: owns the axis pools and the sampling/dedup logic.

    Each axis is sampled independently → combinatorial diversity grows
    multiplicatively even with small per-axis pools.  A sliding-window
    deduplication buffer prevents the same combination from being retried
    in close succession, nudging the search away from local minima.
    """

    def __init__(self, max_memory: int = 25):
        self._rn_seed: int = int(time.time() * 1000) % 2**32
        self._tried_combinations: list[dict[str, str]] = []
        self._max_memory = max_memory  # sliding window size for dedup

        # ── Axis 1: HOW to reason about the problem ──────────────────────────
        # These are epistemic stances, not prescriptions for what to build.
        self.thinking_lenses: list[tuple[str, str]] = [
            ("first_principles",
             "Reason from first principles. Forget all conventional multi-agent patterns. "
             "Ask: what is the minimal information flow that solves this task? Build up from there."),
            ("adversarial_red_team",
             "Think like a red-teamer trying to break this workflow. Where would you inject "
             "a misleading input to cause cascading failure? Design the fix around that attack vector."),
            ("information_theoretic",
             "Think in terms of information: where is entropy highest? Where does the workflow "
             "lose bits? Where does it need bits it doesn't have? Redesign information flow, not agent count."),
            ("systems_biology",
             "Model this workflow as a biological system — agents are cells, messages are signals. "
             "What feedback loops are missing? Where is the homeostasis? What's the stress response?"),
            ("cognitive_science",
             "Apply cognitive science: each agent has limited working memory and attention. "
             "What cognitive biases might agents exhibit? How to prevent confirmation bias in sequential chains?"),
            ("engineering_reliability",
             "Think like a reliability engineer. What is the single point of failure? "
             "What's the mean time to recovery? Design for graceful degradation, not just happy-path."),
            ("economist",
             "Think about marginal returns. Which agent adds the least value per token spent? "
             "Where is over-investment in one capability starving another? Reallocate the compute budget."),
            ("ecologist",
             "Think about the workflow as an ecosystem. Are agents competing for the same niche? "
             "Is there an empty niche that no agent fills? What would a keystone agent look like?"),
            ("philosopher_of_science",
             "What are the hidden assumptions? What hypothesis is the current workflow implicitly "
             "testing, and is there a completely different hypothesis it should test instead?"),
            ("reverse_engineer",
             "Start from the desired output and work backwards. What's the last transformation needed? "
             "The one before that? Design the pipeline in reverse — you may find the forward order was wrong."),
        ]

        # ── Axis 2: WHAT topological property to change toward ───────────────
        # Abstract pressures, not prescriptions.  The LLM decides how to honour them.
        self.structural_mutations: list[tuple[str, str]] = [
            ("widen_before_narrow",
             "The workflow should explore more before committing. "
             "Generate multiple candidate paths early, then converge late."),
            ("narrow_before_widen",
             "The workflow should commit to a direction early, then explore "
             "variations within that commitment. Depth-first, not breadth-first."),
            ("flatten_hierarchy",
             "Remove one level of indirection. If agents delegate to agents, "
             "can the inner agents do the outer agent's job directly?"),
            ("add_tension",
             "Introduce constructive tension: two agents with deliberately "
             "different objectives whose outputs must be reconciled."),
            ("remove_bottleneck",
             "Find where all information flows through a single agent. "
             "Is that bottleneck necessary, or can work flow in parallel around it?"),
            ("invert_control",
             "Flip the control flow. Instead of early agents deciding what "
             "late agents do, let late agents pull what they need."),
            ("shorten_path",
             "The critical path is probably too long. What happens if you "
             "remove the two least essential agents entirely?"),
            ("add_memory",
             "The workflow is memoryless between steps. What if agents could "
             "read a shared scratchpad updated by all previous agents?"),
            ("decouple_stages",
             "Agents are too coupled — one agent's failure format dictates "
             "the next agent's behavior. Add an explicit interface contract between stages."),
            ("challenge_agent_count",
             "Is the number of agents justified? Would fewer agents with "
             "richer prompts outperform many agents with thin prompts?"),
        ]

        # ── Axis 3: HOW the prompt is voiced ────────────────────────────────
        # Changes the LLM's starting manifold — same facts, different aperture.
        self.prompt_voices: list[tuple[str, str]] = [
            ("surgical",
             "Be extremely precise. Identify the single weakest link in the chain. "
             "Make the minimum viable change that addresses it. No flourish, no extra agents."),
            ("creative_divergent",
             "Be bold and unconventional. The current approach has fundamental assumptions "
             "that may be wrong. What's a completely different way to structure this?"),
            ("skeptical_minimalist",
             "You are skeptical that more agents help. Prove that every agent earns its keep. "
             "If you can't justify an agent's existence in one sentence, eliminate it."),
            ("teacher_explaining",
             "Explain to yourself, step by step, what each agent is supposed to learn from "
             "the previous one. Where does that learning break down? Fix the pedagogy."),
            ("debugger_systematic",
             "Approach this like a systematic debugging session. Form a hypothesis about why "
             "it failed. Design a minimal experiment (workflow change) that tests exactly that hypothesis."),
            ("architect_refactoring",
             "This workflow accumulated accidental complexity. Refactor it. What's the clean "
             "architecture hiding underneath the current mess?"),
            ("provocateur",
             "What if the opposite of the current approach works better? If agents go A→B→C, "
             "what happens with C→B→A? If there are 5 agents, what about 2? Challenge everything."),
            ("pragmatist",
             "Forget elegance. What's the dirtiest, simplest hack that would actually make this work? "
             "Sometimes a crude solution that works beats a beautiful one that doesn't."),
        ]

        # ── Axis 4: WHAT to stop / question / remove ────────────────────────
        # Constraint inversions trigger lateral thinking by banning a default move.
        self.constraint_inversions: list[tuple[str, str]] = [
            ("ban_sequential",
             "CONSTRAINT: Do NOT use a purely sequential pipeline this time. "
             "At least two agents must operate on the same input independently."),
            ("ban_single_pass",
             "CONSTRAINT: The workflow must revisit at least one agent's output "
             "with new information. No single-pass-through designs."),
            ("ban_generic_agents",
             "CONSTRAINT: Every agent name must reference a specific domain concept, "
             "not a generic role like 'validator' or 'analyzer'. Specificity forces better prompts."),
            ("ban_long_prompts",
             "CONSTRAINT: Each agent's instruction must be under 150 words. "
             "If you need more words, you need more agents. Brevity forces clarity."),
            ("ban_json_handoff",
             "CONSTRAINT: Agents must communicate in natural language this time, "
             "not structured JSON. This forces robust understanding over brittle parsing."),
            ("ban_error_agents",
             "CONSTRAINT: No dedicated error-handling agents. Every agent must handle "
             "its own errors internally. Fewer agents, more resilient agents."),
            ("question_goal_decomposition",
             "CONSTRAINT: Re-read the goal. Are you solving the right problem? "
             "Maybe the previous workflow was well-built but solving a subtly wrong interpretation."),
            ("remove_weakest_agent",
             "CONSTRAINT: You must remove the lowest-value agent from the previous workflow "
             "AND still improve performance. Prove the workflow was over-engineered."),
            ("question_tool_assignment",
             "CONSTRAINT: Re-examine which tools each agent has. Are agents using "
             "the right tools? Would swapping tool assignments between agents help?"),
        ]

        # ── Axis 5: HOW to conceptually decompose the problem ────────────────
        self.decomposition_strategies: list[tuple[str, str]] = [
            ("by_uncertainty",
             "Decompose by uncertainty: separate what you know from what you don't. "
             "Agents that gather information vs. agents that reason about it."),
            ("by_timescale",
             "Decompose by timescale: separate fast operations (lookups, transforms) "
             "from slow ones (reasoning, search). Don't mix them in one agent."),
            ("by_reversibility",
             "Decompose by reversibility: separate destructive operations (summarization, "
             "filtering) from reversible ones. Put quality gates before destructive steps."),
            ("by_stakeholder",
             "Decompose by stakeholder: who consumes the output? Design backward from "
             "their needs, not forward from the data source."),
            ("by_failure_mode",
             "Decompose by failure mode: group operations that fail together, "
             "separate those that fail independently. Each failure domain gets its own recovery."),
            ("by_data_type",
             "Decompose by data type: numeric data, text data, and code should flow "
             "through different specialized agents, not one generalist."),
            ("by_confidence",
             "Decompose by confidence level: high-confidence outputs go straight through, "
             "low-confidence outputs go through additional validation."),
            ("by_abstraction_level",
             "Decompose by abstraction level: strategic planning agents vs. tactical execution "
             "agents vs. verification agents. Don't mix levels."),
        ]

    def compose(self, seed: int | None = None) -> Perturbation:
        """
        Sample one item from every axis and return a ready-to-use Perturbation.

        Tries up to 10 times to avoid a recently-seen combination.  If all
        retries are exhausted (unlikely with 5 independent axes), the last
        generated combo is returned anyway — diversity is a soft constraint.

        Args:
            seed: explicit RNG seed; if None, uses and increments internal seed.
        """
        effective_seed = seed if seed is not None else self._rn_seed
        self._rn_seed += 1

        rng = random.Random(effective_seed)
        combo = self._sample_unique(rng)
        return Perturbation(combo)

    def format_tried_strategies_block(self, n: int = 5) -> str:
        """
        Render the last *n* tried combinations as a prompt section so the LLM
        can consciously avoid retreading the same ground.
        """
        if not self._tried_combinations:
            return ""
        recent = self._tried_combinations[-n:]
        lines = ["## PREVIOUSLY TRIED DIRECTIONS (avoid repeating these):"]
        for i, combo in enumerate(recent, 1):
            tags = ", ".join(f"{k}={v}" for k, v in combo.items())
            lines.append(f"  {i}. [{tags}]")
        lines.append("Take a genuinely different direction this time.\n")
        return "\n".join(lines)

    def _sample_unique(self, rng: random.Random, max_retries: int = 10) -> dict[str, tuple[str, str]]:
        """Sample axis values, retrying if the signature was recently tried."""
        combo: dict[str, tuple[str, str]] = {}
        for _ in range(max_retries):
            combo = {
                "lens":         rng.choice(self.thinking_lenses),
                "structure":    rng.choice(self.structural_mutations),
                "voice":        rng.choice(self.prompt_voices),
                "constraint":   rng.choice(self.constraint_inversions),
                "decomposition": rng.choice(self.decomposition_strategies),
            }
            sig = {k: v[0] for k, v in combo.items()}
            if sig not in self._tried_combinations:
                self._register(sig)
                return combo
        # Exhausted retries — register and return whatever we have
        self._register({k: v[0] for k, v in combo.items()})
        return combo

    def _register(self, sig: dict[str, str]) -> None:
        """Add a signature to the sliding window, evicting the oldest if full."""
        self._tried_combinations.append(sig)
        if len(self._tried_combinations) > self._max_memory:
            self._tried_combinations.pop(0)