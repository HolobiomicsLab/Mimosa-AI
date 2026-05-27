
"""
VariationEngine: search-schedule and prompt assembly for LLM-guided workflow evolution.
"""

from .workflow_info import WorkflowInfo

from sources.cli.pretty_print import (
    print_info,
    CYAN, GREEN, YELLOW, RED, DIM, RESET, BOLD,
)

import numpy as np

class VariationEngine:
    """
    Orchestrates iterative LLM-driven workflow search via structured prompt mutation.

    Each call to mutation_prompt() or crossover_prompt() produces a prompt that:
      - Anchors the LLM on concrete execution feedback (agent answers, judge eval).
      - Injects a freshly-sampled multi-dimensional perturbation (via Mutagen)
      - Applies a phase-aware annealing schedule that governs exploration breadth
        and permitted topology complexity as iterations progress.
    """

    def __init__(self):
        self.diagnosis_history = []

    def _sample_agent_count(self, progress: float, lo: int, hi: int, concentration: float = 4.0) -> int:
        """
        Sample agent random count within [lo, hi], biased upward by progress.
        """
        if lo == hi:
            return lo
        target_mean = lo + progress * (hi - lo)
        p = np.clip((target_mean - lo) / (hi - lo), 0.05, 0.95)
        alpha = p * concentration
        beta = (1 - p) * concentration
        prob = np.random.beta(alpha, beta)
        return lo + int(np.random.binomial(hi - lo, prob))

    def _diagnosis_similarity(self, a: str, b: str) -> float:
        """Token-level Jaccard. Cheap stand-in for embedding cosine; no model load."""
        if not a or not b:
            return 0.0
        tokens_a = set(a.lower().split())
        tokens_b = set(b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    def _compute_stagnation(self, window: int = 5) -> float:
        """Mean pairwise Jaccard over the last `window` diagnoses, ∈ [0, 1].

        High value ⇒ the LLM-mutator is cycling on similar failure modes
        (mode collapse). Used to drive stochastic phase regression.
        """
        recent = self.diagnosis_history[-window:]
        if len(recent) < 2:
            return 0.0
        sims = [
            self._diagnosis_similarity(recent[i], recent[j])
            for i in range(len(recent))
            for j in range(i + 1, len(recent))
        ]
        return sum(sims) / len(sims) if sims else 0.0

    def _sample_phase_regression(
        self, stagnation: float, max_regression: float = 0.45, concentration: float = 4.0
    ) -> float:
        """Beta-sampled regression amount, mean ≈ stagnation · max_regression.

        Mirrors `_sample_agent_count`'s shape so the schedule keeps some
        randomness instead of snapping deterministically on the threshold.
        """
        if stagnation <= 0.0:
            return 0.0
        p = float(np.clip(stagnation, 0.05, 0.95))
        alpha = p * concentration
        beta = (1 - p) * concentration
        return float(np.random.beta(alpha, beta)) * max_regression

    def _get_temperature_phase(
        self,
        iteration_count,
        max_iterations=10,
        score=0.0,
        alpha=0.5,
        last_failure_mode=None,
    ):
        """
        EV-ranked phase schedule: single-agent → tools → prompt → (only if needed)
        decomposition → integrative tuning → polish.

        When recent failure diagnoses cluster (LLM mutator cycling on a basin),
        stochastically regress progress so the next iteration drops back into a
        more exploratory phase. Regression magnitude is Beta-sampled.
        """
        if max_iterations <= 1:
            progress = 0.5
        else:
            progress = (iteration_count / max(max_iterations - 1, 1)) ** (1 - alpha * score)

        stagnation = self._compute_stagnation()
        regression = self._sample_phase_regression(stagnation)
        progress = max(0.0, progress - regression)

        stuck_note = (
            f"[stagnation={stagnation:.2f} → regress {regression:.2f}; "
            "recent diagnoses cluster — try a structural change, not a prompt tweak]\n"
            if regression > 0.05 else ""
        )

        i, n, p = iteration_count + 1, max_iterations, progress
        diag = stuck_note + (f"Prior diagnosis: {last_failure_mode}\n" if last_failure_mode else "")

        if progress < 0.25:
            n_agents = self._sample_agent_count(progress, 1, 2)
            return (
                f"## PHASE: SEED  [{i}/{n} | {p:.0%}]\n{diag}"
                f"Priority mutations: prompt only. Max agent count: {n_agents}.\n"
                "Goal: build the strongest possible single-agent workflow and see how far it gets.\n"
                "Do: one agent with a domain-specific role prompt; attach every relevant tool.\n"
            )
        elif progress < 0.50:
            n_agents = self._sample_agent_count(progress, 1, 3)
            return (
                f"## PHASE: ANCHOR  [{i}/{n} | {p:.0%}]\n{diag}"
                f"Permitted mutations: prompt (primary), topology, tools. Agent count: {n_agents}.\n"
                "Goal: rewrite the agent's prompt or decompse verification/steps across agents. This phase gets the largest iteration budget.\n"
                "Each variant should change ONE thing from the previous best:\n"
                "  domain vocabulary, role definition, required output shape, level of formality,\n"
                "  explicit step listing, requirement to preserve task wording verbatim.\n"
                "Why: prompts steer the model into the right way of thinking about the task. "
            )
        elif progress < 0.65:
            n_agents = self._sample_agent_count(progress, 2, 5)
            return (
                f"## PHASE: DECOMPOSE  [{i}/{n} | {p:.0%}]\n{diag}"
                f"Permitted mutations: topology, prompt, handoff format. Agent count: {n_agents} maximum.\n"
                "Why: see if multi-agent shape might outperforms a well-prompted single agent."
            )
        elif progress < 0.85:
            n_agents = self._sample_agent_count(progress, 3, 7)
            return (
                f"## PHASE: ENGAGE  [{i}/{n} | {p:.0%}]\n{diag}"
                f"Permitted mutations: prompt, handoff format, agent count (restricted) {n_agents} maximum.\n"
                "Goal: tighten the workflow you have; remove any agent that isn't earning its place."
                "Solver / executor / single-purpose agents need sharper, more specific prompts so they commit confidently to one approach.\n"
            )
        else:
            n_agents = self._sample_agent_count(progress, 5, 7)
            return (
                f"## PHASE: POLISH  [{i}/{n} | {p:.0%}]\n{diag}"
                f"Permitted mutations: prompt only. Agent count: {n_agents}.\n"
                "Goal: one prompt fix per iteration, targeting the single most concrete failure.\n"
                "Do: trace the failure to one agent and edit that agent's prompt."
                "Don't: change topology, add tools, or rewrite multiple prompts at once.\n"
                "Why: at this point, any large change risks breaking what works. Make small, precise edits."
            )

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_agent_answers(wf_state: dict | None) -> str:
        """Flatten per-agent answers from workflow state into a readable string."""
        if not wf_state or "answers" not in wf_state:
            return "No agent answers captured."
        answers = wf_state["answers"]
        if isinstance(answers, list):
            return "\n".join(
                f"agent {name}: {str(answer)[:2048]}..."
                for name, answer in zip(wf_state["step_name"], answers, strict=True)
            )
        return str(answers)

    # ── Prompt builders ───────────────────────────────────────────────────────

    def seed_genome_prompt(self, goal: str) -> str:
        """
        Prompt for very first workflow generation (generation 0).
        """
        return (
            "## First workflow generation\n"
            f"Goal to assemble a workflow for:\n{goal}\n"
            "Build the minimal workflow for the task with maximum 2 agents.\n"
        )

    def mutation_prompt(
        self,
        goal: str,
        wf_info: WorkflowInfo | None,
        genotype: str | None,
        run_stderr: str,
        iteration_count: int,
        max_iterations: int = 10,
    ) -> str:
        """
        Build a prompt for one mutation step in the evolutionary search.

        The prompt has three layers:
          1. Voice framing   — sets the LLM's reasoning tone for this iteration.
          2. Execution grounding — previous code, agent answers, and judge eval.
        """
        score      = wf_info.overall_score        if wf_info else 0.0
        diagnosis  = wf_info.abstracted_diagnosis if wf_info else ""
        wf_state   = wf_info.state_result         if wf_info else None

        # ── Execution evidence ───────────────────────────────────────────────
        agent_answers = self._extract_agent_answers(wf_state)
        diagnosis_block = (
            diagnosis.strip()
            if diagnosis and diagnosis.strip()
            else (run_stderr or "").strip()
            or "No diagnosis captured."
        )
        self.diagnosis_history.append(diagnosis_block[:512])
        phase_block   = self._get_temperature_phase(iteration_count, max_iterations, score)

        if genotype is None:
            body = "Previous attempt failed. Fix syntax errors."
        else:
            body = "\n".join([
                "## WORKFLOW EVOLUTION STEP",
                "",
                phase_block,
                "",
                "Your previous workflow attempt did not reach the success threshold.",
                f"Goal: {goal}",
                "",
                "## Previous workflow code:",
                "<python>",
                genotype,
                "</python>",
                "",
                "## EXECUTION RESULTS:",
                "<agents_answers>",
                agent_answers,
                "</agents_answers>",
                "<diagnosis>",
                "",
                diagnosis_block,
                "</diagnosis>",
                "",
                "## Task: apply a single mutation to the workflow code.",
            ])

        return "\n".join([
            f"Attempt {iteration_count + 1} of workflow generation.",
            body,
            "\nTarget goal:",
            goal,
        ])

    def crossover_prompt(
        self,
        goal: str,
        wf_infos: list[WorkflowInfo],
        genotypes: list[str],
        run_stderrs: list[str],
        iteration_count: int,
        max_iterations: int = 10,
    ) -> str:
        """
        Build a prompt for a crossover step: recombine N parent workflows into one offspring.
        Parents are sorted best → worst so the LLM sees the strongest candidates first.
        """
        # ── Assemble parent records ──────────────────────────────────────────
        parents = []
        for i, (wf_info, genotype, stderr) in enumerate(
            zip(wf_infos, genotypes, run_stderrs)
        ):
            score     = wf_info.overall_score        if wf_info else 0.0
            diagnosis = wf_info.abstracted_diagnosis if wf_info else ""
            # Layer 1: rubric-blind diagnosis instead of raw judge log.
            # Fall back to stderr tail only when no diagnosis exists.
            diagnosis = diagnosis.strip() or (stderr or "").strip()[-1024:]
            answers   = self._extract_agent_answers(wf_info.state_result if wf_info else None)
            parents.append({
                "index":     i + 1,
                "score":     score,
                "code":      genotype,
                "diagnosis": diagnosis,
                "answers":   answers,
            })

        # Best parents first — LLM primacy bias helps inherit strong traits
        parents.sort(key=lambda p: p["score"], reverse=True)

        parents_block = []
        for p in parents:
            parents_block.append("\n".join([
                f"### Parent {p['index']} (score={p['score']:.2f})",
                "<python>",
                p["code"] or "(construction failed)",
                "</python>",
                "<agents_answers>",
                p["answers"],
                "</agents_answers>"
            ]))

        return "\n".join([
            f"Attempt {iteration_count + 1}: CROSSOVER — synthesize from {len(parents)} parent workflows.",
            f"Goal: {goal}",
            "",
            "## PARENT WORKFLOWS:",
            "\n\n".join(parents_block),
            "",
            "## YOUR TASK:",
            "Do not pick one parent and patch it. Genuinely recombine.",
            "1. Identify which structural decisions worked in each parent (look at agent answers, not just score).",
            "2. Identify which decisions failed and why.",
            "3. Build a new workflow that inherits the strongest sub-structures across parents",
            "   and discards the weakest, even if that means a topology none of the parents used.",
            "",
            f"Target goal:\n{goal}",
        ])


if __name__ == "__main__":
    ve = VariationEngine()
    assert ve._compute_stagnation() == 0.0, "empty history → no stagnation"

    ve.diagnosis_history = ["verification failed: claim X not supported"] * 5
    s_high = ve._compute_stagnation()
    assert s_high > 0.9, f"clustered diagnoses → high stagnation, got {s_high}"

    ve.diagnosis_history = [
        "verification failed: claim X not supported",
        "tool returned empty result for query Y",
        "import error: matplotlib missing",
        "agent emitted no answer at step 2",
        "syntax error in generated code",
    ]
    s_low = ve._compute_stagnation()
    assert s_low < 0.3, f"diverse diagnoses → low stagnation, got {s_low}"

    np.random.seed(0)
    samples = [ve._sample_phase_regression(0.8) for _ in range(500)]
    mean = sum(samples) / len(samples)
    assert 0.28 < mean < 0.44, f"E[regression@0.8] ≈ 0.36, got {mean:.3f}"

    ve.diagnosis_history = ["stuck same failure"] * 6
    block = ve._get_temperature_phase(iteration_count=30, max_iterations=35)
    assert "stagnation=" in block, "high-stagnation phase prompt should mention it"

    print(f"✅ stagnation high={s_high:.3f}, low={s_low:.3f}, "
          f"mean regression@0.8={mean:.3f}")
    print("✅ variation_engine smoke check passed.")