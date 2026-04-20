
"""
VariationEngine: search-schedule and prompt assembly for LLM-guided workflow evolution.

Owns three concerns:
  1. Annealing schedule  — _get_temperature_phase()  decides what to do each iteration
  2. Prompt construction — seed_genome_prompt / mutation_prompt / crossover_prompt
  3. Mutagen delegation  — all perturbation sampling is done by self.mutagen
"""

from .mutagen import Mutagen
from .workflow_info import WorkflowInfo

from sources.cli.pretty_print import (
    print_info,
    CYAN, GREEN, YELLOW, RED, DIM, RESET, BOLD,
)


class VariationEngine:
    """
    Orchestrates iterative LLM-driven workflow search via structured prompt mutation.

    Each call to mutation_prompt() or crossover_prompt() produces a prompt that:
      - Anchors the LLM on concrete execution feedback (agent answers, judge eval).
      - Injects a freshly-sampled multi-dimensional perturbation (via Mutagen) to
        escape local minima.
      - Applies a phase-aware annealing schedule that governs exploration breadth
        and permitted topology complexity as iterations progress.
    """

    def __init__(self):
        # Mutagen is the single source of truth for perturbation diversity.
        self.mutagen = Mutagen()

    def _get_temperature_phase(
        self,
        iteration_count: int,
        max_iterations: int = 10,
        score: float = 0.0,
        alpha: float = 0.5,
    ) -> str:
        """
        Combined simulated-annealing + complexity-curriculum schedule.

        Two forces evolve together as iteration progress increases:
          • Temperature (exploration → exploitation): how bold/divergent to be.
          • Complexity budget (simple → rich): how many agents / how deep a topology
            is permitted. Complexity is *earned* — early iterations must prove a
            simple baseline works before the search is allowed to grow the workflow.

        The progress exponent is modulated by the current best score (alpha):
          - High score → slower progress → stay exploratory longer.
          - Low score  → faster progress → commit to convergence sooner.

        Phases (by progress ∈ [0, 1]):
          [0.00, 0.10) SEED       — 2-3 agents, prove solvability.
          [0.10, 0.25) BOOTSTRAP  — 3-5 agents, establish linear baseline.
          [0.25, 0.40) DIVERGE    — 4-7 agents, explore structural diversity.
          [0.40, 0.60) SCALE      — 5+ agents, purposeful depth addition.
          [0.60, 0.80) CONVERGE   — freeze count, tighten prompts.
          [0.80, 1.00] POLISH     — frozen architecture, micro-improvements only.
        """
        if max_iterations <= 1:
            progress = 0.5
        else:
            progress = (iteration_count / max(max_iterations - 1, 1)) ** (1 - alpha * score)

        i, n, p = iteration_count + 1, max_iterations, progress

        if progress < 0.10:
            return (
                f"## PHASE: SEED  [iteration {i}/{n}  |  progress {p:.0%}]\n"
                "Complexity budget: 2–3 agents maximum.\n"
                "Objective: prove the task is solvable at all with the minimum viable workflow.\n"
                "Rules:\n"
                "  • Build the simplest possible chain that could plausibly produce the answer.\n"
                "  • No error-handling agents, no validators, no critics — just the core transformation.\n"
                "  • If a single agent can attempt the whole task, do that first.\n"
                "Why: debugging a 1-agent failure is 5× faster than a 5-agent one. "
                "Earn complexity by demonstrating that simple approaches are genuinely insufficient."
            )
        elif progress < 0.25:
            return (
                f"## PHASE: BOOTSTRAP  [iteration {i}/{n}  |  progress {p:.0%}]\n"
                "Complexity budget: 3–5 agents maximum.\n"
                "Objective: establish a working linear baseline — one that produces an answer.\n"
                "Rules:\n"
                "  • Add at most ONE agent compared to last iteration.\n"
                "  • Each new agent must address a specific failure observed in the SEED phase.\n"
                "  • Topology must be sequential (A → B → C). No branching yet.\n"
            )
        elif progress < 0.40:
            return (
                f"## PHASE: DIVERGE  [iteration {i}/{n}  |  progress {p:.0%}]\n"
                "Complexity budget: 4–7 agents. Branching and loops are now permitted.\n"
                "Objective: explore structurally different approaches — breadth of search, not depth of refinement.\n"
                "Rules:\n"
                "  • Try a topology qualitatively different from all previous attempts\n"
                "    (e.g., if you tried linear, try parallel fan-out; if fan-out, try debate).\n"
                "  • Keep agent prompts short. Complexity should come from structure, not instructions.\n"
                "  • Do NOT polish an approach you've already tried — the goal is diversity.\n"
                "Why: the search space is still largely unexplored."
            )
        elif progress < 0.60:
            return (
                f"## PHASE: SCALE  [iteration {i}/{n}  |  progress {p:.0%}]\n"
                "Complexity budget: 5+ agents. Additional depth is now justified.\n"
                "Objective: take the most promising topology and add ONE purposeful layer —\n"
                "  a validator, a critic, a fallback path, or a specialised sub-agent.\n"
                "Rules:\n"
                "  • Identify the single weakest link in the best workflow seen so far.\n"
                "  • Add exactly one new agent that directly addresses that weakness.\n"
                "  • Each agent must justify its existence in one sentence — or remove it.\n"
                "Why: complexity now has a proven foundation."
            )
        elif progress < 0.80:
            return (
                f"## PHASE: CONVERGE  [iteration {i}/{n}  |  progress {p:.0%}]\n"
                "Complexity budget: keep the current agent count. Do not add new agents.\n"
                "Objective: make the existing structure work better — tighter prompts, fixed handoffs.\n"
                "Rules:\n"
                "  • No new agents. No structural changes.\n"
                "  • Pick the agent whose output quality is lowest and rewrite its prompt.\n"
                "  • Add a guard condition if error propagation is detected.\n"
                "  • Prefer precision over addition: a sharper prompt outperforms an extra step.\n"
                "Why: the topology is likely correct."
            )
        else:
            return (
                f"## PHASE: POLISH  [iteration {i}/{n}  |  progress {p:.0%}]\n"
                "Complexity budget: frozen. The architecture is final.\n"
                "Objective: micro-improvements only — wording, output format, edge-case handling.\n"
                "Rules:\n"
                "  • Do not change topology or agent count.\n"
                "  • Fix the single most concrete failure visible in the evaluation results.\n"
                "  • Trace any failed assertion to the responsible agent and fix its prompt.\n"
                "  • If no clear failures remain, improve output formatting or add an explicit\n"
                "    success/failure status to the final agent's output.\n"
                "Why: you are near the optimum for this topology."
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
                f"agent {name}: {str(answer)[:256]}..."
                for name, answer in zip(wf_state["step_name"], answers, strict=True)
            )
        return str(answers)

    # ── Prompt builders ───────────────────────────────────────────────────────

    def seed_genome_prompt(self, goal: str) -> str:
        """
        Prompt for the very first workflow generation (generation 0).

        No prior code or evaluation exists yet, so this is intentionally
        minimal — just the goal and a nudge toward simplicity.  The annealing
        schedule's SEED phase rules apply implicitly; they are not injected
        here to keep the first prompt clean and unbiased.
        """
        return (
            "## First workflow generation\n"
            f"Goal to assemble a workflow for:\n{goal}\n"
            "Build the minimal workflow for the task.\n"
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
          3. Change pressures — a freshly-sampled Perturbation from Mutagen that
             nudges the LLM away from local minima without being prescriptive.

        The annealing schedule (_get_temperature_phase) constrains complexity
        budget so the search obeys a simple-to-complex curriculum.
        """
        score      = wf_info.overall_score      if wf_info else 0.0
        judge_eval = wf_info.judge_evaluation   if wf_info else None
        wf_state   = wf_info.state_result       if wf_info else None

        # ── Perturbation sampling ────────────────────────────────────────────
        perturbation  = self.mutagen.compose()
        voice_opening, voice_closing = perturbation.get_voice_framing()
        perturbation_block = perturbation.format_block()
        tried_block = self.mutagen.format_tried_strategies_block()

        # Log for debugging / reproducibility
        print_info("DIVERSITY PERTURBATION")
        for k, v in perturbation.signature.items():
            print_info(f"  {k:>15}: {v}")

        # ── Execution evidence ───────────────────────────────────────────────
        agent_answers = self._extract_agent_answers(wf_state)
        exec_result   = (judge_eval or run_stderr)[-2048:].strip()
        phase_block   = self._get_temperature_phase(iteration_count, max_iterations, score)

        if genotype is None:
            # No prior workflow exists yet (shouldn't normally reach here —
            # use seed_genome_prompt() for generation 0).
            body = "Previous attempt failed. Learn from mistakes and improve the multi-agent workflow."
        else:
            body = "\n".join([
                "## WORKFLOW EVOLUTION STEP",
                "",
                voice_opening,
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
                "<evaluation>",
                exec_result or "No evaluation captured.",
                "</evaluation>",
                "",
                "## CHANGE PRESSURES:",
                "Randomly sampled to push you away from local minima.",
                "You don't have to follow all of them literally — let them *influence* your thinking.",
                "",
                perturbation_block,
                "",
                tried_block,
                "## YOUR TASK:",
                "1. Diagnose what went wrong (code bug? wrong approach? bad decomposition? weak agent prompts?).",
                "2. Let the change pressures above shift your perspective on the problem.",
                "3. Generate an improved workflow that is meaningfully different from the previous one.",
                "   Not a cosmetic tweak — a structural or strategic change.",
                "",
                voice_closing,
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

        Parents are sorted best → worst so the LLM sees the strongest candidates
        first (primacy bias works in our favour here).  The LLM is explicitly
        instructed to *recombine*, not to pick-and-patch one parent — genuine
        structural inheritance is the goal.

        Unlike mutation_prompt(), crossover does not inject a Mutagen perturbation.
        The structural diversity already comes from the parent pool; adding random
        pressures on top would risk obscuring the recombination signal.
        """
        # ── Assemble parent records ──────────────────────────────────────────
        parents = []
        for i, (wf_info, genotype, stderr) in enumerate(
            zip(wf_infos, genotypes, run_stderrs)
        ):
            score     = wf_info.overall_score    if wf_info else 0.0
            judge_eval = (wf_info.judge_evaluation if wf_info else None) or stderr.strip()
            answers   = self._extract_agent_answers(wf_info.state_result if wf_info else None)
            parents.append({
                "index":   i + 1,
                "score":   score,
                "code":    genotype,
                "eval":    judge_eval,
                "answers": answers,
            })

        # Best parents first — LLM primacy bias helps inherit strong traits
        parents.sort(key=lambda p: p["score"], reverse=True)

        phase_block = self._get_temperature_phase(
            iteration_count,
            max_iterations,
            score=parents[0]["score"] if parents else 0.0,
        )

        parents_block = []
        for p in parents:
            parents_block.append("\n".join([
                f"### Parent {p['index']} (score={p['score']:.2f})",
                "<python>",
                p["code"] or "(construction failed)",
                "</python>",
                "<agents_answers>",
                p["answers"],
                "</agents_answers>",
                "<evaluation>",
                p["eval"] or "None.",
                "</evaluation>",
            ]))

        return "\n".join([
            f"Attempt {iteration_count + 1}: CROSSOVER — synthesize from {len(parents)} parent workflows.",
            "",
            phase_block,
            "",
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