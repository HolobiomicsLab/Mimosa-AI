
"""
VariationEngine: search-schedule and prompt assembly for LLM-guided workflow evolution.
"""

import math
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from .workflow_info import WorkflowInfo

from sources.cli.pretty_print import (
    print_info, print_ok, print_warn, print_err,
    CYAN, GREEN, YELLOW, RED, DIM, RESET, BOLD,
)

import numpy as np

class VariationEngine:
    """
    Orchestrates iterative LLM-driven workflow search via structured prompt mutation.
    Each call to mutation_prompt() or crossover_prompt() produces a prompt that:
      - Anchors the LLM on concrete execution feedback (agent answers, judge eval).
      - Applies a stagnation-driven mutation scope that widens exploration
        breadth and grows the agent budget as recent offspring keep failing
        the same way, damped by parent score so near-winners stay protected.
    """

    def __init__(self) -> None:
        """Initialise empty history buffers and lazy embedder state."""
        self.prompt_gradient_history: list[tuple[str, bool]] = []
        # Per-offspring (child_score, best_before, is_failure) for the
        # Rechenberg 1/5 success rule. ``None`` slots mark offspring whose
        # scores were not supplied (back-compat path).
        self.score_history: list[tuple[float | None, float | None, bool]] = []
        self.agent_count_history: list[int] = []
        self.max_possible_agents = 7
        self._embedder: SentenceTransformer | None = None

    def record_offspring_gradient(
        self,
        gradient: str,
        *,
        is_failure: bool = False,
        child_score: float | None = None,
        best_before: float | None = None,
    ) -> None:
        """Append an offspring's prompt-gradient diagnosis and outcome to history.

        Args:
            gradient: Free-text diagnosis of the offspring's failure mode.
                Empty values are replaced with a sentinel placeholder.
            is_failure: Whether the offspring failed to execute at all
                (excluded from semantic stagnation and success-rate stats).
            child_score: Overall score of the produced offspring, in ``[0, 1]``.
                ``None`` when unavailable; such entries do not contribute to
                the success-rate signal.
            best_before: Best score achieved across the population *before*
                this offspring was produced. Used together with
                ``child_score`` to decide whether the offspring is a "success"
                in the Rechenberg sense.
        """
        text = (gradient or "").strip() or "UNKNOWN_GRADIENT: No feedback captured."
        self.prompt_gradient_history.append((text, bool(is_failure)))
        self.score_history.append((child_score, best_before, bool(is_failure)))

    def _sample_agent_count(self, stagnation: float, lo: int, hi: int, concentration: float = 4.0) -> int:
        """Sample a random agent count within ``[lo, hi]``, biased upward by stagnation.

        Args:
            stagnation: Stagnation level in ``[0, 1]`` pulling the mean toward
                ``hi``.
            lo: Inclusive lower bound on the agent count.
            hi: Inclusive upper bound on the agent count.
            concentration: Beta concentration parameter; higher values
                tighten the sample around the target mean.

        Returns:
            An integer in ``[lo, hi]`` sampled from a Beta-Binomial.
        """
        if lo == hi:
            return lo
        target_mean = lo + stagnation * (hi - lo)
        p = np.clip((target_mean - lo) / (hi - lo), 0.05, 0.95)
        alpha = p * concentration
        beta = (1 - p) * concentration
        prob = np.random.beta(alpha, beta)
        return lo + int(np.random.binomial(hi - lo, prob))

    def _prompt_gradient_similarity(self, a: str, b: str) -> float:
        """Cosine similarity over MiniLM-encoded diagnoses.

        Args:
            a: First diagnosis text.
            b: Second diagnosis text.

        Returns:
            Cosine similarity in ``[-1, 1]``, or ``0.0`` when either text is empty.
        """
        if not a or not b:
            return 0.0
        if self._embedder is None:
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2", token=False)
        emb_a = self._embedder.encode(a, convert_to_tensor=True, show_progress_bar=False)
        emb_b = self._embedder.encode(b, convert_to_tensor=True, show_progress_bar=False)
        return F.cosine_similarity(emb_a, emb_b, dim=0).item()

    def _compute_stagnation(self, window: int = 4) -> float:
        """Mean pairwise cosine over recent non-failure offspring gradients, ∈ [0, 1].

        Args:
            window: How many recent semantic gradients to consider.

        Returns:
            Stagnation in ``[0, 1]``; ``0.0`` when fewer than two non-failure
            gradients are available.
        """
        semantic = [g for g, is_failure in self.prompt_gradient_history if not is_failure]
        recent = semantic[-window:]
        if len(recent) < 2:
            return 0.0
        sims = [
            self._prompt_gradient_similarity(recent[i], recent[j])
            for i in range(len(recent))
            for j in range(i + 1, len(recent))
        ]
        raw = sum(sims) / len(sims) if sims else 0.0
        # MiniLM unrelated baseline ≈ 0.4; treat 0.8+ as fully stagnated.
        return float(np.clip((raw - 0.4) / 0.4, 0, 1))

    # The Rechenberg success threshold — fraction of recent offspring that
    # must improve on the best-so-far for the search to be considered
    # "making progress". Below this, step size is grown; above, damped.
    # 0.20 is the classical 1/5 success rule (Rechenberg 1973).
    _SUCCESS_RULE_THRESHOLD = 0.20

    def _compute_success_rate(self, window: int = 5) -> float | None:
        """Fraction of recent scored offspring that improved on best-so-far.

        Implements the success-counting half of Rechenberg's 1/5 success
        rule (1973): a high fraction of recent improvements means step
        size is too small; a low fraction means we are stuck and should
        escalate.

        Offspring marked ``is_failure`` and those without recorded
        scores are excluded.

        Args:
            window: Number of recent scored offspring to consider.

        Returns:
            Success rate in ``[0, 1]``, or ``None`` when the history holds
            fewer than two scored offspring with comparable ``best_before``.
        """
        scored = [
            (c, b)
            for c, b, fail in self.score_history
            if not fail and c is not None and b is not None
        ]
        recent = scored[-window:]
        if len(recent) < 2:
            return None
        # Strict improvement; tiny epsilon to absorb float noise.
        return sum(1 for c, b in recent if c > b + 1e-6) / len(recent)

    def _get_prompt_step_size(self, parent_score: float = 0.0) -> str:
        """Evidence-based mutation scope (Rechenberg 1/5 success rule, 1973).

        Boldness is driven by two evidence signals, not by the parent's
        absolute score:

        - ``raw_stagnation`` — cosine similarity of recent prompt gradients.
          High → the search keeps diagnosing the same failure.
        - ``success_rate`` — fraction of recent offspring that beat the
          running best. Below the Rechenberg 1/5 threshold → step size is
          too small / search is stuck → grow scope. Above → damp scope.

        ``parent_score`` no longer multiplies the whole signal (that was a
        state-based damper that locked high-score lineages into "tiny
        tweak" mode even when the gradient repeated identically). It only
        re-enters as a *near-finish* soft floor in the last 5 % of score
        range, where a single regression could blow up a workflow about to
        hit the early-stop threshold.

        Updates ``self.agent_count_history`` as a side effect.

        Args:
            parent_score: Parent reward in ``[0, 1]``.

        Returns:
            A one-line human-readable mutation-scope directive embeddable in
            the LLM prompt.
        """
        raw_stagnation = self._compute_stagnation()
        success_rate = self._compute_success_rate()
        parent_score = float(np.clip(parent_score, 0.0, 1.0))

        # ── Rechenberg 1/5 rule, projected onto a [0, 1] boldness scalar ──
        thr = self._SUCCESS_RULE_THRESHOLD
        if success_rate is None:
            # Cold start — no improvement evidence yet. Trust gradient
            # repetition alone; also the back-compat path when the caller
            # does not pass scores.
            effective = raw_stagnation
        elif success_rate < thr:
            # Search is stuck (or has never improved). Below 1/5, escalate
            # at least up to the deficit even when gradient repetition is mild.
            deficit = (thr - success_rate) / thr  # ∈ [0, 1]
            effective = max(raw_stagnation, deficit)
        else:
            # Above 1/5 — real progress. Damp boldness in proportion to
            # how far above threshold we are; at success_rate ≥ 0.80
            # boldness collapses regardless of stagnation.
            progress = min(1.0, (success_rate - thr) / (0.80 - thr))
            effective = raw_stagnation * (1.0 - progress)

        # Near-finish floor: only in the last 5 % of the score range do we
        # re-introduce a mild parent_score damper, so the optimiser does
        # not gamble away a 0.96 parent one generation before early-stop.
        near_finish = max(0.0, (parent_score - 0.95) / 0.05)
        effective *= (1.0 - 0.5 * near_finish)
        effective = float(np.clip(effective, 0.0, 1.0))

        curr = self.agent_count_history[-1] if self.agent_count_history else 1
        budget = curr + round(effective * (self.max_possible_agents - curr))
        n_agents = self._sample_agent_count(effective, 1, budget)
        self.agent_count_history.append(n_agents)

        sr_repr = "n/a" if success_rate is None else f"{success_rate:.2f}"
        msg = (
            f"Boldness effective={effective:.2f} "
            f"(raw_stagnation={raw_stagnation:.2f}, "
            f"success_rate={sr_repr}, parent_score={parent_score:.2f})."
        )
        if effective > 0.5:
            print_warn(f"{msg} Increasing mutation boldness and agent budget.")
        else:
            print_info(f"{msg} Mutation scope and agent budget remain moderate.")

        bands = [
            (0.20, "prompt-only little tweak"),
            (0.40, "prompt, handoff information and tool change - improve the information flow"),
            (0.60, "topology, prompts, handoff format, tools — significant redesign while keeping topology"),
            (0.80, "bold rewire — restructure or grow the agent set"),
            (1.01, "complete rethink — discard inherited topology/prompts and innovate freely"),
        ]
        scope = next(label for threshold, label in bands if effective < threshold)
        return f"Mutation scope: {scope}. Boldness: {effective*100:.2f}%. Use at most {n_agents} agent(s).\n"

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_agent_answers(wf_state: dict | None) -> str:
        """Flatten per-agent answers from workflow state into a readable string.

        Args:
            wf_state: Workflow state dict containing ``answers`` and (when a
                list) parallel ``step_name`` entries.

        Returns:
            A newline-joined ``agent <name>: <truncated answer>...`` block, or
            the raw ``answers`` string for non-list values; a sentinel string
            when no answers were captured.
        """
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

    def random_topology_prompt(self) -> str:
        """Pick a random workflow-topology suggestion as a short label.

        Returns:
            One of a curated set of human-readable topology descriptions used
            to seed initial workflow generations.
        """
        return np.random.choice([
            "simple linear chain",
            "hub-and-spoke with 3-4 agents",
            "fully connected mesh",
            "single-agent",
            "debate (two+ agents argue, judge decides)",
            "reflection pair (actor + critic loop)",
            "blackboard (shared debate scratchpad, no direct messaging)",
            "map-reduce (fan-out subtasks, aggregator merges)",
            "sequential pipeline (output of one is input to next)",
            "round-robin group chat (shared conversation thread)",
            "mixture-of-experts (router picks expert per step)",
            "verifier-generator (generator proposes, verifier gates)"
        ])

    def seed_genome_prompt(self, goal: str) -> str:
        """Build the prompt for the very first workflow generation (generation 0).

        Args:
            goal: Task description the seeded workflow should target.

        Returns:
            A prompt suggesting a random topology and a small starting agent budget.
        """
        n_agents = self._sample_agent_count(0.5, 1, 4)  # start with small random agent count
        topology = self.random_topology_prompt()
        return (
            "## First workflow generation\n"
            f"Goal to assemble a workflow for:\n{goal}\n"
            f"Suggested initial topology: {topology}.\n"
            f"Build the minimal workflow for the task with maximum {n_agents} agents.\n"
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
        """Build a prompt for one mutation step in the evolutionary search.

        The prompt has three layers:
          1. Voice framing   — sets the LLM's reasoning tone for this iteration.
          2. Execution grounding — previous code, agent answers, and judge eval.

        Args:
            goal: Task description for the mutated workflow.
            wf_info: Parent workflow info supplying score, gradient and state;
                ``None`` for cold-start variants.
            genotype: Source code of the parent workflow; ``None`` means the
                previous attempt failed before producing code.
            run_stderr: Stderr tail used as a fallback gradient when no
                semantic diagnosis is available.
            iteration_count: Zero-based index of the current attempt.
            max_iterations: Total planned attempts (used for prompt context).

        Returns:
            The fully assembled mutation prompt string.
        """
        score      = wf_info.overall_score        if wf_info else 0.0
        prompt_gradient  = wf_info.abstracted_prompt_gradient if wf_info else ""
        wf_state   = wf_info.state_result         if wf_info else None

        # ── Execution evidence ───────────────────────────────────────────────
        agent_answers = self._extract_agent_answers(wf_state)
        fail_msg = "FAILURE:Last run likely failed with no feedback captured. Focus on fixing syntax errors or langraph patterns."
        prompt_gradient_block = (
            prompt_gradient.strip()
            if prompt_gradient and prompt_gradient.strip()
            else (run_stderr or fail_msg).strip()
            or "This is a fresh attempt, no execution feedback is available yet. Create the first workflow based on the goal alone."
        ).replace('_', ' ')[:2048]
        step_block = self._get_prompt_step_size(parent_score=score)

        if genotype is None:
            body = "Previous attempt failed. Fix syntax errors."
        else:
            body = "\n".join([
                "## WORKFLOW EVOLUTION STEP",
                "",
                "Your previous workflow attempt did not reach the success threshold.",
                "",
                "## Previous workflow code:",
                "<python>",
                genotype,
                "</python>",
                "",
                "## EXECUTION RESULTS:",
                #"<agents_answers>",
                #agent_answers,
                #"</agents_answers>",
                "<diagnosis>",
                "",
                prompt_gradient_block,
                "</diagnosis>",
                "<boldness>",
                step_block,
                "</boldness>",
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
        """Build a prompt for a crossover step: recombine N parent workflows into one offspring.

        Parents are sorted best → worst so the LLM sees the strongest candidates first.

        Args:
            goal: Task description for the recombined workflow.
            wf_infos: Parent workflow infos, one per genotype.
            genotypes: Parent source code strings, parallel to `wf_infos`.
            run_stderrs: Parent stderr tails used as fallback gradients.
            iteration_count: Zero-based index of the current attempt.
            max_iterations: Total planned attempts (used for prompt context).

        Returns:
            The fully assembled crossover prompt string.
        """
        # ── Assemble parent records ──────────────────────────────────────────
        parents = []
        for i, (wf_info, genotype, stderr) in enumerate(
            zip(wf_infos, genotypes, run_stderrs)
        ):
            score     = wf_info.overall_score        if wf_info else 0.0
            prompt_gradient = wf_info.abstracted_prompt_gradient if wf_info else ""
            # Layer 1: rubric-blind prompt_gradient instead of raw judge log.
            # Fall back to stderr tail only when no prompt_gradient exists.
            prompt_gradient = prompt_gradient.strip() or (stderr or "").strip()[-1024:]
            answers   = self._extract_agent_answers(wf_info.state_result if wf_info else None)
            parents.append({
                "index":     i + 1,
                "score":     score,
                "code":      genotype,
                "prompt_gradient": prompt_gradient,
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
            "Do not create a workflow with more agents than the workflow with the highest agent count among the parents. Hard cap to prevent runaway complexity.",
            "1. Identify which structural decisions worked in each parent (look at agent answers, not just score).",
            "2. Identify which decisions failed and why.",
            "3. Build a new workflow that inherits the strongest sub-structures across parents",
            "   and discards the weakest, even if that means a topology none of the parents used.",
        ])


if __name__ == "__main__":
    np.random.seed(0)

    # ── _compute_stagnation: failures are excluded ────────────────────────
    ve = VariationEngine()
    for _ in range(4):
        ve.record_offspring_gradient("anything", is_failure=True)
    assert ve._compute_stagnation() == 0.0
    assert ve._compute_success_rate() is None  # no scored offspring

    # ── _compute_stagnation: repeated gradient ⇒ high cosine ──────────────
    ve = VariationEngine()
    for _ in range(4):
        ve.record_offspring_gradient("INCONSISTENT_MULTITASK_SPLIT repeating")
    assert ve._compute_stagnation() > 0.8

    # ── Plateau case: same gradient, zero improvement, high parent_score.
    #    Old behaviour (state-based damping) wrongly returned "tiny tweak".
    #    New behaviour (1/5 rule) must escalate scope.
    ve = VariationEngine()
    for _ in range(5):
        ve.record_offspring_gradient(
            "DATA_LEAKAGE: same diagnosis again",
            child_score=0.92,
            best_before=0.92,  # no improvement
        )
    assert ve._compute_success_rate() == 0.0, "5/5 non-improving offspring"
    plateau_step = ve._get_prompt_step_size(parent_score=0.92)
    assert "tweak" not in plateau_step, (
        "1/5 rule must lift scope past the smallest band when stuck:\n"
        + plateau_step
    )

    # ── Real progress: every offspring beats best_before ⇒ damp boldness ──
    ve = VariationEngine()
    prev_best = 0.5
    for inc in (0.05, 0.07, 0.09, 0.11, 0.13):
        ve.record_offspring_gradient(
            "PROGRESS: different diagnosis " + str(inc),
            child_score=prev_best + inc,
            best_before=prev_best,
        )
        prev_best += inc
    assert ve._compute_success_rate() == 1.0
    progress_step = ve._get_prompt_step_size(parent_score=0.5)
    assert "tweak" in progress_step or "information flow" in progress_step, (
        "When the search is improving steadily, scope should stay small:\n"
        + progress_step
    )

    # ── Near-finish floor: at parent_score≥0.95 we damp by 50 %  ──────────
    ve = VariationEngine()
    for _ in range(5):
        ve.record_offspring_gradient(
            "stuck near finish",
            child_score=0.96,
            best_before=0.96,
        )
    near_finish_step = ve._get_prompt_step_size(parent_score=0.96)
    # Boldness should still be > 0 (we are stuck), but capped.
    print("near-finish step:", near_finish_step)

    # ── Cold start (no scores supplied): fall back to gradient-only ───────
    ve = VariationEngine()
    for g in (
        "DEEPCHEM_API_MISMATCH:The workflow produced a usable two-task probability prediction table, but the final training script is not reliably runnable because it calls an unavailable DeepChem model API, and it also lacks a clear held-out classification metric report.",
        "INCONSISTENT_MULTITASK_SPLIT:The workflow generated plausible probability predictions, but its script and outputs were internally inconsistent, with duplicate molecule rows and weak evidence that the provided train/test split, ECFP features, and both ClinTox endpoints were actually used in a true two-output multitask classifier.",
        "INCONSISTENT_MULTITASK_SPLIT:The workflow produced a plausible multitask ClinTox prediction table, but it showed serious integrity issues around endpoint/positive-class mapping, possible train-test contamination, and unclear alignment between molecules and their predicted probabilities.",
    ):
        ve.record_offspring_gradient(g)
        stag = ve._compute_stagnation()
        step = ve._get_prompt_step_size(parent_score=0.5)
        print(f"Gradient prompt: {g}\nStagnation: {stag:.2f}\nPrompt step:\n{step}\n{'-'*40}")
    print("smoke OK")