
"""
VariationEngine: search-schedule and prompt assembly for LLM-guided workflow evolution.
"""

import math
from .workflow_info import WorkflowInfo

from sources.cli.pretty_print import (
    print_info, print_ok, print_warn, print_err,
    CYAN, GREEN, YELLOW, RED, DIM, RESET, BOLD,
)
from sources.core.llm_provider import LLMConfig, LLMProvider

import numpy as np

class VariationEngine:
    """Assemble mutation/crossover prompts and pick mutation scope from
    Rechenberg 1/5 success rate and a non-improvement plateau counter."""

    def __init__(self, config) -> None:
        """Initialise empty history buffers."""
        self.textual_gradient_history: list[tuple[str, bool]] = []
        # Per-offspring (child_score, best_before, is_failure) for the Rechenberg 1/5 success rule.
        self.score_history: list[tuple[float | None, float | None, bool]] = []
        self.agent_count_history: list[int] = []
        self.max_possible_agents = 7
        self.last_variation_state: dict = {}
        self.config = config
        self.llm_config = None
        self.bands = [
            (
                0.35, "Slight mutation (small step, exploit known good structure)"
            ),
            (
                0.50, "Roleplay shift (moderate step, explore new persona or reasoning style)"
            ),
            (
                0.65, "Prompt and roleplay shift (moderate step, more explicit instructions, more direct framing, change persona and reasoning mode)"
            ),
            (
                0.90, "Topology mutation (larger step, explore new agent arrangement or workflow structure)"
            ),
            (
                1.01, "Bolder mutation (explore new agent persona arrangement or workflow structure)"
            ),
        ]
        self.setup_llm(config)

    def setup_llm(self, config):
        self.judge_model = config.workflow_llm_model
        try:
            provider, model = self.judge_model.split("/", 1) if "/" in self.judge_model else ("openai", self.judge_model)
            self.llm_config = LLMConfig().from_dict({
                "model": model,
                "provider": provider,
                "temperature": 1.2,
                "reasoning_effort": config.reasoning_effort,
                "max_tokens": getattr(config, 'max_tokens', 8192),
                "openrouter_provider": config.openrouter_provider_for(self.judge_model),
                "openrouter_quantizations": config.openrouter_quantizations_for(self.judge_model),
            })
        except Exception as e:
            raise Exception(f"Failed to initialize LLM configuration: {str(e)}") from e


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
                (excluded from plateau-counter and success-rate stats).
            child_score: Overall score of the produced offspring, in ``[0, 1]``.
                ``None`` when unavailable; such entries do not contribute to
                the success-rate signal.
            best_before: Best score achieved across the population *before*
                this offspring was produced. Used together with
                ``child_score`` to decide whether the offspring is a "success"
                in the Rechenberg sense.
        """
        text = (gradient or "").strip() or "UNKNOWN_GRADIENT: No feedback captured."
        self.textual_gradient_history.append((text, bool(is_failure)))
        self.score_history.append((child_score, best_before, bool(is_failure)))

    def _sample_agent_count(self, boldness: float, lo: int, hi: int, concentration: float = 4.0) -> int:
        """Sample a random agent count within ``[lo, hi]``, biased upward by boldness.

        Args:
            boldness: Boldness level in ``[0, 1]`` pulling the mean toward ``hi``.
            lo: Inclusive lower bound on the agent count.
            hi: Inclusive upper bound on the agent count.
            concentration: Beta concentration parameter; higher values
                tighten the sample around the target mean.

        Returns:
            An integer in ``[lo, hi]`` sampled from a Beta-Binomial.
        """
        if lo == hi:
            return lo
        target_mean = lo + boldness * (hi - lo)
        p = np.clip((target_mean - lo) / (hi - lo), 0.05, 0.95)
        alpha = p * concentration
        beta = (1 - p) * concentration
        prob = np.random.beta(alpha, beta)
        return lo + int(np.random.binomial(hi - lo, prob))

    def _iters_since_improvement(self) -> int:
        """Length of the current run of scored offspring that did not beat best-so-far.

        Failures and ``None``-scored entries are skipped (no count, no break).
        Returns ``0`` when the most recent scored offspring improved.
        """
        count = 0
        for c, b, is_failure in reversed(self.score_history):
            if is_failure or c is None or b is None:
                continue
            if c > b + 1e-6:
                break
            count += 1
        return count

    _SUCCESS_RULE_THRESHOLD = 0.20   # Classical Rechenberg 1/5 rule.
    _PLATEAU_PATIENCE = 6
    _RESPECIATION_PATIENCE = 8
    _RESPECIATION_CLAMP = 0.89       # Just below the 0.90 RE-SPECIATION band.

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
        """Pick a boldness level and matching scope band for the next mutation.

        Blends two fitness-grounded signals: ``success_rate`` (Rechenberg 1/5)
        and ``plateau`` (``iters_since_improvement`` over ``_PLATEAU_PATIENCE``).
        ``parent_score`` enters only as a near-finish damper in the last 5 %
        of range. The top RE-SPECIATION band is hysteresis-gated: both
        ``iters_since_improvement >= _RESPECIATION_PATIENCE`` and
        ``success_rate in {None, 0.0}`` must hold.

        Updates ``self.agent_count_history`` and ``self.last_variation_state``.

        Args:
            parent_score: Parent reward in ``[0, 1]``.

        Returns:
            One-line mutation-scope directive embeddable in the LLM prompt.
        """
        iters_since_improvement = self._iters_since_improvement()
        plateau = min(1.0, iters_since_improvement / self._PLATEAU_PATIENCE)
        success_rate = self._compute_success_rate()
        parent_score = float(np.clip(parent_score, 0.0, 1.0))

        thr = self._SUCCESS_RULE_THRESHOLD
        if success_rate is None:
            effective = 0.3 * plateau                            # cold start cap
        elif success_rate >= thr:
            progress = min(1.0, (success_rate - thr) / (0.80 - thr))
            effective = plateau * (1.0 - progress)
        else:
            deficit = (thr - success_rate) / thr
            effective = 0.5 * deficit + 0.5 * plateau

        near_finish = max(0.0, (parent_score - 0.95) / 0.05)
        effective *= (1.0 - 0.5 * near_finish)
        effective = float(np.clip(effective, 0.0, 1.0))

        respeciation_allowed = (
            iters_since_improvement >= self._RESPECIATION_PATIENCE
            and (success_rate is None or success_rate == 0.0)
        )
        if not respeciation_allowed:
            effective = min(effective, self._RESPECIATION_CLAMP)

        curr = self.agent_count_history[-1] if self.agent_count_history else 1
        budget = curr + round(effective * (self.max_possible_agents - curr))
        n_agents = self._sample_agent_count(effective, 1, budget)
        self.agent_count_history.append(n_agents)

        sr_repr = "n/a" if success_rate is None else f"{success_rate:.2f}"
        msg = (
            f"Boldness effective={effective:.2f} "
            f"(plateau={plateau:.2f}, iters_no_improve={iters_since_improvement}, "
            f"success_rate={sr_repr}, parent_score={parent_score:.2f})."
        )
        if effective > 0.5:
            print_warn(f"{msg} Increasing mutation boldness and agent budget.")
        else:
            print_info(f"{msg} Mutation scope and agent budget remain moderate.")

        scope = next(label for threshold, label in self.bands if effective < threshold)

        self.last_variation_state = {
            "iters_since_improvement": int(iters_since_improvement),
            "plateau": float(plateau),
            "success_rate": None if success_rate is None else float(success_rate),
            "effective_boldness": float(effective),
            "parent_score": float(parent_score),
            "respeciation_gate_open": bool(respeciation_allowed),
            "agent_budget": int(n_agents),
        }
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

    def seed_genome_prompt(self, goal: str) -> str:
        """Build the prompt for the very first workflow generation (generation 0).

        Args:
            goal: Task description the seeded workflow should target.

        Returns:
            A prompt suggesting a random topology and a small starting agent budget.
        """
        n_agents = self._sample_agent_count(0.5, 1, 4)  # start with small random agent count
        return (
            "## First workflow generation\n"
            f"Goal to assemble a workflow for:\n{goal}\n"
            f"Build the minimal workflow for the task with maximum {n_agents} agents.\n"
        )
    
    def llm_think_mutation_directive(self, agent_answers: str, textual_gradient_block: str, step_block: str) -> str:
        sys_msg = """
YOu are an expert at pinpointing the root cause of failures in multi-agent workflows.
Your task is to analyze these inputs and provide a clear, concise directive for the next mutation step.
Focus on identifying what worked, what didn't, and why. Suggest specific changes to improve the next workflow's performance.
You will be given the previous workflow's agent answers and a textual gradient block that summarizes the failure modes.
You will also be given a <boldness> block that indicates how much change incentive you are allowed to suggest for the next workflow iteration.
"""
        prompt = ''.join([
            "## EXECUTION RESULTS:",
            "<agents_answers>",
            agent_answers,
            "</agents_answers>",
            "<diagnosis>",
            "",
            textual_gradient_block,
            "</diagnosis>",
            "<boldness>",
            step_block,
            "</boldness>",
            "Suggest a mutation directive for the next workflow iteration"
            "Higher boldness mean the same failure more was identified multiple times."
            "Example directive:"
            "- 'Focus on improving the data preprocessing step, as the agent answers indicate that the current approach is causing data leakage. Consider adding a validation step to check for data integrity before proceeding to the next agent.'"
            "- 'The agent are subborn, they are not following the instructions. Consider changing the agent's persona to be more compliant.'"
            "- Tweak the prompt of agent X to put the agent on a more domain-specific manifold, to avoid them to be stuck in the same local minima."
        ])
        provider = LLMProvider(
            system_msg=sys_msg,
            config=self.llm_config,
        )
        return provider(prompt)


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
        textual_gradient  = wf_info.abstracted_textual_gradient if wf_info else ""
        wf_state   = wf_info.state_result         if wf_info else None

        # ── Execution evidence ───────────────────────────────────────────────
        agent_answers = self._extract_agent_answers(wf_state)
        fail_msg = "FAILURE:Last run likely failed with no feedback captured. Focus on fixing syntax errors or langraph patterns."
        textual_gradient_block = (
            textual_gradient.strip()
            if textual_gradient and textual_gradient.strip()
            else (run_stderr or fail_msg).strip()
            or "This is a fresh attempt, no execution feedback is available yet. Create the first workflow based on the goal alone."
        ).replace('_', ' ')[:2048]
        step_block = self._get_prompt_step_size(parent_score=score)

        if genotype is None:
            directive = "Previous attempt failed completly. Fix syntax errors."
        else:
            directive = self.llm_think_mutation_directive(
                agent_answers=agent_answers,
                textual_gradient_block=textual_gradient_block,
                step_block=step_block
            )
        return "\n".join([
            f"Attempt {iteration_count + 1} of workflow generation.",
            "## GOAL:",
            goal,
            "## WORKFLOW EVOLUTION STEP",
            "Previous workflow code:",
            "<python>",
            genotype,
            "</python>",
            "Your previous workflow attempt did not reach the success threshold.",
            "<directive>",
            directive,
            "</directive>",
            "Follow directive as guideline regarding what to change in the workflow code.",
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
            textual_gradient = wf_info.abstracted_textual_gradient if wf_info else ""
            # Layer 1: rubric-blind textual_gradient instead of raw judge log.
            # Fall back to stderr tail only when no textual_gradient exists.
            textual_gradient = textual_gradient.strip() or (stderr or "").strip()[-1024:]
            answers   = self._extract_agent_answers(wf_info.state_result if wf_info else None)
            parents.append({
                "index":     i + 1,
                "score":     score,
                "code":      genotype,
                "textual_gradient": textual_gradient,
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
                "<error_diagnosis>",
                p["textual_gradient"],
                "</error_diagnosis>"
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

    # ── _iters_since_improvement: empty history ──────────────────────────
    ve = VariationEngine()
    assert ve._iters_since_improvement() == 0
    assert ve._compute_success_rate() is None

    # ── _iters_since_improvement: failures and unscored entries skipped ──
    ve = VariationEngine()
    for _ in range(4):
        ve.record_offspring_gradient("anything", is_failure=True)
    ve.record_offspring_gradient("no scores attached")  # child_score=None
    assert ve._iters_since_improvement() == 0
    assert ve._compute_success_rate() is None

    # ── _iters_since_improvement: counts only consecutive non-improvers ──
    ve = VariationEngine()
    ve.record_offspring_gradient("improve", child_score=0.30, best_before=0.20)
    ve.record_offspring_gradient("flat",    child_score=0.30, best_before=0.30)
    ve.record_offspring_gradient("flat",    child_score=0.30, best_before=0.30)
    ve.record_offspring_gradient("crash",   is_failure=True)  # transparent
    ve.record_offspring_gradient("flat",    child_score=0.30, best_before=0.30)
    assert ve._iters_since_improvement() == 3, ve._iters_since_improvement()

    # ── _iters_since_improvement: latest improvement resets streak to 0 ──
    ve = VariationEngine()
    for _ in range(4):
        ve.record_offspring_gradient("flat", child_score=0.5, best_before=0.5)
    ve.record_offspring_gradient("up", child_score=0.6, best_before=0.5)
    assert ve._iters_since_improvement() == 0

    # ── Plateau case: 5 non-improving offspring at 0.92.
    #    Hysteresis gate keeps us out of RE-SPECIATION (iters=5 < 8) but
    #    boldness must clear the smallest band.
    ve = VariationEngine()
    for _ in range(5):
        ve.record_offspring_gradient(
            "DATA_LEAKAGE: same diagnosis again",
            child_score=0.92,
            best_before=0.92,
        )
    assert ve._compute_success_rate() == 0.0
    plateau_step = ve._get_prompt_step_size(parent_score=0.92)
    state = ve.last_variation_state
    assert state["effective_boldness"] >= 0.35, state
    assert state["effective_boldness"] < 0.90, state
    assert state["respeciation_gate_open"] is False, state

    # ── Hysteresis gate opens at iters_since_improvement ≥ 8 + success=0. ──
    ve = VariationEngine()
    for _ in range(8):
        ve.record_offspring_gradient(
            "stuck", child_score=0.5, best_before=0.5,
        )
    deep_stuck_step = ve._get_prompt_step_size(parent_score=0.5)
    state = ve.last_variation_state
    assert state["respeciation_gate_open"] is True, state

    # ── Real progress: improvements drop boldness to the smallest band. ──
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
    assert ve.last_variation_state["effective_boldness"] < 0.35, ve.last_variation_state

    # ── Near-finish floor: at parent=1.0 the damper halves the pre-clamp
    #    boldness. Use a 3-iter streak so the result stays well below the
    #    hysteresis clamp at both parent scores — otherwise the clamp
    #    masks the comparison.
    ve = VariationEngine()
    for _ in range(3):
        ve.record_offspring_gradient(
            "stuck near finish", child_score=0.5, best_before=0.5,
        )
    ve._get_prompt_step_size(parent_score=0.50)
    bold_low_val = ve.last_variation_state["effective_boldness"]
    ve._get_prompt_step_size(parent_score=1.00)
    bold_high_val = ve.last_variation_state["effective_boldness"]
    assert bold_low_val < 0.89 and bold_high_val < 0.89, (bold_low_val, bold_high_val)
    assert abs(bold_high_val - 0.5 * bold_low_val) < 1e-6, (bold_high_val, bold_low_val)

    # ── Cold start (no scores supplied): boldness stays ≤ 0.30. ───────────
    ve = VariationEngine()
    for g in ("alpha", "beta", "gamma"):
        ve.record_offspring_gradient(g)
        step = ve._get_prompt_step_size(parent_score=0.5)
        assert ve.last_variation_state["effective_boldness"] <= 0.3 + 1e-9, ve.last_variation_state
        print(f"Cold-start gradient '{g}': {step}")
    print("smoke OK")