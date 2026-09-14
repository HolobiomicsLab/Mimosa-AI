
"""
VariationEngine: prompt assembly and search-state observation for LLM-guided
workflow evolution.
"""

import numpy as np

from sources.cli.pretty_print import (
    print_info,
)
from sources.core.llm_provider import LLMConfig, LLMProvider

from .workflow_info import WorkflowInfo


class VariationEngine:
    """Assemble mutation/crossover prompts and expose search-state diagnostics.

    Mutation magnitude is implicit: the directive LLM judges how bold the
    next change should be from the deterministic ``<search_state>`` block
    (parent score, iteration progress, plateau streak, success rate,
    score trajectory). Deterministic guardrails (agent-count caps, the
    <=1-agent rule, the keep-90% mandate) stay in code.
    """

    def __init__(self, config) -> None:
        """Initialise empty history buffers."""
        self.textual_gradient_history: list[tuple[str, bool]] = []
        # Per-offspring (child_score, best_before, is_failure) for the
        # search-state observer diagnostics (plateau streak, success rate).
        self.score_history: list[tuple[float | None, float | None, bool]] = []
        self.agent_count_history: list[int] = []
        self.max_possible_agents = 7
        self.last_variation_state: dict = {}
        self.config = config
        self.llm_config = None
        self.setup_llm(config)

    def setup_llm(self, config):
        self.judge_model = config.workflow_llm_model
        try:
            provider, model = self.judge_model.split("/", 1) if "/" in self.judge_model else ("openai", self.judge_model)
            self.llm_config = LLMConfig().from_dict({
                "model": model,
                "provider": provider,
                "temperature": 1.0,
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
                ``child_score`` to decide whether the offspring improved on
                the best-so-far (observer diagnostics only).
        """
        text = (gradient or "").strip() or "UNKNOWN_GRADIENT: No feedback captured."
        self.textual_gradient_history.append((text, bool(is_failure)))
        self.score_history.append((child_score, best_before, bool(is_failure)))

    def _sample_agent_count(self, lo: int, hi: int) -> int:
        """Sample a uniformly random agent count within ``[lo, hi]``.

        Bounds are clamped to the hard caps ``[1, max_possible_agents]``.

        Args:
            lo: Inclusive lower bound on the agent count.
            hi: Inclusive upper bound on the agent count.

        Returns:
            An integer in ``[lo, hi]``.
        """
        lo = max(1, min(lo, self.max_possible_agents))
        hi = max(lo, min(hi, self.max_possible_agents))
        if lo == hi:
            return lo
        return int(np.random.randint(lo, hi + 1))

    def _sample_mutation_agent_budget(self, parent_agents: int) -> int:
        """Sample a mutation agent budget around the PARENT's agent count.

        The window is ``[max(1, parent_agents - 1),
        min(max_possible_agents, parent_agents + 1)]`` with a uniform draw
        inside it, so complexity drifts by at most one agent per mutation
        regardless of any schedule. Appends the sample to
        ``self.agent_count_history``.

        Args:
            parent_agents: Number of distinct agents in the parent workflow.

        Returns:
            The sampled agent budget, within ``[1, max_possible_agents]`` and
            within one agent of the parent count.
        """
        lo = max(1, parent_agents - 1)
        hi = min(self.max_possible_agents, parent_agents + 1)
        n_agents = self._sample_agent_count(lo, hi)
        self.agent_count_history.append(n_agents)
        return n_agents

    @staticmethod
    def _count_parent_agents(wf_state: dict | None) -> int | None:
        """Count distinct agents in the parent's last executed state.

        ``step_name`` may contain repeat visits from retry loops, so the
        count is over unique names.

        Args:
            wf_state: Parent ``state_result`` mapping, or ``None``.

        Returns:
            Number of distinct agents, or ``None`` when unknown.
        """
        if not wf_state:
            return None
        names = wf_state.get("step_name")
        if isinstance(names, list) and names:
            return max(1, len({str(n) for n in names}))
        return None

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

    _PLATEAU_PATIENCE = 6   # Observer-only: normalises the plateau streak.

    def _compute_success_rate(self, window: int = 5) -> float | None:
        """Fraction of recent scored offspring that improved on best-so-far.

        Diagnostic observer only (historically the success-counting half
        of Rechenberg's 1/5 rule): the value is reported in the
        ``<search_state>`` block and in telemetry; it no longer drives any
        controller arithmetic.

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

    def _recent_score_trajectory(self, window: int = 5) -> list[float]:
        """Return the last ``window`` recorded offspring scores.

        Failed and unscored offspring are skipped.

        Args:
            window: Maximum number of recent scores to return.

        Returns:
            Recent child scores, oldest first; possibly empty.
        """
        return [
            c for c, _b, fail in self.score_history
            if not fail and c is not None
        ][-window:]

    def _search_state_block(
        self, parent_score: float, iteration_count: int, max_iterations: int,
    ) -> str:
        """Assemble the deterministic, read-only search-state block.

        Feeds the directive LLM plain search statistics — parent score,
        iteration progress, plateau streak, recent success rate and a
        short score-only trajectory. Contains no rubric or diagnosis text,
        preserving the verifier's rubric-blindness firewall. Also refreshes
        ``self.last_variation_state`` for telemetry.

        Args:
            parent_score: Parent reward in ``[0, 1]``.
            iteration_count: Zero-based index of the current attempt.
            max_iterations: Total planned attempts.

        Returns:
            A multi-line, read-only ``<search_state>`` payload (without the
            enclosing tags).
        """
        iters_since_improvement = self._iters_since_improvement()
        plateau = min(1.0, iters_since_improvement / self._PLATEAU_PATIENCE)
        success_rate = self._compute_success_rate()
        trajectory = self._recent_score_trajectory()
        parent_score = float(np.clip(parent_score, 0.0, 1.0))

        self.last_variation_state = {
            "iters_since_improvement": int(iters_since_improvement),
            "plateau": float(plateau),
            "success_rate": None if success_rate is None else float(success_rate),
            "parent_score": float(parent_score),
        }

        sr_repr = "n/a" if success_rate is None else f"{success_rate:.2f}"
        traj_repr = ", ".join(f"{s:.2f}" for s in trajectory) or "n/a"
        block = (
            f"parent_score: {parent_score:.2f}\n"
            f"iteration: {iteration_count + 1} of {max_iterations}\n"
            f"iterations_since_improvement: {iters_since_improvement} "
            f"(plateau level {plateau:.2f})\n"
            f"success_rate_last_5: {sr_repr}\n"
            f"recent_scores_oldest_first: {traj_repr}"
        )
        print_info(
            f"Search state: parent_score={parent_score:.2f}, "
            f"iters_no_improve={iters_since_improvement}, success_rate={sr_repr}."
        )
        return block

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
        n_agents = self._sample_agent_count(1, 4)  # small random starting count
        return (
            "## First workflow generation\n"
            f"Goal to assemble a workflow for:\n{goal}\n"
            f"Build the minimal workflow for the task with maximum {n_agents} agents.\n"
        )

    def llm_think_mutation_directive(
        self,
        agent_answers: str,
        textual_gradient_block: str,
        search_state: str,
        goal: str,
    ) -> str:
        """Ask a dedicated LLM for the next mutation directive.

        The directive decides *implicitly* how bold the next change should
        be: the system prompt tells the LLM to judge magnitude itself from
        the deterministic search state (recent improvements -> small tweak;
        long plateau or 0% success -> bolder restructuring). There is no
        boldness level or scope band to obey — only the search facts.

        Args:
            agent_answers: Flattened per-agent answers of the parent run.
            textual_gradient_block: Rubric-blind diagnosis from the verifier.
            search_state: Read-only ``<search_state>`` payload built by
                ``_search_state_block`` (scores only, no rubric text).
            goal: Task description.

        Returns:
            The directive text (≤ 3 sentences, one named issue).
        """
        sys_msg = """
You are an expert at pinpointing the root cause of failures in multi-agent workflows.
Your task is to analyze these inputs and provide a clear, concise directive for the next mutation step.
Focus on identifying what worked, what didn't, and why. Suggest specific changes to improve the next workflow's performance.
You will be given the previous workflow's agent answers (agents_answers) and a textual gradient block (diagnosis) that summarizes the failure.
The diagnosis is a summary of  deterministic ground truth verification using rubric-based scoring, and may include hints about what went wrong.
The diagnosis is trusted and should be used to inform your directive.
The agent cannot be fully trusted and may have provided misleading or incomplete answers. Use your judgment to weigh the agent's answers against the diagnosis.
You will also be given a read-only <search_state> block with deterministic search statistics: parent score, iteration progress, iterations since the last improvement, recent success rate, and the recent score trajectory.
You decide yourself how bold the next change should be, justified by the search state: recent improvements and a rising score trajectory call for small incremental tweaks; a long plateau, a 0% success rate, or repeated identical failures call for bolder restructuring.
Do not add or remove more than 1 agent at a time.
Most of the time, suggest small, incremental changes to the workflow. Only suggest larger changes if the diagnosis and the search state indicate that the current approach is fundamentally flawed.
"""
        prompt = ''.join([
            "## GOAL:",
            goal,
            "## EXECUTION RESULTS:",
            "<agents_answers>",
            agent_answers,
            "</agents_answers>",
            "<diagnosis>",
            "",
            textual_gradient_block,
            "</diagnosis>",
            "The diagnosis contains failure assertions, never report a diagnosis claim as something good, diagnosis only report issues to be fixed."
            "<search_state>",
            search_state,
            "</search_state>",
            "Suggest a mutation directive for the next workflow iteration"
            "Example directive:"
            "- 'Focus on improving the data preprocessing step, as the agent answers indicate that the current approach is causing data leakage. Consider adding a validation step to check for data integrity before proceeding to the next agent.'"
            "- Add a agent X that will ..."
            "- Tweak the prompt of agent X to be more domain-specific, to avoid them to be stuck in the same reasoning pattern."
            "Keep it short and focused on one issue, no more than 3 sentences. Do not include any code or workflow structure in your directive."
            "Specify the kind of mutation you are suggesting (e.g., prompt tweak, agent persona change, topology change), its intended magnitude (small tweak, component rewrite, or structural redesign), and the rationale behind it justified by the search state."
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
        # Deterministic search-state observer (no controller arithmetic).
        search_state = self._search_state_block(
            parent_score=score,
            iteration_count=iteration_count,
            max_iterations=max_iterations,
        )

        # Parent-centered agent budget: uniform sample within ±1 agent of
        # the parent's count, hard-capped to [1, max_possible_agents].
        parent_agents = self._count_parent_agents(wf_state)
        if parent_agents is None:
            parent_agents = (
                self.agent_count_history[-1] if self.agent_count_history else 1
            )
        agent_budget = self._sample_mutation_agent_budget(parent_agents)
        self.last_variation_state["agent_budget"] = int(agent_budget)

        if genotype is None:
            directive = "Previous attempt failed completly. Fix syntax errors."
        else:
            directive = self.llm_think_mutation_directive(
                agent_answers=agent_answers,
                textual_gradient_block=textual_gradient_block,
                search_state=search_state,
                goal=goal,
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
            "## MUTATION INSTRUCTIONS:",
            "- Follow exactly the directive as guideline regarding what to change in the workflow code.",
            f"- Do not add or remove more than 1 agent at a time, and use at most {agent_budget} agents in total.",
            "- Do not change the workflow's overall topology unless the directive explicitly suggests it.",
            "- Do not change prompt instructions outside the scope of the directive.",
            "- Do not add code sample or overly precise instructions. Let agent reason and do the instructed work."
            "- You must keep 90% of the previous workflow prompts and code unchanged, only modify the parts that are relevant to the directive.",
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
            zip(wf_infos, genotypes, run_stderrs, strict=False)
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

    class _SmokeConfig:
        workflow_llm_model = "openai/gpt-4o-mini"
        reasoning_effort = "low"
        max_tokens = 8192

        def openrouter_provider_for(self, model):
            return None

        def openrouter_quantizations_for(self, model):
            return None

    # ── _iters_since_improvement: empty history ──────────────────────────
    ve = VariationEngine(_SmokeConfig())
    assert ve._iters_since_improvement() == 0
    assert ve._compute_success_rate() is None

    # ── _iters_since_improvement: failures and unscored entries skipped ──
    ve = VariationEngine(_SmokeConfig())
    for _ in range(4):
        ve.record_offspring_gradient("anything", is_failure=True)
    ve.record_offspring_gradient("no scores attached")  # child_score=None
    assert ve._iters_since_improvement() == 0
    assert ve._compute_success_rate() is None

    # ── _iters_since_improvement: counts only consecutive non-improvers ──
    ve = VariationEngine(_SmokeConfig())
    ve.record_offspring_gradient("improve", child_score=0.30, best_before=0.20)
    ve.record_offspring_gradient("flat",    child_score=0.30, best_before=0.30)
    ve.record_offspring_gradient("flat",    child_score=0.30, best_before=0.30)
    ve.record_offspring_gradient("crash",   is_failure=True)  # transparent
    ve.record_offspring_gradient("flat",    child_score=0.30, best_before=0.30)
    assert ve._iters_since_improvement() == 3, ve._iters_since_improvement()

    # ── _iters_since_improvement: latest improvement resets streak to 0 ──
    ve = VariationEngine(_SmokeConfig())
    for _ in range(4):
        ve.record_offspring_gradient("flat", child_score=0.5, best_before=0.5)
    ve.record_offspring_gradient("up", child_score=0.6, best_before=0.5)
    assert ve._iters_since_improvement() == 0

    # ── Search-state block: observer fields, no controller fields. ────────
    ve = VariationEngine(_SmokeConfig())
    for _ in range(5):
        ve.record_offspring_gradient(
            "DATA_LEAKAGE: same diagnosis again",
            child_score=0.92,
            best_before=0.92,
        )
    assert ve._compute_success_rate() == 0.0
    block = ve._search_state_block(
        parent_score=0.92, iteration_count=4, max_iterations=10
    )
    state = ve.last_variation_state
    assert state["iters_since_improvement"] == 5, state
    assert abs(state["plateau"] - 5 / 6) < 1e-9, state
    assert state["success_rate"] == 0.0, state
    assert state["parent_score"] == 0.92, state
    assert set(state) == {
        "iters_since_improvement",
        "plateau",
        "success_rate",
        "parent_score",
    }, state
    assert "0.92" in block and "iteration: 5 of 10" in block
    assert "recent_scores_oldest_first: 0.92" in block

    # ── Parent-centered agent budget: within ±1 of the parent count. ──────
    ve = VariationEngine(_SmokeConfig())
    budget = ve._sample_mutation_agent_budget(3)
    assert 2 <= budget <= 4, budget
    assert ve.agent_count_history[-1] == budget
    for _ in range(50):
        assert 1 <= ve._sample_mutation_agent_budget(1) <= 2   # floor cap
        assert 6 <= ve._sample_mutation_agent_budget(7) <= 7   # ceiling cap

    print("smoke OK")
