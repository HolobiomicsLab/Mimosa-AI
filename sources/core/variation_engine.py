
"""Variation prompts and search-state observation for the evolution loop.

Layer 2: ``VariationEngine`` assembles mutation and crossover prompts plus
the read-only ``<search_state>`` block the directive LLM conditions on.
``sources.core.evolution_engine`` imports the engine and ``exploration_state``
and calls them once per offspring, after the verifier scored the parent run.
Module helpers package frontier-probe results for the ``<exploration>``
prompt block and for persistence. Prompt templates live inline; deterministic
guardrails (agent caps, the <=1-agent rule, the keep-90% mandate) stay in
code, never in the LLM's hands.
"""

import logging

import numpy as np

from sources.cli.pretty_print import (
    print_info,
)
from sources.core.llm_provider import LLMConfig, LLMProvider

from .workflow_info import WorkflowInfo


def frontier_target(
    value: float,
    direction: str,
    bounds: tuple[float, float] | list[float] | None = None,
    integer: bool = False,
    delta: float = 0.25,
    min_step: float = 0.1,
) -> float | int | None:
    """Sign-aware exploratory target for one metric.

    ``render_exploration_block`` calls it when a metric spec marks an
    objective with a known direction; the result becomes an "arithmetic
    target" line in the ``<exploration>`` block. ``target = value ∓
    max(|value|·delta, min_step)`` in the improving direction; a signed
    value is never multiplied.

    Args:
        value: float — current measured value of the metric, from the
            verifier's metric summary.
        direction: str — improving direction, ``"higher"`` or ``"lower"``,
            from the metric spec; other values yield ``None``.
        bounds: tuple[float, float] | list[float] | None — ``[lo, hi]``
            limits the metric cannot leave; ``None`` skips clipping.
        integer: bool — round the target to a whole number when True,
            from the metric spec.
        delta: float — relative step size as a fraction of ``|value|``.
        min_step: float — absolute floor for the step; covers
            ``value == 0``.

    Returns:
        float | int | None — a strictly better target inside ``bounds``;
        ``None`` when ``direction`` is unknown or no strictly better value
        exists (the bound is already reached).
    """
    if direction not in ("higher", "lower"):
        return None
    v = float(value)
    step = max(abs(v) * float(delta), float(min_step))
    target = v + step if direction == "higher" else v - step
    if bounds:
        lo, hi = float(bounds[0]), float(bounds[1])
        target = min(max(target, lo), hi)
    if integer:
        target = int(round(target))
        if not (target > v if direction == "higher" else target < v):
            target = target + 1 if direction == "higher" else target - 1
        if bounds:
            lo, hi = float(bounds[0]), float(bounds[1])
            target = min(max(target, lo), hi)
    better = target > v if direction == "higher" else target < v
    return target if better else None


def exploration_state(probe: dict | None) -> dict:
    """Compact, JSON-persistable summary of one probe evaluation.

    ``sources.core.evolution_engine`` calls it after each frontier probe and
    persists the result under ``variation_state.exploration`` for telemetry
    and later inspection.

    Args:
        probe: dict | None — probe result from
            ``sources.core.evolution_engine.evaluate_exploration_probe``;
            ``None`` when no evaluation happened.

    Returns:
        dict — ``{"triggered": bool, "reason": str, "metrics": [...]}``,
        the exact shape persisted under ``variation_state.exploration``.
    """
    probe = probe or {}
    return {
        "triggered": bool(probe.get("triggered")),
        "reason": str(probe.get("reason") or ""),
        "metrics": list(probe.get("metrics") or []),
    }


def render_exploration_block(
    probe: dict | None,
    metric_specs: dict[str, dict] | None = None,
) -> str:
    """Render the labelled ``<exploration>`` prompt block.

    ``VariationEngine.mutation_prompt`` calls it with the frontier-probe
    state. The block is a SEARCH-POLICY request, never a measurement: it
    states the deterministic search state, asks for one bolder change, and
    forbids changing reported numbers without a producing method.

    Args:
        probe: dict | None — probe state with ``triggered``/``reason``/
            ``champion``/``since_change``/``n_pairs``/
            ``generations_compared``/``metrics``; an absent or untriggered
            probe renders "".
        metric_specs: dict[str, dict] | None — ``{metric_id: spec}`` with
            ``role``, ``direction``, ``bounds`` and ``integer`` from the
            verifier's metric summary. When a spec marks an ``objective``
            metric with a known direction, a sign-aware arithmetic target
            from ``frontier_target`` is appended. ``None`` renders
            text-only (no invented numbers).

    Returns:
        str — the multi-line ``<exploration>…</exploration>`` block, or ""
        when the probe is absent or untriggered.
    """
    if not probe or not probe.get("triggered"):
        return ""
    state_bits: list[str] = []
    champion = probe.get("champion")
    since = probe.get("since_change")
    if champion:
        tag = str(champion)[-8:] if len(str(champion)) > 8 else str(champion)
        state_bits.append(f"champion {tag} unchanged for {since} generations")
    metrics = list(probe.get("metrics") or [])
    if metrics:
        values = "; ".join(f"{m['id']} = {m['value']} everywhere" for m in metrics)
        n_pairs = probe.get("n_pairs")
        compared = f", {n_pairs} pairs compared" if n_pairs else ""
        gens = probe.get("generations_compared")
        window = f"last {gens} generations" if gens else "recent generations"
        state_bits.append(
            f"every measured metric is identical across the {window} "
            f"({values}{compared})"
        )
    state = "; ".join(state_bits) or "no measured state available"
    names = "/".join(str(m["id"]) for m in metrics) if metrics else "the current values"
    head = f"{metrics[0]['value']}" if metrics else ""
    cannot = (
        f"The verifier cannot tell whether {head} is optimal."
        if metrics
        else "The verifier cannot tell whether the current values are optimal."
    )
    lines = [
        "<exploration>",
        "EXPLORATION PROBE — search policy, NOT a measured loss.",
        f"State: {state}.",
        f"{cannot} Propose ONE bolder change that would either improve "
        f"{names} or independently confirm the bound (e.g. a different "
        "algorithm, a proof/bound argument, or a stronger independent "
        "validation).",
    ]
    for metric in metrics:
        spec = (metric_specs or {}).get(str(metric["id"]))
        if not spec or str(spec.get("role") or "objective") != "objective":
            continue  # invariants/reports are never pushed (defect C5)
        target = frontier_target(
            metric["value"],
            str(spec.get("direction") or ""),
            spec.get("bounds"),
            bool(spec.get("integer")),
        )
        if target is not None:
            lines.append(
                f"Arithmetic target to attempt to reach: "
                f"{metric['id']} = {target}."
            )
    lines.append(
        "Do not change reported numbers without a method that produces them."
    )
    lines.append("</exploration>")
    return "\n".join(lines)


class VariationEngine:
    """Assemble mutation/crossover prompts and expose search-state diagnostics.

    ``sources.core.evolution_engine`` constructs one engine per session and
    calls ``mutation_prompt``/``crossover_prompt`` per offspring. Mutation
    magnitude stays implicit: the directive LLM judges how bold the next
    change should be from the deterministic ``<search_state>`` block
    (parent score, iteration progress, plateau streak, success rate, score
    trajectory). Deterministic guardrails (agent-count caps, the <=1-agent
    rule, the keep-90% mandate) stay in code.
    """

    def __init__(self, config) -> None:
        """Create empty history buffers and wire the directive LLM.

        Called once per session by the evolution engine.

        Args:
            config: session config supplying ``workflow_llm_model`` and the
                LLM knobs used by ``setup_llm``.

        Returns:
            None.
        """
        self.textual_gradient_history: list[tuple[str, bool]] = []
        self.score_history: list[tuple[float | None, float | None, bool]] = []
        self.agent_count_history: list[int] = []
        self.max_possible_agents = 7
        self.last_variation_state: dict = {}
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.llm_config = None
        self.setup_llm(config)

    def setup_llm(self, config):
        """Build the LLM client used for mutation directives.

        ``__init__`` calls it with the session config; later directive calls
        go through ``self.llm_config``.

        Args:
            config: session config supplying ``workflow_llm_model``,
                ``reasoning_effort``, ``max_tokens`` and the OpenRouter
                routing helpers.

        Returns:
            None.

        Raises:
            Exception: when the LLM configuration cannot be built from the
                config values.
        """
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
        """Append one offspring's textual-gradient diagnosis and outcome.

        The evolution engine calls it after each offspring is scored so the
        search-state observer can compute plateau streaks, success rate and
        the score trajectory.

        Args:
            gradient: str — free-text diagnosis of the offspring's failure
                mode, from the verifier's textual gradient; empty values are
                replaced with a sentinel placeholder.
            is_failure: bool — True when the offspring failed to execute at
                all; such entries are excluded from plateau and success-rate
                stats.
            child_score: float | None — overall score of the produced
                offspring in [0, 1]; ``None`` entries do not contribute to
                the success-rate signal.
            best_before: float | None — best score across the population
                before this offspring; paired with ``child_score`` to decide
                improvement (observer diagnostics only).

        Returns:
            None.
        """
        text = (gradient or "").strip() or "UNKNOWN_GRADIENT: No feedback captured."
        self.textual_gradient_history.append((text, bool(is_failure)))
        self.score_history.append((child_score, best_before, bool(is_failure)))

    def _sample_agent_count(self, lo: int, hi: int) -> int:
        """Sample one agent count uniformly within ``[lo, hi]``.

        Mutation budgeting calls it; bounds are clamped to the hard caps
        ``[1, max_possible_agents]``.

        Args:
            lo: int — inclusive lower bound on the agent count.
            hi: int — inclusive upper bound on the agent count.

        Returns:
            int — the sampled count in ``[lo, hi]`` after clamping; ``lo``
            itself when the clamped bounds meet.
        """
        lo = max(1, min(lo, self.max_possible_agents))
        hi = max(lo, min(hi, self.max_possible_agents))
        if lo == hi:
            return lo
        return int(np.random.randint(lo, hi + 1))

    def _sample_mutation_agent_budget(self, parent_agents: int) -> int:
        """Sample a mutation agent budget around the parent's agent count.

        ``mutation_prompt`` calls it per offspring so complexity drifts by
        at most one agent per mutation; the draw is appended to
        ``agent_count_history``.

        Args:
            parent_agents: int — number of distinct agents in the parent
                workflow, from ``_count_parent_agents``.

        Returns:
            int — budget within ``[1, max_possible_agents]`` and within one
            agent of the parent count.
        """
        lo = max(1, parent_agents - 1)
        hi = min(self.max_possible_agents, parent_agents + 1)
        n_agents = self._sample_agent_count(lo, hi)
        self.agent_count_history.append(n_agents)
        return n_agents

    @staticmethod
    def _count_parent_agents(wf_state: dict | None) -> int | None:
        """Count distinct agents in the parent's last executed state.

        Mutation budgeting calls it to anchor the agent budget. Repeat
        visits share one name, so the count is over unique ``step_name``
        values.

        Args:
            wf_state: dict | None — parent ``state_result`` mapping from the
                run record; ``None`` when no state exists.

        Returns:
            int | None — number of distinct agents, or ``None`` when the
            state is missing or carries no step names.
        """
        if not wf_state:
            return None
        names = wf_state.get("step_name")
        if isinstance(names, list) and names:
            return max(1, len({str(n) for n in names}))
        return None

    def _iters_since_improvement(self) -> int:
        """Length of the current run of scored offspring that did not beat best-so-far.

        ``_search_state_block`` calls it for the plateau signal. Failures
        and entries without both scores are skipped; the count stops at the
        first offspring that beat its ``best_before``.

        Returns:
            int — consecutive non-improving scored offspring; ``0`` when
            the most recent scored offspring improved.
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
        """Fraction of recent scored offspring that beat best-so-far.

        ``_search_state_block`` calls it with the default window. Entries
        marked ``is_failure`` and entries without both scores are excluded.

        Args:
            window: int — number of recent scored offspring to consider
                (default 5).

        Returns:
            float | None — success rate in [0, 1]; ``None`` when fewer than
            two comparable entries exist in the window.
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
        """Return the last recorded offspring scores, oldest first.

        ``_search_state_block`` calls it to show the score trajectory in
        the search-state block. Failures and unscored offspring are
        skipped.

        Args:
            window: int — maximum number of recent scores to return
                (default 5).

        Returns:
            list[float] — recent child scores, oldest first; possibly
            empty.
        """
        return [
            c for c, _b, fail in self.score_history
            if not fail and c is not None
        ][-window:]

    def _search_state_block(
        self, parent_score: float, iteration_count: int, max_iterations: int,
    ) -> str:
        """Assemble the read-only ``<search_state>`` block for the directive.

        ``mutation_prompt`` calls it right before the directive LLM runs.
        The block carries parent score, iteration progress, plateau streak,
        recent success rate and a short score-only trajectory, and it
        refreshes ``last_variation_state`` for telemetry.

        Args:
            parent_score: float — parent reward in [0, 1], clipped here.
            iteration_count: int — zero-based index of the current attempt.
            max_iterations: int — total planned attempts.

        Returns:
            str — multi-line ``<search_state>`` payload without the
            enclosing tags.
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
        """Flatten per-agent answers from workflow state into text.

        Mutation and crossover prompts call it so the directive LLM reads
        compact agent answers instead of raw state; each answer is
        truncated to 16384 characters.

        Args:
            wf_state: dict | None — workflow state dict with ``answers``
                and, for list answers, parallel ``step_name`` entries.

        Returns:
            str — newline-joined ``agent <name>: <answer>...`` lines for
            list answers, the raw string otherwise, or a sentinel line
            when no answers were captured.
        """
        if not wf_state or "answers" not in wf_state:
            return "No agent answers captured."
        answers = wf_state["answers"]
        if isinstance(answers, list):
            return "\n".join(
                f"agent {name}: {str(answer)[:16384]}..."
                for name, answer in zip(wf_state["step_name"], answers, strict=True)
            )
        return str(answers)

    # ── Prompt builders ───────────────────────────────────────────────────────

    def seed_genome_prompt(self, goal: str) -> str:
        """Build the generation-0 seed prompt.

        The engine calls it before the first workflow exists, to obtain a
        small random-topology starting genome for the goal.

        Args:
            goal: str — task description the seeded workflow should
                target, from the session goal.

        Returns:
            str — prompt suggesting a random topology and a small starting
            agent budget.
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
        exploration_block: str = "",
    ) -> str:
        """Ask the directive LLM for the next mutation directive.

        ``mutation_prompt`` calls it with the parent evidence; the system
        prompt tells the LLM to judge the change magnitude itself from the
        deterministic search state. Degenerate output (shorter than 20
        characters) triggers one retry, then a deterministic fallback
        directive.

        Args:
            agent_answers: str — flattened parent answers from
                ``_extract_agent_answers``.
            textual_gradient_block: str — rubric-blind diagnosis from the
                verifier's textual gradient.
            search_state: str — read-only ``<search_state>`` payload from
                ``_search_state_block``.
            goal: str — task description for the session.
            exploration_block: str — rendered ``<exploration>`` block;
                empty when no probe fired.

        Returns:
            str — directive text of at most three sentences naming one
            issue; never empty.
        """
        sys_msg = """
You are an expert at pinpointing the root cause of failures in multi-agent workflows.
Your task is to analyze these inputs and provide a clear, concise directive for the next mutation step.
Focus on identifying what worked, what didn't, and why. Suggest specific changes to improve the next workflow's performance.
You will be given the previous workflow's agent answers (agents_answers) and a textual gradient block (diagnosis) that summarizes the failure.
The diagnosis is a summary of  deterministic ground truth verification using rubric-based scoring, and may include hints about what went wrong.
The diagnosis is trusted and should be used to inform your directive.
An <exploration> block, when present, is a search-policy request, not a measured fact; the <diagnosis> remains the only measured evidence.
The agent cannot be fully trusted and may have provided misleading or incomplete answers. Use your judgment to weigh the agent's answers against the diagnosis.
You will also be given a read-only <search_state> block with deterministic search statistics: parent score, iteration progress, iterations since the last improvement, recent success rate, and the recent score trajectory.
You decide yourself how bold the next change should be, justified by the search state: recent improvements and a rising score trajectory call for small incremental tweaks; a long plateau, a 0% success rate, or repeated identical failures call for bolder restructuring (such as adding agents, pressure to use another approach or algorithm, rewriting prompt, changing prompt persona, etc...).
Do not add or remove more than 1 agent at a time. Ask to add an agent whenever decomposition of the problem might help.
Most of the time, suggest small, incremental changes to the workflow. Suggest large change if the diagnosis and the search state indicate that the current approach is flawed.
If 'no agent answers' were captured, or that you see a python/syntax error, simply instruct to carefully craft the syntax for workflow creation, select tools correctly, and respect the generation guidelines (you do not make guidelines, the system reading your directive will know what it is; never specify what to fix, just give a general hint at fixing the syntax)
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
            *([exploration_block] if exploration_block else []),
            "The diagnosis contains fail/pass assertions with details, never report passing claims, never leak the exact claim being checked. only  diagnosis the failure and suggest pragmatic switch in approach for fix."
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
        directive = provider(prompt)
        # Guard against truncated/degenerate directive output
        if not directive or len(directive.strip()) < 20:
            self.logger.warning(
                "Directive LLM returned empty/degenerate output "
                f"({directive!r:.60}); retrying with output-only nudge."
            )
            directive = provider(
                prompt + "\n\nOutput ONLY the directive text now — no "
                "analysis, no preamble, max 3 sentences.",
                use_cache=False,
            )
        if not directive or len(directive.strip()) < 20:
            self.logger.warning(
                "Directive still degenerate after retry; using deterministic "
                "fallback directive."
            )
            directive = (
                "Make one small incremental improvement to the weakest step "
                "flagged in the diagnosis; keep the rest of the workflow "
                "unchanged."
            )
        return directive

    def mutation_prompt(
        self,
        goal: str,
        wf_info: WorkflowInfo | None,
        genotype: str | None,
        run_stderr: str,
        iteration_count: int,
        max_iterations: int = 10,
        exploration_probe: dict | None = None,
    ) -> str:
        """Build the full mutation prompt for one offspring.

        The evolution engine calls it per iteration with the parent's
        record. It assembles execution evidence (agent answers, textual
        gradient or stderr fallback), the search-state block and the
        exploration block, asks the directive LLM for a mutation directive
        (or a fixed syntax-fix directive when ``genotype`` is None), and
        appends the hard mutation instructions.

        Args:
            goal: str — task description for the mutated workflow.
            wf_info: WorkflowInfo | None — parent info supplying score,
                abstracted textual gradient and state; ``None`` on a cold
                start.
            genotype: str | None — parent source code; ``None`` means the
                previous attempt produced no code.
            run_stderr: str — stderr tail used as fallback gradient when
                no textual gradient exists.
            iteration_count: int — zero-based index of the current
                attempt.
            max_iterations: int — total planned attempts (prompt context).
            exploration_probe: dict | None — frontier-probe state;
                ``render_exploration_block`` decides whether a block is
                rendered; ``None`` injects nothing.

        Returns:
            str — the fully assembled mutation prompt.
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
        ).replace('_', ' ')
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

        # Frontier probe (review §4.3): labelled search-policy request.
        exploration_block = render_exploration_block(exploration_probe)
        self.last_variation_state["exploration"] = exploration_state(
            exploration_probe
        )

        if genotype is None:
            directive = "Previous attempt failed completly. Fix syntax errors."
        else:
            directive = self.llm_think_mutation_directive(
                agent_answers=agent_answers,
                textual_gradient_block=textual_gradient_block,
                search_state=search_state,
                goal=goal,
                exploration_block=exploration_block,
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
            *([exploration_block] if exploration_block else []),
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
        exploration_probe: dict | None = None,
    ) -> str:
        """Build a crossover prompt recombining N parents into one offspring.

        The engine calls it when parent selection chose crossover. Parents
        are sorted best-first so the LLM sees strong candidates first, and
        the prompt caps agents at the largest parent count.

        Args:
            goal: str — task description for the recombined workflow.
            wf_infos: list[WorkflowInfo] — parent infos, one per genotype.
            genotypes: list[str] — parent source code strings, parallel to
                ``wf_infos``.
            run_stderrs: list[str] — parent stderr tails used as fallback
                gradients.
            iteration_count: int — zero-based index of the current
                attempt.
            max_iterations: int — total planned attempts (prompt context).
            exploration_probe: dict | None — frontier-probe state; the
                labelled ``<exploration>`` block is injected after the
                parent diagnoses when triggered; ``None`` injects nothing.

        Returns:
            str — the fully assembled crossover prompt.
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
        """Minimal config stub for this offline smoke check."""
        workflow_llm_model = "openai/gpt-4o-mini"
        reasoning_effort = "low"
        max_tokens = 8192

        def openrouter_provider_for(self, model):
            """Return None; routing is not exercised by this smoke check."""
            return None

        def openrouter_quantizations_for(self, model):
            """Return None; quantization is not exercised by this smoke check."""
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
