"""
Mutagen: Multi-dimensional prompt perturbations for LLM workflow improvement.
"""

import random
import time

from .workflow_info import WorkflowInfo

from sources.cli.pretty_print import (
    print_ok, print_warn, print_err, print_info,
    CYAN, GREEN, YELLOW, RED, DIM, RESET, BOLD,
)

class Mutagen:
    """
    Artificially induce diversity in LLM-guided workflow search using multi-dimensional prompt perturbations.
    The point is not whenever the sampled perturbation really make sense but to push the LLM to explore different regions of the workflow design space, escaping local minima.
    Each axis is sampled independently, yielding combinatorial diversity even with small pools per axis.
    """

    def __init__(self):
        self.rn_seed = int(time.time() * 1000) % 2**32
        self._tried_combinations: list[dict[str, str]] = []
        self._max_memory = 25  # remember last N tried combinations

        # HOW to reason about the problem (not what to build)
        self.thinking_lenses = [
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

        # WHAT property the topology should change toward (abstract, not prescriptive)
        self.structural_mutations = [
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

        # HOW the improvement prompt itself is worded — changes LLM starting manifold
        self.prompt_voices = [
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

        # What to STOP doing, REMOVE, or QUESTION (triggers lateral thinking)
        self.constraint_inversions = [
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

        # HOW to break the problem apart conceptually
        self.decomposition_strategies = [
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

    def get_flow_answers(self, wf_state: any) -> str:
        """Extract the answers from the workflow state."""
        if not wf_state or "answers" not in wf_state:
            return ""
        flow_answers = (
            "\n".join(f"agent {n}: {str(x)[:256]}..." for (n, x) in zip(wf_state["step_name"], wf_state["answers"], strict=True))
            if isinstance(wf_state["answers"], list)
            else wf_state["answers"]
        )
        return flow_answers

    def _sample_one(self, axis: list[tuple[str, str]], rng: random.Random) -> tuple[str, str]:
        """Sample one item from an axis."""
        return rng.choice(axis)

    def _compose_perturbation(self, seed: int | None = None) -> dict[str, tuple[str, str]]:
        """
        Sample one item from each orthogonal axis to compose a unique
        multi-dimensional perturbation. Avoids recently-tried combinations.
        """
        rng = random.Random(seed)
        max_retries = 10

        for _ in range(max_retries):
            combo = {
                "lens": self._sample_one(self.thinking_lenses, rng),
                "structure": self._sample_one(self.structural_mutations, rng),
                "voice": self._sample_one(self.prompt_voices, rng),
                "constraint": self._sample_one(self.constraint_inversions, rng),
                "decomposition": self._sample_one(self.decomposition_strategies, rng),
            }
            # Check if this combination was recently tried (by axis keys)
            combo_signature = {k: v[0] for k, v in combo.items()}
            if combo_signature not in self._tried_combinations:
                self._tried_combinations.append(combo_signature)
                if len(self._tried_combinations) > self._max_memory:
                    self._tried_combinations.pop(0)
                return combo
        # If all retries exhausted (unlikely), return last generated
        return combo

    def _format_perturbation_block(self, perturbation: dict[str, tuple[str, str]]) -> str:
        """Format the multi-dimensional perturbation as a prompt section."""
        sections = [
            ("THINKING LENS", perturbation["lens"]),
            ("STRUCTURAL PRESSURE", perturbation["structure"]),
            ("CONSTRAINT", perturbation["constraint"]),
            ("DECOMPOSITION ANGLE", perturbation["decomposition"]),
        ]
        lines = []
        for title, (name, desc) in sections:
            lines.append(f"**{title}** [{name}]: {desc}")
        return "\n".join(lines)

    def _get_voice_framing(self, voice: tuple[str, str]) -> tuple[str, str]:
        """Return (opening_frame, closing_frame) that reshape the prompt's tone."""
        name, desc = voice
        opening = (
            f"**YOUR APPROACH THIS ROUND** [{name}]: {desc}\n"
            f"Adopt this mindset for your entire analysis below. "
            f"Let it shape what you notice and what you change.\n"
        )
        closing = f"Remember: stay in [{name}] mode. {desc.split('.')[0]}."
        return opening, closing

    def _format_tried_strategies_block(self) -> str:
        """Format recently tried strategies so the LLM can avoid them."""
        if not self._tried_combinations:
            return ""
        recent = self._tried_combinations[-5:]  # show last 5
        lines = ["## PREVIOUSLY TRIED DIRECTIONS (avoid repeating these):"]
        for i, combo in enumerate(recent, 1):
            tags = ", ".join(f"{k}={v}" for k, v in combo.items())
            lines.append(f"  {i}. [{tags}]")
        lines.append("Take a genuinely different direction this time.\n")
        return "\n".join(lines)

    def _get_temperature_phase(self, iteration_count: int, max_iterations: int = 10, score: float = 0.0, alpha: float = 0.5) -> str:
        """
        Combined simulated-annealing + complexity-curriculum schedule.

        Two forces act simultaneously as iteration progress increases:
          - Temperature (exploration → exploitation): how bold/divergent to be.
          - Complexity budget (simple → rich): how many agents / how deep a topology
            is permitted. Complexity is earned — early iterations must prove that a
            simple baseline works before the search is allowed to grow the workflow.
        """
        if max_iterations <= 1:
            progress = 0.5
        else:
            progress = (iteration_count / max(max_iterations - 1, 1)) ** (1 - alpha * score)

        if progress < 0.10:
            return (
                "## PHASE: SEED  [iteration {i}/{n}  |  progress {p:.0%}]\n"
                "Complexity budget: 2–3 agents maximum.\n"
                "Objective: prove the task is solvable at all with the minimum viable workflow.\n"
                "Rules:\n"
                "  • Build the simplest possible chain that could plausibly produce the answer.\n"
                "  • No error-handling agents, no validators, no critics — just the core transformation.\n"
                "  • If a single agent can attempt the whole task, do that first.\n"
                "Why: debugging a 1-agent failure is 5× faster than a 5-agent one. "
                "Earn complexity by demonstrating that simple approaches are genuinely insufficient."
            ).format(i=iteration_count + 1, n=max_iterations, p=progress)

        elif progress < 0.25:
            return (
                "## PHASE: BOOTSTRAP  [iteration {i}/{n}  |  progress {p:.0%}]\n"
                "Complexity budget: 3–5 agents maximum.\n"
                "Objective: establish a working linear baseline — one that produces an answer, "
                "Rules:\n"
                "  • Add at most ONE agent compared to last iteration.\n"
                "  • Each new agent must address a specific failure observed in the SEED phase.\n"
                "  • Topology must be sequential (A → B → C). No branching yet.\n"
            ).format(i=iteration_count + 1, n=max_iterations, p=progress)

        elif progress < 0.40:
            return (
                "## PHASE: DIVERGE  [iteration {i}/{n}  |  progress {p:.0%}]\n"
                "Complexity budget: 4–7 agents. Branching and loops are now permitted.\n"
                "Objective: explore structurally different approaches to see which topology "
                "is most promising — breadth of search, not depth of refinement.\n"
                "Rules:\n"
                "  • Try a topology that is qualitatively different from all previous attempts "
                "  (e.g., if you tried linear, try parallel fan-out; if you tried fan-out, try debate).\n"
                "  • Keep agent prompts short and focused. Complexity should come from structure, "
                "  not from lengthy per-agent instructions.\n"
                "  • Do NOT polish an approach you've already tried — the goal is diversity.\n"
                "Why: the search space is still largely unexplored. "
            ).format(i=iteration_count + 1, n=max_iterations, p=progress)

        elif progress < 0.60:
            return (
                "## PHASE: SCALE  [iteration {i}/{n}  |  progress {p:.0%}]\n"
                "Complexity budget: 5+ agents. Additional layer of depth could now be justified.\n"
                "Objective: take the most promising topology found so far and add ONE purposeful "
                "layer — a validator, a critic, a fallback path, or a specialised sub-agent.\n"
                "Rules:\n"
                "  • Identify the single weakest link in the best workflow seen so far.\n"
                "  • Add exactly one new agent (or structural feature) that directly addresses "
                "  that weakness. Do not add agents 'just in case'.\n"
                "  • Each agent must justify its existence: if you cannot state in one sentence "
                "  what unique transformation it performs, remove it.\n"
                "Why: complexity now has a proven foundation. "
            ).format(i=iteration_count + 1, n=max_iterations, p=progress)

        elif progress < 0.80:
            return (
                "## PHASE: CONVERGE  [iteration {i}/{n}  |  progress {p:.0%}]\n"
                "Complexity budget: keep the current agent count. Do not add new agents.\n"
                "Objective: make the existing structure work better — tighten agent prompts, "
                "fix handoff contracts, improve routing conditions.\n"
                "Rules:\n"
                "  • No new agents. No structural changes.\n"
                "  • Pick the one agent whose output quality is lowest and rewrite its prompt.\n"
                "  • If the workflow is failing due to error propagation, add a guard condition\n"
                "  • Prefer precision over addition: a sharper prompt outperforms an extra step.\n"
                "Why: the topology is likely correct. "
            ).format(i=iteration_count + 1, n=max_iterations, p=progress)

        else:
            return (
                "## PHASE: POLISH  [iteration {i}/{n}  |  progress {p:.0%}]\n"
                "Complexity budget: frozen. The architecture is final.\n"
                "Objective: micro-improvements only — wording, output format, edge-case handling.\n"
                "Rules:\n"
                "  • Do not change the workflow topology or agent count.\n"
                "  • Fix the single most concrete failure visible in the evaluation results.\n"
                "  • If the judge evaluation mentions a specific assertion that failed, "
                "  trace which agent is responsible and fix its prompt for that case.\n"
                "  • If there are no clear failures, improve output formatting or add "
                "  an explicit success/failure status to the final agent's output.\n"
                "Why: you are near the optimum for this topology. "
            ).format(i=iteration_count + 1, n=max_iterations, p=progress)

    def improvement_prompt(
        self,
        goal: str,
        wf_info: WorkflowInfo,
        flow_code: str,
        run_stderr: str,
        iteration_count: int,
        max_iterations: int = 10
    ) -> str:
        exec_result = ""
        agents_answers = None
        wf_state = wf_info.state_result if wf_info else None
        judge_eval = wf_info.judge_evaluation if wf_info else None
        score = wf_info.overall_score if wf_info else 0.0

        # Compose multi-dimensional perturbation
        perturbation = self._compose_perturbation(seed=self.rn_seed)
        self.rn_seed += 1

        # Extract voice for prompt framing
        voice_opening, voice_closing = self._get_voice_framing(perturbation["voice"])
        perturbation_block = self._format_perturbation_block(perturbation)
        tried_block = self._format_tried_strategies_block()
        temperature_phase = self._get_temperature_phase(iteration_count, max_iterations, score)

        if wf_state:
            agents_answers = self.get_flow_answers(wf_state)

        if judge_eval:
            exec_result = judge_eval
        else:
            exec_result = run_stderr.strip()

        # Log the perturbation for debugging
        combo_sig = {k: v[0] for k, v in perturbation.items()}
        print_info(f"DIVERSITY PERTURBATION")
        for k, v in combo_sig.items():
            print_info(f"  {k:>15}: {v}")

        improv_prompt = "Previous attempt failed. Learn from mistakes and improve the multi-agent workflow."
        if flow_code is not None:
            improv_prompt = "\n".join([
                "## WORKFLOW EVOLUTION STEP",
                "",
                voice_opening,
                temperature_phase,
                "",
                "Your previous workflow attempt did not reach the success threshold.",
                "Goal: " + goal,
                "",
                "## Previous workflow code:",
                "<python>",
                flow_code,
                "</python>",
                "",
                "## EXECUTION RESULTS:",
                "<agents_answers>",
                agents_answers if wf_state else "No agent answers captured.",
                "</agents_answers>",
                "<evaluation>",
                exec_result,
                "</evaluation>",
                "",
                "## CHANGE PRESSURES:",
                "The following pressures are randomly sampled to push you away from local minima.",
                "You don't have to follow all of them literally, but let them *influence* your thinking.",
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

        return "".join(
            [
                f"Attempt {iteration_count + 1} of workflow generation.\n",
                improv_prompt,
                "\nTarget goal:\n",
                goal,
            ]
        )