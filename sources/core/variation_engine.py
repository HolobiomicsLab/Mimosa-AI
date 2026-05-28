
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
      - Applies a step-aware annealing schedule that governs exploration breadth
        and permitted topology complexity as progress stagnate.
    """

    def __init__(self):
        # Each entry is (gradient_text, is_failure). Failures are kept in the
        # log but skipped by the stagnation signal so identical fallback text
        # ("…failed to generate or execute…") cannot peg cosine at 1.0.
        self.prompt_gradient_history: list[tuple[str, bool]] = []
        self.agent_count_history = []
        self.max_possible_agents = 7
        self._embedder = None

    def record_offspring_gradient(self, gradient: str, *, is_failure: bool = False) -> None:
        """Record the diagnosis of a freshly evaluated offspring.

        The engine must call this after each evaluation. Stagnation is
        measured over *offspring* gradients, not parent gradients — when
        the QD archive collapses to a single parent (common in practice),
        appending the parent's gradient at every mutation would trivially
        peg the signal at 1.0.
        """
        text = (gradient or "").strip() or "NO_prompt_gradient:No prompt_gradient captured."
        self.prompt_gradient_history.append((text, bool(is_failure)))

    def _sample_agent_count(self, stagnation: float, lo: int, hi: int, concentration: float = 4.0) -> int:
        """
        Sample agent random count within [lo, hi], biased upward by stagnation.
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
        """Cosine similarity over MiniLM-encoded diagnoses."""
        if not a or not b:
            return 0.0
        if self._embedder is None:
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2", token=False)
        emb_a = self._embedder.encode(a, convert_to_tensor=True, show_progress_bar=False)
        emb_b = self._embedder.encode(b, convert_to_tensor=True, show_progress_bar=False)
        return F.cosine_similarity(emb_a, emb_b, dim=0).item()

    def _compute_stagnation(self, window: int = 4) -> float:
        """Mean pairwise cosine over the last `window` *semantic* offspring
        diagnoses, ∈ [0, 1]. High value ⇒ the LLM-mutator is cycling on
        similar failure modes (mode collapse). Failure-fallback gradients
        are filtered out because their identical sentinel text would
        falsely saturate the signal.
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
        stagnation = float(np.clip((raw - 0.4) / 0.4, 0, 1))
        return stagnation

    def _get_prompt_step_size(self, parent_score: float = 0.0) -> str:
        """
        Adapt mutation boldness to current stagnation. Higher stagnation widens
        both the mutation scope and the permitted agent budget. A near-winning
        parent damps boldness so we don't rewire a workflow that almost
        succeeded — `effective = raw_stagnation · (1 − parent_score)`.
        """
        raw_stagnation = self._compute_stagnation()
        parent_score = float(np.clip(parent_score, 0.0, 1.0))
        stagnation = raw_stagnation * (1.0 - parent_score)

        curr = self.agent_count_history[-1] if self.agent_count_history else 1
        budget = curr + round(stagnation * (self.max_possible_agents - curr))
        n_agents = self._sample_agent_count(stagnation, 1, budget)
        self.agent_count_history.append(n_agents)

        msg = (
            f"Stagnation effective={stagnation:.2f} "
            f"(raw={raw_stagnation:.2f}, parent_score={parent_score:.2f})."
        )
        if stagnation > 0.5:
            print_warn(f"{msg} Increasing mutation boldness and agent budget.")
        else:
            print_info(f"{msg} Mutation scope and agent budget remain moderate.")

        bands = [
            (0.20, "prompt-only tweak"),
            (0.40, "prompt, optional tool change"),
            (0.60, "topology, prompts, handoff format"),
            (0.80, "bold rewire — restructure or grow the agent set"),
            (1.01, "complete rethink — discard inherited topology"),
        ]
        scope = next(label for threshold, label in bands if stagnation < threshold)
        return f"Mutation scope: {scope}. Stagnation: {stagnation*100:.2f}%. Use at most {n_agents} agent(s).\n"

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

    def random_topology_prompt(self) -> str:
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
        """
        Prompt for very first workflow generation (generation 0).
        """
        n_agents = self._sample_agent_count(0.5, 1, 5)  # start with small random agent count
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
        """
        Build a prompt for one mutation step in the evolutionary search.

        The prompt has three layers:
          1. Voice framing   — sets the LLM's reasoning tone for this iteration.
          2. Execution grounding — previous code, agent answers, and judge eval.
        """
        score      = wf_info.overall_score        if wf_info else 0.0
        prompt_gradient  = wf_info.abstracted_prompt_gradient if wf_info else ""
        wf_state   = wf_info.state_result         if wf_info else None

        # ── Execution evidence ───────────────────────────────────────────────
        agent_answers = self._extract_agent_answers(wf_state)
        prompt_gradient_block = (
            prompt_gradient.strip()
            if prompt_gradient and prompt_gradient.strip()
            else (run_stderr or "FAILURE:unknown failure").strip()
            or "NO_prompt_gradient:No prompt_gradient captured."
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
                "<agents_answers>",
                agent_answers,
                "</agents_answers>",
                "<prompt_gradient>",
                "",
                prompt_gradient_block,
                "</prompt_gradient>",
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
            "1. Identify which structural decisions worked in each parent (look at agent answers, not just score).",
            "2. Identify which decisions failed and why.",
            "3. Build a new workflow that inherits the strongest sub-structures across parents",
            "   and discards the weakest, even if that means a topology none of the parents used.",
        ])


if __name__ == "__main__":
    np.random.seed(0)
    _FAILURE_SENTINEL = (
        "workflow code failed to generate or execute; "
        "ensure code is properly formatted and that the workflow runs without crashing"
    )

    # ── Fix #3: identical failure sentinels must NOT peg stagnation ──────
    ve = VariationEngine()
    for _ in range(4):
        ve.record_offspring_gradient(_FAILURE_SENTINEL, is_failure=True)
    assert ve._compute_stagnation() == 0.0, "failure sentinels must be filtered"

    # ── Fix #2: a near-winning parent damps effective stagnation ─────────
    ve = VariationEngine()
    for _ in range(4):
        ve.record_offspring_gradient("INCONSISTENT_MULTITASK_SPLIT: same failure mode repeating")
    raw = ve._compute_stagnation()
    block_high_score = ve._get_prompt_step_size(parent_score=0.97)
    block_low_score = ve._get_prompt_step_size(parent_score=0.10)
    assert raw > 0.8, f"expected high raw stagnation, got {raw:.2f}"
    assert "prompt-only tweak" in block_high_score, (
        f"high-score parent should damp scope; got:\n{block_high_score}"
    )
    assert "rethink" in block_low_score or "rewire" in block_low_score, (
        f"low-score parent under high stagnation should widen scope; got:\n{block_low_score}"
    )

    # ── Fix #1 (semantic): diverse offspring gradients keep stagnation low ─
    ve = VariationEngine()
    diverse = [
        "TIMEOUT_AGENT_3: agent 3 hit the wall-clock limit while iterating over the full dataset.",
        "JUDGE_REJECTED_FORMAT: CSV used semicolons as separators; enforce commas.",
        "EMPTY_HANDOFF: agent 2 returned empty string; add validation gate.",
        "INFINITE_LOOP_PLANNER: planner re-emitted same plan; route critic output back.",
    ]
    for g in diverse:
        ve.record_offspring_gradient(g)
    assert ve._compute_stagnation() < 0.5, (
        f"diverse offspring gradients should not stagnate; got {ve._compute_stagnation():.2f}"
    )

    print("smoke OK: failure-sentinel filter, parent-score damper, diverse-offspring ↦ low stagnation")