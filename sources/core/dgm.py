"""
Darwinian Evolution of multi-agent workflows improvements.
"""

import json
import logging
import os
import re
import time
from pathlib import Path

import random

from sources.utils.notify import PushNotifier
from sources.utils.pricing import PricingCalculator
from sources.utils.shared_visualization import SharedVisualizationData
from sources.utils.visualization import VisualizationUtils
from sources.evaluation.scenario_loader import ScenarioLoader

from .orchestrator import WorkflowOrchestrator
from .workflow_info import WorkflowInfo
from .workflow_selection import WorkflowSelector
from .schema import IndividualRun, ImprovementLog
from .improvement_validator import ImprovementValidator
from sources.cli.pretty_print import (
    print_ok, print_warn, print_err, print_info,
    print_phase, print_section,
    print_iteration_header, print_box,
    print_summary, print_agent_answers,
    CYAN, GREEN, YELLOW, RED, DIM, RESET, BOLD,
)


def check_answer_success(answer: str) -> bool:
    """Check if an answer indicates success using pattern matching.
    Args:
        answer: The answer string to check
    Returns:
        bool: True if answer indicates success, False otherwise
    """
    answer_lower = str(answer).lower()
    failure_patterns = [
        r'\bfailed\b', r'\berror\b', r'\bfailure\b',
    ]
    return all(not re.search(pattern, answer_lower) for pattern in failure_patterns)

def evaluate_workflow_success(wf_info: WorkflowInfo, answers: list) -> bool:
    """
    Evaluate workflow success using multiple criteria.
    Args:
        wf_info: WorkflowInfo object containing state and evaluation results
        answers: List of answers from workflow agents
    Returns:
        bool: True if workflow is considered successful, False otherwise
    """
    if wf_info.state_result and 'evaluation' in wf_info.state_result:
        eval_data = wf_info.state_result['evaluation']

        if 'scenario' in eval_data and eval_data['scenario']:
            passed = eval_data['scenario'].get('passed_assertions', 0)
            total = eval_data['scenario'].get('total_assertions', 1)
            return (passed / total) >= 0.8
        if 'generic' in eval_data and eval_data['generic']:
            score = eval_data['generic'].get('overall_score', 0.0)
            return score >= 0.7
    if answers:
        return check_answer_success(answers[-1])
    return False

class PromptGradient:
    def __init__(self):
        """
        hints to provide diversity in generated workflows, we sample 3 of them at random for each workflow generation
        provide an approximate search direction to the LLM to improve workflow quality and diversity
        """
        self.rn_seed = int(time.time() * 1000) % 2**32
        self.hints = [
            # === TOPOLOGY & ORCHESTRATION ===
            ("parallel_fanout", "Explore a parallel fan-out topology where independent subtasks run concurrently (or divide tasks sequentially) before a merge agent consolidates."),
            ("debate_topology", "Consider a debate topology: two agents independently produce outputs, a third arbitrates and synthesizes the best elements."),
            ("map_reduce", "Explore a map-reduce pattern where a splitter agent decomposes the input, workers process chunks, a reducer assembles the final result."),
            ("dynamic_routing", "Add adaptive routing that selects the next agent based on intermediate uncertainty signals rather than fixed paths."),
            ("hierarchical_decomposition", "Use hierarchical decomposition: high-level agents break complex goals into subtasks and delegate to specialized workers."),
            # === QUALITY / VALIDATION ===
            ("adversarial_agent", "Add a dedicated adversarial agent whose only job is to find flaws, edge cases, and failure modes in prior agents' outputs."),
            ("confidence_scoring", "Include a confidence-scoring step where agents explicitly rate their own output certainty before passing downstream."),
            ("peer_review_loop", "Consider a peer-review loop: the second agent critiques the first's output before a third agent makes the final call."),
            ("cross_verification", "Implement cross-verification where agents check each other's work using different reasoning paths to catch blind spots."),
            ("structured_metadata_passing", "Require agents to pass uncertainty bounds, assumptions, and limitations as explicit metadata—not just outputs."),
            # === FALLBACK / RESILIENCE ===
            ("heuristic_fallback", "Design fallback paths that skip expensive agents and use a cheaper heuristic when upstream confidence is low."),
            ("circuit_breaker", "Consider circuit-breaker logic: if an agent fails twice on the same input, escalate to a more capable model instead of retrying."),
            ("triage_routing", "Add a triage agent at entry that classifies input complexity and routes to a fast-path or deep-analysis path accordingly."),
            ("budget_enforcement", "Enforce explicit token/call budgets per agent to prevent resource grabbing races that starve the system."),
            ("checkpoint_resume", "Design checkpointing into long workflows so partial states can resume after failure without full restart."),
            # === SPECIALIZATION & CONTEXT ===
            ("orthogonal_specialization", "Consider decomposing the problem domain into orthogonal concerns, each owned by a hyper-specialized agent."),
            ("normalization_first", "Explore using one agent purely for data extraction/normalization before any reasoning agents touch the content."),
            ("explicit_planning", "Add a planning agent that produces an explicit step-by-step execution plan that downstream agents must follow and can annotate."),
            ("context_scoping", "Scope context intentionally—execution agents get narrow, relevant context rather than full conversation history to prevent drift."),
            ("role_boundary_guardrails", "Enforce strict role boundaries so agents don't silently assume each other's responsibilities."),
            # === OUTPUT QUALITY ===
            ("refinement_loop", "Consider a multi-pass refinement loop where the final agent can send output back for one revision cycle if quality threshold isn't met."),
            ("audience_rewrite", "Add an agent that rewrites the final output for the target audience's format and vocabulary before delivery."),
            ("deduplication", "Include a deduplication/consolidation agent to merge redundant findings when multiple agents explore overlapping territory."),
            ("synthesis_failure_guard", "Watch for synthesis failure: when parallel agents return contradictory or uneven outputs, trigger a reconciliation sub-workflow."),
            # === EFFICIENCY ===
            ("gating_agent", "Consider a gating agent that decides whether the full workflow is needed or if a cached/simple answer suffices."),
            ("early_termination", "Add explicit termination conditions with verification—prevent premature exits before objectives are actually met."),
            ("redundancy_deduplication", "Track which experimental configurations have already been explored to prevent redundant computation across agents."),
            # === SCIENTIFIC-SPECIFIC ===
            ("hypothesis_tracking", "Maintain a shared hypothesis registry so agents know what has been tested and what remains open—prevents redundant exploration."),
            ("provenance_logging", "Require every output to include provenance: which data sources, which assumptions, which agent chain produced it."),
            ("uncertainty_quantification", "Add explicit uncertainty propagation—when agents combine findings, they must aggregate uncertainty, not just point estimates."),
            ("reproducibility_bundle", "Bundle code, data references, and random seeds with every result so downstream agents can verify or reproduce."),
            ("domain_validator", "Include a domain-specific validator agent that checks outputs against scientific conventions (units, significant figures, citation formats)."),
            ("instrumental_alignment_check", "Watch for instrumental goal alignment failure—agents hiding information to appear better than they are."),
            ("world_model_sync", "Periodically force agents to synchronize their internal world models to prevent competing assumptions from diverging."),
            ("error_propagation_barrier", "Design error propagation barriers—one agent's bad assumption shouldn't cascade through the entire system unchecked."),
            ("convergence_budget", "Set a maximum iteration budget for consensus-building to prevent infinite negotiation loops between agents."),
            ("silent_deadlock_detector", "Monitor for silent deadlocks where agents all wait for someone else to act—implement timeout escalation."),
            # === SEARCH DIRECTION & EXPLORATION ===
            ("simulated_annealing_search", "Treat exploration like simulated annealing: start with high-temperature, highly divergent agent prompts for brainstorming, then systematically lower temperature for convergent refinement."),
            ("dead_end_backtracking", "Implement an explicit 'negative memory' cache. When agents hit a dead end in a research path, log the failure so parallel/future agents don't retread the same useless search space."),
            ("diversity_forcing", "Combat agent echo chambers by injecting explicit orthogonal biases into parallel search agents (e.g., 'Agent A favors biological explanations, Agent B favors chemical ones') to fully map the hypothesis space."),
            # === AVOIDING UNNECESSARY COMPLEXITY ===
            ("single_agent_baseline", "Always establish a single-agent baseline first. Only introduce multi-agent orchestration when the single agent demonstrably fails due to context limits or competing objectives."),
            ("agent_consolidation", "Avoid 'micro-service' bloat. If two agents always operate sequentially without conditional branching, human-in-the-loop, or distinct tool needs, combine them into one prompt."),
            ("communication_overhead_tax", "Monitor the token-ratio of formatting/handshakes versus actual scientific reasoning. If agents spend more tokens packing/unpacking JSON than thinking, simplify the workflow."),
            ("avoid_premature_abstraction", "Don't build generalized hierarchical frameworks for a specific scientific pipeline until you've successfully hardcoded the exact path at least once."),
            # === SMOOTH MANIFOLD TRANSITIONS (AGENT HANDOFFS) ===
            ("semantic_impedance_matching", "Ensure smooth transitions by defining strict, validated schema contracts (e.g., Pydantic) between agents. Don't rely on unstructured natural language for critical data handoffs."),
            ("sliding_context_window", "Create a smooth cognitive transition by passing the previous agent's summarized 'train of thought' alongside the final output, preventing abrupt context shifts downstream."),
            ("shared_ontology_sync", "Initialize the entire agent swarm with a shared scientific ontology/glossary. This prevents downstream agents from hallucinating or misinterpreting specialized terminology used by upstream agents."),
            ("state_dictionary_continuity", "Maintain a continuous, system-level 'experiment state' dictionary (JSON/YAML) independent of the conversational thread. Update it mutably across agent transitions so quantitative data doesn't degrade in natural language translation."),
            ("lossless_data_pointers", "Never force agents to copy-paste large datasets or matrices in their text outputs. Pass file paths or database pointers during handoffs to maintain data fidelity and smooth the transition manifold.")
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

    def sample_workflow_hints(self, hints: list[tuple[str, str]], n: int, seed: int | None = None) -> list[tuple[str, str]]:
        rng = random.Random(seed)
        sampled = rng.sample(hints, min(n, len(hints)))
        return sampled

    def get_hints(self) -> tuple[list[str], list[str]]:
        sampled_hints = self.sample_workflow_hints(self.hints, n=3, seed=self.rn_seed)
        self.rn_seed += 1  # Advance seed so next call samples different hints
        hints_names = [hint[0] for hint in sampled_hints]
        hints = [hint[1] for hint in sampled_hints]
        return hints, hints_names

    def improvement_prompt(
        self,
        goal: str,
        wf_info: WorkflowInfo,
        flow_code: str,
        run_stderr: str,
        iteration_count: int
    ) -> str:
        exec_result = ""
        agents_answers = None
        wf_state = wf_info.state_result if wf_info else None
        judge_eval = wf_info.judge_evaluation if wf_info else None
        hints, hints_names = self.get_hints()
        hints_display = "\n".join(f"{i+1}. {name}: {desc}" for i, (name, desc) in enumerate(zip(hints_names, hints), start=1))

        if wf_state: # alternative: use agents answer dict
            agents_answers = self.get_flow_answers(wf_state)

        if judge_eval: # use judge eval if possible
            exec_result = judge_eval
        else: # use stdout/stderr if execution failed
            exec_result = run_stderr.strip()

        print(f"\n\033[95m{'WORKFLOW IMPROVEMENT HINTS':^80}\033[0m")
        print(hints_display)

        improv_prompt = "Previous attempt failed. Learn from mistakes and improve the multi-agent workflow."
        if flow_code is not None:
            improv_prompt = "\n".join([
                "## SELF-IMPROVEMENT STEP",
                "Your previous attempt at generating a workflow did not succeed or didn't reach success threshold.",
                "Your goal was: ",
                goal,
                "\n",
                "## Previous workflow code:",
                "<python>",
                flow_code,
                "</python>",
                "\n",
                "## EXECUTION RESULTS::",
                "This is the answer from each agent during the last execution.",
                "<agents_answers>",
                agents_answers if wf_state else "No agent answers captured.",
                "</agents_answers>",
                "This is the judge evaluation if available, otherwise this is the execution error or stderr output.",
                "<results>",
                exec_result,
                "</results>",
                "Reflect on your previous attempt and identify what went wrong.",
                "\n",
                "## ANALYZE FAILURES:",
                "1. If the workflow code execution failed, analyze the error and fix the python code. If no workflow was generated, this is most likely because prompt were too long,  invalid syntax or ``` delimitation missing (```python...```)",
                "2. If the workflow executed but didn't achieve the goal, analyze the agent answers and judge evaluation to identify which part of the workflow underperformed and why.",
                "3. Identify the failure mode: was it a reasoning failure, an agent miscommunication, a missing tool, or an inefficient orchestration?",
                "4. Identify one key improvement that would most likely fix the failure mode. It can be a change in the workflow topology, a new agent specialization, an added validation step, or a more efficient orchestration pattern. Be creative and use the hints below for inspiration.",
                "\n",
                "## SUGGESTED IMPROVEMENT:",
                hints_display,
                "These are only random suggestions to inspire you, you don't have to use them but they might help you find a good improvement idea. Select one if it seems relevant to the failure mode you identified, or ignore if not relevant.",
                "## GENERATE ONE IMPROVEMENT:",
                "Choose one SINGLE most impactful change.",
                "Apply ONLY that one change to the code.",
                "Explain what will improve and why.",
                "\n",
                "Your improvement will be empirically validated against the baseline.",
            ])

        return "".join(
            [
                f"Attempt {iteration_count + 1} of workflow generation.\n",
                improv_prompt,
                "Target goal:\n",
                goal,
            ]
        )

class DarwinMachine:
    """Darwin Machine for evolution of workflow workflows."""
    def __init__(
        self,
        config,
        viz_utils: VisualizationUtils = None,
        process_id: int = None,
    ) -> None:
        from sources.evaluation.evaluator import WorkflowEvaluator
        self.config = config
        self.workflow_dir = config.workflow_dir
        self.model_pricing = config.model_pricing
        self.workflow_selector = WorkflowSelector(config)
        self.orchestrator = WorkflowOrchestrator(config)
        self.judge = WorkflowEvaluator(config)
        self.improvement_validator = ImprovementValidator(min_improvement_threshold=0.05)
        self.prompt_gradient = PromptGradient()
        self.notifier = PushNotifier(config.pushover_token, config.pushover_user)
        self.viz_utils = viz_utils or VisualizationUtils()
        self.process_id = process_id
        self.pricing = PricingCalculator(config)
        self.logger = logging.getLogger(__name__)

    def load_wf_state_result(self, uuid: str) -> any:
        """Load the result of a previously executed workflow state.
        Args:
            uuid: UUID of the workflow state to load
        Returns:
            str: The output of the workflow state if found, None otherwise
        """
        try:
            with open(f"{self.workflow_dir}/{uuid}/state_result.json") as f:
                return json.loads(f.read().strip())
        except FileNotFoundError:
            print(f"Workflow state for UUID {uuid} not found in {self.workflow_dir}.")
            return None
        except Exception as e:
            raise ValueError(f"❌ Error reading workflow state: {str(e)}") from e

    def load_workflow_code(self, workflow_id: str) -> str:
        """
        Load the workflow code for a given workflow ID.
        """
        workflow_path = f"{self.workflow_dir}/{workflow_id}"
        if not os.path.exists(workflow_path):
            raise ValueError(
                f"❌ Workflow for ID {workflow_id} not found in {self.workflow_dir}."
            )

        try:
            with open(f"{workflow_path}/workflow_code_{workflow_id}.py") as f:
                return f.read()
        except FileNotFoundError as e:
            raise ValueError(
                f"❌ Workflow code file not found for ID {workflow_id} in {workflow_path}."
            ) from e
        except Exception as e:
            raise ValueError(f"❌ Error reading workflow code: {str(e)}") from e

    def get_total_rewards(self, wf_state: any, eval_type: str) -> float:
        """Calculate the total rewards from the workflow state."""
        if not wf_state or not eval_type:
            return 0.0
        if eval_type == "generic":
            return wf_state["evaluation"]["generic"]["overall_score"]
        elif eval_type == "scenario":
            return wf_state["evaluation"]["scenario"]["score"]
        else:
            return 0.0

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

    def show_answers(self, flow_answers):
        print_box(flow_answers, title="Workflow Agents Answers", color=YELLOW)


    def select_workflow_template(self, goal, template_uuid: str = None) -> WorkflowInfo:
        """Select and load a workflow template from the workflow directory or by UUID.
        Args:
            template_uuid: Optional UUID of workflow template to load
        Returns:
            str: The workflow template content if found, None otherwise
        """
        if not os.path.exists(self.workflow_dir):
            print(f"Workflow directory {self.workflow_dir} does not exist.")
            return None
        workflows = [f for f in os.listdir(self.workflow_dir)]
        if not workflows:
            print(f"No workflows found in {self.workflow_dir}.")
            return None

        # default to selecting best workflow if no template UUID provided
        if template_uuid is None:
            candidates = self.workflow_selector.select_best_workflows(
                goal=goal,
                threshold_similarity=0.8,
                threshold_score=0.0,
            )
            print_section("🎯 WORKFLOW SELECTION")
            print_info(f"Selected {len(candidates)} candidate(s)")
            print_info(f"Top candidate: {candidates[0].uuid if candidates else 'None'}")
            return WorkflowInfo(candidates[0].uuid, Path(f"{self.workflow_dir}/{candidates[0].uuid}")) if candidates else None
        return WorkflowInfo(template_uuid, Path(f"{self.workflow_dir}/{template_uuid}"))

    def get_craft_instructions(self, goal, wf):
        if wf:
            return self.prompt_gradient.improvement_prompt(
                goal, wf, wf.code, "", 0
            )
        else:
            return goal

    async def start_dgm(
        self,
        goal: str,
        template_uuid: str | None = None,
        judge: bool = True,
        scenario_rubric: str = None,
        max_iteration: int = 1,
        learning_mode: bool = False,
        original_task: str = None,
        single_agent_mode: bool = False,
        mockup_mode: bool = False
    ) -> list[IndividualRun]:
        """
        Start the learning process for achieving a specified goal.
        Args:
        - goal (str): The primary goal or objective to be accomplished (may be knowledge-wrapped).
         template_uuid (str | None, optional): UUID of a workflow template to use.
        - judge (bool, optional): Whether to enable judging mode for evaluation.
        - scenario_rubric (str, optional): ID of scenario for evaluation.
        - max_iteration (int): Maximum number of iterations.
        - learning_mode (bool): Whether in learning mode. Will keep attempt at improving workflow score even if all agents report success state.
        - original_task (str, optional): Original unwrapped task for similarity matching.
        - mockup_mode (bool, optional): If True, use existing workflow data from select_workflow_template
            instead of calling orchestrate_workflow. Useful for testing and debugging.
        """
        if max_iteration is None:
            max_iteration = 1
        if learning_mode:
            max_iteration = self.config.max_learning_evolve_iterations if max_iteration <= 1 else max_iteration

        wf = self.select_workflow_template(
            goal, template_uuid=template_uuid
        )

        # Mockup mode: use existing workflow data without calling orchestrate_workflow
        if mockup_mode:
            if wf is None:
                raise ValueError("❌ Mockup mode requires a valid workflow template. "
                                 "Please provide a template_uuid or ensure workflows exist in the workflow directory.")
            print_phase("MOCKUP MODE", color="\033[93m")
            print_info(f"Using existing workflow data from: {wf.uuid}")
            mock_run = IndividualRun(
                goal=wf.goal or goal,
                prompt=wf.code or goal,
                template_uuid=template_uuid or wf.uuid,
                workflow_template=wf,
                max_depth=max_iteration,
                judge=judge,
                scenario_rubric=scenario_rubric,
                original_task=original_task or wf.original_task,
                current_uuid=wf.uuid,
                answers=wf.answers,
                state_result=wf.state_result,
                reward=wf.overall_score,
                iteration_count=1
            )
            flow_answers = self.get_flow_answers(wf.state_result)
            self.show_answers(flow_answers)
            print_ok(f"Mockup run completed with reward: {wf.overall_score:.1f}")
            return [mock_run]

        craft_instructions = self.get_craft_instructions(goal, wf)

        rewards_history = []
        assertion_history = []  # Track [passed, total] per iteration
        self.viz_utils.create_rewards_curve_plot(goal)

        run0 = IndividualRun(
            goal=goal,
            prompt=craft_instructions,
            template_uuid=template_uuid,
            workflow_template=wf,
            max_depth=max_iteration,
            judge=judge,
            scenario_rubric=scenario_rubric,
            original_task=original_task
        )

        return await self.recursive_self_evolution(
            [run0],
            rewards_history=rewards_history,
            assertion_history=assertion_history,
            learning_mode=learning_mode,
            single_agent_mode=single_agent_mode
        )

    async def recursive_self_evolution(
        self,
        runs: list[IndividualRun],
        rewards_history: list[float] = None,
        assertion_history: list[list[int]] = None,
        learning_mode: bool = False,
        single_agent_mode: bool = False
    ):
        """Run a self-improvement loop for the workflow."""
        self._log_iteration_start(runs[-1].goal, runs[-1].iteration_count, runs[-1].max_depth)

        iteration_start_time = time.time()
        on_error = False
        uuid = None
        current_iteration_cost = 0.0  # Cost for this iteration only, not cumulative

        # Execute workflow
        print_info(f"Run {runs[-1].iteration_count + 1} of {runs[-1].max_depth}")
        run_stdout, uuid, workflow_code, executed = await self.orchestrator.orchestrate_workflow(
            goal=runs[-1].goal,
            craft_instructions=runs[-1].prompt,
            original_task=runs[-1].original_task,
            single_agent_mode=single_agent_mode
        )
        wf_info = WorkflowInfo(uuid, Path(f"{self.workflow_dir}/{uuid}"))
        if "WORKFLOW_GENERATION_ERROR" in run_stdout:
            print_err(f"Workflow generation failed:\n{run_stdout[256:]}")
            on_error = True
        if workflow_code:
            # Evaluate and calculate costs
            eval_type, current_iteration_cost = await self._evaluate_and_calculate_cost(
                executed, runs[-1].judge, uuid, runs[-1].answers, runs[-1].scenario_rubric, assertion_history
            )
            runs[-1].reward = wf_info.overall_score

        runs[-1].current_uuid = uuid
        runs[-1].answers = wf_info.answers
        runs[-1].state_result = wf_info.state_result
        flow_answers = self.get_flow_answers(wf_info.state_result)
        self.show_answers(flow_answers)
        rewards_history.append(wf_info.overall_score)

        if runs[-1].iteration_count > 0:
            validation_result = self.improvement_validator.validate_improvement(
                baseline_runs=runs[-5:],
                new_runs=runs[-1:],
                threshold=0.05  # 5% improvement threshold
            )

            improvement_log = ImprovementLog(
                from_iteration=runs[-2].iteration_count,
                to_iteration=runs[-1].iteration_count,
                improvement_type=self.improvement_validator.get_improvement_type(runs[-2], runs[-1]),
                delta_reward=validation_result["absolute_improvement"],
                is_validated=validation_result["valid"],
                confidence=validation_result["confidence"]
            )

            if not hasattr(runs[-1], 'improvement_history'):
                runs[-1].improvement_history = []
            runs[-1].improvement_history.append(improvement_log)

            self.logger.info(f"[IMPROVEMENT VALIDATION] {improvement_log}")

        # Update visualizations
        self._update_visualizations(
            rewards_history, assertion_history,
            runs[-1].goal, runs[-1].scenario_rubric, uuid
        )

        # Calculate cumulative cost and update runs[-1].cost for accurate tracking
        runs[-1].cost = runs[-1].cost + current_iteration_cost

        # Log and notify completion (show per-iteration cost, not cumulative)
        self._log_iteration_completion(
            runs[-1].iteration_count, runs[-1].max_depth, iteration_start_time,
            wf_info.overall_score, current_iteration_cost, runs[-1].goal, uuid, wf_info.state_result, rewards_history
        )

        all_success = evaluate_workflow_success(wf_info, runs[-1].answers)

        # Check termination conditions
        if runs[-1].iteration_count >= runs[-1].max_depth-1 and not on_error:
            print_info("Maximum recursive depth reached.")
            return runs
        if learning_mode and wf_info.overall_score > self.config.learned_score_threshold:
            # reach learning threshold
            print_ok("DGM done learning task.")
            self._save_final_plots(assertion_history, rewards_history, uuid)
            self.notifier.send_message(
                f"Done learning task: {wf_info.goal[:256]} \n"
                f"Final UUID: {uuid}\n"
                f"Iterations: {runs[-1].iteration_count + 1}/{runs[-1].max_depth}\n",
                title="Evolution done learning task.",
                priority=0
            )
            return runs
        elif not on_error:
            if not learning_mode and all_success:
                self._save_final_plots(assertion_history, rewards_history, uuid)
                print_ok("DGM completed task successfully.")
                self.notifier.send_message(
                    f"Task completed successfully!\n"
                    f"Goal: {runs[-1].goal[:128]}...\n"
                    f"Final UUID: {uuid}\n"
                    f"Iterations: {runs[-1].iteration_count + 1}/{runs[-1].max_depth}\n"
                    f"All workflows successful!",
                    title=f"Evolution success - {uuid}",
                    priority=0
                )
                return runs

        # select and use best scoring workflow
        wf_info_best = self.select_workflow_template(
            runs[-1].goal, template_uuid=None
        )
        code = wf_info_best.code if wf_info_best else ""
        runs[-1].prompt = self.prompt_gradient.improvement_prompt(
            runs[-1].original_task or runs[-1].goal, wf_info_best, code, run_stdout, runs[-1].iteration_count
        )

        runs.append(IndividualRun(
            goal=runs[-1].goal,
            prompt=runs[-1].prompt,
            cost=runs[-1].cost,  # Correct cumulative cost
            current_uuid=uuid,
            template_uuid=None,
            workflow_template=runs[-1].workflow_template if wf_info.state_result else None,
            iteration_count=runs[-1].iteration_count + 1,
            max_depth=runs[-1].max_depth,
            judge=runs[-1].judge,
            answers=wf_info.answers,
            state_result=wf_info.state_result,
            scenario_rubric=runs[-1].scenario_rubric,
            original_task=runs[-1].original_task  # PRESERVE original_task for workflow selection
        ))

        runs = await self.recursive_self_evolution(
            runs,
            rewards_history=rewards_history,
            assertion_history=assertion_history,
            learning_mode=learning_mode,
            single_agent_mode=single_agent_mode
        )

        runs[-1].plot = self._save_final_plots(assertion_history, rewards_history, uuid)
        return runs

    def _get_human_validation(self) -> bool:
        """Get human validation for continuing the workflow."""
        human_validation = input("Attempt to retry task? (yes/no): ").strip().lower()
        if human_validation not in ["yes", "y"]:
            print("Exiting self-improvement loop.")
            return False
        return True

    def _log_iteration_start(self, goal: str, iteration_count: int, max_depth: int):
        """Log the start of an iteration."""
        logger = logging.getLogger(__name__)
        print_iteration_header(iteration_count + 1, max_depth)
        print_box(goal, title="📋 CURRENT TASK", truncate=256)
        logger.info(f"[ITERATION START] {iteration_count + 1}/{max_depth} - {goal[:50]}...")

    async def _evaluate_and_calculate_cost(
        self, executed: bool, judge: bool, uuid: str,
        agent_answers: str, scenario_rubric: str, assertion_history: list
    ) -> tuple[str, float]:
        """Evaluate workflow and calculate cost."""
        logger = logging.getLogger(__name__)
        eval_type = None
        exec_cost = 0.0

        if judge and uuid:
            agent_answers = agent_answers if executed else "workflow failed to execute."
            eval_type = await self._evaluate_workflow(uuid, agent_answers, scenario_rubric, assertion_history)
        # Calculate cost regardless of execution success
        cost_start = time.time()
        exec_cost = self.pricing.calculate_cost(uuid)
        cost_time = time.time() - cost_start
        logger.info(f"[WORKFLOW COST] {uuid} cost calculated in {cost_time:.3f}s")

        return eval_type, exec_cost

    async def _evaluate_workflow(
        self, uuid: str, agent_answers: str, scenario_rubric: str, assertion_history: list
    ) -> str:
        """Evaluate the workflow and update assertion history."""
        logger = logging.getLogger(__name__)
        print_phase("⚖️  WORKFLOW EVALUATION PHASE")
        eval_start = time.time()
        eval_result = self.judge.evaluate(uuid=uuid, agent_answers=agent_answers, scenario_rubric=scenario_rubric)
        eval_type = 'scenario' if scenario_rubric else 'generic'
        eval_time = time.time() - eval_start
        logger.info(f"[WORKFLOW EVALUATION] {uuid}:\n{json.dumps(eval_result, indent=2)}")
        print_ok(f"Workflow evaluation completed in {eval_time:.3f}s")
        # Track assertion progress for scenario evaluation
        if scenario_rubric and isinstance(eval_result, dict) and assertion_history is not None:
            self._update_assertion_history(eval_result, assertion_history)
        return eval_type

    def _update_assertion_history(self, eval_result: dict, assertion_history: list):
        """Update assertion history with evaluation results."""
        passed = eval_result.get('passed_assertions', eval_result.get('earned_points', 0))
        total = eval_result.get('total_assertions', eval_result.get('total_points', 100))
        assertion_history.append([passed, total])
        pct = passed / total * 100 if total > 0 else 0
        print_info(f"📊 Assertions progress: {passed}/{total} ({pct:.0f}%)")

    def _update_visualizations(
        self, rewards_history: list, assertion_history: list,
        goal: str, scenario_rubric: str, uuid: str
    ):
        """Update all visualizations with current data."""
        if assertion_history:
            self._update_assertion_plot(assertion_history, scenario_rubric, uuid)
        elif rewards_history:
            self._update_rewards_plot(rewards_history)

    def _update_rewards_plot(self, rewards_history):
        self.viz_utils.update_rewards_curve(rewards_history)

    def _update_assertion_plot(
        self, assertion_history: list,
        scenario_rubric: str, uuid: str
    ):
        """Update assertion progress plot."""
        from sources.evaluation.scenario_loader import ScenarioLoader
        scenario = ScenarioLoader().load_scenario(scenario_rubric)
        total_assertions = len(scenario.get("assertions", [])) if scenario else 0
        self.viz_utils.update_assertion_progress_plot(assertion_history, total_assertions)
        plot_filename = f"{self.workflow_dir}/{uuid}/assertion_progress.png"
        self.viz_utils.save_plot(plot_filename)
        print_info(f"📊 Assertion progress plot updated: {plot_filename}")

    def _log_iteration_completion(
        self, iteration_count: int, max_depth: int, iteration_start_time: float,
        wf_rewards: float, exec_cost: float, goal: str, uuid: str,
        wf_state: any, rewards_history: list
    ):
        """Log iteration completion and send notification."""
        logger = logging.getLogger(__name__)
        iteration_time = time.time() - iteration_start_time
        logger.info(
            f"[ITERATION END] {iteration_count + 1}/{max_depth} completed in {iteration_time:.3f}s - "
            f"Rewards: {wf_rewards:.1f}, Cost: {exec_cost:.3f} USD"
        )
        print_summary(
            f"ITERATION {iteration_count + 1}/{max_depth} COMPLETE",
            [
                ("Rewards", f"{wf_rewards:.1f}"),
                ("Cost", f"${exec_cost:.6f}"),
                ("Time", f"{iteration_time:.3f}s"),
            ],
        )
        self.notifier.send_message(
            f"Iteration {iteration_count + 1} completed.\n"
            f"Goal: {goal[:128]}...\n"
            f"Cost: {exec_cost:.6f} USD.\n"
            f"Rewards history: {rewards_history}"
            f"Answers: {self.get_flow_answers(wf_state)}\n",
            title=f"Workflow {uuid} completed.",
        )

    def _save_final_plots(self, assertion_history: list, reward_history: list, uuid: str) -> str:
        """Save final assertion plots."""
        plot_filename = ""
        if assertion_history or reward_history:
            plot_filename = f"{self.workflow_dir}/{uuid}/reward_progress.png"
            self.viz_utils.save_plot(plot_filename)
        print_info(f"📊 Reward progress plot saved: {plot_filename}")
        return plot_filename
