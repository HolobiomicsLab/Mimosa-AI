"""Shared data model for Mimosa-AI: enums and dataclasses crossing all layers.

This module owns the plain data types every layer passes around:
``TaskStatus`` and ``TaskComplexity`` enums; ``SelectionLog`` and
``IndividualRun`` for the Layer 2 evolution engine; ``Plan``, ``PlanStep``,
and ``Task`` for the Layer 0 planner. The dataclasses carry state, validate
their own invariants on construction, and hold no behaviour beyond ``__str__``
summaries. ``sources.core`` re-exports them, and the planner, the evolution
engine, the mock-data utility, the planner visualization, and the tests
import them from here. The module loads at import time; its types are
constructed throughout every run.
"""

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

class TaskStatus(Enum):
    """Lifecycle status of a plan step or task.

    The planner sets it when a step starts, finishes, or fails, and the
    visualization and reporting read it. Valid values are exactly "pending",
    "running", "completed", "failed", and "skipped"; only COMPLETED counts
    as done for dependency gating in the planner.
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class TaskComplexity(Enum):
    """Complexity label a plan step can carry.

    The planner LLM proposes it per step, ``PlanStep`` validates against it,
    and the console plan display shows it. Valid values are exactly "low",
    "medium", and "high".
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# class for tracking validated improvements
@dataclass
class SelectionLog:
    """Record of one validated improvement applied during evolution.

    The evolution engine constructs it when a variation is admitted and
    stores it on the owning ``IndividualRun.selection_log``. It keeps the
    from/to iterations, the improvement kind, the reward delta, and the
    validation verdict with its confidence.

    Attributes:
        from_iteration: Iteration index the improvement came from (int).
        to_iteration: Iteration index the improvement was applied to (int).
        improvement_type: Free-text kind of improvement (str); comes from
            the variation operator.
        delta_reward: Reward change attributed to the improvement (float).
        is_validated: Whether the improvement passed validation (bool).
        confidence: Validator confidence in [0, 1] (float, default 0.0).
        admit_rejected: Whether an improvement rejected by validation was
            still admitted (bool, default False).
        timestamp: Construction time (datetime, set automatically).
    """
    from_iteration: int
    to_iteration: int
    improvement_type: str
    delta_reward: float
    is_validated: bool
    confidence: float = 0.0
    admit_rejected: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        """Return a human-readable summary of the selection log entry.

        Logging and console output call it. Returns ``str``: validation
        status, improvement type, signed reward delta, and confidence.
        """
        status = "✅ VALIDATED" if self.is_validated else "⚠️ NOT VALIDATED"
        return (f"SelectionLog({status}, type={self.improvement_type}, "
                f"delta={self.delta_reward:+.3f}, confidence={self.confidence:.0%})")

@dataclass
class IndividualRun:
    """One evolved workflow run and its evaluation outcome.

    The evolution engine constructs one per generated workflow and passes
    instances to the planner, the evaluators, and the persistence layer.
    Each instance carries the goal and prompt, cost and reward, lineage
    (``template_uuid``, ``parent_uuids``, ``evolution_kind``), and the
    verifier state (``answers``, ``state_result``). Invariant: ``reward``
    is the capped reward used for selection, while ``reward_uncapped``
    keeps the raw value when a cap applies.

    Attributes:
        goal: Task goal this run solves (str).
        prompt: Prompt actually sent to the workflow (str).
        cost: Total LLM cost of the run (float, default 0.0).
        reward: Selection reward; capped when applicable (float, default 0.0).
        reward_uncapped: Raw reward before any cap (float, default 0.0).
        reward_is_fallback: True when the reward is a fallback value rather
            than a verifier verdict (bool, default False).
        max_depth: Workflow depth limit used by the run (int, default 3).
        iteration_count: Workflow iterations executed (int, default 0).
        judge: Whether judge evaluation ran (bool, default False).
        current_uuid: This run's workflow uuid (str | None).
        template_uuid: Uuid of the template the run evolved from
            (str | None).
        workflow_template: Workflow code or template text (str | None).
        scenario_rubric: Scenario rubric text when a scenario ran
            (str | None).
        eval_type: Evaluator channel that scored the run (str | None).
        answers: Final answers produced by the run (list[str] | None).
        state_result: Verifier state payload (dict | None).
        plot: Plot path or text placeholder (str | None, default "").
        original_task: Unwrapped task text for similarity matching
            (str | None); set by the planner.
        code: Generated workflow code (str | None).
        selection_log: Validated-improvement record when the engine applied
            one (SelectionLog | None).
        parent_uuids: Parent run uuids; empty for seeds (list[str]).
        evolution_kind: How the run was produced: "seed", "mutation", or
            "crossover" (str, default "seed").
    """
    goal: str
    prompt: str
    cost: float = 0.0
    reward: float = 0.0
    reward_uncapped: float = 0.0
    reward_is_fallback: bool = False
    max_depth: int = 3
    iteration_count: int = 0
    judge: bool = False
    current_uuid: str | None = None
    template_uuid: str | None = None
    workflow_template: str | None = None
    scenario_rubric: str | None = None
    eval_type: str | None = None
    answers: list[str] | None = None
    state_result: dict | None = None
    plot: str | None = ""
    original_task: str | None = None  # Original unwrapped task for similarity matching
    code: str | None = None
    selection_log: "SelectionLog | None" = None
    # Lineage: parents that produced this individual and the variation operator
    # used. Set by EvolutionEngine before orchestrate_workflow runs and
    # persisted via sources.core.lineage.record_lineage afterwards.
    parent_uuids: list[str] = field(default_factory=list)
    evolution_kind: str = "seed"  # "seed" | "mutation" | "crossover"

    def __str__(self) -> str:
        """Return a verbose, debug-style summary of the run.

        Debug logging and tests call it. Returns ``str``: goal, prompt,
        cost, reward, depth, iterations, answers, verifier state, uuids,
        and eval type.
        """
        return (f"IndividualRun(goal='{self.goal}', prompt='{self.prompt}', "
                f"cost={self.cost}, reward={self.reward}, max_depth={self.max_depth}, "
                f"iteration_count={self.iteration_count}, answers={self.answers}, "
                f"state_result={self.state_result}, judge={self.judge}, "
                f"current_uuid={self.current_uuid}, template_uuid={self.template_uuid}, "
                f"eval_type={self.eval_type})")

@dataclass
class PlanStep:
    """One executable step of a plan, with dependencies and an I/O contract.

    The planner builds steps from LLM plan output, and the planner executor
    and the visualization read them. Invariant: ``name`` and ``task`` are
    non-empty, ``complexity`` is a valid :class:`TaskComplexity` value, and
    inside a ``Plan`` names are unique with every ``depends_on`` entry
    pointing to an existing step.

    Attributes:
        name: Unique step name (str).
        goal_context: The plan goal this step serves (str).
        task: Free-text instruction executed for this step (str).
        cost: Accumulated execution cost (int).
        score: Last evaluation score (float).
        depends_on: Names of steps that must complete first (list[str]).
        required_inputs: Files the step needs in the workspace (list[str]).
        expected_outputs: Files the step must produce (list[str]).
        complexity: One of "low", "medium", "high" (str, default "medium").
        status: Execution status (:class:`TaskStatus`, starts PENDING).
    """
    name: str
    goal_context: str
    task: str
    cost: int
    score: float
    depends_on: list[str] = field(default_factory=list)
    required_inputs: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    complexity: str = "medium"
    status: TaskStatus = TaskStatus.PENDING

    def __post_init__(self) -> None:
        """Validate name, task, and complexity right after construction.

        Dataclass machinery calls it automatically when a ``PlanStep`` is
        built (planner parsing, tests). It rejects an empty name or task
        and an unknown complexity value.

        Raises:
            ValueError: When ``name`` or ``task`` is empty, or ``complexity``
                is not one of :class:`TaskComplexity` values.
        """
        if not self.name:
            raise ValueError("Plan step name cannot be empty")
        if not self.task:
            raise ValueError("Plan step task cannot be empty")
        if self.complexity not in [c.value for c in TaskComplexity]:
            raise ValueError(f"Invalid complexity: {self.complexity}")

@dataclass
class Plan:
    """A complete execution plan: a goal plus its ordered steps.

    The planner builds it from validated LLM output and executes its steps
    in order. Invariant: the goal is non-empty, at least one step exists,
    step names are unique, and every ``depends_on`` entry names a step in
    this plan.

    Attributes:
        goal: The canonical goal text (str).
        steps: Executable steps in execution order (list[PlanStep]).
    """
    goal: str
    steps: list[PlanStep] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate goal, steps, name uniqueness, and dependencies.

        Dataclass machinery calls it automatically when a ``Plan`` is built
        (planner parsing, tests). It rejects an empty goal or step list,
        duplicate step names, and dependencies that point outside the plan.

        Raises:
            ValueError: When ``goal`` or ``steps`` are empty, step names are
                not unique, or any step depends on a step not present in the
                plan.
        """
        if not self.goal:
            raise ValueError("Plan goal cannot be empty")
        if not self.steps:
            raise ValueError("Plan must contain at least one step")

        # Validate step names are unique
        step_names = [step.name for step in self.steps]
        if len(step_names) != len(set(step_names)):
            raise ValueError("Plan step names must be unique")

        # Validate dependencies exist
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_names:
                    raise ValueError(f"Step '{step.name}' depends on non-existent step '{dep}'")

# Task class used in the planner module for keeping track of tasks and their states.
@dataclass
class Task:
    """One executed plan task and the evolution runs produced for it.

    The planner constructs one per executed plan step and appends it to
    ``task_history``; the visualization reads that history. It carries the
    step contract (dependencies, inputs, outputs), the status, the
    accumulated cost, and the last run's answers and uuids.

    Attributes:
        name: Name of the plan step this task executed (str).
        description: Wrapped task text given to the engine (str).
        run_id: Run counter (int, default 0).
        evolve_runs: Evolution runs produced for this task
            (list[IndividualRun]).
        final_answers: Answers of the last evolution run (list[str]).
        cost: Accumulated execution cost (float, default 0).
        final_uuid: Uuid of the last evolution run (str | None).
        workflow_uuid: Uuid of the last workflow template (str | None).
        status: Execution status (:class:`TaskStatus`, starts PENDING).
        depends_on: Names of tasks that must complete first (list[str]).
        required_inputs: Files this task needs in the workspace (list[str]).
        expected_outputs: Files this task should produce (list[str]).
        complexity: One of "low", "medium", "high" (str, default "medium").
        produced_outputs: Files that appeared in the workspace while the
            task ran (list[str]).
    """
    name: str
    description: str
    run_id: int = 0
    evolve_runs: list[IndividualRun] = field(default_factory=list) # evolution run result for task
    final_answers: list[str] = field(default_factory=list) # last evolution run answers
    cost: float = 0
    final_uuid: str | None = None # last evolution run uuid
    workflow_uuid: str | None = None # last workflow uuid
    status: TaskStatus = TaskStatus.PENDING
    depends_on: list[str] = field(default_factory=list)
    required_inputs: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    complexity: str = "medium"
    produced_outputs: list[str] = field(default_factory=list)  # Actual outputs produced
