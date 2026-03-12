"""
This class handles the creation and assembly of Langraph-SmolAgent workflow generation.
"""

import logging
import random
import os
import re
import time
import uuid

from sources.modules import state_schema

from .llm_provider import LLMConfig, LLMProvider, extract_model_pattern
from .tools_manager import ToolManager


class WorkflowFactory:
    """Handles the creation and management of Langraph-SmolAgent workflow generation"""

    def __init__(self, config) -> None:
        """Initialize the workflow crafting system.
        Args:
            config: Configuration object containing paths and settings
        """
        self.workflow_dir = config.workflow_dir
        self.memory_dir = config.memory_dir
        self.schema_code_path = config.schema_code_path
        self.smolagent_factory_code_path = config.smolagent_factory_code_path
        self.prompt_workflow_creator = config.prompt_workflow_creator
        self.config = config
        self.logger = logging.getLogger(__name__)

        # hints to provide diversity in generated workflows, we sample 3 of them at random for each workflow generation
        # provide an approximate search direction to the LLM to improve workflow quality and diversity

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

    def get_system_prompt(self) -> str:
        """Load the system prompt for workflow generation.
        Returns:
            str: The system prompt content
        """
        try:
            with open(self.prompt_workflow_creator) as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to load system prompt: {str(e)}") from e

    @staticmethod
    def extract_python_code(code: str) -> str:
        """Extract Python code blocks from text.
        Args:
            code: Text potentially containing Python code blocks
        Returns:
            str: Extracted Python code
        """
        code_blocks = []
        in_code_block = False
        for line in code.splitlines():
            if line.startswith("```python"):
                in_code_block = True
                continue
            if line.startswith("```") and in_code_block:
                in_code_block = False
                continue
            if in_code_block:
                code_blocks.append(line)
        return "\n".join(code_blocks)

    async def load_tools_code(self) -> tuple[str, str]:
        """Discover all MCP servers and format their client code.
        Returns:
            str: Combined code for all MCP clients.
            str: Prompt of discovered MCP names for workflow generation tools-awareness.
        """
        tools_code = ""
        existing_tool_prompt = ""
        tool_manager = ToolManager(self.config)
        try:
            tool_setup = False
            while tool_setup == False:
                mcps = await tool_manager.discover_mcp_servers()
                tool_setup = await tool_manager.verify_tools()
        except Exception as e:
            self.logger.error(f"load_tools_code: Failed to discover MCP servers: {str(e)}")
            raise RuntimeError(f"Failed to discover MCP servers: {str(e)}") from e
        if not mcps:
            raise ValueError(
                "\n" + "=" * 80 +
                "\n🚨  FATAL ERROR: No MCP Servers Found! 🚨"
                "\n" + "-" * 80 +
                "\nPlease ensure at least one MCP instance is running on Toolomics."
                "\nRetrying until MCPs detected.... use CTRL+C to stop."
                "\n" + "=" * 80 + "\n"
            )
        for mcp in mcps:
            client_code = tool_manager.get_client_code(mcp)
            client_prompt = tool_manager.get_client_prompt(mcp)
            tools_code += client_code + "\n"
            existing_tool_prompt += client_prompt + "\n"
        print(f"🔧 Discovered {len(mcps)} MCP servers capabilities. Workflow generation can start.")
        return tools_code, existing_tool_prompt

    def remove_imports(self, code: str) -> str:
        # remove attempt from LLM to import modules/class
        lines = code.splitlines()
        return "\n".join(
            line
            for line in lines
            if not (
                line.strip().startswith("import ") or line.strip().startswith("from ")
            )
        )

    def llm_make_workflow(
        self,
        system_prompt: str,
        craft_instructions: str,
        existing_tool_prompt: str,
        hints: list[tuple[str, str]],
        path: str,
        allow_cache: bool
    ) -> str:
        """Generate a workflow using the LLM."""

        prompt = f"""
# INSTRUCTIONS:

{craft_instructions}

# AVAILABLE TOOLS:

The following tools packages are available for agents:
{existing_tool_prompt}

# ADVICES/SEARCH DIRECTION:
{hints}
You may or may not consider these hints, but they are provided to inspire diverse and creative workflow structures that go beyond common patterns.

Proceed to generate the workflow in Python code using the LangGraph library. Follow the instructions and constraints carefully.
        """

        provider, model = extract_model_pattern(self.config.workflow_llm_model)
        llm_config = LLMConfig(
            model=model,
            provider=provider,
            reasoning_effort=self.config.reasoning_effort,
            max_tokens=getattr(self.config, 'max_tokens', 8192)
        )
        return LLMProvider("workflow_creator", path, system_prompt, llm_config)(prompt, use_cache=allow_cache)


    def sample_workflow_hints(self, hints: list[tuple[str, str]], n: int, seed: int | None = None) -> str:
        rng = random.Random(seed)
        sampled = rng.sample(hints, min(n, len(hints)))
        return sampled

    def get_hints(self) -> str:
        rn_seed = int(time.time() * 1000) % 2**32
        sampled_hints = self.sample_workflow_hints(self.hints, n=3, seed=rn_seed)
        hints_names = [hint[0] for hint in sampled_hints]
        hints = "\n".join(f"{i+1}. {hint[1]}" for i, hint in enumerate(sampled_hints))
        return hints, hints_names

    def create_workflow_code(
        self, craft_instructions: str, existing_tool_prompt: str, path: str, allow_cache: bool
    ) -> str:
        """Generate and validate workflow code.
        Args:
            craft_instructions: The goal description
            existing_tool_prompt: Description of available tools
        Returns:
            str: Validated workflow code
        """
        self.logger.info("Generating workflow code with LLM...")
        system_prompt = self.get_system_prompt()
        try:
            hints, hints_names = self.get_hints()
            print(f"🎲 Sampled workflow hints for this generation: {', '.join(hints_names)}")
            print("🔧 Generating workflow code...")
            llm_output = self.llm_make_workflow(
                system_prompt, craft_instructions, existing_tool_prompt, hints, path, allow_cache
            )
            workflow_code = self.extract_python_code(llm_output)
            commentary = llm_output.replace(workflow_code, "").split("```python")[0]
            print("💬 LLM commentary on workflow:")
            print(commentary)

            workflow_code = f'# {", ".join(hints_names)} ' + "\n\n" + workflow_code
            workflow_code = self.remove_imports(workflow_code)
            if not workflow_code.strip():
                raise ValueError("LLM did not return valid workflow code")
        except Exception as e:
            self.logger.error(f"create_workflow_code: LLM workflow generation/extraction failed: {str(e)}")
            raise ValueError(f"LLM workflow generation/extraction failed: {str(e)}") from e

        # Validate syntax before returning
        try:
            compile(workflow_code, "<workflow>", "exec")
        except SyntaxError as e:
            self.logger.error(f"\n🚨 Invalid workflow code 🚨\n{'='*40}\n\033[91m{workflow_code}\033[0m\n{'='*40}\n{e}")
            raise ValueError(f"LLM generated invalid Python syntax: {e}") from e

        self.logger.info("LLM generated workflow code successfully")
        return workflow_code

    def validate_workflow_structure(self, workflow_code: str) -> None:
        """Validate LangGraph workflow structure before execution."""
        self.logger.info("Validating workflow structure...")

        # Pre-compile regex patterns for efficiency
        patterns = {
            "state_graph": r"workflow = StateGraph\(WorkflowState\)",
            "start_edge": r"workflow\.add_edge\(START,\s*[\"'](\w+)[\"']\)",
            "nodes": r"workflow\.add_node\([\"'](\w+)[\"'],.*?\)",
            "conditional_edges": r"workflow\.add_conditional_edges\(",
            "edge_mappings": r'workflow\.add_conditional_edges\(\s*["\'](\w+)["\'],\s*(\w+),\s*\{([^}]+)\}',
            "router_returns": r'return\s+["\']([^"\']+)["\']',
            "agent_factory": r"SmolAgentFactory\(",
            "node_factory": r"WorkflowNodeFactory\.create_agent_node\(",
        }

        # Basic structure validation
        required_checks = [
            (
                patterns["state_graph"],
                "Missing 'workflow = StateGraph(WorkflowState)' initialization",
            ),
            (
                patterns["conditional_edges"],
                "No conditional edges found - workflows require conditional routing",
            ),
            (patterns["agent_factory"], "No SmolAgentFactory usage found"),
            (patterns["node_factory"], "No WorkflowNodeFactory usage found"),
        ]

        for pattern, error_msg in required_checks:
            if not re.search(pattern, workflow_code):
                raise ValueError(error_msg)

        # Extract and validate core components
        start_match = re.search(patterns["start_edge"], workflow_code)
        if not start_match:
            raise ValueError(
                "Graph must have entry point: workflow.add_edge(START, 'node_name')"
            )

        nodes = set(re.findall(patterns["nodes"], workflow_code))
        if not nodes:
            raise ValueError("No workflow nodes found")
        self.logger.debug(f"📋 Workflow nodes discovered: {', '.join(sorted(nodes))}")

        # Validate START edge target exists
        entry_node = start_match.group(1)
        if entry_node not in nodes:
            raise ValueError(f"START targets non-existent node '{entry_node}'")
        self.logger.debug(f"🚀 Workflow entry point: START → {entry_node}")

        self.logger.info("✅ Workflow structure validation passed")

    def create_folder_structure(self, uuid_str: str) -> tuple[str]:
        """Create directory structure for new workflow.
        Args:
            uuid_str: Unique identifier for the workflow
        Returns:
            str: Path to created workflow directory
        """
        workflow_path = os.path.join(self.workflow_dir, uuid_str)
        self.logger.info(f"Created workflow directory: {workflow_path}")
        os.makedirs(workflow_path, exist_ok=True)
        memory_path = os.path.join(self.memory_dir, uuid_str)
        os.makedirs(memory_path, exist_ok=True)
        self.logger.info(f"Created memory directory: {memory_path}")
        return workflow_path, memory_path

    def assemble_workflow(
        self,
        tools_code: str,
        state_code: str,
        smolagent_factory_code: str,
        workflow_code: str,
        workflow_path: str,
        memory_path: str,
        uuid_str: str,
        goal: str,
    ) -> str:
        """Assemble the complete workflow code.
        Args:
            tools_code: Code for all MCP clients
            state_code: Code for the workflow state schema
            smolagent_factory_code: Code for the SmolAgent factory
            workflow_code: Generated workflow code by LLM
            workflow_path: Path to save the workflow
            memory_path: Path to save the workflow memory
            uuid_str: Unique identifier for the workflow
            goal: The goal for the workflow
        Returns:
            str: Complete workflow code ready for execution
        """
        from pathlib import Path
        script_dir = Path(__file__).resolve().parent.parent.parent
        memory_path = str((script_dir / memory_path).resolve())
        workflow_path = str((script_dir / workflow_path).resolve())
        initial_state = {
            key: (
                uuid_str
                if key == "workflow_uuid"
                else self.config.smolagent_model_id
                if key == "model_id"
                else goal
                if key == "goal"
                else []
            )
            for key in state_schema.WorkflowState.__annotations__
        }
        return f"""
import os
import sys
import re
import json
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from pydantic import BaseModel

MEMORY_PATH = {memory_path!r}
WORKFLOW_PATH = {workflow_path!r}
MODEL_ID = {self.config.smolagent_model_id!r}
ENGINE_NAME = {self.config.engine_name!r}
GOAL = {goal!r}

# Load tools
{tools_code}

# Load state schema
{state_code}

# Load smolagent factory
{smolagent_factory_code}

# Generated workflow
{workflow_code}

print("worflow run: compiling workflow...")
app = workflow.compile()

# Initialize and execute workflow
initial_state = {initial_state}

try:
    if WORKFLOW_PATH:
        try:
            png = app.get_graph().draw_mermaid_png()
            print("workflow run: saving workflow graph as PNG at ", WORKFLOW_PATH)
            with open(os.path.join(WORKFLOW_PATH, "workflow_{uuid_str}.png"), "wb") as f:
                print("workflow run: writing PNG file...")
                f.write(png)
                print("PNG saved at ", os.path.join(WORKFLOW_PATH, "workflow_{uuid_str}.png"))
        except Exception as e:
            RuntimeError(f"Could not save workflow graph:" + str(e))
except Exception as e:
    print(f"❌ Error saving PNG workflow:" + str(e))
    pass

print("workflow run: invoking workflow...")
try:
    result_state = app.invoke(initial_state)
except KeyboardInterrupt:
    print("Workflow execution interrupted by user")
    pass
print("workflow run: workflow execution completed for UUID:", "{uuid_str}")

if WORKFLOW_PATH:
    print("workflow run: saving workflow state JSON at :", WORKFLOW_PATH)
    try:
        with open(os.path.join(WORKFLOW_PATH, "state_result.json"), "w") as f:
            json.dump(result_state, f, indent=2)
    except Exception as e:
        raise(f"Could not save workflow data:" + str(e))
"""

    async def craft_workflow(
        self,
        goal: str,
        craft_instructions: str,
        save_workflow: bool = True,
        original_task: str = None,
    ) -> tuple[str, str]:
        """Main method to craft a complete workflow.
        Args:
            goal: The goal description (may be knowledge-wrapped)
            craft_instructions: The instructions for crafting the workflow
            template_workflow: pre-existing workflow template UUID
            save_workflow: Whether to save the workflow
            original_task: The original unwrapped task for similarity matching
        Returns:
            str: Complete executable workflow code
        """
        # Generate chronologically sortable workflow ID: YYYYMMDD_HHMMSS_shortUUID
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        uuid_str = f"{timestamp}_{short_uuid}"
        try:
            tools_code, existing_tool_prompt = await self.load_tools_code()
        except Exception as e:
            self.logger.error(f"craft_workflow: Failed to load tools code: {str(e)}")
            raise RuntimeError(f"Failed to load tools code: {str(e)}") from e

        try:
            workflow_path, memory_path = (
                self.create_folder_structure(uuid_str)
                if save_workflow
                else (
                    os.path.join(self.workflow_dir, uuid_str),
                    os.path.join(self.memory_dir, uuid_str),
                )
            )
        except Exception as e:
            self.logger.error(f"craft_workflow: Failed to create workflow directories: {str(e)}")
            raise RuntimeError(f"Failed to create workflow directories: {str(e)}") from e

        try:
            with open(self.schema_code_path) as f:
                state_code = f.read()
            with open(self.smolagent_factory_code_path) as f:
                smolagent_factory_code = f.read()
        except Exception as e:
            self.logger.error(f"craft_workflow: Failed to load required code files: {str(e)}")
            raise RuntimeError(f"Failed to load required code files: {str(e)}") from e
        allow_cache = goal == craft_instructions # if goal and craft instructions are the same it mean last workflow didn't fail (dgm level)
        try:
            workflow_code = self.create_workflow_code(
                craft_instructions, existing_tool_prompt, memory_path, allow_cache
            ) # Generate workflow code - let Evolution handle retries
        except Exception as e:
            raise e # raise error for dgm-level to handle
        # Save workflow code immediately so learning layer can access it even if validation fails
        if save_workflow and isinstance(workflow_code, str):
            self.save_workflow_files(workflow_path, uuid_str, workflow_code, goal, original_task)

        try:
            self.validate_workflow_structure(workflow_code)
        except Exception as e:
            self.logger.error(f"craft_workflow: Workflow structure validation failed: {str(e)}")
            raise ValueError(f"UUID:{uuid_str}|{str(e)}") from e

        # Assemble complete workflow
        complete_code = self.assemble_workflow(
            tools_code,
            state_code,
            smolagent_factory_code,
            workflow_code,
            workflow_path,
            memory_path,
            uuid_str,
            goal,
        )

        self.logger.info("Workflow generation completed")

        self.logger.debug(f"Workflow path: {workflow_path}")
        self.logger.debug(f"Memory path: {memory_path}")

        return complete_code, workflow_code, uuid_str

    def _extract_original_from_goal(self, goal: str) -> str:
        """Extract original task from knowledge-wrapped goal.

        Args:
            goal: Goal text that may be wrapped with knowledge context

        Returns:
            str: Extracted original task or goal if not wrapped
        """
        if not goal:
            return ""

        # Pattern: "...Now, use this knowledge to complete:\n<actual_task>"
        # This is the pattern used by planner._build_knowledge_aware_task()
        match = re.search(r'Now, use this knowledge to complete:\s*\n(.*)', goal, re.DOTALL)
        if match:
            return match.group(1).strip()

        # If no wrapper pattern found, return goal as-is
        return goal

    def save_workflow_files(
        self, path: str, uuid_str: str, workflow_code: str, goal: str, original_task: str = None
    ) -> None:
        """Save workflow code and metadata to files.

        Args:
            path: Directory path to save files
            uuid_str: Unique workflow identifier
            workflow_code: Generated workflow code
            goal: The goal description (may be knowledge-wrapped)
            original_task: The original unwrapped task for similarity matching
        """
        try:
            with open(os.path.join(path, f"workflow_code_{uuid_str}.py"), "w") as f:
                f.write(workflow_code)
            self.logger.info(
                f"Saved workflow code to: {path}/workflow_code_{uuid_str}.py"
            )
        except Exception as e:
            self.logger.error(f"Failed to save workflow code: {str(e)}")

        try:
            with open(os.path.join(path, f"system_prompt_{uuid_str}.md"), "w") as f:
                f.write(self.get_system_prompt())
            self.logger.info(
                f"Saved system prompt to: {path}/system_prompt_{uuid_str}.md"
            )
        except Exception as e:
            self.logger.error(f"Failed to save system prompt: {str(e)}")

        try:
            with open(os.path.join(path, f"goal_{uuid_str}.txt"), "w") as f:
                f.write(goal)
            self.logger.info(f"Saved goal to: {path}/goal_{uuid_str}.txt")
        except Exception as e:
            self.logger.error(f"Failed to save goal: {str(e)}")

        # Save original task for better similarity matching
        # Extract from goal if not provided explicitly
        task_to_save = original_task if original_task else self._extract_original_from_goal(goal)
        if task_to_save:
            try:
                with open(os.path.join(path, f"original_task_{uuid_str}.txt"), "w") as f:
                    f.write(task_to_save)
                self.logger.info(f"Saved original task to: {path}/original_task_{uuid_str}.txt")
            except Exception as e:
                self.logger.error(f"Failed to save original task: {str(e)}")

    async def craft_single_agent(self, goal: str, original_task: str = None):
        """
        For crafting single agent with cost tracking support.

        Args:
            goal: The goal description (may be knowledge-wrapped)
            original_task: The original unwrapped task for similarity matching

        Returns:
            tuple[str, str, str]: (complete_code, workflow_code, uuid)
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        uuid_str = f"single_agent_{timestamp}_{short_uuid}"

        try:
            tools_code, existing_tool_prompt = await self.load_tools_code()
        except Exception as e:
            self.logger.error(f"craft_single_agent: Failed to load tools code: {str(e)}")
            raise RuntimeError(f"Failed to load tools code: {str(e)}") from e

        # Create folder structure for cost tracking (like multi-agent mode)
        workflow_path, memory_path = self.create_folder_structure(uuid_str)

        INSTRUCTIONS = ". ".join([
            "TASK:",
            goal,
            "",
            "Address complaints from the last agent informations if any.",
            "",
            "CONSTRAINTS:",
            "- No placeholder or example values.",
            "- No assumptions about missing data. Investigate first using available workspace data.",
            "- Never plot anything to the user. Plotting causes: 'terminating due to uncaught exception of type NSException'.",
            "- Save outputs instead of plotting.",
            "- Only use execute_command to install packages.",
            "- You are only allowed to use tools to create and execute the code required to accomplish the goal.",
            "- Use python/code editing tools when available.",
            "- Wrap any command that may take significant time (>5 minutes) in a timeout.",
            "",
            "INITIAL STEP:",
            "- Assess the workspace by running: ls -la"
        ])

        # Resolve absolute paths (like craft_workflow does)
        from pathlib import Path
        script_dir = Path(__file__).resolve().parent.parent.parent
        memory_path_abs = str((script_dir / memory_path).resolve())
        workflow_path_abs = str((script_dir / workflow_path).resolve())

        mcp_vars = sorted(set(
            re.findall(r"\bMCP_\d+_TOOLS\b", tools_code)
        ))
        mcps_string = "MCPS = [\n" + ",\n".join(f"    {name}" for name in mcp_vars) + "\n]"

        code = f"""
import os
import json
from dataclasses import asdict
from typing import List

import smolagents
from smolagents import CodeAgent, LiteLLMModel, ActionStep
from smolagents.models import get_dict_from_nested_dataclasses
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = {self.config.smolagent_model_id!r}
GOAL = {goal!r}
INSTRUCTIONS = {INSTRUCTIONS!r}
MEMORY_PATH = {memory_path_abs!r}
WORKFLOW_PATH = {workflow_path_abs!r}

engine = LiteLLMModel(
    model_id=MODEL_ID,
    temperature=1.0,
    max_tokens=8096,
)

{tools_code}
{mcps_string}

all_tools = []
for mcp_tools in MCPS:
    all_tools.extend(mcp_tools)

agent = CodeAgent(
    tools=all_tools,
    model=engine,
    name="single_agent",
    max_steps=256,
    additional_authorized_imports=["requests", "bs4", "json"],
)

def save_agent_memories(agent, memory_path: str, agent_name: str):
    print(f"Saving agent memory to: {{{{memory_path}}}}")
    try:
        memories = []
        for idx, step in enumerate(agent.memory.steps):
            if isinstance(step, ActionStep):
                action_step = step.dict()
                action_step["model_input_messages"] = (
                    get_dict_from_nested_dataclasses(
                        [asdict(msg) if hasattr(msg, '__dataclass_fields__') else msg for msg in step.model_input_messages],
                        ignore_key="raw"
                    )
                    if step.model_input_messages
                    else None
                )
                action_step["model_output_message"] = (
                    get_dict_from_nested_dataclasses(
                        step.model_output_message, ignore_key="raw"
                    )
                    if step.model_output_message
                    else None
                )
                memories.append(action_step)

        os.makedirs(memory_path, exist_ok=True)
        agent_task_path = os.path.join(memory_path, f"task_{{agent_name}}.json")
        with open(agent_task_path, "w") as f:
            json.dump(memories, f, indent=2)
        print(f"✅ Agent memories saved successfully to {{{{agent_task_path}}}}")
    except Exception as e:
        print(f"⚠️  Failed to save memory: {{{{str(e)}}}}")

# Run agent
result = agent.run(INSTRUCTIONS)

# Save agent memories for cost tracking
save_agent_memories(agent, MEMORY_PATH, "single_agent")

# Save state_result.json for cost tracking and evaluation
state_result = {{
    "model_id": MODEL_ID,
    "goal": GOAL,
    "workflow_uuid": "{uuid_str}",
    "single_agent_mode": True,
    "step_name": ["single_agent"],
    "answers": [str(result)],
    "success": [True]  # Assume success if no exception
}}

try:
    with open(os.path.join(WORKFLOW_PATH, "state_result.json"), "w") as f:
        json.dump(state_result, f, indent=2)
    print(f"✅ Saved state_result.json to {{WORKFLOW_PATH}}")
except Exception as e:
    print(f"❌ Failed to save state_result.json: {{e}}")
        """

        # Save metadata files (like multi-agent mode)
        self.save_workflow_files(
            workflow_path,
            uuid_str,
            code,  # Save the single agent code
            goal,
            original_task
        )

        return code, code, uuid_str
