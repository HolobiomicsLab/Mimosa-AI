"""LLM-driven synthesis of LangGraph + SmolAgents workflow source code.

This file owns code generation for workflows: it prompts the workflow LLM,
extracts and cleans the generated Python, checks the LangGraph structure,
and assembles a runnable script that wires in the MCP tools, the state
schema, and the SmolAgents factory. ``WorkflowOrchestrator`` constructs
one factory per campaign and awaits
:meth:`WorkflowFactory.craft_workflow` for each new candidate; the
produced source is what the Layer 3 runner (``workflow_runner.py``)
executes. It runs during evolution whenever a new workflow genotype is
needed.
"""

import logging
import os
import random
import re
import time
import uuid

from sources.modules import state_schema

from .factory import Factory
from .llm_provider import LLMConfig, LLMProvider, extract_model_pattern
from .tools_manager import ToolManager
from sources.cli.pretty_print import (
    print_ok, print_warn, print_err, print_info,
    print_phase, print_section,
    print_iteration_header, print_box,
    print_summary, print_agent_answers,
    CYAN, GREEN, YELLOW, RED, DIM, RESET, BOLD,
)



class WorkflowFactory(Factory):
    """Build executable LangGraph + SmolAgents workflow scripts.

    Drives the LLM to author the workflow ``StateGraph`` body, validates the
    resulting code, then assembles a runnable Python module that wires in the
    state schema, MCP tool clients, and the SmolAgents factory.
    ``WorkflowOrchestrator`` constructs one factory per campaign and awaits
    :meth:`craft_workflow` for each new candidate. Key invariant: nothing is
    returned to the caller before the generated body and the assembled script
    both compile.
    """

    def __init__(self, config: "Config") -> None:
        """Copy the config paths this factory reads and writes.

        The orchestrator calls it once at campaign startup, passing the shared
        :class:`Config` built from the campaign config file. It only stores
        references; all slow work happens in :meth:`craft_workflow`.

        Args:
            config: Live :class:`Config` object providing the workflow and memory
                directories, the schema and SmolAgents factory code file paths,
                the workflow-creator prompt path, and the LLM settings.
        """
        self.workflow_dir = config.workflow_dir
        self.memory_dir = config.memory_dir
        self.schema_code_path = config.schema_code_path
        self.smolagent_factory_code_path = config.smolagent_factory_code_path
        self.prompt_workflow_creator = config.prompt_workflow_creator
        self.config = config
        self.logger = logging.getLogger(__name__)

    def get_system_prompt(self) -> str:
        """Load the system prompt that steers the workflow-creator LLM.

        ``create_workflow_genotype_code`` calls it before every generation
        request. It reads the prompt file named by
        ``config.prompt_workflow_creator`` from disk.

        Returns:
            str: Full text of the system prompt file.

        Raises:
            ValueError: If the prompt file cannot be read from disk.
        """
        try:
            with open(self.prompt_workflow_creator) as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to load system prompt: {str(e)}") from e

    @staticmethod
    def extract_python_code(code: str) -> str:
        """Collect the Python code blocks from raw LLM output.

        ``create_workflow_genotype_code`` calls it on every LLM completion; the
        input is model text that may mix commentary with ```python fences. It
        scans the text line by line and keeps only lines inside fenced Python
        blocks.

        Args:
            code: Raw LLM output text, straight from the provider response.

        Returns:
            str: All fenced Python lines joined with newlines, in file order;
            empty string when the output contains no Python fence.
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

    def remove_imports(self, code: str) -> str:
        """Remove import lines from generated workflow code.

        ``create_workflow_genotype_code`` calls it right after code extraction;
        the assembled script already provides the needed imports. It drops every
        line whose stripped form starts with ``import `` or ``from ``.

        Args:
            code: Generated workflow source, already extracted from the LLM
                reply.

        Returns:
            str: The code with every import line removed.
        """
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
        path: str,
        allow_cache: bool,
    ) -> str:
        """Send one workflow-generation request to the LLM.

        ``create_workflow_genotype_code`` calls it with the system prompt, the
        current instructions, and the tool catalogue. It builds the user prompt,
        picks a random temperature for diversity (capped at 1.0 for Anthropic and
        Claude models), and lets the provider cache the response.

        Args:
            system_prompt: Workflow-creator system prompt loaded from disk.
            craft_instructions: Goal-level crafting instructions for this
                candidate, produced by the evolution layer.
            existing_tool_prompt: Pre-formatted description of the available MCP
                tools, built by the tool manager.
            path: Directory context passed to ``LLMProvider`` for its cache and
                log files; the workflow's memory directory.
            allow_cache: When True, the provider may return a cached response for
                identical prompts.

        Returns:
            str: Raw LLM completion text, normally containing one or more
            ```python fences.
        """
        prompt = f"""
# INSTRUCTIONS:

{craft_instructions}

# AVAILABLE TOOLS:

The following tools packages are available for agents:
{existing_tool_prompt}

Proceed to generate the workflow in Python code using the LangGraph library. Follow the instructions and constraints carefully.
        """

        provider, model = extract_model_pattern(self.config.workflow_llm_model)
        temperature = random.uniform(0.7, 1.3) # enhance workflow diversity and avoid error repetition.
        if provider == "anthropic" or "claude" in model.lower():
            temperature = min(temperature, 1.0)
        self.logger.info(f"Workflow LLM temperature: {temperature:.2f}")
        llm_config = LLMConfig(
            model=model,
            provider=provider,
            temperature=temperature,
            reasoning_effort=self.config.reasoning_effort,
            max_tokens=32000,
            openrouter_provider=None, # use default
        )
        return LLMProvider("workflow_creator", path, system_prompt, llm_config)(prompt, use_cache=allow_cache)

    def create_workflow_genotype_code(
        self, craft_instructions: str, existing_tool_prompt: str, path: str, allow_cache: bool
    ) -> str:
        """Produce a cleaned, syntax-checked workflow genotype from the LLM.

        ``craft_workflow`` calls it once per candidate; inputs are the current
        crafting instructions and the tool catalogue, plus the memory directory
        for provider files. It runs :meth:`llm_make_workflow`, extracts the
        Python fences, strips imports, prints the LLM design commentary, and
        compiles the result to catch syntax errors early.

        Args:
            craft_instructions: Goal-level instructions for the new workflow.
            existing_tool_prompt: Description of the available MCP tools.
            path: Directory passed to ``llm_make_workflow`` for provider cache
                and log files.
            allow_cache: When True, the provider may reuse a cached completion.

        Returns:
            str: The validated workflow body, that is Python source with no
            import lines.

        Raises:
            ValueError: If the LLM call fails, produces no usable code, or the
                extracted code does not compile.
        """
        self.logger.info("Generating workflow code with LLM...")
        system_prompt = self.get_system_prompt()
        try:
            print_info("Generating workflow code...")
            llm_output = self.llm_make_workflow(
                system_prompt, craft_instructions, existing_tool_prompt, path, allow_cache
            )
            print_ok("Workflow code generated by LLM")
            workflow_genotype_code = self.extract_python_code(llm_output)
            if not workflow_genotype_code:
                print_err("No code generated by LLM.")
            commentary = llm_output.replace(workflow_genotype_code, "").split("```python")[0]
            print_box(commentary, title="LLM Commentary on Workflow Design", color=CYAN)

            workflow_genotype_code = self.remove_imports(workflow_genotype_code)
            if not workflow_genotype_code.strip():
                raise ValueError("LLM did not return valid workflow code")
        except Exception as e:
            self.logger.error(f"create_workflow_genotype_code: LLM workflow generation/extraction failed: {str(e)}")
            raise ValueError(f"LLM workflow generation/extraction failed: {str(e)}") from e

        # Validate syntax before returning
        try:
            compile(workflow_genotype_code, "<workflow>", "exec")
        except SyntaxError as e:
            print_err(f"\n🚨 Invalid workflow code 🚨\n{'='*40}\n\033[91m{workflow_genotype_code}\033[0m\n{'='*40}\n{e}")
            raise ValueError(f"LLM generated invalid Python syntax: {e}") from e

        self.logger.info("LLM generated workflow code successfully")
        return workflow_genotype_code

    def validate_workflow_structure(self, workflow_genotype_code: str) -> None:
        """Check the generated code for required LangGraph structure.

        ``craft_workflow`` calls it after generation and before assembly, so a
        malformed genotype fails fast with a clear message. It regex-checks the
        source for the required elements: ``StateGraph(WorkflowState)``
        initialization, conditional edges, SmolAgentFactory and
        WorkflowNodeFactory usage, a START edge, and at least one node with a
        valid Python name, and it confirms the START target exists.

        Args:
            workflow_genotype_code: Generated workflow source to check.

        Raises:
            ValueError: If any required element is missing, a node name is a
                keyword or not a valid identifier, or the START edge points to a
                non-existent node.
        """
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
            if not re.search(pattern, workflow_genotype_code):
                raise ValueError(error_msg)

        # Extract and validate core components
        start_match = re.search(patterns["start_edge"], workflow_genotype_code)
        if not start_match:
            raise ValueError(
                "Graph must have entry point: workflow.add_edge(START, 'node_name')"
            )

        nodes = set(re.findall(patterns["nodes"], workflow_genotype_code))
        if not nodes:
            raise ValueError("No workflow nodes found")
        import keyword as _keyword
        invalid_ids = {n for n in nodes if _keyword.iskeyword(n) or not n.isidentifier()}
        if invalid_ids:
            raise ValueError(
                f"Workflow node names are Python keywords or invalid identifiers: {sorted(invalid_ids)}"
            )
        self.logger.debug(f"Workflow nodes discovered: {', '.join(sorted(nodes))}")

        # Validate START edge target exists
        entry_node = start_match.group(1)
        if entry_node not in nodes:
            raise ValueError(f"START targets non-existent node '{entry_node}'")
        self.logger.debug(f"Workflow entry point: START → {entry_node}")

        self.logger.info("Workflow structure validation passed")

    def assemble_workflow(
        self,
        tools_code: str,
        state_code: str,
        smolagent_factory_code: str,
        workflow_genotype_code: str,
        workflow_path: str,
        memory_path: str,
        uuid_str: str,
        goal: str,
        smolagent_system_prompt: str | None = None,
    ) -> str:
        """Assemble the complete, runnable workflow script.

        ``craft_workflow`` calls it after structural validation passes, passing
        the four code parts plus the target paths. It resolves repo-relative
        paths, picks the default model id from config, and returns one f-string
        module that wires in the tools, the state schema, and the SmolAgents
        factory, embeds config-derived constants (model, engine, timeouts, goal,
        prompts), compiles the LangGraph, tries to render a PNG diagram, invokes
        the graph on an initial state built from the schema fields, and saves
        ``state_result.json``.

        Args:
            tools_code: Source that defines all MCP tool clients; loaded from
                disk.
            state_code: Source that defines the workflow state schema; loaded
                from the configured schema file.
            smolagent_factory_code: Source that defines the SmolAgents factory;
                loaded from the configured factory file.
            workflow_genotype_code: LLM-generated workflow body, already
                validated.
            workflow_path: Directory for workflow artifacts such as the PNG and
                the state JSON; resolved relative to the repo root.
            memory_path: Directory for memory artifacts; resolved relative to the
                repo root.
            uuid_str: Unique workflow identifier stamped into file names and the
                initial state.
            goal: Campaign goal string embedded as ``GOAL`` in the script.
            smolagent_system_prompt: Optional system prompt embedded as
                ``SYSTEM_PROMPT``; ``None`` leaves the field unset in the script.

        Returns:
            str: Complete Python source of the runnable workflow script.
        """
        from pathlib import Path
        script_dir = Path(__file__).resolve().parent.parent.parent
        memory_path = str((script_dir / memory_path).resolve())
        workflow_path = str((script_dir / workflow_path).resolve())
        default_model = self.config.smolagent_model_id[0] if isinstance(self.config.smolagent_model_id, list) else self.config.smolagent_model_id
        # `WorkflowState.model_id` is a single string and is persisted to
        # `state_result.json`, where cost accounting reads it back; store the
        # resolved default rather than the raw (possibly list) config value.
        initial_state = {
            key: (
                uuid_str
                if key == "workflow_uuid"
                else default_model
                if key == "model_id"
                else goal
                if key == "goal"
                else []
            )
            for key in state_schema.WorkflowState.__annotations__
        }
        providers = self.config.openrouter_provider_for(default_model) if self.config.orchestrator_choose_model == False else None
        engine = "mlx" if "mlx-community/" in default_model else self.config.engine_name
        return f"""
import os
import sys
import re
import json
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

# because LLM like to use random typing
import operator
from typing import Any, Optional, Union, List, Dict, Tuple, Callable
from typing import ClassVar, Final, Literal, Protocol, TypedDict
from typing import Annotated

MEMORY_PATH = {memory_path!r}
WORKFLOW_PATH = {workflow_path!r}
MODEL_ID = {default_model!r}
ENGINE_NAME = {engine!r}
OPENROUTER_PROVIDER = {providers!r}
AGENT_EXECUTION_TIMEOUT = {self.config.agent_execution_timeout!r}
SAVE_LOGPROBS = {self.config.save_logprobs!r}
GOAL = {goal!r}
SYSTEM_PROMPT = {smolagent_system_prompt!r}

# Load tools
{tools_code}

# Load state schema
{state_code}

# Load smolagent factory
{smolagent_factory_code}

# Generated workflow
{workflow_genotype_code}

app = workflow.compile()

# Initialize and execute workflow
initial_state = {initial_state}

try:
    if WORKFLOW_PATH:
        try:
            png = app.get_graph().draw_mermaid_png()
            with open(os.path.join(WORKFLOW_PATH, "workflow_{uuid_str}.png"), "wb") as f:
                f.write(png)
        except Exception as e:
            RuntimeError(f"Could not save workflow graph:" + str(e))
except Exception as e:
    print(f"❌ Error saving PNG workflow:" + str(e))
    pass

try:
    result_state = app.invoke(initial_state)
except KeyboardInterrupt:
    print("Workflow execution interrupted by user")
    pass
print("workflow run: workflow execution completed for UUID:", "{uuid_str}")

if WORKFLOW_PATH:
    print("workflow run: saving workflow state JSON at :", WORKFLOW_PATH)
    try:
        _state_path = os.path.join(WORKFLOW_PATH, "state_result.json")
        _tmp_path = _state_path + ".tmp"
        with open(_tmp_path, "w") as f:
            json.dump(result_state, f, indent=2)
        os.replace(_tmp_path, _state_path)
    except Exception as e:
        raise(Exception(f"Could not save workflow data:" + str(e)))
"""

    async def craft_workflow(
        self,
        goal: str,
        craft_instructions: str,
        save_workflow: bool = True,
        original_task: str | None = None,
    ) -> tuple[str, str, str]:
        """Craft one complete workflow from a goal, end to end.

        The orchestrator awaits it for every new candidate, passing the possibly
        knowledge-wrapped goal and the crafting instructions from the evolution
        layer. It makes a sortable id, loads tool code and template files, asks
        the LLM for the genotype, saves the genotype early, validates and
        assembles the full script, and compiles the final code before returning.

        Args:
            goal: The task goal; may be wrapped with retrieved knowledge.
            craft_instructions: Instructions for the LLM. When equal to
                ``goal``, the previous attempt did not fail, so the provider
                cache is allowed.
            save_workflow: When True, persist workflow and memory artifacts under
                fresh per-UUID directories; when False, artifacts are not written
                and only default paths are computed.
            original_task: The unwrapped original task, stored with the workflow
                files for similarity checks; ``None`` when not available.

        Returns:
            tuple[str, str, str]: ``(complete_code, workflow_genotype_code,
            uuid_str)``, where ``complete_code`` is the assembled runnable
            script, ``workflow_genotype_code`` is the LLM-produced body, and
            ``uuid_str`` is the workflow identifier.

        Raises:
            RuntimeError: If loading tool code, creating directories, or reading
                the template code files fails.
            ValueError: If generation, structural validation, or final
                compilation fails; validation and compile errors carry the
                workflow UUID prefix.
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
        allow_cache = goal == craft_instructions # if goal and craft instructions are the same it mean last workflow didn't fail (evolve level)
        try:
            workflow_genotype_code = self.create_workflow_genotype_code(
                craft_instructions, existing_tool_prompt, memory_path, allow_cache
            ) # Generate workflow code - let Evolution handle retries
        except Exception as e:
            raise e # raise error for evolve-level to handle
        # Save workflow code immediately so learning layer can access it even if validation fails
        if save_workflow and isinstance(workflow_genotype_code, str):
            self.save_workflow_files(workflow_path, uuid_str, workflow_genotype_code, goal, original_task)

        try:
            self.validate_workflow_structure(workflow_genotype_code)
        except Exception as e:
            self.logger.error(f"craft_workflow: Workflow structure validation failed: {str(e)}")
            raise ValueError(f"UUID:{uuid_str}|{str(e)}") from e

        smolagent_system_prompt = await self.load_single_agent_system_prompt()
        # Assemble complete workflow
        complete_code = self.assemble_workflow(
            tools_code,
            state_code,
            smolagent_factory_code,
            workflow_genotype_code,
            workflow_path,
            memory_path,
            uuid_str,
            goal,
            smolagent_system_prompt
        )

        try:
            compile(complete_code, "<assembled_workflow>", "exec")
        except SyntaxError as e:
            raise ValueError(f"UUID:{uuid_str}|Assembled workflow has invalid syntax: {e}") from e

        self.logger.info("Workflow generation completed")

        self.logger.debug(f"Workflow path: {workflow_path}")
        self.logger.debug(f"Memory path: {memory_path}")

        return complete_code, workflow_genotype_code, uuid_str
