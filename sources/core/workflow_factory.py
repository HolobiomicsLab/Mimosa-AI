"""
This class handles the creation and assembly of Langraph-SmolAgent workflow generation.
"""

import logging
import os
import re
import time
import uuid

from sources.modules import state_schema

from .factory import Factory
from .llm_provider import LLMConfig, LLMProvider, extract_model_pattern
from .tools_manager import ToolManager
from sources.cli.pretty_print import print_ok, print_info, print_warn, print_err


class WorkflowFactory(Factory):
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
            print("🔧 Generating workflow code...")
            llm_output = self.llm_make_workflow(
                system_prompt, craft_instructions, existing_tool_prompt, path, allow_cache
            )
            workflow_code = self.extract_python_code(llm_output)
            commentary = llm_output.replace(workflow_code, "").split("```python")[0]
            print_info("LLM commentary on workflow:")
            print(commentary)

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
        self.logger.debug(f"Workflow nodes discovered: {', '.join(sorted(nodes))}")

        # Validate START edge target exists
        entry_node = start_match.group(1)
        if entry_node not in nodes:
            raise ValueError(f"START targets non-existent node '{entry_node}'")
        self.logger.debug(f"Workflow entry point: START → {entry_node}")

        self.logger.info("✅ Workflow structure validation passed")

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
        smolagent_system_prompt: str = None
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
from pydantic import BaseModel

# because LLM like to use random typing
from typing import Any, Optional, Union, List, Dict, Tuple, Callable
from typing import ClassVar, Final, Literal, Protocol, TypedDict
from typing import Annotated

MEMORY_PATH = {memory_path!r}
WORKFLOW_PATH = {workflow_path!r}
MODEL_ID = {self.config.smolagent_model_id!r}
ENGINE_NAME = {self.config.engine_name!r}
GOAL = {goal!r}
SYSTEM_PROMPT = {smolagent_system_prompt!r}

# Load tools
{tools_code}

# Load state schema
{state_code}

# Load smolagent factory
{smolagent_factory_code}

# Generated workflow
{workflow_code}

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
        with open(os.path.join(WORKFLOW_PATH, "state_result.json"), "w") as f:
            json.dump(result_state, f, indent=2)
    except Exception as e:
        raise(Exception(f"Could not save workflow data:" + str(e)))
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

        smolagent_system_prompt = await self.load_single_agent_system_prompt()
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
            smolagent_system_prompt
        )

        self.logger.info("Workflow generation completed")

        self.logger.debug(f"Workflow path: {workflow_path}")
        self.logger.debug(f"Memory path: {memory_path}")

        return complete_code, workflow_code, uuid_str
