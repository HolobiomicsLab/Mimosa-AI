"""
This class handles the creation and assembly of Langraph-SmolAgent workflow generation.
"""

import os
import uuid

from sources.modules import state_schema

from .llm_provider import LLMProvider
from .tools_manager import ToolManager


class WorkflowFactory:
    """Handles the creation and management of Langraph-SmolAgent workflow generation.

    Attributes:
        tools_dir (str): Directory containing tool modules
        workflow_dir (str): Base directory for workflow storage
    """

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

    def llm_make_workflow(
        self,
        system_prompt: str,
        craft_instructions: str,
        existing_tool_prompt: str,
        path: str,
    ) -> str:
        prompt = f"""
You are an expert in generating LangGraph workflows using SmolAgent nodes.

The following tools packages are available for agents:
{existing_tool_prompt}

Your task is to create a LangGraph-SmolAgent workflow for the task:
{craft_instructions}
        """
        history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return LLMProvider().openai_completion(
            history, "orchestrator", path, verbose=False
        )

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
        """Load all tool code from the tools directory.

        Returns:
            Tuple[str, str]: Tuple containing (tools_code, existing_tool_prompt)
        """
        tools_code = ""
        existing_tool_prompt = ""
        tool_manager = ToolManager(self.config)
        mcps = await tool_manager.discover_mcp_servers()
        if not mcps:
            raise ValueError(
                "\nNo MCP servers found."
                "Please ensure at least one MCP is running on toolomics."
            )
        for mcp in mcps:
            client_code = tool_manager.get_client_code(mcp)
            client_prompt = tool_manager.get_client_prompt(mcp)
            tools_code += client_code + "\n"
            existing_tool_prompt += client_prompt + "\n"
        return tools_code, existing_tool_prompt

    def create_workflow_code(
        self, craft_instructions: str, existing_tool_prompt: str, path: str
    ) -> str:
        """Generate and validate workflow code.

        Args:
            craft_instructions: The goal description
            existing_tool_prompt: Description of available tools
        Returns:
            str: Validated workflow code
        """
        print("🧠 Generating workflow code with LLM...")
        system_prompt = self.get_system_prompt()
        llm_output = self.llm_make_workflow(
            system_prompt, craft_instructions, existing_tool_prompt, path
        )
        workflow_code = self.extract_python_code(llm_output)
        if not workflow_code.strip():
            raise ValueError("LLM did not return valid workflow code")
        print("✅ LLM generated workflow code successfully")
        return workflow_code

    def create_folder_structure(self, uuid_str: str) -> tuple[str]:
        """Create directory structure for new workflow.

        Args:
            uuid_str: Unique identifier for the workflow
        Returns:
            str: Path to created workflow directory
        """
        workflow_path = os.path.join(self.workflow_dir, uuid_str)
        print(f"✅ Created workflow directory: {workflow_path}")
        os.makedirs(workflow_path, exist_ok=True)
        memory_path = os.path.join(self.memory_dir, uuid_str)
        os.makedirs(memory_path, exist_ok=True)
        print(f"✅ Created memory directory: {memory_path}")
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
        goal_prompt: str,
    ) -> str:
        initial_state = {
            key: (
                uuid_str
                if key == "workflow_uuid"
                else self.config.smolagent_model_id
                if key == "model_id"
                else goal_prompt
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

MEMORY_PATH = "{memory_path}"
WORKFLOW_PATH = "{workflow_path}"
MODEL_ID = {self.config.smolagent_model_id!r}
GOAL = "{goal_prompt}"

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
        print("workflow run: saving workflow graph as PNG at ", WORKFLOW_PATH)
        try:
            png = app.get_graph().draw_mermaid_png()
            with open(os.path.join(WORKFLOW_PATH, "workflow_{uuid_str}.png"), "wb") as f:
                f.write(png)
            mermaid_code = app.get_graph().draw_mermaid()
            with open(os.path.join(WORKFLOW_PATH, "mermaid.txt"), "w") as f:
                f.write(mermaid_code)
        except Exception as e:
            raise(f"Could not save workflow graph:" + str(e))
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
        goal_prompt: str,
        template_workflow: str | None = None,
        template_uuid: str | None = None,
        save_workflow: bool = True,
    ) -> tuple[str, str]:
        """Main method to craft a complete workflow.

        Args:
            craft_instructions: The goal description
            template_workflow: pre-existing workflow template UUID
            save_workflow: Whether to save the workflow
        Returns:
            str: Complete executable workflow code
        """
        uuid_str = (
            str(uuid.uuid4()).replace("-", "")
            if template_uuid is None
            else template_uuid
        )
        tools_code, existing_tool_prompt = await self.load_tools_code()

        workflow_path, memory_path = (
            self.create_folder_structure(uuid_str)
            if save_workflow
            else (
                os.path.join(self.workflow_dir, uuid_str),
                os.path.join(self.memory_dir, uuid_str),
            )
        )

        with open(self.schema_code_path) as f:
            state_code = f.read()
        with open(self.smolagent_factory_code_path) as f:
            smolagent_factory_code = f.read()
        workflow_code = (
            template_workflow
            if template_workflow
            else self.create_workflow_code(
                goal_prompt, existing_tool_prompt, memory_path
            )
        )
        if workflow_code is None or workflow_code.strip() == "":
            print("Generated workflow:\n", workflow_code)
            raise ValueError("❌ Generated workflow code is empty or invalid")

        complete_code = self.assemble_workflow(
            tools_code,
            state_code,
            smolagent_factory_code,
            workflow_code,
            workflow_path,
            memory_path,
            uuid_str,
            goal_prompt,
        )

        print(f"workflow path {workflow_path}")
        print(f"workflow path {memory_path}")

        if save_workflow and isinstance(workflow_code, str):
            self.save_workflow_files(
                workflow_path, uuid_str, workflow_code, goal_prompt
            )
        return complete_code, uuid_str

    def save_workflow_files(
        self, path: str, uuid_str: str, workflow_code: str, goal_prompt: str
    ) -> None:
        """Save workflow code and metadata to files."""
        try:
            with open(os.path.join(path, f"workflow_code_{uuid_str}.py"), "w") as f:
                f.write(workflow_code)
            print(f"✅ Saved workflow code to: {path}/workflow_code_{uuid_str}.py")
        except Exception as e:
            print(f"❌ Failed to save workflow code: {str(e)}")

        try:
            with open(os.path.join(path, f"system_prompt_{uuid_str}.md"), "w") as f:
                f.write(self.get_system_prompt())
            print(f"✅ Saved system prompt to: {path}/system_prompt_{uuid_str}.md")
        except Exception as e:
            print(f"❌ Failed to save system prompt: {str(e)}")
