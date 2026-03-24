
"""
Base factory class for both single agent and workflow generation.
Child classes:
- SingleAgentFactory: Handles crafting of single agent code with cost tracking support.
- WorkflowFactory: Handles crafting of multi-agent workflows with dynamic tool integration.
"""

import os
import logging
import re

from .tools_manager import ToolManager

class Factory:
    """Base factory class both for single agent and workflow generation. Handles common tasks like loading tools, creating folders, and saving files."""

    def __init__(self, config) -> None:
        """Initialize the workflow crafting system.
        Args:
            config: Configuration object containing paths and settings
        """
        self.workflow_dir = config.workflow_dir
        self.memory_dir = config.memory_dir
        self.config = config
        self.logger = logging.getLogger(__name__)

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

    async def load_single_agent_system_prompt(self) -> str:
        """Load the system prompt for single agent mode.
        Returns:
            str: The system prompt content
        """
        try:
            with open(self.config.prompt_smolagent) as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to load single agent system prompt: {str(e)}") from e

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