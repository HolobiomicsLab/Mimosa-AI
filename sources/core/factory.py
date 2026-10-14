
"""Shared plumbing for crafting runnable agent code in Mimosa-AI.

This module owns the code-generation basics both emitters reuse: MCP
tool loading, run-folder creation, and metadata saving.
:class:`WorkflowFactory` (multi-agent workflows) and
:class:`SingleAgentFactory` (single-agent baseline) subclass
:class:`Factory` and add their own script assembly. The orchestrator and
the evolution engine reach the subclasses at crafting time, before any
generated code runs (Layers 2-3).
"""

import os
import logging
import re

from .tools_manager import ToolManager


class Factory:
    """Base class for the workflow and single-agent code factories.

    The two factory subclasses are constructed with the global ``Config``
    by the orchestrator. This base owns the shared plumbing: MCP tool
    loading, run-folder creation, and metadata saving. Key invariant:
    every run gets one UUID that names its folder pair and its saved
    files.
    """

    def __init__(self, config: "Config") -> None:
        """Store the config, the output folders, and a module logger.

        Called by subclass constructors at crafting time.

        Args:
            config: Global ``Config`` object; supplies ``workflow_dir``,
                ``memory_dir``, prompt paths, and engine settings.
        """
        self.workflow_dir = config.workflow_dir
        self.memory_dir = config.memory_dir
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def load_tools_code(self) -> tuple[str, str]:
        """Discover all MCP servers and format their client code.

        Both factories call this before rendering a script, so the
        generated code can reach every available MCP tool. Discovery
        retries with exponential backoff for up to 300 seconds while MCP
        containers start.

        Returns:
            Tuple ``(tools_code, existing_tool_prompt)``:
                - ``tools_code``: Combined Python code for all MCP
                  clients.
                - ``existing_tool_prompt``: Prompt listing discovered MCP
                  names for tool awareness during generation.

        Raises:
            RuntimeError: If MCP servers do not become ready in time.
            ValueError: If no MCP servers are found.
        """
        tools_code = ""
        existing_tool_prompt = ""
        import asyncio as _asyncio
        import time as _time
        tool_manager = ToolManager(self.config)
        # MCP servers can take minutes to come up (Docker image builds, cold
        # starts, a server brought up by hand). Bound on total wait, not on a
        # fixed attempt count. Exponential backoff capped at 30 s per wait.
        max_wait_seconds = 300.0
        backoff = 1.0
        deadline = _time.monotonic() + max_wait_seconds
        attempt = 0
        mcps = []
        while True:
            attempt += 1
            try:
                mcps = await tool_manager.discover_mcp_servers()
                tool_setup = await tool_manager.verify_tools()
                if tool_setup and mcps:
                    break
                reason = "MCP servers not ready yet"
            except Exception as e:
                reason = str(e)
            if _time.monotonic() >= deadline:
                self.logger.error(
                    f"load_tools_code: MCP discovery failed after {max_wait_seconds:.0f}s "
                    f"({attempt} attempts): {reason}"
                )
                raise RuntimeError(
                    f"MCP servers did not become ready within {max_wait_seconds:.0f}s: {reason}"
                )
            self.logger.warning(
                f"MCP discovery attempt {attempt} failed ({reason}). Retrying in {backoff:.1f}s…"
            )
            await _asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
        if not mcps:
            raise ValueError(
                "\n" + "=" * 80 +
                "\n🚨  FATAL ERROR: No MCP Servers Found! 🚨"
                "\n" + "-" * 80 +
                "\nPlease ensure at least one MCP instance is running on Toolomics."
                "\n" + "=" * 80 + "\n"
            )
        for mcp in mcps:
            client_code = tool_manager.get_client_code(mcp)
            client_prompt = tool_manager.get_client_prompt(mcp)
            tools_code += client_code + "\n"
            existing_tool_prompt += client_prompt + "\n"
        return tools_code, existing_tool_prompt

    async def load_single_agent_system_prompt(self) -> str:
        """Load the system prompt text for single-agent mode.

        :class:`SingleAgentFactory` calls it while rendering the
        single-agent script, and the workflow factory reads it too when
        assembling agent prompts.

        Returns:
            The prompt text read from ``config.prompt_smolagent`` on
            disk.

        Raises:
            ValueError: If the system-prompt file cannot be read.
        """
        try:
            with open(self.config.prompt_smolagent) as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to load single agent system prompt: {str(e)}") from e

    def create_folder_structure(self, uuid_str: str) -> tuple[str, str]:
        """Create the workflow and memory folders for one run.

        Both factories call this right before saving generated code,
        using the run UUID as the folder name.

        Args:
            uuid_str: Unique run identifier minted by the caller.

        Returns:
            Tuple ``(workflow_path, memory_path)``: the two created
            directory paths under ``workflow_dir`` and ``memory_dir``.
        """
        workflow_path = os.path.join(self.workflow_dir, uuid_str)
        self.logger.info(f"Created workflow directory: {workflow_path}")
        os.makedirs(workflow_path, exist_ok=True)
        memory_path = os.path.join(self.memory_dir, uuid_str)
        os.makedirs(memory_path, exist_ok=True)
        self.logger.info(f"Created memory directory: {memory_path}")
        return workflow_path, memory_path

    def _extract_original_from_goal(self, goal: str) -> str:
        """Extract the original task from a knowledge-wrapped goal string.

        :meth:`save_workflow_files` calls this when the caller did not
        pass an explicit original task, so run folders keep the raw task
        for similarity matching. It looks for the marker ``"Now, use this
        knowledge to complete:"`` produced by
        ``planner._build_knowledge_aware_task()`` and returns the text
        that follows it. When the marker is absent, it returns the goal
        unchanged.

        Args:
            goal: Goal text from the planner; may be knowledge-wrapped.

        Returns:
            The extracted original task, or the goal itself when not
            wrapped. Returns an empty string when ``goal`` is falsy.
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
        self, path: str, uuid_str: str, workflow_genotype_code: str, goal: str, original_task: str | None = None
    ) -> None:
        """Write the generated code and its metadata files to a run folder.

        Both factories call this after rendering a script, so every run
        keeps its genotype, system prompt, goal, and original task on
        disk. It writes ``workflow_genotype_<uuid>.py``,
        ``system_prompt_<uuid>.md``, ``goal_<uuid>.txt`` and, when
        available, ``original_task_<uuid>.txt`` into ``path``. Each write
        failure is logged and skipped without raising.

        Args:
            path: Run folder created by :meth:`create_folder_structure`.
            uuid_str: Unique workflow identifier; names each saved file.
            workflow_genotype_code: Generated code from the factory.
            goal: The goal description (may be knowledge-wrapped).
            original_task: Original unwrapped task from the caller. When
                ``None``, it is extracted from ``goal`` instead.
        """
        try:
            with open(os.path.join(path, f"workflow_genotype_{uuid_str}.py"), "w") as f:
                f.write(workflow_genotype_code)
            self.logger.info(
                f"Saved workflow code to: {path}/workflow_genotype_{uuid_str}.py"
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