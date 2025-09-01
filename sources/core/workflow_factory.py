"""
This class handles the creation and assembly of Langraph-SmolAgent workflow generation.
"""

import logging
import os
import re
import time
import uuid

from sources.modules import state_schema

from .llm_provider import LLMProvider
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
        """Generate a workflow using the LLM."""
        prompt = f"""
You are an expert in generating LangGraph workflows using SmolAgent nodes.

The following tools packages are available for agents:
{existing_tool_prompt}

CRITICAL CONSTRAINT: Agents can ONLY use the tools listed above. If a task requires capabilities not available in the listed tools, you MUST either:
1. Find alternative approaches using available tools (e.g., use shell commands instead of web_search)  
2. Create agents with empty tool lists [] that rely only on Python code execution
3. Clearly state that the task cannot be completed with available tools

Do NOT assume any tools exist beyond what is explicitly listed above.

Your task is to create a LangGraph-SmolAgent workflow for the task:
{craft_instructions}
        """
        return LLMProvider("workflow_creator", path, system_prompt)(prompt)

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

    def remove_imports(self, code: str) -> str:
        # remove attempt from LLM to import modules/class
        lines = code.splitlines()
        return "\n".join(
            line for line in lines 
            if not (line.strip().startswith("import ") or line.strip().startswith("from "))
        )

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
        self.logger.info("Generating workflow code with LLM")
        system_prompt = self.get_system_prompt()
        llm_output = self.llm_make_workflow(
            system_prompt, craft_instructions, existing_tool_prompt, path
        )
        workflow_code = self.extract_python_code(llm_output)
        workflow_code = self.remove_imports(workflow_code)
        if not workflow_code.strip():
            raise ValueError("LLM did not return valid workflow code")
        
        # Validate syntax before returning
        try:
            compile(workflow_code, '<workflow>', 'exec')
        except SyntaxError as e:
            self.logger.error(f"Generated workflow has syntax error: {e}")
            raise ValueError(f"LLM generated invalid Python syntax: {e}") from e
        
        self.logger.info("LLM generated workflow code successfully")
        return workflow_code

    def validate_workflow_structure(self, workflow_code: str) -> None:
        """Validate LangGraph workflow structure before execution."""
        self.logger.info("Validating workflow structure...")
        
        # Pre-compile regex patterns for efficiency
        patterns = {
            'state_graph': r"workflow = StateGraph\(WorkflowState\)",
            'start_edge': r"workflow\.add_edge\(START,\s*[\"'](\w+)[\"']\)",
            'nodes': r"workflow\.add_node\([\"'](\w+)[\"'],.*?\)",
            'routers': r"def\s+(\w*router\w*)\s*\(",
            'conditional_edges': r"workflow\.add_conditional_edges\(",
            'edge_mappings': r'workflow\.add_conditional_edges\(\s*["\'](\w+)["\'],\s*(\w+),\s*\{([^}]+)\}',
            'router_returns': r'return\s+["\']([^"\']+)["\']',
            'agent_factory': r"SmolAgentFactory\(",
            'node_factory': r"WorkflowNodeFactory\.create_agent_node\("
        }
        
        # Basic structure validation
        required_checks = [
            (patterns['state_graph'], "Missing 'workflow = StateGraph(WorkflowState)' initialization"),
            (patterns['conditional_edges'], "No conditional edges found - workflows require conditional routing"),
            (patterns['agent_factory'], "No SmolAgentFactory usage found"),
            (patterns['node_factory'], "No WorkflowNodeFactory usage found")
        ]
        
        for pattern, error_msg in required_checks:
            if not re.search(pattern, workflow_code):
                raise ValueError(error_msg)
        
        # Extract and validate core components
        start_match = re.search(patterns['start_edge'], workflow_code)
        if not start_match:
            raise ValueError("Graph must have entry point: workflow.add_edge(START, 'node_name')")
        
        nodes = set(re.findall(patterns['nodes'], workflow_code))
        if not nodes:
            raise ValueError("No workflow nodes found")
        self.logger.debug(f"📋 Workflow nodes discovered: {', '.join(sorted(nodes))}")
        
        routers = set(re.findall(patterns['routers'], workflow_code))
        if not routers:
            raise ValueError("No router functions found")
        self.logger.debug(f"🔀 Router functions found: {', '.join(sorted(routers))}")
        
        # Validate START edge target exists
        entry_node = start_match.group(1)
        if entry_node not in nodes:
            raise ValueError(f"START targets non-existent node '{entry_node}'")
        self.logger.debug(f"🚀 Workflow entry point: START → {entry_node}")
        
        # Validate routing consistency
        self._validate_routing_consistency(workflow_code, patterns, nodes, routers)
        
        self.logger.info("✅ Workflow structure validation passed")
    
    def _validate_routing_consistency(self, workflow_code: str, patterns: dict, nodes: set, routers: set) -> None:
        """Validate routing logic consistency."""
        conditional_edges = re.findall(patterns['edge_mappings'], workflow_code)
        all_mapping_keys = set()
        used_routers = set()
        
        self.logger.debug(f"🔗 Analyzing {len(conditional_edges)} conditional edges:")
        
        for i, (source_node, _router_func, mapping_content) in enumerate(conditional_edges, 1):
            if source_node not in nodes:
                raise ValueError(f"Conditional edge source '{source_node}' doesn't exist")
            
            used_routers.add(_router_func)
            
            # Parse and display mapping in human-readable format
            mapping_pairs = re.findall(r'["\']([^"\']+)["\']:\s*([^,}]+)', mapping_content)
            readable_mappings = []
            
            for key, target in mapping_pairs:
                all_mapping_keys.add(key)
                target_clean = target.strip().strip('"\'')
                
                if target_clean == "START":
                    raise ValueError("Router mapping contains START - use node names or END")
                if target_clean not in nodes and target_clean != "END":
                    raise ValueError(f"Router target '{target_clean}' doesn't exist")
                
                readable_mappings.append(f"'{key}' → {target_clean}")
            
            self.logger.debug(f"   {i}. {source_node} --({_router_func})--> {{ {', '.join(readable_mappings)} }}")
        
        self.logger.debug(f"🗝️  Available routing keys: {', '.join(sorted(all_mapping_keys))}")
        self.logger.debug(f"⚙️  Routers in use: {', '.join(sorted(used_routers))}")
        
        # Validate router functions exist
        missing_routers = used_routers - routers
        if missing_routers:
            raise ValueError(f"Missing router functions: {sorted(missing_routers)}")
        
        # Check router return values don't use START
        router_returns = re.findall(patterns['router_returns'], workflow_code)
        unique_returns = sorted(set(router_returns))
        self.logger.debug(f"🔄 Router return values used: {', '.join(unique_returns)}")
        
        for return_val in router_returns:
            if return_val == "START":
                raise ValueError("Router returns 'START' - use mapping keys or END")
            if return_val != "END" and return_val not in all_mapping_keys:
                raise ValueError(f"Router returns invalid key '{return_val}'")
    


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
        goal_prompt: str,
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
            goal_prompt: The goal description for the workflow
        Returns:
            str: Complete workflow code ready for execution
        """

        
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

MEMORY_PATH = {memory_path!r}
WORKFLOW_PATH = {workflow_path!r}
MODEL_ID = {self.config.smolagent_model_id!r}
ENGINE_NAME = {self.config.engine_name!r}
GOAL = {goal_prompt!r}

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
        goal_prompt: str,
        template_workflow: str | None = None,
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
        # Generate chronologically sortable workflow ID: YYYYMMDD_HHMMSS_shortUUID
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        uuid_str = f"{timestamp}_{short_uuid}"
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
        # Generate workflow code - let DGM handle retries
        if template_workflow:
            workflow_code = template_workflow
        else:
            workflow_code = self.create_workflow_code(
                goal_prompt, existing_tool_prompt, memory_path
            )
            # Save workflow code immediately so DGM can access it even if validation fails
        if save_workflow and isinstance(workflow_code, str):
            self.save_workflow_files(
                workflow_path, uuid_str, workflow_code, goal_prompt
            )
        try :
            # Validate workflow structure before assembly
            self.validate_workflow_structure(workflow_code)
        except Exception as e:
            # Include UUID in exception so orchestrator can return it
            raise ValueError(f"UUID:{uuid_str}|{str(e)}") from e
        
        # Assemble complete workflow
        complete_code = self.assemble_workflow(
            tools_code, state_code, smolagent_factory_code,
            workflow_code, workflow_path, memory_path, uuid_str, goal_prompt
        )
        
        self.logger.info("Workflow generation completed")

        self.logger.debug(f"Workflow path: {workflow_path}")
        self.logger.debug(f"Memory path: {memory_path}")

        return complete_code, uuid_str

    def save_workflow_files(
        self, path: str, uuid_str: str, workflow_code: str, goal_prompt: str
    ) -> None:
        """Save workflow code and metadata to files."""
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
                f.write(goal_prompt)
            self.logger.info(
                f"Saved goal to: {path}/goal_{uuid_str}.txt"
            )
        except Exception as e:
            self.logger.error(f"Failed to save goal: {str(e)}")
