import uuid
import os
from typing import Optional, Tuple
from core.provider import LLMProvider


class WorkflowCrafting:
    """Handles the creation and management of AI workflow generation.
    
    Attributes:
        tools_dir (str): Directory containing tool modules
        workflow_dir (str): Base directory for workflow storage
    """
    
    def __init__(self, tools_dir: str = "modules/tools", workflow_dir: str = "workflows") -> None:
        """Initialize the workflow crafting system.
        
        Args:
            tools_dir: Path to directory containing tool modules
            workflow_dir: Base directory for workflow storage
        """
        self.tools_dir = tools_dir
        self.workflow_dir = workflow_dir

    def get_system_prompt(self) -> str:
        """Load the system prompt for workflow generation.
        
        Returns:
            str: The system prompt content
        """
        try:
            with open("prompts/workflow_creator.md", 'r') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to load system prompt: {str(e)}")

    def llm_make_workflow(self, system_prompt: str, goal_prompt: str, existing_tool_prompt: str) -> str:
        """Generate workflow code using LLM.
        
        Args:
            goal_prompt: The goal description for the workflow
            existing_tool_prompt: Description of available tools
        Returns:
            str: Generated workflow code from LLM
        """
        prompt = f"""
You are an expert in generating LangGraph workflows using SmolAgent nodes.

The following tools packages are available for agents:
{existing_tool_prompt}

Your task is to create a LangGraph-SmolAgent workflow that achieves:
{goal_prompt}
        """
        history = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ]
        return LLMProvider().openai_completion(history, verbose=False)

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

    def load_tools_code(self) -> Tuple[str, str]:
        """Load all tool code from the tools directory.
        
        Returns:
            Tuple[str, str]: Tuple containing (tools_code, existing_tool_prompt)
        """
        if not os.path.exists(self.tools_dir):
            raise ValueError(f"❌ Tools directory '{self.tools_dir}' does not exist")
            
        tools_code = ""
        existing_tool_prompt = ""
        
        for filename in os.listdir(self.tools_dir):
            if not filename.endswith('.py'):
                continue
                
            filepath = os.path.join(self.tools_dir, filename)
            base_name = os.path.splitext(filename)[0]
            
            try:
                with open(filepath, 'r') as f:
                    code = f.read()
            except Exception as e:
                raise ValueError(f"❌ Error reading tool file {filename}: {str(e)}")
                
            tools_code += code + '\n'
            tool_var_name = f"{base_name.upper()}_TOOLS"
            tools_code += f"\n{tool_var_name} = tools\n"
            existing_tool_prompt += f"{tool_var_name}\n"
            
        print(f"✅ Loaded {len(os.listdir(self.tools_dir))} tools from {self.tools_dir}")
        return tools_code, existing_tool_prompt

    def create_workflow_code(self, goal_prompt: str, existing_tool_prompt: str) -> str:
        """Generate and validate workflow code.
        
        Args:
            goal_prompt: The goal description
            existing_tool_prompt: Description of available tools
        Returns:
            str: Validated workflow code
        """
        print("🧠 Generating workflow code with LLM...")
        system_prompt = self.get_system_prompt()
        llm_output = self.llm_make_workflow(system_prompt, goal_prompt, existing_tool_prompt)
        workflow_code = self.extract_python_code(llm_output)
        if not workflow_code.strip():
            raise ValueError("LLM did not return valid workflow code")
        print("✅ LLM generated workflow code successfully")
        return workflow_code

    def create_folder_structure(self, uuid_str: str) -> str:
        """Create directory structure for new workflow.
        
        Args:
            uuid_str: Unique identifier for the workflow
        Returns:
            str: Path to created workflow directory
        """
        workflow_path = os.path.join(self.workflow_dir, uuid_str)
        os.makedirs(workflow_path, exist_ok=True)
        print(f"✅ Created workflow directory: {workflow_path}")
        return workflow_path

    def assemblate_workflow(self, goal_prompt: str,
                            tools_code: str,
                            state_code: str,
                            smolagent_factory_code: str,
                            workflow_code: str,
                            path: str,
                            uuid_str: str
                           ) -> str:
        return f'''
from smolagents import CodeAgent, tool, HfApiModel
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Tuple, Any, Dict, Union, Optional, Callable
import json
import os

# Load tools
{tools_code}

# Load state schema
{state_code}

# Load smolagent factory
{smolagent_factory_code}

# Generated workflow
{workflow_code}

# Initialize and execute workflow
initial_state = {{
    "step_name": ["{goal_prompt}"],
    "actions": [],
    "observations": [],
    "rewards": [],
    "answers": [],
    "success": []
}}

try:
    png = app.get_graph().draw_mermaid_png()
    with open(os.path.join("{path}", "workflow_{uuid_str}.png"), "wb") as f:
        f.write(png)
except Exception as e:
    print(f"Could not save workflow graph:" + str(e))

try:
    result_state = app.invoke(initial_state)
except KeyboardInterrupt:
    print("Workflow execution interrupted by user")
    pass

try:
    with open(os.path.join("{path}", "state_result_{uuid_str}.json"), "w") as f:
        json.dump(result_state, f, indent=2)
except Exception as e:
    print(f"Could not save workflow data:" + str(e))
'''

    def craft_workflow(
        self,
        goal_prompt: str,
        template_workflow: Optional[str] = None,
        save_workflow: bool = True
    ) -> str:
        """Main method to craft a complete workflow.
        
        Args:
            goal_prompt: The goal description
            template_workflow: Optional pre-existing workflow template
            save_workflow: Whether to save the workflow
        Returns:
            str: Complete executable workflow code
        """
        uuid_str = str(uuid.uuid4()).replace("-", "")
        tools_code, existing_tool_prompt = self.load_tools_code()
        
        state_code = open("modules/state_schema.py").read()
        smolagent_factory_code = open("modules/smolagent_factory.py").read()
        workflow_code = (
            template_workflow 
            if template_workflow 
            else self.create_workflow_code(goal_prompt, existing_tool_prompt)
        )
        
        path = self.create_folder_structure(uuid_str) if save_workflow else ""
        
        complete_code = self.assemblate_workflow(
            goal_prompt,
            tools_code,
            state_code,
            smolagent_factory_code,
            workflow_code,
            path,
            uuid_str
        )

        if save_workflow:
            try:
                with open(os.path.join(path, f"workflow_code_{uuid_str}.py"), 'w') as f:
                    f.write(workflow_code)
                print(f"✅ Saved workflow code to: {path}/workflow_code_{uuid_str}.py")
            except Exception as e:
                print(f"❌ Failed to save workflow code: {str(e)}")
        
        try:
            with open(os.path.join(path, f"system_prompt_{uuid_str}.md"), 'w') as f:
                f.write(self.get_system_prompt())
            print(f"✅ Saved system prompt to: {path}/system_prompt_{uuid_str}.md")
        except Exception as e:
            print(f"❌ Failed to save system prompt: {str(e)}")
        return complete_code

def craft_workflow(goal_prompt: str, template_workflow=None, save_workflow=True) -> str:
    return WorkflowCrafting().craft_workflow(goal_prompt, template_workflow, save_workflow)
