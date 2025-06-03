#!/usr/bin/env python3
"""
Meta-Agent Prototype: LLM-Generated LangGraph Workflows with SmolAgent Nodes
============================================================================

This prototype demonstrates the core concept of using an LLM to generate
LangGraph workflows that use SmolAgent instances as nodes.
"""

import os
import json
from typing import Dict, Any, List
from e2b_code_interpreter import Sandbox
import openai
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def create_sandbox_with_dependencies():
    """Create and configure the sandbox with all required dependencies"""
    sandbox = Sandbox()
    
    # Install required packages
    print("Installing dependencies...")
    sandbox.commands.run("pip install smolagents")
    sandbox.commands.run("pip install langgraph")
    sandbox.commands.run("pip install langchain")
    sandbox.commands.run("pip install langchain-core")
    sandbox.commands.run("pip install requests")
    sandbox.commands.run("pip install beautifulsoup4")
    
    return sandbox

def run_code_raise_errors(sandbox, code: str, verbose: bool = False) -> str:
    """Execute code in sandbox and raise errors if execution fails"""
    execution = sandbox.run_code(
        code,
        envs={'HF_TOKEN': os.getenv('HF_TOKEN')}
    )
    if execution.error:
        execution_logs = "\n".join([str(log) for log in execution.logs.stdout])
        logs = execution_logs
        logs += execution.error.traceback
        raise ValueError(logs)
    return "\n".join([str(log) for log in execution.logs.stdout])

def llm_generate_code(task_prompt: str) -> str:
    """
    Use LLM to generate LangGraph workflow code with SmolAgent nodes
    """
    system_prompt = """
You are an expert in creating multi-agent workflows using LangGraph and SmolAgent.
Generate Python code that creates a LangGraph workflow where each node is a SmolAgent instance.

## SmolAgent documentation
SmolAgent is an autonomous coding agent that executes tasks through Python code generation and tool usage.

### CodeAgent documentation (from smolagents library)

```python
from smolagents import CodeAgent, tool
from smolagents import HfApiModel, InferenceClientModel

# CodeAgent Constructor Parameters:
# - tools: List[callable] - List of tool functions decorated with @tool
# - model: Model instance (HfApiModel or InferenceClientModel) 
# - name: str - Agent identifier
# - description: str (optional) - Agent description
# - max_steps: int (default=10) - Maximum execution steps
# - additional_authorized_imports: List[str] (optional) - Extra imports allowed

agent = CodeAgent(
   tools=[tool1, tool2],
   model=model_instance,
   name="agent_name",
   max_steps=15
)

CodeAgent.run() method:
 - Takes instruction string as input
 - Returns string result after task completion
 - Executes action/observation loop internally:
   * Action: Generate and execute Python code using tools
   * Observation: Receive tool execution results
   * Continues until task complete or max_steps reached
result = agent.run("Your task instruction here")

### Integration Pattern
```python
from smolagents import CodeAgent, tool
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

model = "Qwen/Qwen2.5-Coder-32B-Instruct"

class WorkflowState(TypedDict):
    messages: list
    current_url: str
    extracted_data: dict
    next_action: str

def create_smolagent_node(tools, name, instructions):
    agent = CodeAgent(tools=tools, model=model, name=name)
    
    def node_function(state: WorkflowState):
        result = agent.run(instructions)
        return {"messages": state["messages"] + [result]}
    return node_function

### LangGraph documentation

from langgraph.graph import StateGraph, START, END
from typing import TypedDict

### Define State Schema
class YourWorkflowState(TypedDict):
    # Define all state fields that will be passed between nodes
    field1: str
    field2: list
    field3: dict
    # State is automatically merged - each node returns partial state updates

### Create Graph
workflow = StateGraph(YourWorkflowState)

### StateGraph Methods:
# - add_node(node_name: str, node_function: callable)
# - add_edge(from_node: str, to_node: str) 
# - add_conditional_edges(source: str, condition_func: callable, mapping: dict)
# - set_entry_point(node_name: str) or add_edge(START, node_name)
# - set_finish_point(node_name: str) or add_edge(node_name, END)

### Build workflow
workflow.add_node("node_name", node_function)
workflow.add_edge(START, "node_name")
```

### Node function

Node functions must:
- Accept state as first parameter with correct TypedDict type
- Return dictionary with state updates (partial state)
- State updates are merged into existing state

def node_function(state: YourWorkflowState) -> dict:
    # Process current state
    current_data = state.get("field1", "")
    # Perform node logic (e.g., run SmolAgent)
    result = some_processing(current_data)
    # Return state updates
    return {
        "field1": updated_value,
        "field2": new_list_value
    }

A node function could be a use smolagent.run() method to execute a task with an agent.

### Condition functions

Condition functions must:
- Accept state as parameter
- Return string key that matches mapping dictionary keys

def routing_condition(state: YourWorkflowState) -> str:
if state.get("some_field") == "success":
    return "success_path"
else:
    return "retry_path"

### Add conditional edges

workflow.add_conditional_edges(
    source="source_node",
    condition=routing_condition,
    mapping={
        "success_path": "next_node",
        "retry_path": "retry_node",
        "end": END
    }
)

# Example

from smolagents import CodeAgent, tool, HfApiModel
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 1. Define State Schema
class WorkflowState(TypedDict):
    messages: list
    current_url: str
    extracted_data: dict
    next_action: str
    error_count: int

# 2. Define Tools if and only if no available tools can be used
@tool
def example_tool(param: str) -> str:
    return f"processed: {param}"

# 4. Create SmolAgent Node Factory
def create_smolagent_node(tools: list, agent_name: str, instructions_template: str):
    agent = CodeAgent(
        tools=tools,
        model=model,
        name=agent_name,
        max_steps=10
    )
    
    def node_function(state: WorkflowState) -> dict:
        try:
            # Format instructions with current state
            instructions = instructions_template.format(**state)
            
            # Execute SmolAgent
            result = agent.run(instructions)
            
            # Parse result and update state
            return {
                "messages": state["messages"] + [f"{agent_name}: {result}"],
                "last_result": result
            }
        except Exception as e:
            return {
                "messages": state["messages"] + [f"{agent_name} error: {str(e)}"],
                "error_count": state.get("error_count", 0) + 1
            }
    
    return node_function

# 5. Build Workflow
workflow = StateGraph(WorkflowState)

# Add nodes
node1 = create_smolagent_node([tool1, tool2], "agent1", "Do task 1 with {current_url}")
workflow.add_node("agent1", node1)

# Add edges and routing
workflow.add_edge(START, "agent1")
workflow.add_edge("agent1", END)

# 6. Compile and Execute
app = workflow.compile()
result = app.invoke({
    "messages": [],
    "current_url": "",
    "extracted_data": {},
    "next_action": "start",
    "error_count": 0
})

## Available Tools for smolagent:
- go_to_url(url: str) -> bool: Navigate to URL
- get_page_text() -> str: Get page text content  
- get_navigable_links() -> List[str]: Get clickable links
- is_link_valid(url: str) -> bool: Check if link is valid
- get_form_inputs() -> List[str]: Get form input fields
- fill_form(values: List[str]) -> bool: Fill form with values
- screenshot() -> str: Take page screenshot

## Requirements:
1. Create a complete, runnable Python script
2. Use proper LangGraph StateGraph structure
3. Each node should be a SmolAgent with specific tools and instructions
4. Include proper state management between nodes
5. Add conditional routing based on agent outputs
6. Include error handling and logging

## Output Format:
Provide ONLY the Python code, no explanations or markdown. The code should be ready to execute.
No markdown formatting or explanations
Code must be complete and ready to run
Include all necessary imports
Use the exact tool signatures provided
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a workflow for this task: {task_prompt}"}
        ],
        temperature=0.1,
        max_tokens=2000
    )
    
    return response.choices[0].message.content.strip()

def create_mock_tools_setup() -> str:
    """Create mock tools setup code for the sandbox"""
    # NOTE make all tools as MCP server ?
    return """
# Mock tools for demonstration (replace with actual browser tools in production)
from smolagents import tool
from typing import List
import time
import random

@tool
def go_to_url(url: str) -> bool:
    '''Navigate to a specified URL'''
    print(f"Navigating to: {url}")
    time.sleep(0.5)  # Simulate navigation delay
    return True

@tool
def get_page_text() -> str:
    '''Get text content of current page'''
    mock_content = '''
    Welcome to Example Website
    This is a sample page with some content.
    Here you can find information about our services.
    Contact us for more details.
    '''
    print("Extracted page text")
    return mock_content.strip()

@tool  
def get_navigable_links() -> List[str]:
    '''Get navigable links from current page'''
    mock_links = [
        "https://example.com/about",
        "https://example.com/services", 
        "https://example.com/contact"
    ]
    print(f"Found {len(mock_links)} navigable links")
    return mock_links

@tool
def is_link_valid(url: str) -> bool:
    '''Check if a link is valid'''
    print(f"Validating link: {url}")
    return not url.endswith("/broken")

@tool
def get_form_inputs() -> List[str]:
    '''Get form inputs from current page'''
    mock_inputs = ["[email]()", "[name]()", "[message]()"]
    print(f"Found {len(mock_inputs)} form inputs")
    return mock_inputs

@tool
def fill_form(values: List[str]) -> bool:
    '''Fill form with provided values'''
    print(f"Filling form with {len(values)} values")
    return True

@tool
def screenshot() -> str:
    '''Take screenshot of current page'''
    screenshot_path = "/tmp/screenshot.png"
    print(f"Screenshot saved to: {screenshot_path}")
    return screenshot_path
"""

def main():
    """Main execution function"""
    print("🚀 Starting Meta-Agent Prototype...")
    
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    if not os.getenv('HF_TOKEN'):
        print("⚠️  Warning: HF_TOKEN not set, using mock model")
    
    print("📦 Setting up sandbox environment...")
    sandbox = create_sandbox_with_dependencies()
    mock_tools = create_mock_tools_setup()
    
    task_prompt = f"""
You are an expert in creating multi-agent workflows using LangGraph and SmolAgent.
You are given these tools:
- go_to_url(url: str) -> bool: Navigate to URL
- get_page_text() -> str: Get page text content  
- get_navigable_links() -> List[str]: Get clickable links
- is_link_valid(url: str) -> bool: Check if link is valid
- get_form_inputs() -> List[str]: Get form input fields
- fill_form(values: List[str]) -> bool: Fill form with values
- screenshot() -> str: Take page screenshot

Tool are already defined, you just need to use them in the workflow.

Create a sophisticated web browsing agent logic using langraph.
The workflow should be robust, handle errors gracefully, and provide detailed logging.
Create a complete LangGraph workflow where each major step is handled by a specialized SmolAgent node.
"""
    
    print("🧠 Generating workflow code with LLM...")
    try:
        agent_code = llm_generate_code(task_prompt)
        print("=" * 50)
        print(agent_code)
        print("=" * 50)
        complete_code = f"""
{mock_tools}

{agent_code}

try:
    print("\\n🔄 Executing generated workflow...")
    if 'workflow' in locals() or 'app' in locals():
        # Use the compiled workflow
        workflow_app = locals().get('app') or locals().get('workflow')
        if hasattr(workflow_app, 'invoke'):
            initial_state = {{
                "messages": [],
                "current_url": "https://example.com",
                "extracted_data": {{}},
                "next_action": "start"
            }}
            
            print("Initial state:", initial_state)
            result = workflow_app.invoke(initial_state)
            print("\\n✅ Workflow execution completed!")
            print("Final result:", result)
        else:
            print("✅ Workflow structure created successfully!")
            print("Workflow object:", type(workflow_app))
    else:
        print("⚠️  Workflow not found in generated code")
except Exception as e:
    print(f"❌ Workflow execution error: {{e}}")
    import traceback
    traceback.print_exc()
"""
        
        print("\n🔧 Executing generated workflow in sandbox...")
        execution_logs = run_code_raise_errors(sandbox, complete_code, verbose=True)
        print("\n📊 Execution Results:")
        print("=" * 60)
        print(execution_logs)
        print("=" * 60)
        print("\n✅ Meta-Agent prototype completed successfully!")
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Cleaning up sandbox...")
        try:
            sandbox.close()
        except:
            pass

if __name__ == "__main__":
    main()