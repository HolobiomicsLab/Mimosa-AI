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
from anthropic import Anthropic
import openai
from openai import OpenAI

model_choice = "openai"  # Change to "deepseek" or "anthropic" as needed

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

def deepseek_fn(history, verbose=False):
    """
    Use deepseek api to generate text.
    """
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=history,
            stream=False
        )
        thought = response.choices[0].message.content
        if verbose:
            print(thought)
        return thought
    except Exception as e:
        raise Exception(f"Deepseek API error: {str(e)}") from e

def openai_fn(history, verbose=False):
    """
    Use openai to generate text.
    """
    client = OpenAI(api_key=openai_api_key)

    try:
        response = client.chat.completions.create(
            model="o3-2025-04-16",
            messages=history,
        )
        if response is None:
            raise Exception("OpenAI response is empty.")
        thought = response.choices[0].message.content
        if verbose:
            print(thought)
        return thought
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}") from e

def anthropic_fn(history, verbose=False):
    """
    Use Anthropic to generate text with streaming.
    """
    client = Anthropic(api_key=anthropic_api_key)
    system_message = None
    messages = []
    for message in history:
        clean_message = {'role': message['role'], 'content': message['content']}
        if message['role'] == 'system':
            system_message = message['content']
        else:
            messages.append(clean_message)

    try:
        response = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=12000,
            messages=messages,
            system=system_message
        )
        if response is None:
            raise Exception("Anthropic response is empty.")
        thought = response.content[0].text
        if verbose:
            print(thought)
        return thought
    except Exception as e:
        raise Exception(f"Anthropic API error: {str(e)}") from e


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
    sandbox.commands.run("pip install grandalf")
    
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

# 2. Create SmolAgent Node Factory
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

# 3. Build Workflow
workflow = StateGraph(WorkflowState)

# Add nodes
node1 = create_smolagent_node([tool1, tool2], "agent1", "Do task 1 with {current_url}")
workflow.add_node("agent1", node1)

# Add edges and routing
workflow.add_edge(START, "agent1")
workflow.add_edge("agent1", END)

# draw graph
Image(workflow.get_graph().draw_mermaid_png())

# 6. Compile and Execute
app = workflow.compile()
result = app.invoke({
    "messages": [],
    "current_url": "",
    "extracted_data": {},
    "next_action": "start",
    "error_count": 0
})

7. print the result
print("Workflow result:", result)


## Available Tools for smolagent:
- go_to_url(url: str) -> bool: Navigate to URL
- get_page_text() -> str: Get page text content  
- get_navigable_links() -> List[str]: Get clickable links
- is_link_valid(url: str) -> bool: Check if link is valid
- get_form_inputs() -> List[str]: Get form input fields
- fill_form(values: List[str]) -> bool: Fill form with values
- screenshot() -> str: Take page screenshot
These tool are fully available, do not redefine.

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

    history = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': task_prompt}]
    if model_choice == "deepseek":
        return deepseek_fn(history)
    elif model_choice == "anthropic":
        return anthropic_fn(history)
    return openai_fn(history)

def create_mock_tools_setup() -> str:
    """Create mock tools setup code for the sandbox"""
    # NOTE make all tools as MCP server ?
    return '''
# Mock tools for demonstration (replace with actual browser tools in production)
from smolagents import tool
from typing import List
import time
import random

@tool
def go_to_url(url: str) -> bool:
    """Navigate to a specified URL in the mock browser.
    
    Args:
        url: The URL to navigate to
        
    Returns:
        bool: True if navigation was successful (always True in mock implementation)
        
    Side effects:
        Prints navigation message and simulates delay
    """
    print(f"Navigating to: {url}")
    time.sleep(0.5)  # Simulate navigation delay
    return True

@tool
def get_page_text() -> str:
    """Get the text content of the current mock webpage.
    
    Returns:
        str: The mock webpage content as a formatted string
    """
    return None

@tool  
def get_navigable_links() -> List[str]:
    """Get all clickable links from the current mock webpage.
    
    Returns:
        List[str]: A list of mock URLs that would be found on a webpage
        
    Side effects:
        Prints the number of links found
    """
    mock_links = [
        "https://example.com/about",
        "https://example.com/services", 
        "https://example.com/contact"
    ]
    print(f"Found {len(mock_links)} navigable links")
    return mock_links

@tool
def is_link_valid(url: str) -> bool:
    """Check if a URL is valid in the mock browser.
    
    Args:
        url: The URL to validate
        
    Returns:
        bool: True if URL is valid (not ending with "/broken"), False otherwise
        
    Side effects:
        Prints validation message
    """
    print(f"Validating link: {url}")
    return not url.endswith("/broken")

@tool
def get_form_inputs() -> List[str]:
    """Get all input fields from the current mock webpage form.
    
    Returns:
        List[str]: A list of mock form input field names
        
    Side effects:
        Prints the number of inputs found
    """
    mock_inputs = ["[email]()", "[name]()", "[message]()"]
    print(f"Found {len(mock_inputs)} form inputs")
    return mock_inputs

@tool
def fill_form(values: List[str]) -> bool:
    """Fill the mock webpage form with provided values.
    
    Args:
        values: List of values to fill into form inputs
        
    Returns:
        bool: True if form was filled successfully (always True in mock implementation)
        
    Side effects:
        Prints number of values being filled
    """
    print(f"Filling form with {len(values)} values")
    return True

@tool
def screenshot() -> str:
    """Take a screenshot of the current mock webpage.
    
    Returns:
        str: Path to the saved screenshot file
        
    Side effects:
        Prints the path where screenshot was saved
    """
    screenshot_path = "/tmp/screenshot.png"
    print(f"Screenshot saved to: {screenshot_path}")
    return screenshot_path
'''

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
        if agent_code is None or agent_code.strip() == "":
            raise ValueError("LLM did not return any code")
        complete_code = f"""
# pre-defined tools
{mock_tools}

# LLM generated graph code
{agent_code}
"""
        print("=" * 50)
        print(complete_code)
        print("=" * 50)
        exit(1)
        
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
