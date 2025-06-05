
from smolagents import CodeAgent, tool
from smolagents.agents import ActionStep
from smolagents import (
    HfApiModel,
    DuckDuckGoSearchTool
)
from smolagents import CodeAgent, tool, HfApiModel
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# pre-defined tools
# Mock tools for demonstration (replace with actual browser tools in production)
from typing import List
import time
import random
from smolagents import (
    tool,
    HfApiModel,
    DuckDuckGoSearchTool
)

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

tools = [
    go_to_url,
    get_page_text,
    get_navigable_links,
    is_link_valid,
    get_form_inputs,
    fill_form,
    screenshot,
    DuckDuckGoSearchTool(),
]

model = "Qwen/Qwen2.5-Coder-32B-Instruct"

class WorkflowState(TypedDict):
    goal: str
    step_name: str
    actions: List[str]
    observations: List[str]
    rewards: List[float]

def create_smolagent_node(state: WorkflowState) -> dict:
    instructions_template = '''
    Target goal is to {goal}.
    You previously took the following actions: {actions}.
    You last made the following observations: {observations}.
    Current task: Continue working towards the goal using available tools.
    '''

    agent = CodeAgent(
        tools=tools, # tools declared in mock_tools.py
        model=model,
        name=state["step_name"],
        max_steps=5,
    )
    try:
        # Format the template with current state
        instructions = instructions_template.format(
            goal=state["goal"][-1] if state["goal"] else "complete the task",
            actions=state["actions"][-3:] if state["actions"] else ["none"],
            observations=state["observations"][-3:] if state["observations"] else ["none"]
        )
        result = agent.run(instructions)
        return {
            "goal": state["goal"],
            "actions": state["actions"] + [str(result)],
            "observations": state["observations"] + [str(result)],
            "rewards": state["rewards"] + [1],
        }
    except Exception as e:
        return {
            "goal": state["goal"],
            "actions": state["actions"] + ["error"],
            "observations": state["observations"] + [str(e)],
            "rewards": state["rewards"] + [0],
        }

# LLM generated graph code
from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END

# Tools are assumed to be pre-imported:
# go_to_url, get_page_text, get_navigable_links, is_link_valid,
# get_form_inputs, fill_form, screenshot, DuckDuckGoSearchTool

# ------------------------------------------------------------------
# STATE DEFINITION (pre-defined in system, referenced here only)
class WorkflowState(TypedDict):
    goal: List[str]
    step_name: str
    status: str
    actions: List[str]
    observations: List[str]
    rewards: List[float]
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# NODE IMPLEMENTATIONS
# ------------------------------------------------------------------
def search_web(state: WorkflowState) -> WorkflowState:
    query = state["goal"][-1]
    try:
        links = DuckDuckGoSearchTool(query)
        return {
            **state,
            "step_name": "search_web",
            "status": "running",
            "actions": state["actions"] + [f"search_web: {query}"],
            "observations": state["observations"] + [str(links)],
            "rewards": state["rewards"] + [1.0],
        }
    except Exception as e:
        return {
            **state,
            "step_name": "search_web",
            "status": "failed",
            "actions": state["actions"] + ["search_web: error"],
            "observations": state["observations"] + [str(e)],
            "rewards": state["rewards"] + [0.0],
        }


def navigate_page(state: WorkflowState) -> WorkflowState:
    links = []
    print(f"navigate page state: {state}")
    try:
        links = eval(state["observations"][-1])
    except Exception:
        pass

    target_url = None
    for link in links:
        if is_link_valid(link):
            target_url = link
            break

    if not target_url:
        return {
            **state,
            "step_name": "navigate_page",
            "status": "failed",
            "actions": state["actions"] + ["navigate_page: no_valid_link"],
            "observations": state["observations"] + ["No valid link found"],
            "rewards": state["rewards"] + [0.0],
        }

    try:
        success = go_to_url(target_url)
        return {
            **state,
            "step_name": "navigate_page",
            "status": "running" if success else "failed",
            "actions": state["actions"] + [f"navigate_page: {target_url}"],
            "observations": state["observations"] + [f"Navigation {'succeeded' if success else 'failed'}"],
            "rewards": state["rewards"] + [1.0 if success else 0.0],
        }
    except Exception as e:
        return {
            **state,
            "step_name": "navigate_page",
            "status": "failed",
            "actions": state["actions"] + ["navigate_page: error"],
            "observations": state["observations"] + [str(e)],
            "rewards": state["rewards"] + [0.0],
        }


def extract_content(state: WorkflowState) -> WorkflowState:
    try:
        page_text = get_page_text()
        return {
            **state,
            "step_name": "extract_content",
            "status": "running",
            "actions": state["actions"] + ["extract_content"],
            "observations": state["observations"] + [page_text[:500]],  # limit log size
            "rewards": state["rewards"] + [1.0],
        }
    except Exception as e:
        return {
            **state,
            "step_name": "extract_content",
            "status": "failed",
            "actions": state["actions"] + ["extract_content: error"],
            "observations": state["observations"] + [str(e)],
            "rewards": state["rewards"] + [0.0],
        }


def handle_links(state: WorkflowState) -> WorkflowState:
    try:
        links = get_navigable_links()
        next_url = None
        for l in links:
            if is_link_valid(l):
                next_url = l
                break
        if not next_url:
            raise ValueError("No valid link to follow.")
        go_to_url(next_url)
        return {
            **state,
            "step_name": "handle_links",
            "status": "running",
            "actions": state["actions"] + [f"handle_links: {next_url}"],
            "observations": state["observations"] + [f"Followed link {next_url}"],
            "rewards": state["rewards"] + [1.0],
        }
    except Exception as e:
        return {
            **state,
            "step_name": "handle_links",
            "status": "failed",
            "actions": state["actions"] + ["handle_links: error"],
            "observations": state["observations"] + [str(e)],
            "rewards": state["rewards"] + [0.0],
        }


def handle_form(state: WorkflowState) -> WorkflowState:
    try:
        inputs = get_form_inputs()
        values = ["test"] * len(inputs)
        success = fill_form(values)
        return {
            **state,
            "step_name": "handle_form",
            "status": "running" if success else "failed",
            "actions": state["actions"] + [f"handle_form: {len(inputs)} fields"],
            "observations": state["observations"] + [f"Form {'submitted' if success else 'submission failed'}"],
            "rewards": state["rewards"] + [1.0 if success else 0.0],
        }
    except Exception as e:
        return {
            **state,
            "step_name": "handle_form",
            "status": "failed",
            "actions": state["actions"] + ["handle_form: error"],
            "observations": state["observations"] + [str(e)],
            "rewards": state["rewards"] + [0.0],
        }


def take_screenshot(state: WorkflowState) -> WorkflowState:
    try:
        img_path = screenshot()
        return {
            **state,
            "step_name": "take_screenshot",
            "status": "done",
            "actions": state["actions"] + ["take_screenshot"],
            "observations": state["observations"] + [f"screenshot_saved:{img_path}"],
            "rewards": state["rewards"] + [1.0],
        }
    except Exception as e:
        return {
            **state,
            "step_name": "take_screenshot",
            "status": "failed",
            "actions": state["actions"] + ["take_screenshot: error"],
            "observations": state["observations"] + [str(e)],
            "rewards": state["rewards"] + [0.0],
        }

# ------------------------------------------------------------------
# ROUTING FUNCTIONS
# ------------------------------------------------------------------
def route_after_navigate(state: WorkflowState) -> str:
    return "failure" if state["status"] == "failed" else "continue"


def route_after_extract(state: WorkflowState) -> str:
    last_obs = state["observations"][-1].lower()
    if any(word in last_obs for word in ["form", "input"]):
        return "handle_form"
    elif "http" in last_obs and "link" in last_obs:
        return "handle_links"
    else:
        return "take_screenshot"


# ------------------------------------------------------------------
# WORKFLOW CONSTRUCTION
# ------------------------------------------------------------------
initial_state: WorkflowState = {
    "goal": ["Find the official LangChain documentation website"],
    "step_name": "",
    "status": "init",
    "actions": [],
    "observations": [],
    "rewards": [],
}

workflow = StateGraph(WorkflowState)

# Node registrations
workflow.add_node("search_web", search_web)
workflow.add_node("navigate_page", navigate_page)
workflow.add_node("extract_content", extract_content)
workflow.add_node("handle_links", handle_links)
workflow.add_node("handle_form", handle_form)
workflow.add_node("take_screenshot", take_screenshot)

# Edge definitions
workflow.add_edge(START, "search_web")
workflow.add_edge("search_web", "navigate_page")

workflow.add_conditional_edges(
    "navigate_page",
    route_after_navigate,
    {
        "continue": "extract_content",
        "failure": END,
    },
)

workflow.add_conditional_edges(
    "extract_content",
    route_after_extract,
    {
        "handle_form": "handle_form",
        "handle_links": "handle_links",
        "take_screenshot": "take_screenshot",
    },
)

workflow.add_edge("handle_form", "extract_content")
workflow.add_edge("handle_links", "navigate_page")
workflow.add_edge("take_screenshot", END)

# Compile workflow
app = workflow.compile()

# ------------------------------------------------------------------
# EXECUTION (example)
# ------------------------------------------------------------------
if __name__ == "__main__":
    final_state = app.invoke(initial_state)
    print("----- FINAL STATE -----")
    print(final_state)

initial_state: WorkflowState = {
    "goal": ["Search the web for the latest news on AI advancements and summarize the findings."],
    "actions": [],
    "observations": [],
    "rewards": [],
}

result_state = app.invoke(initial_state)