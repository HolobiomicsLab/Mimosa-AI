

from typing import List
import requests

from smolagents import CodeAgent, tool
from smolagents import (
    DuckDuckGoSearchTool
)

from tools_client.tools_state_bridge import (
    print_action,
    print_observation,
    print_reward
)

API_BASE_URL = "http://localhost:5000"

@tool
def search(query: str) -> List[str]:
    """
    Perform a search using DuckDuckGo and return the results.

    Args:
        query (str): The search query.

    Returns:
        List[str]: A list of URLs from the search results.
    """
    obs = ""
    action = f"search({query})"
    print_action([action])
    try:
        obs = DuckDuckGoSearchTool(query)
        reward = 1.0 if obs else 0.0
    except Exception as e:
        obs = f"Error during search: {str(e)}"
        reward = 0.0
    print_observation(obs)
    print_reward(reward)
    return obs

@tool
def go_to_url(url: str) -> bool:
    """Navigate to a specified URL.
    Args:
        url (str): The URL to navigate to.
    Returns:
        bool: True if navigation was successful, False otherwise.
    """
    action = f"go_to_url({url})"
    print_action([action])
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/browser/navigate",
            json={"url": url}
        )
        data = response.json()
        obs = f"navigated to {url}" if data.get("status") == "success" else f"failed to navigate to {url}"
        reward = 1.0 if obs else 0.0
    except Exception as e:
        obs = f"Error navigating to URL: {str(e)}"
        reward = 0.0
    print_observation(obs)
    print_reward(reward)
    return obs

@tool
def get_page_text() -> str:
    """
    Retrieves the text content from the current web page using the browser instance.

    Returns:
        str: The text content of the current page, or "No text found on the page." if no text is available.
    """
    action = "get_page_text()"
    print_action([action])
    try:
        response = requests.get(f"{API_BASE_URL}/api/browser/content")
        data = response.json()
        obs = data.get("content", "No text found on the page.")
        reward = 1.0 if obs != "No text found on the page." else 0.0
    except Exception as e:
        obs = f"Error getting page text: {str(e)}"
        reward = 0.0
    print_observation(obs)
    print_reward(reward)
    return obs
    
@tool
def get_navigable_links() -> List[str]:
    """
    Retrieves a list of navigable links from the browser.

    Returns:
        List[str]: A list of URLs or link texts that can be navigated to from the current browser context.
    """
    action = "get_navigable_links()"
    print_action([action])
    try:
        response = requests.get(f"{API_BASE_URL}/api/browser/links")
        data = response.json()
        obs = data.get("links", [])
        reward = 1.0 if obs else 0.0
    except Exception as e:
        obs = f"Error getting navigable links: {str(e)}"
        reward = 0.0
    print_observation(obs)
    print_reward(reward)
    return obs
    
@tool
def is_link_valid(url: str) -> bool:
    """
    Check if a link is valid for navigation.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if the link is valid, False otherwise.
    """
    action = f"is_link_valid({url})"
    print_action([action])
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/browser/link_valid",
            json={"url": url}
        )
        data = response.json()
        obs = data.get("valid", False)
        reward = 1.0 if obs else 0.0
    except Exception as e:
        obs = f"Error checking link validity: {str(e)}"
        reward = 0.0
    print_observation(obs)
    print_reward(reward)
    return obs
    
@tool
def get_form_inputs() -> List[str]:
    """
    Get all input fields from the current page.

    Returns:
        List[str]: A list of input fields in the format [name](value).
    """
    action = "get_form_inputs()"
    print_action([action])
    try:
        response = requests.get(f"{API_BASE_URL}/api/browser/form_inputs")
        data = response.json()
        obs = data.get("inputs", [])
        reward = 1.0 if obs else 0.0
    except Exception as e:
        obs = f"Error getting form inputs: {str(e)}"
        reward = 0.0
    print_observation(obs)
    print_reward(reward)
    return obs
    
@tool
def fill_form(values: List[str]) -> bool:
    """
    Fill the form with provided values.

    Args:
        values (List[str]): A list of input fields in the format [name](value).

    Returns:
        bool: True if the form was filled successfully, False otherwise.
    """
    action = f"fill_form({values})"
    print_action([action])
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/browser/fill_form",
            json={"inputs": values}
        )
        data = response.json()
        obs = data.get("status") == "success"
        reward = 1.0 if obs else 0.0
    except Exception as e:
        obs = f"Error filling form: {str(e)}"
        reward = 0.0
    print_observation(obs)
    print_reward(reward)
    return obs

@tool
def screenshot() -> str:
    """
    Take a screenshot of the current page.

    Returns:
        str: The path to the saved screenshot.
    """
    action = "screenshot()"
    print_action([action])
    try:
        response = requests.get(f"{API_BASE_URL}/api/browser/screenshot")
        data = response.json()
        obs = data.get("filename", "")
        reward = 1.0 if obs else 0.0
    except Exception as e:
        obs = f"Error taking screenshot: {str(e)}"
        reward = 0.0
    print_observation(obs)
    print_reward(reward)
    return obs

@tool
def take_note(note: str) -> None:
    """
    Take a notes of the current page.
    Args:
        note (str): The note you want to take.

    Returns:
        str: Note you want to take.
    """
    reward = 1.0
    action = f"take_note({note})"
    obs = note
    print_action([action])
    print_observation(obs)
    print_reward(reward)

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

tools_prompt = f"""
You are given these tools:
- go_to_url(url: str) -> bool: Navigate to URL
- get_page_text() -> str: Get page text content  
- get_navigable_links() -> List[str]: Get clickable links
- is_link_valid(url: str) -> bool: Check if link is valid
- get_form_inputs() -> List[str]: Get form input fields
- fill_form(values: List[str]) -> bool: Fill form with values
- screenshot() -> str: Take page screenshot
- DuckDuckGoSearchTool(query: str) -> List[str]: Search DuckDuckGo
"""


engine = HfApiModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token="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",
    max_tokens=5000,
)

agent = CodeAgent(
    tools=tools,
    model=engine,
    max_steps=10,
)
instruct = "You are a SmolAgent that can perform web searches, navigate pages, and fill forms. Use the tools provided to achieve your goal."
output = agent.run(f"Your goal is Search the web for the latest news on AI advancements and summarize the findings..")
print(output)
    