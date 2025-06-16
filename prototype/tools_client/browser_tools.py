

'''
This module provides a set of tools for interacting with a web browser instance.
It is loaded in a sandbox python environment as part of the crafted workflow.
'''

from typing import List, Any
import requests

from smolagents import (
    Tool,
    DuckDuckGoSearchTool
)

API_BASE_URL = 'http://localhost:5000'

def build_formatted_output(action: str, observation: str, reward: float) -> str:
    action_formatted = action[:256].strip().replace('\n', ' - ')
    observation_formatted = observation[:2048].strip().replace('\n', ' - ')
    return f"""
action: {action_formatted}
observation: {observation_formatted}
reward: {reward}
"""

class SearchTool(Tool):
    name = "search_tool"
    description = "Perform a search using DuckDuckGo and return the results."
    inputs = {"query": {"type": "string", "description": "The search query."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        obs = ''
        action = "search:" + query
        try:
            search_tool = DuckDuckGoSearchTool()  # Instantiate the tool
            obs = search_tool(query)              # Call the instance
            reward = 1.0 if obs else 0.0
        except Exception as e:
            obs = "Search failed for query: " + query + " due to error: " + str(e)
            reward = 0.0
        return build_formatted_output(action, obs, reward)

class GoToUrlTool(Tool):
    name = "go_to_url_tool"
    description = "Navigate to a specified URL."
    inputs = {"url": {"type": "string", "description": "The URL to navigate to."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        action = "go_to_url: " + url
        obs = None
        try:
            route = API_BASE_URL + '/api/browser/navigate'
            response = requests.post(
                route,
                json={'url': url}
            )
            data = response.json()
            obs = f'navigated to ' + url if data.get('status') == 'success' else f'failed to navigate to ' + url
            reward = 1.0 if obs else 0.0
        except Exception as e:
            obs = obs if obs else 'failed to navigate to ' + url + ' due to error: ' + str(e)
            reward = 0.0
        return build_formatted_output(action, obs, reward)

class GetPageTextTool(Tool):
    name = "get_page_text_tool"
    description = "Retrieves the text content from the current web page using the browser instance."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        action = 'get_page_text()'
        try:
            route = API_BASE_URL + '/api/browser/content'
            response = requests.get(route)
            data = response.json()
            obs = data.get('content', 'No text found on the page.')
            reward = 1.0 if obs != 'No text found on the page.' else 0.0
        except Exception as e:
            obs = 'Error getting page text due to error ' + str(e)
            reward = 0.0
        return build_formatted_output(action, obs, reward)

class GetNavigableLinksTool(Tool):
    name = "get_navigable_links_tool"
    description = "Retrieves a list of navigable links from the browser."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        action = 'get_navigable_links()'
        try:
            route = API_BASE_URL + '/api/browser/links'
            response = requests.get(route)
            data = response.json()
            obs = data.get('links', [])
            reward = 1.0 if obs else 0.0
        except Exception as e:
            obs = 'Error getting navigable links due to error ' + str(e)
            reward = 0.0
        return build_formatted_output(action, obs, reward)

class IsLinkValidTool(Tool):
    name = "is_link_valid_tool"
    description = "Check if a link is valid for navigation."
    inputs = {"url": {"type": "string", "description": "The URL to check."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        action = "IsLinkValidTool: " + url
        try:
            route = API_BASE_URL + '/api/browser/link_valid'
            response = requests.post(
                route,
                json={'url': url}
            )
            data = response.json()
            obs = data.get('valid', False)
            reward = 1.0 if obs else 0.0
        except Exception as e:
            obs = 'Error checking link validity due to error ' + str(e)
            reward = 0.0
        return build_formatted_output(action, str(obs), reward)

class ScreenshotTool(Tool):
    name = "screenshot_tool"
    description = "Take a screenshot of the current page."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        action = 'screenshot()'
        try:
            route = API_BASE_URL + '/api/browser/screenshot'
            response = requests.get(route)
            data = response.json()
            obs = data.get('filename', '')
            reward = 1.0 if obs else 0.0
        except Exception as e:
            obs = 'Error taking screenshot due to error ' + str(e)
            reward = 0.0
        return build_formatted_output(action, obs, reward)

search_tool = SearchTool()
go_to_url_tool = GoToUrlTool()
get_page_text_tool = GetPageTextTool()
get_navigable_links_tool = GetNavigableLinksTool()
is_link_valid_tool = IsLinkValidTool()
screenshot_tool = ScreenshotTool()

tools = [
    search_tool,
    go_to_url_tool,
    get_page_text_tool,
    get_navigable_links_tool,
    is_link_valid_tool,
    screenshot_tool
]

tools_name = [tool.name for tool in tools]