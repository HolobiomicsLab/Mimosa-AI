'''
This module provides a set of tools for interacting with a web browser instance.
It is loaded in a sandbox python environment as part of the crafted workflow.
'''

from typing import List, Any
import requests
import asyncio

from fastmcp import Client
from smolagents import (
    Tool
)
import json

API_BASE_URL = 'http://localhost:5002'


def build_formatted_output(action: str, observation: str, reward: float) -> str:
    action_formatted = action[:256].strip().replace('\n', ' - ')
    observation_formatted = observation[:1024].strip().replace('\n', ' - ')
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

    async def _async_search(self, query: str) -> dict:
        async with Client(f"{API_BASE_URL}/mcp") as client:
            buffer = await client.call_tool("search", {"query": query})
            return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

    def forward(self, query: str) -> str:
        obs = ''
        action = "search:" + query
        try:
            result = asyncio.run(self._async_search(query))
            obs = result.get('result', 'No results found')
            reward = 1.0 if obs else 0.0
        except Exception as e:
            print(str(e))
            obs = "Search failed for query: " + query + " due to error: " + str(e)
            reward = 0.0
        return build_formatted_output(action, obs, reward)

class GoToUrlTool(Tool):
    name = "go_to_url_tool"
    description = "Navigate to a specified URL and return the page content."
    inputs = {"url": {"type": "string", "description": "The URL to navigate to."}}
    output_type = "string"

    async def _async_navigate(self, url: str) -> dict:
        async with Client(f"{API_BASE_URL}/mcp") as client:
            buffer = await client.call_tool("navigate", {"url": url})
            return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

    def forward(self, url: str) -> str:
        action = "go_to_url_tool(" + url + ")"
        obs = ''
        reward = 0.0
        try:
            result = asyncio.run(self._async_navigate(url))
        except Exception as e:
            print(str(e))
            obs = f'failed to navigate to {url} due to error: {str(e)}'
            return build_formatted_output(action, obs, reward)
        
        if not result or 'error' in result.get('status', {}):
            return build_formatted_output(action, obs, reward)
        
        title = result.get('title', 'No title found')
        content = result.get('content', 'No content found')
        obs = f'''Tile: {title}
            Start of page:
            {content}
            End of page.
        '''
        reward = 1.0
        return build_formatted_output(action, obs, reward)

class GetNavigableLinksTool(Tool):
    name = "get_navigable_links_tool"
    description = "Retrieves a list of navigable links from the browser."
    inputs = {}
    output_type = "string"

    async def _async_get_links(self) -> dict:
        async with Client(f"{API_BASE_URL}/mcp") as client:
            buffer = await client.call_tool("get_links")
            return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

    def forward(self) -> str:
        action = 'get_navigable_links_tool()'
        obs = ''
        reward = 0.0
        try:
            result = asyncio.run(self._async_get_links())
        except Exception as e:
            print(str(e))
            obs = 'Error getting navigable links due to error ' + str(e)
            return build_formatted_output(action, obs, reward)
        
        if 'error' in result.get('status', {}):
            obs = 'Error getting navigable links: ' + result['status']['error']
            return build_formatted_output(action, obs, reward)
        
        obs = result.get('links', [])
        reward = 1.0 if obs else 0.0
        return build_formatted_output(action, obs, reward)

class ScreenshotTool(Tool):
    name = "screenshot_tool"
    description = "Take a screenshot of the current page."
    inputs = {}
    output_type = "string"

    async def _async_screenshot(self) -> dict:
        async with Client(f"{API_BASE_URL}/mcp") as client:
            buffer = await client.call_tool("screenshot")
            return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

    def forward(self) -> str:
        action = 'screenshot()'
        obs = ''
        reward = 0.0
        try:
            result = asyncio.run(self._async_screenshot())
        except Exception as e:
            print(str(e))
            obs = 'Error taking screenshot due to error ' + str(e)
            reward = 0.0
            return build_formatted_output(action, obs, reward)
        
        if 'error' in result.get('status', {}):
            return build_formatted_output(action, obs, reward)
        
        filename = result.get('filename', 'No screenshot available')
        obs = f'Screenshot saved as {filename}'
        reward = 1.0
        return build_formatted_output(action, obs, reward)

search_tool = SearchTool()
go_to_url_tool = GoToUrlTool()
get_navigable_links_tool = GetNavigableLinksTool()
screenshot_tool = ScreenshotTool()

tools = [
    search_tool,
    go_to_url_tool,
    get_navigable_links_tool,
    screenshot_tool
]

tools_name = [tool.name for tool in tools]

if __name__ == "__main__":
    print("Available tools:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")
    print