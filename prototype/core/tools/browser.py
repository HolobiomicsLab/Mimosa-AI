'''
This module provides a set of tools for interacting with a web browser instance.
It is loaded in a sandbox python environment as part of the crafted workflow.
'''

from typing import List, Any
import requests
import asyncio

from fastmcp import Client
from smolagents import (
    Tool,
    DuckDuckGoSearchTool
)
import json

API_BROWSER_TOOLS_URL = 'http://localhost:5002'

def build_formatted_output(action: str, observation: str, reward: float) -> str:
    output = {
        "action": action[:256].strip().replace('\n', ' - '),
        "observation": observation[:4096],
        "reward": reward
    }
    return f"\n```json\n{json.dumps(output, indent=2)}\n```\n"

async def _async_tool_call(tool_name: str, params: dict) -> dict:
    async with Client(f"{API_BROWSER_TOOLS_URL}/mcp") as client:
        tools = await client.list_tools()
        assert tool_name in tools, "Fatal Error: " + tool_name + " not in tools list for mcp at " + API_BROWSER_TOOLS_URL
        buffer = await client.call_tool(tool_name, params)
        return json.loads(buffer[0].text)

class SearchTool(Tool):
    name = "search_tool"
    description = "Perform a search using DuckDuckGo and return the results."
    inputs = {"query": {"type": "string", "description": "The search query."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        obs = ''
        action = f"search_tool(query='{query}')"
        try:
            result = asyncio.run(_async_tool_call("search", {"query": query}))
            obs = result.get('result', 'No results found')
            reward = 1.0 if obs else 0.0
        except Exception as e:
            print(str(e))
            obs = "Search failed for query: " + query + " due to error: " + str(e)
            reward = 0.0
        return build_formatted_output(action, obs, reward)

class GoToUrlTool(Tool):
    name = "go_to_url_tool"
    description = "Navigate to a specified URL and return the page content as Markdown."
    inputs = {"url": {"type": "string", "description": "The URL to navigate to."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        action = f"go_to_url_tool(url='{url}')"
        obs = ''
        reward = 0.0
        try:
            result = asyncio.run(_async_tool_call("navigate", {"url": url}))
        except Exception as e:
            print(str(e))
            obs = f'failed to navigate to {url} due to error: {str(e)}'
            return build_formatted_output(action, obs, reward)
        
        if not result or not 'success' in result.get('status', {}):
            obs = f'Error navigating to {url}: ' + result.get('message', 'Unknown error')
            return build_formatted_output(action, obs, reward)
        
        title = result.get('title', 'No title found')
        content = result.get('content', 'No content found')
        obs = f'''Tile: {title}
            {content}
        '''
        reward = 1.0
        return build_formatted_output(action, obs, reward)

class GetNavigableLinksTool(Tool):
    name = "get_navigable_links_tool"
    description = "Retrieves a list of navigable links on the current page."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        action = 'get_navigable_links_tool()'
        obs = ''
        reward = 0.0
        try:
            result = asyncio.run(_async_tool_call("get_links", {}))
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

    def forward(self) -> str:
        action = 'screenshot()'
        obs = ''
        reward = 0.0
        try:
            result = asyncio.run(_async_tool_call("screenshot", {}))
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