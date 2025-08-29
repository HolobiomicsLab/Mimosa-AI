import readline
import asyncio
import json
from fastmcp import Client
from config import Config
from sources.core.tools_manager import ToolManager, MCP, Tool

class HumanMode:
    def __init__(self, config):
        self.config = config or Config()
    
    def validate_choice(self, choice: str, attribution_map: dict) -> bool:
        try:
            _ = int(choice)
        except ValueError as e:
            print(f"Invalid value: {str(e)}")
            return False
        if int(choice) not in list(range(0, len(attribution_map))):
            print("Choice not in range or invalid.")
            return False
        return True

    def mcp_selection(self, mcps: list[MCP]) -> MCP:
        attribution_map = {
            str(i): mcp for (i, mcp) in enumerate(mcps)
        }
        print("Available MCP(s):\n", attribution_map)
        for i, mcp in attribution_map.items():
            print(f"[{i}] {mcp}")
        choice = input(f"Choose MCP [0-{len(attribution_map)}]:")
        if not self.validate_choice(choice, attribution_map):
            return None
        return attribution_map[choice]
    
    def tool_selection(self, tools: list) -> Tool:
        attribution_map = {
            str(i): mcp for (i, mcp) in enumerate(tools)
        }
        for i, tool in attribution_map.items():
            print(f"[{i}] {tool.name}: {tool.description}")
        choice = input(f"Choose Tool [0-{len(attribution_map)}]:")
        if not self.validate_choice(choice, attribution_map):
            return None
        return attribution_map[choice]
    
    async def discover_tools(self, mcp):
        addr = mcp.address
        port = mcp.port
        tools = None
        try:
            async with Client(f"http://{addr}:{port}/mcp") as client:
                tools = await client.list_tools()
        except Exception as e:
            print(f"Error in Tools discovery: {str(e)}")
            return None
        return tools
    
    def get_tool_arguments(self):
        arg_str = input("Enter tool schema:")
        try:
            arg_json = json.loads(arg_str)
        except Exception as e:
            raise e
        return arg_json

    async def execute_tool(self, mcp, tool_name):
        addr = mcp.address
        port = mcp.port
        try:
            async with Client(f"http://{addr}:{port}/mcp") as client:
                result = await client.call_tool(
                    tool_name,
                    self.get_tool_arguments(),
                )
                try:
                    dict_result = json.loads(result[0].text) if result else {}
                    print(f"📋 Output: {dict_result}")
                except Exception as e:
                    print("Unknown error: ", str(e))
        except Exception as e:
            raise e

    async def shellLoop(self) -> None:
        """
        Manual usage mode for Mimosa AI.
        """
        tool_manager = ToolManager(config=self.config)
        mcps = await tool_manager.discover_mcp_servers()

        print("Entering manual model (ctrl+c to exit).")
        while True:
            mcp_choice = self.mcp_selection(mcps)
            print(f"Selected MCP: ", mcp_choice)
            tools = await self.discover_tools(mcp_choice)
            if tools is None:
                continue
            tool_choice = self.tool_selection(tools)
            print("Selected tool:", tool_choice.name)
            try:
                await self.execute_tool(mcp_choice, tool_choice.name)
            except Exception as e:
                raise e


