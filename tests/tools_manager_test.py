
import asyncio

from config import Config
from sources.core.tools_manager import ToolManager


def test_tool_manager():
     # Example usage
    config = Config()
    tool_manager = ToolManager(config)
    mcps = asyncio.run(tool_manager.discover_mcp_servers())
    for mcp in mcps:
        print(f"MCP instance: {mcp}")  # Print the MCP instance using the __str__ method
        client_code = tool_manager.get_client_code(mcp)
        print(client_code)  # This would print the generated client code for the MCP server
        client_prompt = tool_manager.get_client_prompt(mcp)
        print(client_prompt)  # This would print the generated client prompt for the MCP server



if __name__ == "__main__":
   test_tool_manager()