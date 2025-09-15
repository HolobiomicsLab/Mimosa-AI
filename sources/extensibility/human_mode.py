import readline
import asyncio
import json
from fastmcp import Client
from config import Config
from sources.core.tools_manager import ToolManager, MCP, Tool

class HumanMode:
    def __init__(self, config):
        self.config = config or Config()
    
    def _print_header(self, title: str, char: str = "="):
        """Print a formatted header."""
        width = 60
        print(f"\n{char * width}")
        print(f"{title:^{width}}")
        print(f"{char * width}")
    
    def _print_section(self, title: str):
        """Print a section divider."""
        print(f"\n{'─' * 40}")
        print(f"🔹 {title}")
        print(f"{'─' * 40}")
    
    def _print_error(self, message: str):
        """Print formatted error message."""
        print(f"❌ {message}")
    
    def _print_success(self, message: str):
        """Print formatted success message."""
        print(f"✅ {message}")
    
    def _print_info(self, message: str):
        """Print formatted info message."""
        print(f"ℹ️  {message}")
    
    def validate_choice(self, choice: str, attribution_map: dict) -> bool:
        try:
            _ = int(choice)
        except ValueError as e:
            self._print_error(f"Invalid input: Please enter a number")
            return False
        if int(choice) not in list(range(0, len(attribution_map))):
            self._print_error("Choice not in range. Please select a valid option.")
            return False
        return True

    def mcp_selection(self, mcps: list[MCP]) -> MCP:
        self._print_section("MCP Server Selection")
        
        attribution_map = {
            str(i): mcp for (i, mcp) in enumerate(mcps)
        }
        
        print("📡 Available MCP Servers:")
        print()
        for i, mcp in attribution_map.items():
            print(f"  [{i}] 🖥️  {mcp}")
        
        print()
        while True:
            choice = input(f"🎯 Choose MCP server [0-{len(attribution_map)-1}]: ").strip()
            if self.validate_choice(choice, attribution_map):
                selected_mcp = attribution_map[choice]
                self._print_success(f"Selected: {selected_mcp}")
                return selected_mcp
            print("   Please try again.\n")
    
    def tool_selection(self, tools: list) -> Tool:
        self._print_section("Tool Selection")
        
        attribution_map = {
            str(i): tool for (i, tool) in enumerate(tools)
        }
        
        print("🔧 Available Tools:")
        print()
        for i, tool in attribution_map.items():
            if not tool.description:
                description = "No description available."
            else:
                description = ''.join(tool.description.split('\n')[:1])
            # Truncate long descriptions
            if len(description) > 70:
                description = description[:67] + "..."
            print(f"  [{i}] ⚡ {tool.name}")
            print(f"      💭 {description}")
            print()
        
        while True:
            choice = input(f"🎯 Choose tool [0-{len(attribution_map)-1}]: ").strip()
            if self.validate_choice(choice, attribution_map):
                selected_tool = attribution_map[choice]
                self._print_success(f"Selected: {selected_tool.name}")
                return selected_tool
            print("   Please try again.\n")
    
    async def discover_tools(self, mcp):
        addr = mcp.address
        port = mcp.port
        tools = None
        
        print(f"🔍 Discovering tools from {mcp}...")
        try:
            async with Client(f"http://{addr}:{port}/mcp") as client:
                tools = await client.list_tools()
            self._print_success(f"Found {len(tools)} tool(s)")
        except Exception as e:
            self._print_error(f"Tools discovery failed: {str(e)}")
            return None
        return tools

    def _convert_value(self, value: str, param_type: str):
        """Convert string input to appropriate type based on schema."""
        if not value.strip():
            return None
            
        try:
            if param_type == "integer":
                return int(value)
            elif param_type == "number":
                return float(value)
            elif param_type == "boolean":
                return value.lower() in ['true', '1', 'yes', 'y']
            elif param_type == "array":
                return [item.strip() for item in value.split(',') if item.strip()]
            else:  # string or unknown
                return value
        except ValueError:
            print(f"⚠️  Warning: Could not convert '{value}' to {param_type}, using as string")
            return value

    def _get_parameter_input(self, param_name: str, param_schema: dict) -> any:
        """Get input for a single parameter based on its schema."""
        param_type = param_schema.get('type', 'string')
        description = param_schema.get('description', '')
        required = param_schema.get('required', False)
        default = param_schema.get('default')
        
        # Build prompt with better formatting
        type_indicator = f"[{param_type}]" if param_type != 'string' else ""
        requirement_indicator = "🔴 Required" if required else "🔵 Optional"
        
        print(f"  📝 {param_name} {type_indicator} - {requirement_indicator}")
        if description:
            print(f"     💭 {description}")
        
        # Build input prompt
        prompt_parts = ["     ➤"]
        if default is not None:
            prompt_parts.append(f"(default: {default})")
        
        prompt = " ".join(prompt_parts) + " "
        
        # Handle special cases
        if param_type == "boolean":
            prompt += "[true/false]: "
        elif param_type == "array":
            prompt += "[comma-separated]: "
        else:
            prompt += ": "
            
        value = input(prompt).strip()
        
        # Handle empty input
        if not value:
            if default is not None:
                print(f"     ✓ Using default: {default}")
                return default
            elif not required:
                print(f"     ⏭️  Skipped")
                return None
            else:
                self._print_error(f"{param_name} is required")
                return self._get_parameter_input(param_name, param_schema)
        
        converted_value = self._convert_value(value, param_type)
        print(f"     ✓ Set to: {converted_value}")
        return converted_value

    def get_tool_arguments(self, tool: Tool) -> dict:
        """Interactive tool argument builder based on tool schema."""
        self._print_section("Parameter Configuration")
        
        if not hasattr(tool, 'inputSchema') or not tool.inputSchema:
            self._print_info("This tool requires no arguments")
            return {}
        
        schema = tool.inputSchema
        properties = schema.get('properties', {})
        required_fields = schema.get('required', [])
        
        if not properties:
            self._print_info("Tool requires no arguments")
            return {}
        
        print(f"🔧 Configuring parameters for: {tool.name}")
        if tool.description:
            print(f"📋 {tool.description}")
        print()
        
        arguments = {}
        
        # Process required parameters first
        if required_fields:
            print("🔴 Required Parameters:")
            for param_name in required_fields:
                if param_name in properties:
                    param_schema = properties[param_name].copy()
                    param_schema['required'] = True
                    arguments[param_name] = self._get_parameter_input(param_name, param_schema)
            print()
        
        # Process optional parameters
        optional_params = [name for name in properties if name not in required_fields]
        if optional_params:
            print("🔵 Optional Parameters (press Enter to skip):")
            for param_name in optional_params:
                param_schema = properties[param_name].copy()
                param_schema['required'] = False
                value = self._get_parameter_input(param_name, param_schema)
                if value is not None:
                    arguments[param_name] = value
            print()
        
        # Clean up None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        
        print("📋 Final Configuration:")
        print(json.dumps(arguments, indent=2))
        print()
        
        return arguments

    def _format_output(self, data) -> str:
        """Format output with proper newline handling and readability."""
        if isinstance(data, dict):
            lines = ["📋 Tool Output:"]
            lines.append("┌" + "─" * 58 + "┐")
            for key, value in data.items():
                if isinstance(value, str) and ('\n' in value or len(value) > 80):
                    lines.append(f"│ {key}:")
                    for line in value.split('\n'):
                        lines.append(f"│   {line}")
                    print(f"Len of {key}: {len(value)}")
                else:
                    formatted_value = json.dumps(value) if not isinstance(value, str) else value
                    lines.append(f"│ {key}: {formatted_value}")
            lines.append("└" + "─" * 58 + "┘")
            return '\n'.join(lines)
        elif isinstance(data, str):
            lines = ["📋 Tool Output:"]
            lines.append("┌" + "─" * 58 + "┐")
            for line in data.split('\n'):
                lines.append(f"│ {line}")
            lines.append("└" + "─" * 58 + "┘")
            return '\n'.join(lines)
        else:
            return f"📋 Tool Output:\n{json.dumps(data, indent=2)}"

    async def execute_tool(self, mcp, tool_name, arguments):
        addr = mcp.address
        port = mcp.port
        
        print(f"🚀 Executing tool: {tool_name}...")
        print("⏳ Please wait...")
        
        try:
            async with Client(f"http://{addr}:{port}/mcp") as client:
                result = await client.call_tool(tool_name, arguments)
                if result:
                    if hasattr(result, 'content') and result.content:
                        if hasattr(result.content[0], 'text'):
                            text_content = result.content[0].text
                        else:
                            text_content = str(result.content[0])
                        try:
                            dict_result = json.loads(text_content)
                            print(self._format_output(dict_result))
                        except json.JSONDecodeError:
                            print(self._format_output(text_content))
                    else:
                        print("Error in parsing tool output, raw output:", (str(result)))
                    self._print_success("Tool execution completed!")
                else:
                    self._print_info("No result returned from tool")
        except Exception as e:
            raise e

    async def shellLoop(self) -> None:
        """Manual usage mode for Mimosa AI."""
        self._print_header("🤖 MIMOSA AI - Interactive Mode", "═")
        print("Welcome to Mimosa AI Interactive Mode!")
        print("Use Ctrl+C to exit at any time.")
        
        tool_manager = ToolManager(config=self.config)
        print("\n🔍 Discovering MCP servers...")
        mcps = await tool_manager.discover_mcp_servers()
        
        if not mcps:
            self._print_error("No MCP servers found. Please check your configuration.")
            return
        
        self._print_success(f"Found {len(mcps)} MCP server(s)")
        
        while True:
            try:
                mcp_choice = self.mcp_selection(mcps)
                if not mcp_choice:
                    continue
                
                tools = await self.discover_tools(mcp_choice)
                if tools is None:
                    print("\n🔄 Let's try again...\n")
                    continue
                
                tool_choice = self.tool_selection(tools)
                if not tool_choice:
                    continue
                
                arguments = self.get_tool_arguments(tool_choice)
                if arguments is None:
                    continue
                
                await self.execute_tool(mcp_choice, tool_choice.name, arguments)
                
                print("\n" + "="*60)
                print("🔄 Ready for next operation...")
                
            except KeyboardInterrupt:
                print("\n")
                self._print_header("👋 Goodbye!", "═")
                print("Thank you for using Mimosa AI!")
                break
            except Exception as e:
                self._print_error(f"An error occurred: {str(e)}")
                print("🔄 Continuing...\n")
                continue