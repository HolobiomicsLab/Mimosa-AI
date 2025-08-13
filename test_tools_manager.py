#!/usr/bin/env python3

"""
Test script to verify tools_manager can detect streamable-http MCP servers
"""

import asyncio
import sys
from pathlib import Path

# Add the sources directory to path
sources_dir = Path(__file__).parent / "sources"
sys.path.insert(0, str(sources_dir))

from core.tools_manager import ToolManager, normalize_mcp_endpoint


class MockConfig:
    """Mock config for testing"""
    discovery_addresses = ["127.0.0.1"]


def test_normalize_endpoint():
    """Test the normalize_mcp_endpoint function"""
    print("Testing normalize_mcp_endpoint function...")
    
    # Test streamable-http normalization - adds /mcp when missing
    url, transport, extras = normalize_mcp_endpoint("http://127.0.0.1:5002", "streamable-http")
    print(f"✅ streamable-http: {url} (transport: {transport})")
    assert url == "http://127.0.0.1:5002/mcp", f"Expected /mcp endpoint, got {url}"
    
    # Test http alias
    url, transport, extras = normalize_mcp_endpoint("http://127.0.0.1:5002", "http")
    print(f"✅ http alias: {url} (transport: {transport})")
    assert url == "http://127.0.0.1:5002/mcp", f"Expected /mcp endpoint, got {url}"
    
    # Test preserving existing /mcp path (no trailing slash)
    url, transport, extras = normalize_mcp_endpoint("http://127.0.0.1:5002/mcp", "streamable-http")
    print(f"✅ preserve /mcp: {url} (transport: {transport})")
    assert url == "http://127.0.0.1:5002/mcp", f"Expected preserved /mcp endpoint, got {url}"
    
    # Test preserving existing /mcp/ path (with trailing slash)
    url, transport, extras = normalize_mcp_endpoint("http://127.0.0.1:5002/mcp/", "streamable-http")
    print(f"✅ preserve /mcp/: {url} (transport: {transport})")
    assert url == "http://127.0.0.1:5002/mcp/", f"Expected preserved /mcp/ endpoint, got {url}"
    
    # Test SSE normalization - should now be supported
    url, transport, extras = normalize_mcp_endpoint("http://127.0.0.1:5002/sse#server", "sse")
    print(f"✅ sse: {url} (transport: {transport})")
    assert transport == "sse", f"Expected sse transport, got {transport}"
    assert url == "http://127.0.0.1:5002/sse", f"Expected /sse endpoint preserved, got {url}"
    
    print("✅ All normalize_mcp_endpoint tests passed!\n")


async def test_toolhive_discovery():
    """Test ToolHive MCP server discovery"""
    print("Testing ToolHive MCP server discovery...")
    
    config = MockConfig()
    tool_manager = ToolManager(config)
    
    try:
        # Test ToolHive discovery (the actual method Mimosa-AI uses)
        mcps = await tool_manager.discover_toolhive_servers()
        
        if mcps:
            print(f"✅ Found {len(mcps)} MCP server(s) via ToolHive:")
            for mcp in mcps:
                print(f"   - {mcp.name} ({mcp.toolhive_name})")
                print(f"     Address: {mcp.address}:{mcp.port}")
                print(f"     Transport: {mcp.transport}")
                print(f"     Client URL: {mcp.client_url}")
                print(f"     Tools: {mcp.tools}")
        else:
            print("❌ No MCP servers found via ToolHive")
            return False
            
    except Exception as e:
        print(f"❌ ToolHive discovery failed: {e}")
        return False
    
    print("✅ ToolHive discovery test completed!\n")
    return True


def main():
    """Run all tests"""
    print("Testing tools_manager.py with ToolHive integration\n")
    
    # Test endpoint normalization
    test_normalize_endpoint()
    
    # Test ToolHive discovery
    print("Testing ToolHive MCP server discovery (requires running ToolHive servers)")
    print("Make sure you have started MCP servers with: thv run toolomics-csv --detach (etc.)")
    print("Press Enter to continue with ToolHive discovery test, or Ctrl+C to skip...")
    try:
        input()
        success = asyncio.run(test_toolhive_discovery())
        if success:
            print("🎉 All tests passed!")
        else:
            print("⚠️  Some tests failed")
    except KeyboardInterrupt:
        print("\n⏭️  Skipping ToolHive discovery test")
        print("✅ Endpoint normalization tests passed!")


if __name__ == "__main__":
    main()