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
    
    # Test streamable-http normalization
    url, transport, extras = normalize_mcp_endpoint("http://127.0.0.1:5002", "streamable-http")
    print(f"✅ streamable-http: {url} (transport: {transport})")
    assert url == "http://127.0.0.1:5002/mcp/", f"Expected /mcp/ endpoint, got {url}"
    
    # Test http alias
    url, transport, extras = normalize_mcp_endpoint("http://127.0.0.1:5002", "http")
    print(f"✅ http alias: {url} (transport: {transport})")
    assert url == "http://127.0.0.1:5002/mcp/", f"Expected /mcp/ endpoint, got {url}"
    
    # Test SSE normalization
    url, transport, extras = normalize_mcp_endpoint("http://127.0.0.1:5002", "sse")
    print(f"✅ sse: {url} (transport: {transport})")
    assert url == "http://127.0.0.1:5002/sse", f"Expected /sse endpoint, got {url}"
    
    print("✅ All normalize_mcp_endpoint tests passed!\n")


async def test_mcp_discovery():
    """Test MCP server discovery"""
    print("Testing MCP server discovery...")
    
    config = MockConfig()
    tool_manager = ToolManager(config)
    
    try:
        # Test discovery on a limited port range (where we expect Toolomics servers)
        mcps = await tool_manager.discover_mcp_at_address(
            address="127.0.0.1",
            port_min=5002,  # Just test CSV server port
            port_max=5002,
            timeout=5.0
        )
        
        if mcps:
            print(f"✅ Found {len(mcps)} MCP server(s):")
            for mcp in mcps:
                print(f"   - {mcp.name} on {mcp.address}:{mcp.port}")
                print(f"     Transport: {mcp.transport}")
                print(f"     Client URL: {mcp.client_url}")
                print(f"     Tools: {mcp.tools}")
        else:
            print("❌ No MCP servers found")
            return False
            
    except Exception as e:
        print(f"❌ Discovery failed: {e}")
        return False
    
    print("✅ MCP discovery test completed!\n")
    return True


def main():
    """Run all tests"""
    print("Testing tools_manager.py with streamable-http transport\n")
    
    # Test endpoint normalization
    test_normalize_endpoint()
    
    # Test MCP discovery
    print("Note: This test requires a Toolomics CSV server running on port 5002")
    print("Run: uv run python3 mcp_host/csv/server.py 5002")
    print("Press Enter to continue with discovery test, or Ctrl+C to skip...")
    try:
        input()
        success = asyncio.run(test_mcp_discovery())
        if success:
            print("🎉 All tests passed!")
        else:
            print("⚠️  Some tests failed")
    except KeyboardInterrupt:
        print("\n⏭️  Skipping discovery test")
        print("✅ Endpoint normalization tests passed!")


if __name__ == "__main__":
    main()