#!/bin/bash

# Tool Discovery Script for Mimosa-AI
# Lists all available MCP tools using ToolHive commands
# Usage: ./list_tools.sh [--format table|json|detailed] [--show-code]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default options
FORMAT="detailed"
SHOW_CODE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --compact)
            FORMAT="compact"
            shift
            ;;
        --show-code)
            SHOW_CODE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--format table|json|detailed|compact] [--compact] [--show-code]"
            echo ""
            echo "Options:"
            echo "  --format FORMAT    Output format: table, json, detailed, or compact (default: detailed)"
            echo "  --compact          Shortcut for --format compact (most concise output)"
            echo "  --show-code        Also show generated client code examples"
            echo "  --help, -h         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if ToolHive is available
check_toolhive() {
    if ! command -v thv &> /dev/null; then
        echo -e "${RED}❌ ToolHive (thv) is not installed or not in PATH${NC}"
        echo -e "${YELLOW}Install with: curl -sSL https://get.toolhive.dev | sh${NC}"
        exit 1
    fi
}

# Get list of running servers
get_running_servers() {
    local servers_json
    servers_json=$(thv list --format json 2>/dev/null || echo "[]")
    echo "$servers_json"
}

# Get tools for a specific server
get_server_tools() {
    local server_url="$1"
    local tools_json
    tools_json=$(thv mcp list tools --server "$server_url" --format json 2>/dev/null || echo '{"tools": []}')
    echo "$tools_json"
}

# Extract server info from JSON
extract_server_info() {
    local servers_json="$1"
    echo "$servers_json" | jq -r '.[] | select(.status == "running") | "\(.name)|\(.url)"' 2>/dev/null
}

# Format output as table
format_table() {
    local servers_json="$1"
    
    echo -e "${CYAN}================================================================================${NC}"
    echo -e "${CYAN}                              AVAILABLE TOOLS${NC}"
    echo -e "${CYAN}================================================================================${NC}"
    
    local server_count=0
    local total_tools=0
    
    while IFS='|' read -r name url; do
        if [[ -z "$name" ]]; then
            continue
        fi
        
        server_count=$((server_count + 1))
        echo ""
        echo -e "${GREEN}🔧 $name${NC}"
        
        # Extract address and port from URL
        local address=$(echo "$url" | sed -E 's|https?://([^:/]+).*|\1|')
        local port=$(echo "$url" | sed -E 's|https?://[^:]+:([0-9]+).*|\1|')
        local transport="sse"
        if [[ "$url" == *"/mcp"* ]]; then
            transport="streamable-http"
        fi
        
        echo -e "   Address: ${address}:${port}"
        echo -e "   Transport: ${transport}"
        echo -e "   URL: ${url}"
        
        # Get tools for this server
        local tools_json
        tools_json=$(get_server_tools "$url")
        local tool_count
        tool_count=$(echo "$tools_json" | jq '.tools | length' 2>/dev/null || echo "0")
        total_tools=$((total_tools + tool_count))
        
        echo -e "   Tools (${tool_count}):"
        if [[ "$tool_count" -gt 0 ]]; then
            for ((i=0; i<tool_count; i++)); do
                local tool_name
                local tool_desc
                tool_name=$(echo "$tools_json" | jq -r ".tools[$i].name" 2>/dev/null || echo "")
                tool_desc=$(echo "$tools_json" | jq -r ".tools[$i].description // \"\"" 2>/dev/null || echo "")
                
                if [[ -n "$tool_name" ]]; then
                    if [[ -n "$tool_desc" && "$tool_desc" != "null" ]]; then
                        # Limit description to first 60 characters
                        local short_desc=$(echo "$tool_desc" | head -n 1 | cut -c1-60)
                        if [[ ${#tool_desc} -gt 60 ]]; then
                            short_desc="${short_desc}..."
                        fi
                        echo "     • ${tool_name} - ${short_desc}"
                    else
                        echo "     • ${tool_name}"
                    fi
                fi
            done
        else
            echo "     No tools available"
        fi
        echo -e "${BLUE}$(printf '%.0s-' {1..40})${NC}"
        
    done <<< "$(extract_server_info "$servers_json")"
    
    echo ""
    echo -e "${PURPLE}Summary: ${server_count} server(s), ${total_tools} total tool(s)${NC}"
}

# Format output as detailed
format_detailed() {
    local servers_json="$1"
    
    echo -e "${CYAN}🔍 DETAILED TOOL DISCOVERY REPORT${NC}"
    echo -e "${CYAN}$(printf '%.0s=' {1..60})${NC}"
    
    local server_count=0
    local total_tools=0
    
    # Count servers first
    local running_servers
    running_servers=$(echo "$servers_json" | jq '[.[] | select(.status == "running")] | length' 2>/dev/null || echo "0")
    echo -e "Found ${running_servers} MCP server(s)"
    echo ""
    
    local counter=1
    while IFS='|' read -r name url; do
        if [[ -z "$name" ]]; then
            continue
        fi
        
        server_count=$((server_count + 1))
        echo -e "${GREEN}[$counter] $name${NC}"
        echo -e "${BLUE}$(printf '%.0s─' $(seq 1 $((${#name} + 4))))${NC}"
        
        # Extract connection details
        local address=$(echo "$url" | sed -E 's|https?://([^:/]+).*|\1|')
        local port=$(echo "$url" | sed -E 's|https?://[^:]+:([0-9]+).*|\1|')
        local transport="sse"
        if [[ "$url" == *"/mcp"* ]]; then
            transport="streamable-http"
        fi
        
        echo -e "${YELLOW}📡 Connection Details:${NC}"
        echo -e "   • Address: ${address}:${port}"
        echo -e "   • Transport: ${transport}"
        echo -e "   • URL: ${url}"
        
        # Get tools for this server
        local tools_json
        tools_json=$(get_server_tools "$url")
        local tool_count
        tool_count=$(echo "$tools_json" | jq '.tools | length' 2>/dev/null || echo "0")
        total_tools=$((total_tools + tool_count))
        
        echo ""
        echo -e "${YELLOW}🛠️  Available Tools (${tool_count}):${NC}"
        
        if [[ "$tool_count" -gt 0 ]]; then
            # Process tools properly using jq array indexing
            for ((i=0; i<tool_count; i++)); do
                local tool_name
                local tool_desc
                tool_name=$(echo "$tools_json" | jq -r ".tools[$i].name" 2>/dev/null || echo "")
                tool_desc=$(echo "$tools_json" | jq -r ".tools[$i].description // \"\"" 2>/dev/null || echo "")
                
                if [[ -n "$tool_name" ]]; then
                    echo -e "   ${PURPLE}[$((i+1))] $tool_name${NC}"
                    if [[ -n "$tool_desc" && "$tool_desc" != "null" ]]; then
                        # Clean up description and limit to first line or sentence
                        local clean_desc=$(echo "$tool_desc" | head -n 1 | cut -c1-100)
                        if [[ ${#tool_desc} -gt 100 ]]; then
                            clean_desc="${clean_desc}..."
                        fi
                        echo "       $clean_desc"
                    else
                        echo "       (No description available)"
                    fi
                    echo ""
                fi
            done
        else
            echo "   (No tools available)"
        fi
        
        echo -e "${CYAN}$(printf '%.0s=' {1..60})${NC}"
        echo ""
        counter=$((counter + 1))
        
    done <<< "$(extract_server_info "$servers_json")"
}

# Format output as compact (most concise)
format_compact() {
    local servers_json="$1"
    
    echo -e "${CYAN}🔧 TOOL SUMMARY${NC}"
    echo -e "${CYAN}$(printf '%.0s=' {1..40})${NC}"
    
    local total_tools=0
    local server_count=0
    
    while IFS='|' read -r name url; do
        if [[ -z "$name" ]]; then
            continue
        fi
        
        server_count=$((server_count + 1))
        
        # Get tools for this server
        local tools_json
        tools_json=$(get_server_tools "$url")
        local tool_count
        tool_count=$(echo "$tools_json" | jq '.tools | length' 2>/dev/null || echo "0")
        total_tools=$((total_tools + tool_count))
        
        echo -e "${GREEN}$name${NC} (${tool_count} tools)"
        
        if [[ "$tool_count" -gt 0 ]]; then
            # Show tool names in a compact list
            local tool_names=()
            for ((i=0; i<tool_count; i++)); do
                local tool_name
                tool_name=$(echo "$tools_json" | jq -r ".tools[$i].name" 2>/dev/null || echo "")
                if [[ -n "$tool_name" ]]; then
                    tool_names+=("$tool_name")
                fi
            done
            
            # Print tools in comma-separated format, wrapping at 70 chars
            local line=""
            for tool in "${tool_names[@]}"; do
                if [[ -z "$line" ]]; then
                    line="  $tool"
                elif [[ ${#line} -lt 60 ]]; then
                    line="$line, $tool"
                else
                    echo "$line"
                    line="  $tool"
                fi
            done
            if [[ -n "$line" ]]; then
                echo "$line"
            fi
        else
            echo "  (No tools)"
        fi
        echo ""
        
    done <<< "$(extract_server_info "$servers_json")"
    
    echo -e "${PURPLE}Total: ${server_count} server(s), ${total_tools} tool(s)${NC}"
}

# Format output as JSON
format_json() {
    local servers_json="$1"
    
    local output='{"servers_found": 0, "servers": []}'
    local server_count=0
    
    output=$(echo "$output" | jq '.servers = []')
    
    while IFS='|' read -r name url; do
        if [[ -z "$name" ]]; then
            continue
        fi
        
        server_count=$((server_count + 1))
        
        # Extract connection details
        local address=$(echo "$url" | sed -E 's|https?://([^:/]+).*|\1|')
        local port=$(echo "$url" | sed -E 's|https?://[^:]+:([0-9]+).*|\1|')
        local transport="sse"
        if [[ "$url" == *"/mcp"* ]]; then
            transport="streamable-http"
        fi
        
        # Get tools for this server
        local tools_json
        tools_json=$(get_server_tools "$url")
        
        # Build server object
        local server_obj
        server_obj=$(jq -n \
            --arg name "$name" \
            --arg address "$address" \
            --arg port "$port" \
            --arg transport "$transport" \
            --arg url "$url" \
            --argjson tools "$(echo "$tools_json" | jq '.tools')" \
            '{
                name: $name,
                address: $address,
                port: ($port | tonumber),
                transport: $transport,
                url: $url,
                tools: $tools
            }')
        
        output=$(echo "$output" | jq --argjson server "$server_obj" '.servers += [$server]')
        
    done <<< "$(extract_server_info "$servers_json")"
    
    output=$(echo "$output" | jq --arg count "$server_count" '.servers_found = ($count | tonumber)')
    echo "$output" | jq '.'
}

# Show client code examples
show_client_code() {
    local servers_json="$1"
    
    echo -e "${CYAN}$(printf '%.0s=' {1..80})${NC}"
    echo -e "${CYAN}                          GENERATED CLIENT CODE${NC}"
    echo -e "${CYAN}$(printf '%.0s=' {1..80})${NC}"
    echo ""
    echo "# Generated MCP Client Code"
    echo "# This code can be used in workflows to connect to discovered MCP servers"
    echo ""
    
    while IFS='|' read -r name url; do
        if [[ -z "$name" ]]; then
            continue
        fi
        
        # Get tools for this server
        local tools_json
        tools_json=$(get_server_tools "$url")
        local tool_names
        tool_names=$(echo "$tools_json" | jq -r '.tools[].name' 2>/dev/null | tr '\n' ', ' | sed 's/,$//')
        
        local transport="sse"
        if [[ "$url" == *"/mcp"* ]]; then
            transport="streamable-http"
        fi
        
        local var_name
        var_name=$(echo "$name" | tr ' ' '_' | tr '[:lower:]' '[:upper:]')_TOOLS
        
        echo "# $name"
        echo "# Tools: $tool_names"
        echo "from smolagents import MCPClient"
        echo "params = {\"url\": \"$url\", \"transport\": \"$transport\"}"
        echo "client = MCPClient(params)"
        echo "tools = client.get_tools()"
        echo "$var_name = tools"
        echo ""
        
    done <<< "$(extract_server_info "$servers_json")"
}

# Main function
main() {
    check_toolhive
    
    echo -e "${BLUE}🔍 Discovering MCP servers and tools...${NC}"
    
    local servers_json
    servers_json=$(get_running_servers)
    
    local running_count
    running_count=$(echo "$servers_json" | jq '[.[] | select(.status == "running")] | length' 2>/dev/null || echo "0")
    
    if [[ "$running_count" -eq 0 ]]; then
        echo -e "${RED}❌ No MCP servers found running. Make sure ToolHive servers are started.${NC}"
        echo -e "${YELLOW}   To start servers: thv start <server-name>${NC}"
        echo -e "${YELLOW}   To list available servers: thv list${NC}"
        exit 1
    fi
    
    case "$FORMAT" in
        "table")
            format_table "$servers_json"
            ;;
        "json")
            format_json "$servers_json"
            ;;
        "detailed")
            format_detailed "$servers_json"
            ;;
        "compact")
            format_compact "$servers_json"
            ;;
        *)
            echo -e "${RED}Unknown format: $FORMAT${NC}"
            exit 1
            ;;
    esac
    
    if [[ "$SHOW_CODE" == true ]]; then
        echo ""
        show_client_code "$servers_json"
    fi
}

# Run main function
main "$@"