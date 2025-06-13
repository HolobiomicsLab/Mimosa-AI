from langgraph.graph import StateGraph, START, END

# ----- TOOLSETS -----
SEARCH_TOOLS = BROWSER_TOOLS_TOOL
EXTRACTION_TOOLS = BROWSER_TOOLS_TOOL + CSV_TOOLS_TOOL

# ----- AGENT INSTRUCTIONS -----
instruct_search = """
You are a dedicated Web Research Agent.

GOAL
- Discover authoritative web sources for the given user goal.

CAPABILITIES
1. Issue search queries using search tools
2. Collect top relevant URLs with short descriptions
3. Validate each URL’s relevance to the goal
4. Store findings in observations

OUTPUT FORMAT
Return a Python list of dicts, each dict containing:
- "url": the link
- "snippet": brief reason for relevance
Ensure the list is neither empty nor exceeds 5 items.
"""

instruct_extract = """
You are a Web Extraction Agent.

GOAL
- Navigate to the provided URL and extract concise, relevant information that answers the user goal.

CAPABILITIES
1. Open the URL using browser tools
2. Read page content
3. Extract key facts, statistics, and insights
4. Structure extracted data into a CSV-friendly string (comma-separated values)

INPUT
The most recent observation contains a list of candidate URLs. Choose the best one (first if unsure).

OUTPUT FORMAT
Return a dict with:
- "chosen_url": the URL visited
- "extracted_info": CSV-compatible string summarizing key information
"""

# ----- WORKFLOW INITIALIZATION -----
workflow = StateGraph(WorkflowState)

# ----- AGENT CREATION -----
smol_search = SmolAgentFactory(instruct_search, SEARCH_TOOLS)
smol_extract = SmolAgentFactory(instruct_extract, EXTRACTION_TOOLS)

# ----- NODE ADDITIONS -----
workflow.add_node("web_searcher", WorkflowNodeFactory.create_agent_node(smol_search))
workflow.add_node("info_extractor", WorkflowNodeFactory.create_agent_node(smol_extract))

# ----- ROUTING FUNCTION -----
def route_after_search(state: WorkflowState) -> str:
    try:
        # Success only if last success flag exists and True, and at least one URL returned
        urls_found = "url" in state["observations"][-1].get("data", "")
        if state["success"] and state["success"][-1] and urls_found:
            return "continue"
        else:
            return "stop"
    except Exception:
        return "stop"

# ----- EDGES -----
workflow.add_edge(START, "web_searcher")
workflow.add_conditional_edges(
    "web_searcher",
    route_after_search,
    {
        "continue": "info_extractor",
        "stop": END
    }
)
workflow.add_edge("info_extractor", END)

# ----- COMPILE WORKFLOW -----
app = workflow.compile()