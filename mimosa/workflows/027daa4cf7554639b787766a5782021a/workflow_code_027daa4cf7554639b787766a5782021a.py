# ================================================================
# LangGraph ‑ SmolAgent Workflow
# Task : “Do a deep research and make a comprehensive report about
#         CNRS goals in AI for European sovereignty”
# ================================================================

from langgraph.graph import StateGraph, START, END

# ----------------------------------------------------------------
# 1. SmolAgent PROMPTS (one atomic responsibility each)
# ----------------------------------------------------------------
search_prompt_web = """
You are WEB_SEARCH_AGENT.
Goal: Find as many high-quality, authoritative web sources that mention
CNRS (French National Centre for Scientific Research) strategy / goals
in Artificial Intelligence specifically in the context of European
technological sovereignty.

TASK STEPS
1. Use browser tools to query relevant keywords
   (e.g. “CNRS AI European sovereignty”, “CNRS artificial intelligence Europe strategy”)
2. Collect at least 5 distinct URLs from official CNRS pages, EU policy notes,
   press releases, and reputable media.
3. For every URL write one short sentence describing why it is relevant.

COMPLETION PROTOCOL (MUST follow exactly)  
If you found ≥5 good sources:
    final_answer("SEARCH_COMPLETE\n" + json_dump({
        "sources":[{"url":URL, "note":SHORT_NOTE}, …]
    }))
If information is insufficient:
    final_answer("SEARCH_FAILURE\n[Explain what you tried and why it failed]")
If you hit a technical/tool problem:
    final_answer("GIVE_UP\n[Describe the technical issue]")
"""

search_prompt_shell = """
You are SHELL_SEARCH_AGENT (fallback).
Goal identical to WEB_SEARCH_AGENT but you may only use shell tools
(e.g. curl, grep) to fetch pages and discover links.

Follow the same COMPLETION PROTOCOL with keywords:
    SHELL_SEARCH_COMPLETE / SHELL_SEARCH_FAILURE / GIVE_UP
"""

extract_prompt = """
You are EXTRACTION_AGENT.
Goal: Read the supplied list of URLs (from state) and extract every sentence
that directly states CNRS objectives, missions, budgets or initiatives
regarding AI for European sovereignty.

OUTPUT exactly:
EXTRACTION_COMPLETE
<json_dump({
   "extracted_facts":[{"fact":TEXT, "source":URL}, …]
})>

On insufficient information:
EXTRACTION_FAILURE
[detailed reason + suggestions]

On tool error:
GIVE_UP
[technical explanation]
"""

validate_prompt = """
You are VALIDATION_AGENT.
Goal: Cross-check the extracted facts for consistency across at least two
independent sources. Mark each fact as VALID or CONFLICTING.

OUTPUT exactly:
VALIDATION_COMPLETE
<json_dump({
  "validated":[{"fact":TEXT,"status":"VALID/CONFLICTING","supporting_sources":[…]}]
})>

If validation impossible (not enough sources):
VALIDATION_FAILURE
[why]

On technical error:
GIVE_UP
[error]
"""

report_prompt = """
You are REPORT_AGENT.
Goal: Write a comprehensive, well-structured report (≈600-800 words) that:
• Summarises CNRS goals in AI for European sovereignty
• Groups validated facts into thematic sections (Research, Funding, Collaboration…)
• Cites sources inline with [#] markers and append a bibliography list.

OUTPUT exactly:
REPORT_COMPLETE
<full_report_text>

If missing data:
REPORT_FAILURE
[explain what is missing]

On technical/tool problem:
GIVE_UP
[error]
"""

# ----------------------------------------------------------------
# 2. Instantiate SmolAgents with appropriate tool packages
# ----------------------------------------------------------------
web_search_agent      = SmolAgentFactory(search_prompt_web,   BROWSER_TOOL_TOOLS)
shell_search_agent    = SmolAgentFactory(search_prompt_shell, SHELL_TOOL_TOOLS)
extraction_agent      = SmolAgentFactory(extract_prompt,      BROWSER_TOOL_TOOLS)
validation_agent      = SmolAgentFactory(validate_prompt,     BROWSER_TOOL_TOOLS)
report_agent          = SmolAgentFactory(report_prompt,       SHELL_TOOL_TOOLS)

# ----------------------------------------------------------------
# 3. Build Workflow Graph
# ----------------------------------------------------------------
workflow = StateGraph(WorkflowState)

# ----- Helper: capped retry counter --------------------------------
def retry_allowed(state: WorkflowState, node_name:str, cap:int=3)->bool:
    return state["step_name"].count(node_name) < cap

# ------------------------- ROUTERS ---------------------------------
def web_search_router(state: WorkflowState) -> str:
    try:
        answer = state["answers"][-1] if state["answers"] else ""
        state["step_name"].append("web_search_router")

        if "SEARCH_COMPLETE" in answer:
            return "extraction_agent"
        if "SEARCH_FAILURE" in answer:
            if retry_allowed(state,"web_search_agent"):
                return "web_search_agent"          # retry same agent
            else:
                return "shell_search_agent"       # fallback path
        if "GIVE_UP" in answer:
            return END
        # Unexpected output – safe retry
        return "web_search_agent"
    except Exception as e:
        print(f"💥 Router error web_search: {e}")
        return END

def shell_search_router(state: WorkflowState) -> str:
    try:
        answer = state["answers"][-1] if state["answers"] else ""
        state["step_name"].append("shell_search_router")

        if "SHELL_SEARCH_COMPLETE" in answer:
            return "extraction_agent"
        if "SHELL_SEARCH_FAILURE" in answer:
            if retry_allowed(state,"shell_search_agent"):
                return "shell_search_agent"
            else:
                return END                         # ultimate failure
        if "GIVE_UP" in answer:
            return END
        return "shell_search_agent"
    except Exception as e:
        print(f"💥 Router error shell_search: {e}")
        return END

def extraction_router(state: WorkflowState) -> str:
    try:
        answer = state["answers"][-1] if state["answers"] else ""
        state["step_name"].append("extraction_router")

        if "EXTRACTION_COMPLETE" in answer:
            return "validation_agent"
        if "EXTRACTION_FAILURE" in answer:
            # go back to web search to fetch more sources
            if retry_allowed(state,"extraction_agent"):
                return "web_search_agent"
            else:
                return END
        if "GIVE_UP" in answer:
            return END
        return "extraction_agent"
    except Exception as e:
        print(f"💥 Router error extraction: {e}")
        return END

def validation_router(state: WorkflowState) -> str:
    try:
        answer = state["answers"][-1] if state["answers"] else ""
        state["step_name"].append("validation_router")

        if "VALIDATION_COMPLETE" in answer:
            return "report_agent"
        if "VALIDATION_FAILURE" in answer:
            if retry_allowed(state,"validation_agent"):
                return "extraction_agent"          # back for more facts
            else:
                return END
        if "GIVE_UP" in answer:
            return END
        return "validation_agent"
    except Exception as e:
        print(f"💥 Router error validation: {e}")
        return END

def report_router(state: WorkflowState) -> str:
    try:
        answer = state["answers"][-1] if state["answers"] else ""
        state["step_name"].append("report_router")

        if "REPORT_COMPLETE" in answer:
            return END
        if "REPORT_FAILURE" in answer:
            if retry_allowed(state,"report_agent"):
                return "validation_agent"          # gather better facts
            else:
                return END
        if "GIVE_UP" in answer:
            return END
        return "report_agent"
    except Exception as e:
        print(f"💥 Router error report: {e}")
        return END

# ------------------------- NODES -----------------------------------
workflow.add_node("web_search_agent",   WorkflowNodeFactory.create_agent_node(web_search_agent))
workflow.add_node("shell_search_agent", WorkflowNodeFactory.create_agent_node(shell_search_agent))
workflow.add_node("extraction_agent",   WorkflowNodeFactory.create_agent_node(extraction_agent))
workflow.add_node("validation_agent",   WorkflowNodeFactory.create_agent_node(validation_agent))
workflow.add_node("report_agent",       WorkflowNodeFactory.create_agent_node(report_agent))

# ------------------------- EDGES -----------------------------------
workflow.add_edge(START, "web_search_agent")

workflow.add_conditional_edges(
    "web_search_agent",
    web_search_router,
    {
        "extraction_agent":    "extraction_agent",
        "web_search_agent":    "web_search_agent",
        "shell_search_agent":  "shell_search_agent",
        END:                   END
    }
)

workflow.add_conditional_edges(
    "shell_search_agent",
    shell_search_router,
    {
        "extraction_agent":    "extraction_agent",
        "shell_search_agent":  "shell_search_agent",
        END:                   END
    }
)

workflow.add_conditional_edges(
    "extraction_agent",
    extraction_router,
    {
        "validation_agent":    "validation_agent",
        "web_search_agent":    "web_search_agent",
        "extraction_agent":    "extraction_agent",
        END:                   END
    }
)

workflow.add_conditional_edges(
    "validation_agent",
    validation_router,
    {
        "report_agent":        "report_agent",
        "extraction_agent":    "extraction_agent",
        "validation_agent":    "validation_agent",
        END:                   END
    }
)

workflow.add_conditional_edges(
    "report_agent",
    report_router,
    {
        END:                   END,
        "validation_agent":    "validation_agent",
        "report_agent":        "report_agent"
    }
)

# ------------------------- COMPILE ---------------------------------
app = workflow.compile()

# The compiled `app` can now be invoked with an initial WorkflowState:
# app.invoke({"step_name":[], "actions":[], "observations":[],
#             "rewards":[], "answers":[], "success":[]})