# -----------------------------------------------
# LangGraph ‑ SmolAgent workflow
# Task :  Search the MetaboT project (Holobiomics Lab – CNRS),  
#         save all project contributors in a CSV file,  
#         then install the project from GitHub.
#
# Tools already provided in the execution context:
#   SHELL_TOOLS , BROWSER_TOOLS , CSV_TOOLS
# -----------------------------------------------

from langgraph.graph import StateGraph, START, END

# ------------------------------------------------
# 1.  WORKFLOW INITIALISATION
# ------------------------------------------------
workflow = StateGraph(WorkflowState)   # ←  State schema already loaded in context


# ------------------------------------------------
# 2.  AGENT DEFINITIONS  (one atomic task each)
# ------------------------------------------------
# ---- 2.1  Web-search agent ---------------------------------
search_prompt = """
You are a Web-search agent.

YOUR SINGLE GOAL
- Locate the official “MetaboT” project created by the Holobiomics Lab (CNRS).
- Identify and return the project’s main public URL *and* its GitHub repository URL.

OUTPUT FORMAT (MANDATORY)
If you located BOTH urls, answer exactly:
SEARCH_COMPLETE
<project_url>
<github_url>

If you could not locate them, answer exactly:
SEARCH_FAILURE

If a tool error or unexpected situation blocks you, answer exactly:
GIVE_UP
"""
agent_web_search = SmolAgentFactory(search_prompt, BROWSER_TOOLS)

# ---- 2.2  Contributor extractor agent ----------------------
extract_prompt = """
You are a Contributor-extraction agent.

YOUR SINGLE GOAL
- Open the GitHub repository url for the MetaboT project (passed to you in context).
- Extract the list of *human* contributor names (no bots) visible on GitHub.

OUTPUT FORMAT (MANDATORY)
If contributors were extracted, answer exactly:
EXTRACTION_COMPLETE
<name1>
<name2>
<name3>
...

If extraction failed, answer exactly:
EXTRACTION_FAILURE

If a tool error or unexpected situation blocks you, answer exactly:
GIVE_UP
"""
agent_contrib_extractor = SmolAgentFactory(extract_prompt, BROWSER_TOOLS)

# ---- 2.3  CSV writer agent ---------------------------------
csv_prompt = """
You are a CSV-writing agent.

YOUR SINGLE GOAL
- Receive a plain list of contributor names (one per line) from context.
- Create/overwrite a CSV file called contributors.csv with header "name"
  and write each contributor on its own row.

OUTPUT FORMAT (MANDATORY)
If the file was written successfully, answer exactly:
CSV_COMPLETE

If it failed, answer exactly:
CSV_FAILURE

If a tool error or unexpected situation blocks you, answer exactly:
GIVE_UP
"""
agent_csv_writer = SmolAgentFactory(csv_prompt, CSV_TOOLS)

# ---- 2.4  Installer agent ----------------------------------
install_prompt = """
You are an installation agent.

YOUR SINGLE GOAL
- Using shell commands, clone the MetaboT GitHub repository into /workspace/metabot
- Run any documented installation commands (e.g. make install, pip install -e . , etc.)
- Ensure the installation finishes without error (exit code 0).

OUTPUT FORMAT (MANDATORY)
If installation succeeded, answer exactly:
INSTALL_COMPLETE

If installation failed, answer exactly:
INSTALL_FAILURE

If a tool error or unexpected situation blocks you, answer exactly:
GIVE_UP
"""
agent_installer = SmolAgentFactory(install_prompt, SHELL_TOOLS)


# ------------------------------------------------
# 3.  ADD AGENT NODES TO THE GRAPH
# ------------------------------------------------
workflow.add_node("web_search",           WorkflowNodeFactory.create_agent_node(agent_web_search))
workflow.add_node("contributors_extractor", WorkflowNodeFactory.create_agent_node(agent_contrib_extractor))
workflow.add_node("csv_writer",           WorkflowNodeFactory.create_agent_node(agent_csv_writer))
workflow.add_node("installer",            WorkflowNodeFactory.create_agent_node(agent_installer))


# ------------------------------------------------
# 4.  ROUTING / ERROR-HANDLING FUNCTIONS
# ------------------------------------------------
def router_search(state: WorkflowState) -> str:
    """Routing after web_search"""
    try:
        answer = state.get("answers", [""])[-1]
        retries = state.get("step_name", []).count("web_search")
        if "SEARCH_COMPLETE" in answer:
            return "contributors_extractor"
        elif "SEARCH_FAILURE" in answer and retries < 3:
            return "web_search"               # retry same agent
        else:
            return "give_up"                  # GIVE_UP or max retries
    except Exception as e:
        print(f"Router search error: {e}")
        return "give_up"

def router_extract(state: WorkflowState) -> str:
    """Routing after contributors_extractor"""
    try:
        answer = state.get("answers", [""])[-1]
        retries = state.get("step_name", []).count("contributors_extractor")
        if "EXTRACTION_COMPLETE" in answer:
            return "csv_writer"
        elif "EXTRACTION_FAILURE" in answer and retries < 3:
            return "contributors_extractor"
        else:
            return "give_up"
    except Exception as e:
        print(f"Router extract error: {e}")
        return "give_up"

def router_csv(state: WorkflowState) -> str:
    """Routing after csv_writer"""
    try:
        answer = state.get("answers", [""])[-1]
        retries = state.get("step_name", []).count("csv_writer")
        if "CSV_COMPLETE" in answer:
            return "installer"
        elif "CSV_FAILURE" in answer and retries < 3:
            return "csv_writer"
        else:
            return "give_up"
    except Exception as e:
        print(f"Router csv error: {e}")
        return "give_up"

def router_install(state: WorkflowState) -> str:
    """Routing after installer"""
    try:
        answer = state.get("answers", [""])[-1]
        retries = state.get("step_name", []).count("installer")
        if "INSTALL_COMPLETE" in answer:
            return "success_end"
        elif "INSTALL_FAILURE" in answer and retries < 3:
            return "installer"
        else:
            return "give_up"
    except Exception as e:
        print(f"Router install error: {e}")
        return "give_up"


# ------------------------------------------------
# 5.  EDGE DEFINITIONS  (with multiple fallbacks)
# ------------------------------------------------
workflow.add_edge(START, "web_search")

workflow.add_conditional_edges(
    "web_search",
    router_search,
    {
        "contributors_extractor": "contributors_extractor",
        "web_search": "web_search",    # retry path
        "give_up": END
    }
)

workflow.add_conditional_edges(
    "contributors_extractor",
    router_extract,
    {
        "csv_writer": "csv_writer",
        "contributors_extractor": "contributors_extractor",   # retry
        "give_up": END
    }
)

workflow.add_conditional_edges(
    "csv_writer",
    router_csv,
    {
        "installer": "installer",
        "csv_writer": "csv_writer",   # retry
        "give_up": END
    }
)

workflow.add_conditional_edges(
    "installer",
    router_install,
    {
        "success_end": END,
        "installer": "installer",     # retry
        "give_up": END
    }
)

# ------------------------------------------------
# 6.  COMPILE WORKFLOW
# ------------------------------------------------
app = workflow.compile()