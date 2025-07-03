###############################################################################
# LANGGRAPH – SMOLAGENT WORKFLOW : “Deep-Dive & macOS Installation of MetaboT”
###############################################################################
#
# GOAL
# ----
# 1. Perform an exhaustive web-search about the “MetaboT” metabolomics project
#    created by the Holobiomics Lab.
# 2. Extract & structure every useful piece of information
# 3. Validate that we truly have:
#       • Project description / publication links
#       • Latest download source  (GitHub, Zenodo, etc.)
#       • Explicit macOS requirements & dependencies
#       • Full step-by-step installation instructions
# 4. If validation passes, produce a bullet-proof macOS install plan
# 5. Execute the plan on the shell (simulation) and confirm success.
#
# ARCHITECTURE
# ------------
#    START ─▶ web_searcher ─┬─▶ info_extractor ─┬─▶ info_validator ─┬─▶ install_planner ─┬─▶ shell_installer ─▶ END
#                           │                   │                   │                    │
#                           │<── retry /───────>│<── retry /───────>│<── retry /────────>│
#                           │   fallback        │   fallback        │    fallback        │
#                           └────── emergency fallback (GIVE_UP)─────────────────────────┘
#
###############################################################################

# ---------------------------------------------------------------------------
# PRE-DEFINED OBJECTS (already in the environment – WE DO NOT REDECLARE them)
# ---------------------------------------------------------------------------
# - WorkflowState   (TypedDict)
# - StateGraph, START, END
# - SmolAgentFactory
# - WorkflowNodeFactory
# - SHELL_TOOLS, BROWSER_TOOLS, CSV_TOOLS
#
###############################################################################

from langgraph.graph import StateGraph, START, END
import re   # allowed standard import for small regex checks

# ---------------------------------------------------------------------------
# 1) AGENT PROMPTS
# ---------------------------------------------------------------------------

# --- Agent 1 : WEB SEARCHER -------------------------------------------------
prompt_web_search = """
You are an expert web-research agent.

#####################   YOUR TASK   #####################
1. Search the web for any information about *MetaboT* metabolomics project
   created by the **Holobiomics Lab**.
2. Collect URLs of papers, repos, or data portals.
3. Provide raw snippets (title + short excerpt) for each URL discovered.

######################  OUTPUT RULES  ###################
SUCCESS → If you have at least:
          • ≥3 distinct, relevant URLs AND
          • one download / code repository link
Return a single line exactly:
SEARCH_COMPLETE
followed by a blank line, then the list of resources in markdown bullet form
Example:
SEARCH_COMPLETE

- https://github.com/holobiomics/MetaboT  – “MetaboT: source code”
- ...

FAILURE → If criteria not met:
SEARCH_FAILURE
then detailed explanation and suggestions.

ERROR → For tool / technical issues:
GIVE_UP
then diagnostics.
"""

# --- Agent 2 : INFO EXTRACTOR ----------------------------------------------
prompt_info_extract = """
You are a data extraction agent that receives raw web search results on MetaboT.

#####################   YOUR TASK   #####################
1. Transform the unstructured list of URLs & snippets into a structured
   JSON block with keys:
      - "project_description"
      - "publication_links"
      - "download_link"
      - "dependencies"
      - "install_snippets"
2. Do NOT invent data. If a field is missing, leave it as an empty string.

######################  OUTPUT RULES  ###################
SUCCESS  → If at least "download_link" AND "install_snippets" not empty:
EXTRACTION_COMPLETE
<blank line>
<the JSON>

FAILURE  → Otherwise:
EXTRACTION_FAILURE
<blank line>
<explanation what is missing>

ERROR    → On technical issues:
GIVE_UP
"""

# --- Agent 3 : INFO VALIDATOR ----------------------------------------------
prompt_validator = """
You are a validation agent ensuring completeness of MetaboT information.

#####################   YOUR TASK   #####################
Validate the JSON received. Check that we have:
  • Non-empty "download_link"
  • At least one macOS compatible instruction in "install_snippets"
  • Non-empty "dependencies"

######################  OUTPUT RULES  ###################
If everything is present and macOS is explicitly mentioned:
VALIDATION_PASS

If information is insufficient:
VALIDATION_FAIL
<blank line>
• List missing items
• What extra search terms could help

ERROR → For technical issues:
GIVE_UP
"""

# --- Agent 4 : INSTALL PLANNER ---------------------------------------------
prompt_planner = """
You are an installation-planner agent for macOS.

#####################   YOUR TASK   #####################
Create a thorough, step-by-step macOS installation guide for MetaboT using
the validated information. Include:
  1. Homebrew / conda commands to install dependencies
  2. Command to clone / download MetaboT
  3. Build / compile / install commands
  4. Example test run to confirm installation

######################  OUTPUT RULES  ###################
If guide finished:
PLAN_COMPLETE
<blank line>
<ordered steps, each on its own line>

If you realise information is missing → produce:
INSUFFICIENT_INFO
<blank line>
<what is missing>

ERROR → Technical problems:
GIVE_UP
"""

# --- Agent 5 : SHELL INSTALLER ---------------------------------------------
prompt_shell = """
You are a shell execution agent working on a macOS environment.

#####################   YOUR TASK   #####################
Run the provided installation plan commands one by one.
After each command, capture the stdout/stderr.

######################  OUTPUT RULES  ###################
If ALL commands succeed (exit code 0):
INSTALL_SUCCESS
<blank line>
• Summarise outputs

If any command fails:
INSTALL_FAIL
<blank line>
• Which command failed
• Relevant stderr
• Suggest fix

ERROR → Tool or environment issues:
GIVE_UP
"""

# ---------------------------------------------------------------------------
# 2) AGENT INSTANTIATION
# ---------------------------------------------------------------------------
agent_web_search   = SmolAgentFactory(prompt_web_search,   BROWSER_TOOLS)
agent_extractor    = SmolAgentFactory(prompt_info_extract, BROWSER_TOOLS)
agent_validator    = SmolAgentFactory(prompt_validator,    CSV_TOOLS)
agent_planner      = SmolAgentFactory(prompt_planner,      BROWSER_TOOLS)
agent_shell        = SmolAgentFactory(prompt_shell,        SHELL_TOOLS)

# ---------------------------------------------------------------------------
# 3) WORKFLOW INITIALISATION
# ---------------------------------------------------------------------------
workflow = StateGraph(WorkflowState)

# ---------------------------------------------------------------------------
# 4) NODE REGISTRATION
# ---------------------------------------------------------------------------
workflow.add_node("web_searcher",   WorkflowNodeFactory.create_agent_node(agent_web_search))
workflow.add_node("info_extractor", WorkflowNodeFactory.create_agent_node(agent_extractor))
workflow.add_node("info_validator", WorkflowNodeFactory.create_agent_node(agent_validator))
workflow.add_node("install_planner",WorkflowNodeFactory.create_agent_node(agent_planner))
workflow.add_node("shell_installer",WorkflowNodeFactory.create_agent_node(agent_shell))

# ---------------------------------------------------------------------------
# 5) ROUTING / ERROR-HANDLING FUNCTIONS
# ---------------------------------------------------------------------------

def _count_retries(state: WorkflowState, node_name: str) -> int:
    return sum(1 for n in state.get("step_name", []) if n.startswith(node_name))

# ---------- Router after web_searcher --------------------------------------
def route_after_search(state: WorkflowState) -> str:
    try:
        answers = state.get("answers", [])
        if not answers:
            print("⚠️  route_after_search: no answer found – retrying search")
            return "web_searcher"

        last = answers[-1]
        if "SEARCH_COMPLETE" in last:
            return "info_extractor"
        if "SEARCH_FAILURE" in last:
            # retry up to 3 times
            if _count_retries(state, "web_searcher") < 3:
                print("🔄  web_searcher retry")
                return "web_searcher"
            else:
                print("❌  web_searcher max retries reached")
                return END
        # unexpected token → treat as failure but retry once
        if _count_retries(state, "web_searcher") < 1:
            print("⚠️  web_searcher unexpected format – one more retry")
            return "web_searcher"
        return END
    except Exception as e:
        print(f"💥 route_after_search error: {e}")
        return END

# ---------- Router after info_extractor ------------------------------------
def route_after_extractor(state: WorkflowState) -> str:
    try:
        last = state.get("answers", [])[-1]
        if "EXTRACTION_COMPLETE" in last:
            return "info_validator"
        if "EXTRACTION_FAILURE" in last:
            if _count_retries(state, "info_extractor") < 3:
                return "info_extractor"
            else:
                # fallback: go back to search to gather more data
                return "web_searcher"
        return END
    except Exception as e:
        print(f"💥 route_after_extractor error: {e}")
        return END

# ---------- Router after info_validator ------------------------------------
def route_after_validator(state: WorkflowState) -> str:
    try:
        last = state.get("answers", [])[-1]
        if "VALIDATION_PASS" in last:
            return "install_planner"
        if "VALIDATION_FAIL" in last:
            # If validator fails we ask extractor to enrich data, but only twice
            if _count_retries(state, "info_extractor") < 2:
                return "info_extractor"
            else:
                # fallback to web search for fresh data
                return "web_searcher"
        return END
    except Exception as e:
        print(f"💥 route_after_validator error: {e}")
        return END

# ---------- Router after install_planner ------------------------------------
def route_after_planner(state: WorkflowState) -> str:
    try:
        last = state.get("answers", [])[-1]
        if "PLAN_COMPLETE" in last:
            return "shell_installer"
        if "INSUFFICIENT_INFO" in last:
            # cycle back to validator to identify gap (max 2)
            if _count_retries(state, "info_validator") < 2:
                return "info_validator"
            else:
                return "web_searcher"
        return END
    except Exception as e:
        print(f"💥 route_after_planner error: {e}")
        return END

# ---------- Router after shell_installer ------------------------------------
def route_after_shell(state: WorkflowState) -> str:
    try:
        last = state.get("answers", [])[-1]
        if "INSTALL_SUCCESS" in last:
            return END
        if "INSTALL_FAIL" in last:
            # Only one retry of planner then shell again
            if _count_retries(state, "install_planner") < 2:
                return "install_planner"
            else:
                return END
        return END
    except Exception as e:
        print(f"💥 route_after_shell error: {e}")
        return END

# ---------------------------------------------------------------------------
# 6) EDGES & CONDITIONAL ROUTING
# ---------------------------------------------------------------------------
workflow.add_edge(START, "web_searcher")

workflow.add_conditional_edges(
    "web_searcher",
    route_after_search,
    {
        "info_extractor": "info_extractor",
        "web_searcher": "web_searcher",
        END: END
    }
)

workflow.add_conditional_edges(
    "info_extractor",
    route_after_extractor,
    {
        "info_validator": "info_validator",
        "info_extractor": "info_extractor",
        "web_searcher": "web_searcher",
        END: END
    }
)

workflow.add_conditional_edges(
    "info_validator",
    route_after_validator,
    {
        "install_planner": "install_planner",
        "info_extractor": "info_extractor",
        "web_searcher": "web_searcher",
        END: END
    }
)

workflow.add_conditional_edges(
    "install_planner",
    route_after_planner,
    {
        "shell_installer": "shell_installer",
        "info_validator": "info_validator",
        "web_searcher": "web_searcher",
        END: END
    }
)

workflow.add_conditional_edges(
    "shell_installer",
    route_after_shell,
    {
        END: END,
        "install_planner": "install_planner"
    }
)

# ---------------------------------------------------------------------------
# 7) COMPILE WORKFLOW
# ---------------------------------------------------------------------------
app = workflow.compile()

###############################################################################
# The `app` object is ready. 
# Use:  app.invoke(initial_state)  where `initial_state` matches WorkflowState.
###############################################################################