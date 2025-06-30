# --------------  LangGraph-SmolAgent WORKFLOW DEFINITION  --------------
#
# Overall goal:
#   1) Plan relevant search queries
#   2) Search web for alternatives to ManusAI (names & links)
#   3) Visit each link and extract:   name | description | local? | repo/website
#   4) Validate data completeness / quality
#   5) Persist the final list to disk as CSV  (file:  alternatives_to_ManusAI.csv)
#
# Tools available in runtime environment (already declared):
#   - SHELL_TOOLS      : basic shell / file utilities
#   - BROWSER_TOOLS    : web search & browsing
#   - CSV_TOOLS        : create / append / write CSV files
#
# NOTE:  All imported classes / factories / packages are already loaded in the interpreter.
# ----------------------------------------------------------------------

from langgraph.graph import StateGraph, START, END

# ----------------------------------------------------------------------
#                       AGENT PROMPTS
# ----------------------------------------------------------------------

# 1. Query-Planner  – produces a list of concrete search queries
instruct_planner = """
You are a strategic query-planning assistant.

## YOUR TASK
- Produce 5-10 distinct web-search queries that will discover *alternatives to ManusAI*
- Queries should emphasise open-source, self-hosted, SaaS, and similar note-taking or knowledge-base tools.
- Return them as a numbered list.

## COMPLETION RULES
SUCCESS → When you have a good query list:
    final_answer("PLANNER_COMPLETE: <one-line rationale> ||| QUERIES: <query1>; <query2>; ...")

FAILURE → If you cannot devise queries:
    final_answer("PLANNER_FAILURE: <explain attempted thoughts & why impossible>")

ERROR → Technical issues:
    final_answer("GIVE_UP")
"""

# 2. Web-Researcher  – performs searches & collects candidate tools
instruct_research = """
You are a web research agent.

## YOUR TASK
Using the queries provided, search the web and compile a rough candidate list of *alternatives to ManusAI*.
For each candidate capture:
- name
- primary repository OR website link

Return a bullet list “name – link”.

## COMPLETION RULES
SUCCESS → Provide list and end with:  RESEARCH_COMPLETE
FAILURE → If search unsuccessful:      RESEARCH_FAILURE
ERROR   → Technical error:             GIVE_UP
"""

# 3. Data-Extractor  – visits each candidate link & extracts structured details
instruct_extractor = """
You are a focused data-extraction agent.

## YOUR TASK
For each candidate (name & link) provided by the Web-Researcher:
- Visit link (repo or site)
- Extract a SHORT description (<= 40 words)
- Determine if software is *local/self-hosted* ("Yes" if runnable locally or self-hosted, else "No")

Return one candidate per line separated by " | " columns:
name | description | local (Yes/No) | link

## COMPLETION RULES
SUCCESS  → Finish with line: EXTRACTION_COMPLETE
INSUFFICIENT_INFORMATION → If any candidates missing data: EXTRACTION_INCOMPLETE
ERROR    → Technical tool issues: GIVE_UP
"""

# 4. Data-Validator  – checks completeness & basic quality
instruct_validator = """
You are a strict data quality validator.

## YOUR TASK
Given the extracted lines (name | description | local | link):
- Verify every field is non-empty
- Ensure at least 3 distinct alternatives exist
- Check 'local' field is 'Yes' or 'No'
If everything is good say VALIDATION_SUCCESS.
If not, list all issues and say VALIDATION_FAILURE.
If data format is wrong or unreadable say GIVE_UP.
"""

# 5. CSV-Writer  – saves the final table to disk
instruct_csv_writer = """
You are a CSV writing agent.

## YOUR TASK
Create / overwrite file  'alternatives_to_ManusAI.csv'
with header: name,description,local,link
Write each validated record on its own row (comma separated, quote fields if needed).

After writing file, respond with FILE_WRITTEN.

If writing fails, respond with WRITE_FAILURE.

For technical errors respond GIVE_UP.
"""

# ----------------------------------------------------------------------
#               AGENT CONSTRUCTION (using predefined factory)
# ----------------------------------------------------------------------

planner_agent     = SmolAgentFactory(instruct_planner,     BROWSER_TOOLS)
research_agent    = SmolAgentFactory(instruct_research,    BROWSER_TOOLS)
extractor_agent   = SmolAgentFactory(instruct_extractor,   BROWSER_TOOLS)
validator_agent   = SmolAgentFactory(instruct_validator,   [])
csv_writer_agent  = SmolAgentFactory(instruct_csv_writer,  CSV_TOOLS + SHELL_TOOLS)

# ----------------------------------------------------------------------
#                   WORKFLOW GRAPH INITIALISATION
# ----------------------------------------------------------------------

workflow = StateGraph(WorkflowState)

# ----------------  Helper: generic retry counter  ---------------------
def _retry_count(state: WorkflowState, node_name: str) -> int:
    return [s for s in state.get("step_name", []) if node_name in s].__len__()

# ----------------------------------------------------------------------
#                   ROUTING FUNCTIONS
# ----------------------------------------------------------------------

MAX_RETRIES = 3   # per agent

def planner_router(state: WorkflowState) -> str:
    try:
        answer = (state.get("answers") or [""])[-1]
        if "PLANNER_COMPLETE" in answer:
            return "web_researcher"
        elif "PLANNER_FAILURE" in answer and _retry_count(state, "query_planner") < MAX_RETRIES:
            return "query_planner"         # retry same agent
        else:
            return END                     # fatal
    except Exception as e:
        print(f"[Planner-Router Error] {e}")
        return END

def research_router(state: WorkflowState) -> str:
    try:
        answer = (state.get("answers") or [""])[-1]
        if "RESEARCH_COMPLETE" in answer:
            return "data_extractor"
        elif _retry_count(state, "web_researcher") < MAX_RETRIES:
            return "web_researcher"
        else:
            return "query_planner"         # fallback: maybe better queries
    except Exception as e:
        print(f"[Research-Router Error] {e}")
        return END

def extraction_router(state: WorkflowState) -> str:
    try:
        answer = (state.get("answers") or [""])[-1]
        if "EXTRACTION_COMPLETE" in answer:
            return "data_validator"
        elif "EXTRACTION_INCOMPLETE" in answer and _retry_count(state, "data_extractor") < MAX_RETRIES:
            return "data_extractor"        # retry extraction
        else:
            return "web_researcher"        # fallback to get more/better links
    except Exception as e:
        print(f"[Extraction-Router Error] {e}")
        return END

def validation_router(state: WorkflowState) -> str:
    try:
        answer = (state.get("answers") or [""])[-1]
        if "VALIDATION_SUCCESS" in answer:
            return "csv_writer"
        elif "VALIDATION_FAILURE" in answer and _retry_count(state, "data_extractor") < MAX_RETRIES:
            return "data_extractor"        # fix issues
        else:
            return END                     # irrecoverable
    except Exception as e:
        print(f"[Validation-Router Error] {e}")
        return END

def csv_router(state: WorkflowState) -> str:
    try:
        answer = (state.get("answers") or [""])[-1]
        if "FILE_WRITTEN" in answer:
            return END
        elif _retry_count(state, "csv_writer") < MAX_RETRIES:
            return "csv_writer"
        else:
            return END
    except Exception as e:
        print(f"[CSV-Router Error] {e}")
        return END

# ----------------------------------------------------------------------
#                       NODE REGISTRATION
# ----------------------------------------------------------------------

workflow.add_node("query_planner",    WorkflowNodeFactory.create_agent_node(planner_agent))
workflow.add_node("web_researcher",   WorkflowNodeFactory.create_agent_node(research_agent))
workflow.add_node("data_extractor",   WorkflowNodeFactory.create_agent_node(extractor_agent))
workflow.add_node("data_validator",   WorkflowNodeFactory.create_agent_node(validator_agent))
workflow.add_node("csv_writer",       WorkflowNodeFactory.create_agent_node(csv_writer_agent))

# -----------------------  EDGE DEFINITIONS  ---------------------------

workflow.add_edge(START, "query_planner")

workflow.add_conditional_edges("query_planner",  planner_router, {
    "web_researcher":  "web_researcher",
    "query_planner":   "query_planner",
    END:               END
})

workflow.add_conditional_edges("web_researcher", research_router, {
    "data_extractor":  "data_extractor",
    "web_researcher":  "web_researcher",
    "query_planner":   "query_planner",
    END:               END
})

workflow.add_conditional_edges("data_extractor", extraction_router, {
    "data_validator":  "data_validator",
    "data_extractor":  "data_extractor",
    "web_researcher":  "web_researcher",
    END:               END
})

workflow.add_conditional_edges("data_validator", validation_router, {
    "csv_writer":      "csv_writer",
    "data_extractor":  "data_extractor",
    END:               END
})

workflow.add_conditional_edges("csv_writer", csv_router, {
    END:               END,
    "csv_writer":      "csv_writer"
})

# -----------------------  COMPILE WORKFLOW  ---------------------------
app = workflow.compile()

# ----------------------------------------------------------------------
# The `app` object is now an executable LangGraph workflow that:
#     • Plans → Searches → Extracts → Validates → Writes CSV
#     • Includes retry loops & fallbacks with max 3 attempts per step
# ----------------------------------------------------------------------