######################################################################
# LangGraph–SmolAgent WORKFLOW : “Best Hikes around Nice, France”   #
######################################################################

# -------------------------------------------------------------------
#  Assumptions
#  - StateGraph, START, END, WorkflowState, SmolAgentFactory,
#    WorkflowNodeFactory, and the three tool-packages
#      • SHELL_TOOLS
#      • BROWSER_TOOLS
#      • CSV_TOOLS
#    are already available in the execution environment.
# -------------------------------------------------------------------

from langgraph.graph import StateGraph, START, END   # (already available)

# ----------------------------------------------------
# 1️⃣  AGENT PROMPTS  (one atomic responsibility each)
# ----------------------------------------------------

# 1. Web Researcher : finds candidate hikes & source URLs
instruct_research = """You are a specialised Web-Research agent.
## TASK
Search the web for hiking routes within ~100 km of Nice, France.
Collect at least 8 promising hikes.
For EACH hike capture:
- Hike name
- Source URL
Return ONLY a newline-separated markdown list “name – url”.

## COMPLETION RULES
SUCCESS → finish your answer with the exact token: RESEARCH_COMPLETE
FAILURE → detailed diagnostic followed by: RESEARCH_FAILURE
ERROR   → detailed error report followed by: GIVE_UP
Strictly use one of the three tokens above at the very end."""

# 2. Detail Extractor : visits each URL & extracts structured info
instruct_extract = """You are a data-extraction agent.
## INPUT
A markdown list of hike names & URLs (provided in the conversation).
## TASK
For EACH hike open its URL and extract:
- name
- brief 1-sentence description
- location (nearest village / landmark)
- total distance in km (number only)
- typical duration in hours (number only)
Return a JSON array, one object per hike.

## COMPLETION RULES
SUCCESS → end with: EXTRACTION_COMPLETE
FAILURE → end with: EXTRACTION_FAILURE
ERROR   → end with: GIVE_UP"""

# 3. Hike Selector : chooses “best” subset (top 5)
instruct_select = """You are a decision agent.
## INPUT
JSON array of hikes with details.
## TASK
Score hikes by overall interest (scenery + variety) and choose the TOP 5.
Return the filtered JSON array (still 5 objects) – nothing else.

## COMPLETION RULES
SUCCESS → end with: SELECTION_COMPLETE
FAILURE → end with: SELECTION_FAILURE
ERROR   → end with: GIVE_UP"""

# 4. CSV Maker : converts JSON to CSV
instruct_csv = """You are a formatting agent that converts JSON → CSV.
## COLUMNS
name,description,location,distance_km,duration_h
Return ONLY valid CSV text (header + 5 rows).

## COMPLETION RULES
SUCCESS → end with: CSV_COMPLETE
FAILURE → end with: CSV_FAILURE
ERROR   → end with: GIVE_UP"""

# 5. Quality Checker : validates CSV integrity
instruct_validate = """You are a QA agent.
## TASK
Validate that the incoming CSV:
- contains exactly 6 rows (header + 5 data rows)
- has no empty fields
If VALID → reply 'VALIDATION_PASS'
If INVALID but fixable → explain issue + suggestions, end 'VALIDATION_FAIL'
If unrecoverable error → explain & end 'GIVE_UP'"""

# ---------------------------------------
# 2️⃣  CREATE SMOLAGENTS WITH PROPER TOOLS
# ---------------------------------------
agent_research  = SmolAgentFactory(instruct_research,  BROWSER_TOOLS)
agent_extract   = SmolAgentFactory(instruct_extract,   BROWSER_TOOLS)
agent_select    = SmolAgentFactory(instruct_select,    SHELL_TOOLS)   # simple text ops OK
agent_csv       = SmolAgentFactory(instruct_csv,       CSV_TOOLS)
agent_validate  = SmolAgentFactory(instruct_validate,  SHELL_TOOLS)

# --------------------------------------------------------
# 3️⃣  BUILD THE WORKFLOW GRAPH WITH ROBUST ROUTING LOGIC
# --------------------------------------------------------
workflow = StateGraph(WorkflowState)

# ----- Helper: generic router ------------------------------------------------
def make_router(expected_token: str, next_step: str, retry_step: str):
    """
    Creates a routing function that looks for an expected success token.
    Returns:
        next_step     – on detected success
        retry_step    – on normal failure (≤3 attempts)
        'end_failure' – on GIVE_UP or too many retries
    """
    def _router(state: WorkflowState) -> str:
        try:
            answers      = state.get("answers", [])
            steps_sofar  = state.get("step_name", [])
            current_name = steps_sofar[-1] if steps_sofar else "unknown"
            attempts     = steps_sofar.count(current_name)

            last_answer  = answers[-1] if answers else ""
            lowered      = last_answer.upper()

            if expected_token in lowered:
                return next_step

            if "GIVE_UP" in lowered:
                return "end_failure"

            # Normal failure path
            if attempts < 3:
                return retry_step
            else:                         # exceeded retries
                return "end_failure"
        except Exception as e:
            print(f"⚠️ Router error ({current_name}): {e}")
            return "end_failure"
    return _router
# -----------------------------------------------------------------------------


# -------------- Add Agent Nodes ---------------------------------------------
workflow.add_node("web_researcher",  WorkflowNodeFactory.create_agent_node(agent_research))
workflow.add_node("detail_extractor",WorkflowNodeFactory.create_agent_node(agent_extract))
workflow.add_node("hike_selector",   WorkflowNodeFactory.create_agent_node(agent_select))
workflow.add_node("csv_maker",       WorkflowNodeFactory.create_agent_node(agent_csv))
workflow.add_node("quality_checker", WorkflowNodeFactory.create_agent_node(agent_validate))

# -------------- START edge ---------------------------------------------------
workflow.add_edge(START, "web_researcher")

# -------------- Conditional edges with fallbacks -----------------------------
workflow.add_conditional_edges(
    "web_researcher",
    make_router("RESEARCH_COMPLETE", next_step="detail_extractor", retry_step="web_researcher"),
    {
        "detail_extractor": "detail_extractor",
        "web_researcher":   "web_researcher",  # retry
        "end_failure":      END
    }
)

workflow.add_conditional_edges(
    "detail_extractor",
    make_router("EXTRACTION_COMPLETE", next_step="hike_selector", retry_step="detail_extractor"),
    {
        "hike_selector":    "hike_selector",
        "detail_extractor": "detail_extractor",  # retry
        "end_failure":      END
    }
)

workflow.add_conditional_edges(
    "hike_selector",
    make_router("SELECTION_COMPLETE", next_step="csv_maker", retry_step="hike_selector"),
    {
        "csv_maker":   "csv_maker",
        "hike_selector": "hike_selector",       # retry
        "end_failure": END
    }
)

workflow.add_conditional_edges(
    "csv_maker",
    make_router("CSV_COMPLETE", next_step="quality_checker", retry_step="csv_maker"),
    {
        "quality_checker": "quality_checker",
        "csv_maker":       "csv_maker",         # retry
        "end_failure":     END
    }
)

workflow.add_conditional_edges(
    "quality_checker",
    make_router("VALIDATION_PASS", next_step=END, retry_step="csv_maker"),
    {
        END:             END,                   # success
        "csv_maker":     "csv_maker",           # fix & retry CSV
        "end_failure":   END
    }
)

# -------------- END node automatically added by the library ------------------
# -----------------------------------------------------------------------------


# ------------------------- COMPILE APP ---------------------------------------
app = workflow.compile()

######################################################################
# The graph now contains:                                            #
#  • 5 specialised SmolAgents (research → extract → select → csv → QA)#
#  • Multi-level retry (≤3) + GIVE_UP → early END                    #
#  • At least two fallback routes in every router                    #
######################################################################