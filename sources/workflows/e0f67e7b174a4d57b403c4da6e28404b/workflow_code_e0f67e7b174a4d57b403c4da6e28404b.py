# ============================================================
# LangGraph – SmolAgent WORKFLOW
# Task: “Create a mock CSV file containing a list of random names”
# ============================================================

# PRE-EXISTING OBJECTS (already available in the interpreter):
#   ‑ WorkflowState  (TypedDict)
#   ‑ SmolAgentFactory
#   ‑ WorkflowNodeFactory
#   ‑ Tool packages :  WEB_BROWSER_MCP_TOOLS, R_COMMAND_MCP_TOOLS,
#                      CSV_MANAGEMENT_TOOLS,  BASH_COMMAND_MCP_TOOLS
# DO **NOT** REDECLARE THEM !

# ------------------------------------------------------------
# MANDATORY imports
# ------------------------------------------------------------
from langgraph.graph import StateGraph, START, END

# ------------------------------------------------------------
# 1)   AGENT PROMPTS (ONE atomic responsibility each)
# ------------------------------------------------------------

# A) Random-name generator  (R tools)
instruct_name_gen = """You are an R-scripting agent that must generate a list of
50 random, realistic first names.

## TASK
1. Use R code only (tool: execute_r_code) to create a character
   vector containing 50 unique first names.
2. Return the vector in plain text, one name per line.

## COMPLETION PROTOCOL
SUCCESS  → end with:  GENERATION_COMPLETE
FAILURE  → end with:  GENERATION_FAILURE
ERROR    → end with:  GIVE_UP

Always output the names BEFORE the completion tag so downstream
agents can parse them."""
# ------------------------------------------------------------

# B) CSV creator  (CSV tools)
instruct_csv_creator = """You are a CSV-management agent.

## TASK
1. Receive a plain-text list of names from the previous step.
2. Create a CSV dataset called “random_names.csv” with ONE column
   `name` and one row per name.  Use the tool create_csv.
3. Save it to the default storage path.

## COMPLETION PROTOCOL
SUCCESS  → end with:  CSV_CREATION_COMPLETE
FAILURE  → end with:  CSV_CREATION_FAILURE
ERROR    → end with:  GIVE_UP

Only the completion tag must be on the last line."""
# ------------------------------------------------------------

# C) CSV validator  (CSV tools – atomic responsibility: verify)
instruct_csv_validator = """You are a validation agent.

## TASK
1. Load the file random_names.csv (tool: load_csv_from_path).
2. Verify that:
   •      the file exists
   •      it has exactly 50 rows
   •      the column name is 'name'
3. Report findings.

## COMPLETION PROTOCOL
SUCCESS  → end with:  CSV_VALIDATION_SUCCESS
INSUFFICIENT_INFORMATION → end with:  CSV_VALIDATION_INSUFFICIENT
FAILURE  → end with:  CSV_VALIDATION_FAILURE
ERROR    → end with:  GIVE_UP"""
# ------------------------------------------------------------

# D) Reporter / summariser (no external tools needed)
instruct_reporter = """You are a reporting agent.

## TASK
Summarise the whole workflow for the user:
– Mention whether the CSV was created and validated
– Provide the storage path
– Show the first 5 names as a preview

## COMPLETION PROTOCOL
Always finish with:  WORKFLOW_DONE
"""

# ------------------------------------------------------------
# 2)   AGENT CREATION
# ------------------------------------------------------------
agent_name_gen   = SmolAgentFactory("name_generator",   instruct_name_gen,   R_COMMAND_MCP_TOOLS)
agent_csv_create = SmolAgentFactory("csv_creator",      instruct_csv_creator, CSV_MANAGEMENT_TOOLS)
agent_csv_check  = SmolAgentFactory("csv_validator",    instruct_csv_validator, CSV_MANAGEMENT_TOOLS)
agent_report     = SmolAgentFactory("reporter",         instruct_reporter,   [])   # no tool needed

# ------------------------------------------------------------
# 3)   WORKFLOW INITIALISATION
# ------------------------------------------------------------
workflow = StateGraph(WorkflowState)

# Add nodes
workflow.add_node("generate_names", WorkflowNodeFactory.create_agent_node(agent_name_gen))
workflow.add_node("create_csv",     WorkflowNodeFactory.create_agent_node(agent_csv_create))
workflow.add_node("validate_csv",   WorkflowNodeFactory.create_agent_node(agent_csv_check))
workflow.add_node("report_result",  WorkflowNodeFactory.create_agent_node(agent_report))

# ------------------------------------------------------------
# 4)   ROUTING FUNCTIONS  (robust, with retries & fallbacks)
# ------------------------------------------------------------
MAX_RETRIES = 3   # hard limit per atomic step


def _retry_count(state: WorkflowState, step: str) -> int:
    """Helper – how many times has <step> occurred so far?"""
    return state.get("step_name", []).count(step)


# ----------  R1 : after name generation  --------------------
def route_after_generation(state: WorkflowState) -> str:
    try:
        raw_ans = state.get("answers", [])
        last_ans = raw_ans[-1] if raw_ans else ""
        
        # SUCCESS path
        if "GENERATION_COMPLETE" in last_ans:
            return "create_csv"
        
        # FAILURE → retry if budget left
        if _retry_count(state, "generate_names") < MAX_RETRIES:
            print("🔄  Regenerating names …")
            return "generate_names"
        
        # Hard failure
        print("❌  Name generation failed too many times.")
        return END
    except Exception as e:
        print(f"🚨 Routing error (generation) : {e}")
        return END


# ----------  R2 : after CSV creation  -----------------------
def route_after_csv_creation(state: WorkflowState) -> str:
    try:
        last_ans = state.get("answers", [])[-1] if state.get("answers") else ""
        
        if "CSV_CREATION_COMPLETE" in last_ans:
            return "validate_csv"
        
        # maybe names were wrong – go all the way back
        if "CSV_CREATION_FAILURE" in last_ans or "INSUFFICIENT" in last_ans:
            if _retry_count(state, "generate_names") < MAX_RETRIES:
                print("🔄  Going back to regenerate names …")
                return "generate_names"
        
        # Retry the same step if quota left
        if _retry_count(state, "create_csv") < MAX_RETRIES:
            print("🔄  Retrying CSV creation …")
            return "create_csv"
        
        print("❌  CSV creation unrecoverable.")
        return END
    except Exception as e:
        print(f"🚨 Routing error (csv creation): {e}")
        return END


# ----------  R3 : after CSV validation  ---------------------
def route_after_validation(state: WorkflowState) -> str:
    try:
        last_ans = state.get("answers", [])[-1] if state.get("answers") else ""
        
        if "CSV_VALIDATION_SUCCESS" in last_ans:
            return "report_result"
        
        if "CSV_VALIDATION_INSUFFICIENT" in last_ans:
            # Something fundamental wrong – go back 2 steps to generator
            if _retry_count(state, "generate_names") < MAX_RETRIES:
                print("🔄  Insufficient information – back to name generation.")
                return "generate_names"
        
        # Normal failure → retry validator
        if _retry_count(state, "validate_csv") < MAX_RETRIES:
            print("🔄  Retrying CSV validation …")
            return "validate_csv"
        
        print("❌  CSV validation failed irrecoverably.")
        return END
    except Exception as e:
        print(f"🚨 Routing error (validation): {e}")
        return END


# ----------  R4 : after reporting  --------------------------
def route_after_report(state: WorkflowState) -> str:
    return END   # Always finish here


# ------------------------------------------------------------
# 5)   EDGE DEFINITIONS
# ------------------------------------------------------------
workflow.add_edge(START, "generate_names")

workflow.add_conditional_edges(
    "generate_names",
    route_after_generation,
    {
        "create_csv":      "create_csv",
        "generate_names":  "generate_names",    # self-loop retry
        END:               END
    }
)

workflow.add_conditional_edges(
    "create_csv",
    route_after_csv_creation,
    {
        "validate_csv":    "validate_csv",
        "generate_names":  "generate_names",    # jump back two steps
        "create_csv":      "create_csv",        # retry
        END:               END
    }
)

workflow.add_conditional_edges(
    "validate_csv",
    route_after_validation,
    {
        "report_result":   "report_result",
        "generate_names":  "generate_names",    # long jump back
        "validate_csv":    "validate_csv",      # retry
        END:               END
    }
)

workflow.add_conditional_edges(
    "report_result",
    route_after_report,
    { END: END }
)

# ------------------------------------------------------------
# 6)   COMPILE WORKFLOW
# ------------------------------------------------------------
app = workflow.compile()