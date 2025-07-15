#######################################################################
# LANGGRAPH – SMOLAGENT WORKFLOW  (v1.2)   ←  Auto-improved iteration
#  • Fixes KeyError caused by retry routes missing in edge-mapping
#  • Keeps granular 6-agent decomposition + multi-level fallbacks
#######################################################################
# CONTEXT OBJECTS ALREADY IN MEMORY:
#   • WorkflowState            • SmolAgentFactory  • WorkflowNodeFactory
#   • Tool-packages: WEB_BROWSER_MCP_TOOLS / BASH_COMMAND_MCP_TOOLS /
#                    CSV_MANAGEMENT_TOOLS  / R_COMMAND_MCP_TOOLS
from langgraph.graph import StateGraph, START, END      # ← provided

#######################################################################
# 1.  PROMPTS  (strict 1 atomic skill each — unchanged)
#######################################################################
prompt_research = """
You are a specialised WEB-RESEARCH agent.

GOAL
1. Find reliable, up-to-date instructions to install MassCube
2. Locate at least one freely-available example dataset for MassCube

OUTPUT (MANDATORY)
Return a concise, numbered list:
  1) Installation steps / dependencies
  2) Example-dataset download link(s) + one-line description

FINISH WITH ONE EXACT TOKEN ON LAST LINE
TASK_SUCCESS:   – all info complete
TASK_FAILURE:   – searched exhaustively but missing info
GIVE_UP:        – blocked site / tool error
"""

prompt_installer = """
You are a secure BASH-EXECUTION agent.

INPUT = previous agent’s installation instructions.
Convert them into safe bash commands with execute_command (one-by-one).
Never use sudo / destructive ops. On failure retry once with safe variant.

Report summary of what was installed & where.

FINISH TOKEN
TASK_SUCCESS: / TASK_FAILURE: / GIVE_UP:
"""

prompt_dataset_dl = """
You are a DATASET-DOWNLOAD agent.

Download the MassCube example dataset link(s) supplied earlier.
Validate link, download, and provide:
  • local file path
  • file size

FINISH TOKEN
TASK_SUCCESS: / TASK_FAILURE: / GIVE_UP:
"""

prompt_csv_loader = """
You are a CSV-MANAGEMENT agent.

Load the downloaded CSV or TSV into storage.
Return:
  • storage dataset name
  • rows × cols
  • preview of first 3 rows (query_csv)

FINISH TOKEN
TASK_SUCCESS: / TASK_FAILURE: / GIVE_UP:
"""

prompt_r_runner = """
You are an R-COMMAND agent (MassCube focus).

Steps:
 1. library(MassCube)
 2. Import dataset from CSV storage
 3. Run ONE illustrative MassCube analysis (e.g. PCA)
 4. Save any plot / summary to file

Return executed R code, output path, and a one-sentence interpretation.

FINISH TOKEN
TASK_SUCCESS: / TASK_FAILURE: / GIVE_UP:
"""

prompt_report = """
You are a MARKDOWN REPORT-GENERATOR agent.

Compose a short markdown report containing:
  • Installation summary
  • Dataset description
  • Analysis run & rationale
  • Key insight with link/path to output
  • Next-step recommendations

FINISH TOKEN
TASK_SUCCESS: / TASK_FAILURE: / GIVE_UP:
"""

#######################################################################
# 2.  CREATE SMOLAGENTS  (one tool-package each)
#######################################################################
agent_research     = SmolAgentFactory("researcher",  prompt_research,   WEB_BROWSER_MCP_TOOLS)
agent_installer    = SmolAgentFactory("installer",   prompt_installer,  BASH_COMMAND_MCP_TOOLS)
agent_dataset_dl   = SmolAgentFactory("downloader",  prompt_dataset_dl, WEB_BROWSER_MCP_TOOLS)
agent_csv_loader   = SmolAgentFactory("csv_loader",  prompt_csv_loader, CSV_MANAGEMENT_TOOLS)
agent_r_runner     = SmolAgentFactory("r_runner",    prompt_r_runner,   R_COMMAND_MCP_TOOLS)
agent_report       = SmolAgentFactory("reporter",    prompt_report,     R_COMMAND_MCP_TOOLS)

#######################################################################
# 3.  WORKFLOW INITIALISATION & NODE REGISTRATION
#######################################################################
workflow = StateGraph(WorkflowState)

workflow.add_node("research",    WorkflowNodeFactory.create_agent_node(agent_research))
workflow.add_node("installer",   WorkflowNodeFactory.create_agent_node(agent_installer))
workflow.add_node("dataset_dl",  WorkflowNodeFactory.create_agent_node(agent_dataset_dl))
workflow.add_node("csv_loader",  WorkflowNodeFactory.create_agent_node(agent_csv_loader))
workflow.add_node("r_runner",    WorkflowNodeFactory.create_agent_node(agent_r_runner))
workflow.add_node("report",      WorkflowNodeFactory.create_agent_node(agent_report))

#######################################################################
# 4.  ROUTING HELPERS  (retry-aware, robust)
#######################################################################
MAX_RETRIES = 3         # per node
FATAL_ROUTE = "!!END"   # maps to END

def _last_answer(state: WorkflowState) -> str:
    return state.get("answers", [])[-1] if state.get("answers") else ""

def _retry_count(state: WorkflowState, step: str) -> int:
    return sum(1 for s in state.get("step_name", []) if s.startswith(step))

def router_factory(step: str, ok: str, fallback: str):
    """
    Creates a router for <step>.
    Possible returns: ok / fallback / step (for retry) / FATAL_ROUTE
    """
    def _router(state: WorkflowState):
        try:
            msg     = _last_answer(state)
            retries = _retry_count(state, step)

            # ---------- SUCCESS ----------
            if "TASK_SUCCESS:" in msg:
                print(f"✅ {step} → {ok}")
                return ok

            # ---------- FAILURE ----------
            if "TASK_FAILURE:" in msg or not msg.strip():
                if retries < MAX_RETRIES:
                    print(f"🔄 {step} retry {retries+1}/{MAX_RETRIES}")
                    state["step_name"].append(f"{step}_retry")
                    return step                      # ← self-route for retry
                else:
                    print(f"⚠️  {step} exhausted retries → {fallback}")
                    return fallback

            # ---------- GIVE_UP ----------
            if "GIVE_UP:" in msg:
                print(f"⛔ {step} gave up → END")
                return FATAL_ROUTE

            # ---------- UNEXPECTED ----------
            print(f"❓ {step} missing marker – treat as failure")
            if retries < MAX_RETRIES:
                state["step_name"].append(f"{step}_retry")
                return step
            else:
                return fallback

        except Exception as e:
            print(f"🚨 Router exception in {step}: {e}")
            return FATAL_ROUTE
    return _router

#######################################################################
# 5.  EDGE-MAPPING UTILITY (now injects <self_step> automatically)
#######################################################################
def _edge_map(step_self: str, *routes):
    """
    Build a mapping dict for add_conditional_edges.
    Ensures:
      • self-step is always routable (for retry)
      • FATAL_ROUTE and END always resolvable
    """
    mapping = {r: r for r in routes}
    mapping[step_self] = step_self            # ← fix for retry KeyError
    mapping[FATAL_ROUTE] = END
    mapping[END] = END
    return mapping

#######################################################################
# 6.  EDGE DEFINITIONS  (with repaired mapping)
#######################################################################
workflow.add_edge(START, "research")

# research  ──► installer
workflow.add_conditional_edges(
    "research",
    router_factory("research", "installer", END),
    _edge_map("research", "installer")
)

# installer ──► dataset_dl  (fallback → research)
workflow.add_conditional_edges(
    "installer",
    router_factory("installer", "dataset_dl", "research"),
    _edge_map("installer", "dataset_dl", "research")
)

# dataset_dl ──► csv_loader  (fallback → research)
workflow.add_conditional_edges(
    "dataset_dl",
    router_factory("dataset_dl", "csv_loader", "research"),
    _edge_map("dataset_dl", "csv_loader", "research")
)

# csv_loader ──► r_runner  (fallback → dataset_dl)
workflow.add_conditional_edges(
    "csv_loader",
    router_factory("csv_loader", "r_runner", "dataset_dl"),
    _edge_map("csv_loader", "r_runner", "dataset_dl")
)

# r_runner ──► report  (fallback → csv_loader)
workflow.add_conditional_edges(
    "r_runner",
    router_factory("r_runner", "report", "csv_loader"),
    _edge_map("r_runner", "report", "csv_loader")
)

# report ──► END
workflow.add_conditional_edges(
    "report",
    router_factory("report", END, END),
    _edge_map("report")
)

#######################################################################
# 7.  COMPILE WORKFLOW
#######################################################################
app = workflow.compile()

#######################################################################
#  The compiled `app` is now ready and the retry-route KeyError is fixed.
#  Invoke with an initial WorkflowState, e.g.:
#     init_state: WorkflowState = {
#         "step_name": [],
#         "actions": [],
#         "observations": [],
#         "answers": [],
#         "success": []
#     }
#     final_state = app.invoke(init_state)
#######################################################################