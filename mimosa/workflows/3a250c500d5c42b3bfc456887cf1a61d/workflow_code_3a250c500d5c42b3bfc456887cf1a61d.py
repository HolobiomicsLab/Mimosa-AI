###############################################################
#  LANGGRAPH – SMOLAGENT  WORKFLOW :   CHECK DISK SPACE
###############################################################
# PREREQUISITES (already provided in execution environment)
# - WorkflowState   (TypedDict)
# - StateGraph, START, END         from langgraph
# - SmolAgentFactory               (creates SmolAgent instances)
# - WorkflowNodeFactory            (wraps SmolAgent as graph nodes)
# - CSV_TOOL_TOOLS, BROWSER_TOOL_TOOLS, SHELL_TOOL_TOOLS
###############################################################

from langgraph.graph import StateGraph, START, END         # noqa: F401  (already available)

################################################################
# 1)  CREATE ALL AGENTS (4 atomic SmolAgents, 1 function node)
################################################################

# ────────────────────────────  A)   OS-PROBE AGENT  ────────────────────────────
os_probe_prompt = """
You are an operating-system probe agent.

## YOUR TASK
1. Identify the underlying operating system type (Linux, macOS, Windows, or Other).
2. Use ONLY shell commands from the provided tool-set to discover the OS.
   • Recommended commands: `uname -s`, `cat /etc/os-release`, `ver`, etc.

## COMPLETION PROTOCOL
SUCCESS:  When you are certain of the OS, call final_answer with:
OS_DETECTED: <one_word_os_name>

FAILURE:   If shell commands fail or result is unclear, call final_answer with:
OS_DETECTION_FAILURE: <what went wrong>

ERROR:     For genuine tool errors, call final_answer with:
GIVE_UP: <technical details>
"""
os_probe_agent = SmolAgentFactory(os_probe_prompt, SHELL_TOOL_TOOLS)

# ────────────────────────────  B)   DISK-FETCH AGENT  ──────────────────────────
disk_fetch_prompt = """
You are a disk-space retrieval agent.

## CONTEXT
The previous step has supplied you with an OS name inside the workflow state under `answers[-1]`.
Choose ONE shell command appropriate for that OS to get the disk-usage summary:
• Linux/macOS → `df -h`
• Windows     → `wmic logicaldisk get size,freespace,caption`
Run the command via the SHELL tool.

## COMPLETION PROTOCOL
SUCCESS:  On success, call final_answer with:
DISK_FETCHED:
<raw_command_output_here>

FAILURE:  If the command executes but output is empty/invalid, call:
DISK_FETCH_FAILURE: <explanation>

ERROR:    For genuine tool errors, call:
GIVE_UP: <technical details>
"""
disk_fetch_agent = SmolAgentFactory(disk_fetch_prompt, SHELL_TOOL_TOOLS)

# ────────────────────────────  C)   WEB-SEARCH (FALLBACK) AGENT  ───────────────
web_search_prompt = """
You are an online research fallback agent.

## YOUR TASK
Search the web for how to check available disk space for the supplied OS type
(found in workflow state `answers[-1]`).  Produce one or two shell commands that
the Disk-Fetch agent could try next.

## COMPLETION PROTOCOL
SUCCESS → final_answer(
SEARCH_COMPLETE:
<suggested_shell_commands>
)

FAILURE → final_answer(
SEARCH_FAILURE:
<explanation_of_what_was_tried_and_why_it_failed>
)

ERROR  → final_answer(
GIVE_UP:
<technical_error_details>
)
"""
web_search_agent = SmolAgentFactory(web_search_prompt, BROWSER_TOOL_TOOLS)

# ────────────────────────────  D)   REPORTER AGENT  ────────────────────────────
report_prompt = """
You are the final reporting agent.

## YOUR TASK
1. Read the workflow state's latest parsed disk-space string (answers[-1]).
2. Transform it into a clear, human-readable summary.

## COMPLETION PROTOCOL
On successful summary output:
REPORT_COMPLETE:
<well-formatted_summary>

If input is insufficient:
REPORT_INSUFFICIENT_DATA:
<describe_missing_information>
"""
report_agent = SmolAgentFactory(report_prompt, CSV_TOOL_TOOLS)   # we just pick a benign tool package

# ────────────────────────────  E)   PARSE FUNCTION NODE  ───────────────────────
def parse_disk_output(state: WorkflowState) -> WorkflowState:
    """
    Extracts one representative line from raw disk-usage output and stores it
    back into state['answers'] while appending success flag.
    """
    try:
        raw_answers = state.get("answers", [])
        output = raw_answers[-1] if raw_answers else ""
        parsed_line = "UNPARSED"
        success_flag = False

        # very naive parsing: choose first line that contains '%' OR 'FreeSpace'
        for ln in output.splitlines():
            if "%" in ln or "FreeSpace" in ln or "Available" in ln:
                parsed_line = ln.strip()
                success_flag = True
                break

        # update state
        state.setdefault("step_name", []).append("parse_disk_output")
        state.setdefault("answers", []).append(parsed_line)
        state.setdefault("success", []).append(success_flag)
        state.setdefault("rewards", []).append(1.0 if success_flag else 0.0)
    except Exception as e:
        state.setdefault("step_name", []).append("parse_disk_output")
        state.setdefault("answers", []).append(f"PARSE_EXCEPTION: {e}")
        state.setdefault("success", []).append(False)
        state.setdefault("rewards", []).append(0.0)
    return state

################################################################
# 2)  CONDITIONAL ROUTERS WITH ROBUST ERROR HANDLING
################################################################
def route_after_os_probe(state: WorkflowState) -> str:
    """
    Decide next step after OS probe.
    """
    try:
        answer = state.get("answers", [""])[-1]
        if "OS_DETECTED" in answer:
            return "disk_fetch"
        elif "OS_DETECTION_FAILURE" in answer:
            # fallback to browser search for guidance
            return "web_search"
        else:
            # treat any other output as error & retry up to 3 times
            retries = state.get("step_name", []).count("os_probe")
            return "os_probe" if retries < 3 else "web_search"
    except Exception:
        return "web_search"

def route_after_web_search(state: WorkflowState) -> str:
    """
    Decide what to do after web search fallback.
    """
    try:
        answer = state.get("answers", [""])[-1]
        if "SEARCH_COMPLETE" in answer:
            # give Disk-Fetch another shot (it will read suggested commands)
            return "disk_fetch"
        elif "SEARCH_FAILURE" in answer:
            return END
        else:
            retries = state.get("step_name", []).count("web_search")
            return "web_search" if retries < 2 else END
    except Exception:
        return END

def route_after_disk_fetch(state: WorkflowState) -> str:
    """
    Decide next step after trying to fetch disk statistics.
    """
    try:
        answer = state.get("answers", [""])[-1]
        if "DISK_FETCHED" in answer:
            return "parser"
        elif "DISK_FETCH_FAILURE" in answer:
            # attempt browser search for alternative commands
            return "web_search"
        else:
            retries = state.get("step_name", []).count("disk_fetch")
            return "disk_fetch" if retries < 3 else "web_search"
    except Exception:
        return "web_search"

def route_after_parser(state: WorkflowState) -> str:
    """
    Send to reporter if parsing succeeded, otherwise retry disk fetch with new commands (if any).
    """
    try:
        if state.get("success", [False])[-1]:
            return "reporter"
        else:
            retries = state.get("step_name", []).count("parser")
            return "disk_fetch" if retries < 2 else END
    except Exception:
        return END

def route_after_report(state: WorkflowState) -> str:
    """
    Terminate if reporting complete, otherwise fallback to END.
    """
    try:
        answer = state.get("answers", [""])[-1]
        if "REPORT_COMPLETE" in answer:
            return END
        else:
            return END
    except Exception:
        return END

################################################################
# 3)  BUILD THE WORKFLOW GRAPH
################################################################
workflow = StateGraph(WorkflowState)

# --- Add Nodes ---
workflow.add_node("os_probe",   WorkflowNodeFactory.create_agent_node(os_probe_agent))
workflow.add_node("disk_fetch", WorkflowNodeFactory.create_agent_node(disk_fetch_agent))
workflow.add_node("web_search", WorkflowNodeFactory.create_agent_node(web_search_agent))
workflow.add_node("reporter",   WorkflowNodeFactory.create_agent_node(report_agent))
workflow.add_node("parser",     parse_disk_output)   # plain function node

# --- Add Edges & Routing ---
workflow.add_edge(START, "os_probe")

workflow.add_conditional_edges(
    "os_probe",
    route_after_os_probe,
    {
        "disk_fetch": "disk_fetch",
        "web_search": "web_search",
        "os_probe":   "os_probe"     # self-loop retry
    }
)

workflow.add_conditional_edges(
    "web_search",
    route_after_web_search,
    {
        "disk_fetch": "disk_fetch",
        "web_search": "web_search",
        END: END
    }
)

workflow.add_conditional_edges(
    "disk_fetch",
    route_after_disk_fetch,
    {
        "parser":     "parser",
        "web_search": "web_search",
        "disk_fetch": "disk_fetch"
    }
)

workflow.add_conditional_edges(
    "parser",
    route_after_parser,
    {
        "reporter":  "reporter",
        "disk_fetch": "disk_fetch",
        END: END
    }
)

workflow.add_conditional_edges(
    "reporter",
    route_after_report,
    {
        END: END
    }
)

# --- Compile App ---
app = workflow.compile()

################################################################
# 4)  EXAMPLE EXECUTION (commented out; remove comments to run)
################################################################
# initial_state: WorkflowState = {
#     "step_name": [],
#     "actions": [],
#     "observations": [],
#     "rewards": [],
#     "answers": [],
#     "success": []
# }
# result_state = app.invoke(initial_state)
# print("FINAL STATE:", result_state)
################################################################