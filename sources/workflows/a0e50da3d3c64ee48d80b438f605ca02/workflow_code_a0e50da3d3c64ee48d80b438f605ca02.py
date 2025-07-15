# ======================================================================
# LangGraph – SmolAgent workflow : Install `llama.cpp` on THIS machine
# ======================================================================
#
#  • 5 atomic SmolAgents
#  • 3 independent routing functions with retry + fallback paths
#  • 2 different fallback mechanisms (auto-retry & manual fallback agent)
#
# ----------------------------------------------------------------------

# Built-ins already provided by execution environment
from langgraph.graph import StateGraph, START, END

# ----------------------------------------------------------------------
# 1️⃣  Workflow initialisation
# ----------------------------------------------------------------------
workflow = StateGraph(WorkflowState)            # <-  State schema already exists

# ----------------------------------------------------------------------
# 2️⃣  SmolAgent DEFINITION (prompt + tools + instantiate)
# ----------------------------------------------------------------------

# ---- Agent-1 : WEB SEARCH ------------------------------------------------
search_prompt = """
You are an internet research specialist.

GOAL
Find the most up-to-date, architecture-specific installation instructions
for compiling and installing `llama.cpp` on the CURRENT machine.

OUTPUT FORMAT
If you find COMPLETE instructions:
    SEARCH_SUCCESS: <bullet-list of sources + short explanation>
If information is incomplete:
    SEARCH_FAILURE: <what is missing, what you tried>
On technical error:
    GIVE_UP: <error explanation>

Include links, exact commands, and mention required dependencies.
"""

search_agent = SmolAgentFactory(
    "search_agent",
    search_prompt,
    WEB_BROWSER_MCP_TOOLS,          # Browser toolkit
)

# ---- Agent-2 : REQUIREMENT PARSER ---------------------------------------
parse_prompt = """
You are a requirements extraction agent.

INPUT
<<CONTEXT>>

TASK
Extract from the context:
  • list of OS dependencies
  • git clone command for llama.cpp
  • build / compile commands (make …)
  • post-installation verification command (`./main –h` or similar)

OUTPUT FORMAT
If everything present:
    PARSE_SUCCESS: <json with keys: deps, clone_cmd, build_cmds, test_cmd>
Otherwise:
    PARSE_FAILURE: <missing items>
"""

parse_agent = SmolAgentFactory(
    "parse_agent",
    parse_prompt,
    CSV_MANAGEMENT_TOOLS           # simple parsing / csv tools (minimal)
)

# ---- Agent-3 : SCRIPT GENERATOR -----------------------------------------
script_prompt = """
You are a bash script generator.

INPUT
<<CONTEXT>>

TASK
Generate a SAFE bash script that:
  1. Installs dependencies (use package manager apt/yum/brew as detected)
  2. Clones llama.cpp to $HOME/llama.cpp
  3. Builds with optimal parameters for detected architecture
  4. Runs verification command

OUTPUT FORMAT
If script ready:
    SCRIPT_SUCCESS: <```bash\n…\n```>
If context insufficient:
    SCRIPT_INSUFFICIENT_INFO: <explain what is missing>
If generation error:
    SCRIPT_FAILURE: <error reason>
"""

script_agent = SmolAgentFactory(
    "script_agent",
    script_prompt,
    BASH_COMMAND_MCP_TOOLS
)

# ---- Agent-4 : EXECUTOR --------------------------------------------------
exec_prompt = """
You are an execution agent.

INPUT
<<CONTEXT>>  (bash script)

TASK
Execute the provided script step-by-step.
Capture stdout/stderr, summarise results.

OUTPUT FORMAT
If every command executed without non-zero exit code:
    EXEC_SUCCESS: <summary & logs>
If any command failed:
    EXEC_FAILURE: <which step failed & logs>
On environment restriction preventing execution:
    EXEC_GIVE_UP: <reason>
"""

exec_agent = SmolAgentFactory(
    "exec_agent",
    exec_prompt,
    BASH_COMMAND_MCP_TOOLS
)

# ---- Agent-5 : MANUAL FALLBACK ------------------------------------------
manual_prompt = """
Automatic installation failed.

TASK
Provide a clear MANUAL step-by-step guide a human can follow to install
llama.cpp on this system, including troubleshooting tips.

OUTPUT FORMAT
    MANUAL_STEPS_PROVIDED: <comprehensive human guide>
"""

manual_agent = SmolAgentFactory(
    "manual_fallback_agent",
    manual_prompt,
    WEB_BROWSER_MCP_TOOLS          # still may fetch docs
)

# ----------------------------------------------------------------------
# 3️⃣  Create SmolAgent NODES
# ----------------------------------------------------------------------
workflow.add_node("search_agent",  WorkflowNodeFactory.create_agent_node(search_agent))
workflow.add_node("parse_agent",   WorkflowNodeFactory.create_agent_node(parse_agent))
workflow.add_node("script_agent",  WorkflowNodeFactory.create_agent_node(script_agent))
workflow.add_node("exec_agent",    WorkflowNodeFactory.create_agent_node(exec_agent))
workflow.add_node("manual_agent",  WorkflowNodeFactory.create_agent_node(manual_agent))

# ----------------------------------------------------------------------
# 4️⃣  ROUTING FUNCTIONS (retry + fallback + logging)
# ----------------------------------------------------------------------
MAX_RETRY = 3   # hard limit per agent


def _count_attempts(state: WorkflowState, agent_name: str) -> int:
    """Helper: how many times an agent already executed"""
    names = state.get("step_name", [])
    return sum(1 for n in names if n.startswith(agent_name))


# ---- Router after SEARCH -------------------------------------------------
def route_after_search(state: WorkflowState) -> str:
    print("🔀  Routing after SEARCH agent")
    try:
        answer = state["answers"][-1] if state.get("answers") else ""
        attempts = _count_attempts(state, "search_agent")
        if "SEARCH_SUCCESS" in answer:
            return "parse_agent"
        elif attempts < MAX_RETRY:
            print(f"🔄  Retrying search (attempt {attempts+1}/{MAX_RETRY})")
            return "search_agent"
        else:
            print("⚠️  Search failed too many times – switching to manual guide")
            return "manual_agent"
    except Exception as e:
        print(f"🚨  Routing error after search: {e}")
        return "manual_agent"


# ---- Router after PARSE --------------------------------------------------
def route_after_parse(state: WorkflowState) -> str:
    print("🔀  Routing after PARSE agent")
    try:
        answer = state["answers"][-1] if state.get("answers") else ""
        attempts = _count_attempts(state, "parse_agent")
        if "PARSE_SUCCESS" in answer:
            return "script_agent"
        elif "PARSE_FAILURE" in answer and attempts < MAX_RETRY:
            print(f"🔄  Missing data – returning to SEARCH for more info (parse attempt {attempts})")
            return "search_agent"
        else:
            print("⚠️  Parsing unrecoverable – manual fallback")
            return "manual_agent"
    except Exception as e:
        print(f"🚨  Routing error after parse: {e}")
        return "manual_agent"


# ---- Router after SCRIPT GENERATION -------------------------------------
def route_after_script(state: WorkflowState) -> str:
    print("🔀  Routing after SCRIPT agent")
    try:
        answer = state["answers"][-1] if state.get("answers") else ""
        attempts = _count_attempts(state, "script_agent")
        if "SCRIPT_SUCCESS" in answer:
            return "exec_agent"
        elif "SCRIPT_INSUFFICIENT_INFO" in answer and attempts < MAX_RETRY:
            print("🔄  Script missing info – go back to PARSE")
            return "parse_agent"
        elif attempts < MAX_RETRY:
            print(f"🔄  Regenerating script (attempt {attempts+1}/{MAX_RETRY})")
            return "script_agent"
        else:
            print("⚠️  Script generation failed repeatedly – manual fallback")
            return "manual_agent"
    except Exception as e:
        print(f"🚨  Routing error after script: {e}")
        return "manual_agent"


# ---- Router after EXECUTION ---------------------------------------------
def route_after_exec(state: WorkflowState) -> str:
    print("🔀  Routing after EXEC agent")
    try:
        answer = state["answers"][-1] if state.get("answers") else ""
        if "EXEC_SUCCESS" in answer:
            return END
        elif "EXEC_FAILURE" in answer:
            print("⚠️  Execution failed – manual fallback.")
            return "manual_agent"
        else:
            print("❔  Unrecognised exec result – finishing for safety.")
            return END
    except Exception as e:
        print(f"🚨  Routing error after exec: {e}")
        return END

# ----------------------------------------------------------------------
# 5️⃣  EDGE DEFINITIONS
# ----------------------------------------------------------------------
workflow.add_edge(START, "search_agent")

#  • From SEARCH agent
workflow.add_conditional_edges(
    "search_agent",
    route_after_search,
    {
        "parse_agent": "parse_agent",
        "search_agent": "search_agent",      # retry
        "manual_agent": "manual_agent"
    }
)

#  • From PARSE agent
workflow.add_conditional_edges(
    "parse_agent",
    route_after_parse,
    {
        "script_agent": "script_agent",
        "search_agent": "search_agent",      # gather more info
        "manual_agent": "manual_agent"
    }
)

#  • From SCRIPT agent
workflow.add_conditional_edges(
    "script_agent",
    route_after_script,
    {
        "exec_agent": "exec_agent",
        "parse_agent": "parse_agent",        # back-prop for missing info
        "script_agent": "script_agent",      # regenerate
        "manual_agent": "manual_agent"
    }
)

#  • From EXEC agent
workflow.add_conditional_edges(
    "exec_agent",
    route_after_exec,
    {
        END: END,
        "manual_agent": "manual_agent"
    }
)

#  • From MANUAL FALLBACK -> always END
workflow.add_edge("manual_agent", END)

# ----------------------------------------------------------------------
# 6️⃣  Compile the graph
# ----------------------------------------------------------------------
install_llama_cpp_workflow = workflow.compile()

# The variable `install_llama_cpp_workflow` is now an executable LangGraph
# ----------------------------------------------------------------------