######################################################################
# SEARCH & INSTALL 𝓵𝓵𝓪𝓶𝓪.cpp  – LangGraph ⚡ SmolAgent WORKFLOW
#
# Goal :  Detect local OS architecture → gather precise install
#          procedure for llama.cpp → execute compilation / installation
#          → verify the binary works.
#
# - 6 ultra-focused agents (single atomic responsibility each)
# - 3 independent routing blocks with multi-level retry + emergency
#   fall-through on unrecoverable errors
# - min 2 explicit fallback mechanisms (alt_web_researcher, re-parse path)
######################################################################

from langgraph.graph import StateGraph, START, END

# -------- 1️⃣  BUILD GRAPH ----------------------------------------------------

workflow = StateGraph(WorkflowState)


# -------- 2️⃣  AGENT PROMPTS & DECLARATIONS ----------------------------------

# 2.1  ARCH DETECTOR  (one job: figure out current CPU architecture)
arch_prompt = """
You are an OS-architecture detection agent.

YOUR ONLY TOOL: execute_command

TASK
1. Run `uname -m` (and if needed `uname -s`) to identify the current CPU /
   OS architecture (e.g., x86_64, arm64, aarch64, etc.).
2. Return the architecture as one word.

COMPLETION PROTOCOL
SUCCESS  → end with: ARCH_DETECTED: <architecture>
FAILURE  → end with: ARCH_FAILURE: <why it failed>
ERROR    → end with: GIVE_UP: <technical reason>
"""
arch_agent     = SmolAgentFactory("arch_detector", arch_prompt, BASH_COMMAND_MCP_TOOLS)
workflow.add_node("arch_detector", WorkflowNodeFactory.create_agent_node(arch_agent))


# 2.2  PRIMARY WEB RESEARCHER
web_prompt = """
You are a focussed web-research agent.

INPUTS
Architecture (may appear in previous messages). Your mission is scoped only to
installation instructions for llama.cpp **for that architecture**.

TASK
1. Search the web for the most reliable, *up-to-date* instructions on building /
   installing llama.cpp for this architecture (GitHub README, blog posts, etc.).
2. Collect download/clone URL, required packages, compiler flags, cmake
   commands, and any architecture-specific tweaks.
3. Summarise steps clearly.

COMPLETION PROTOCOL
SUCCESS  → end with: RESEARCH_COMPLETE:
FAILURE  → end with: RESEARCH_FAILURE:
ERROR    → end with: GIVE_UP:
Always output a detailed summary before the trigger word.
"""
web_agent       = SmolAgentFactory("web_researcher", web_prompt, WEB_BROWSER_MCP_TOOLS)
workflow.add_node("web_researcher", WorkflowNodeFactory.create_agent_node(web_agent))


# 2.3  ALTERNATIVE WEB RESEARCHER  (fallback – broader query & different wording)
alt_web_prompt = """
You are a secondary web research agent, activated when the primary search failed.

Broaden the query:
- Search forums, issues, stack-overflow, Reddit threads, mirrors.
Follow the same completion protocol as the primary researcher.
"""
alt_web_agent = SmolAgentFactory("alt_web_researcher", alt_web_prompt, WEB_BROWSER_MCP_TOOLS)
workflow.add_node("alt_web_researcher", WorkflowNodeFactory.create_agent_node(alt_web_agent))


# 2.4  INSTRUCTION PARSER  (extract clean command list)
parse_prompt = """
You are an instruction-extraction agent.

TASK
Take the preceding research summary and output a STRICT, line-by-line bash
command list required to install llama.cpp on the detected architecture.

• Include prerequisite package install lines (apt, brew, pacman, etc.).
• Include git clone, cmake, make, copy/rename steps.
• Do NOT execute anything – only craft commands.

COMPLETION PROTOCOL
SUCCESS  → end with: INSTRUCTION_PARSED:
INSUFFICIENT_INFORMATION → end with: INSTRUCTION_FAILURE:
ERROR    → end with: GIVE_UP:
"""
parser_agent   = SmolAgentFactory("instruction_parser", parse_prompt, CSV_MANAGEMENT_TOOLS)
workflow.add_node("instruction_parser", WorkflowNodeFactory.create_agent_node(parser_agent))


# 2.5  INSTALLER  (executes commands)
install_prompt = """
You are an installation executor.

TASK
Run the bash commands provided (one by one) to build / install llama.cpp.

REQUIREMENTS
- Halt immediately if any command fails (non-zero exit).
- Capture and log full stdout/stderr for each command.

COMPLETION PROTOCOL
SUCCESS  → end with: INSTALL_SUCCESS:
FAILURE  → end with: INSTALL_FAILURE:
ERROR    → end with: GIVE_UP:
"""
install_agent  = SmolAgentFactory("installer", install_prompt, BASH_COMMAND_MCP_TOOLS)
workflow.add_node("installer", WorkflowNodeFactory.create_agent_node(install_agent))


# 2.6  VERIFIER  (checks binary)
verify_prompt = """
You are an installation verifier.

TASK
1. Check that the llama.cpp binary or executable exists (default build/bin or
   current directory).
2. Run `./llama --help` or `./main --help` depending on build artefact.
3. Report version / help text snippet.

COMPLETION PROTOCOL
SUCCESS  → end with: VERIFY_SUCCESS:
FAILURE  → end with: VERIFY_FAILURE:
ERROR    → end with: GIVE_UP:
"""
verify_agent = SmolAgentFactory("verifier", verify_prompt, BASH_COMMAND_MCP_TOOLS)
workflow.add_node("verifier", WorkflowNodeFactory.create_agent_node(verify_agent))


# -------- 3️⃣  ROUTING UTILITIES ---------------------------------------------

MAX_RETRIES = 3  # per step


def _retry_count(state: WorkflowState, step_key: str) -> int:
    """
    Helper – count how many times we've attempted <step_key>.
    """
    return sum(1 for s in state.get("step_name", []) if s.startswith(step_key))


# --- 3.1  ROUTER AFTER ARCH DETECTION ----------------------------------------
def route_arch(state: WorkflowState) -> str:
    print("🔀 [Router-ARCH] deciding …")
    try:
        answers = state.get("answers", [])
        if not answers:
            return "arch_detector"  # should not happen, but be safe

        last = answers[-1]

        if "ARCH_DETECTED" in last:
            state["success"].append(True)
            return "web_researcher"

        # retry logic
        if _retry_count(state, "arch_detector") < MAX_RETRIES:
            print("↻  Retrying arch detection")
            return "arch_detector"

        # unrecoverable
        print("💥 Arch detection unrecoverable → END")
        state["success"].append(False)
        return END
    except Exception as e:
        print(f"⚠️  Router-ARCH exception: {e}")
        return END


# --- 3.2  ROUTER AFTER WEB RESEARCH (primary or alt) -------------------------
def route_research(state: WorkflowState) -> str:
    print("🔀 [Router-RESEARCH] deciding …")
    try:
        answers = state.get("answers", [])
        if not answers:
            return "web_researcher"

        last = answers[-1]

        if "RESEARCH_COMPLETE" in last:
            state["success"].append(True)
            return "instruction_parser"

        if "RESEARCH_FAILURE" in last:
            # switch to alt researcher if we haven't tried yet
            if _retry_count(state, "web_researcher") < MAX_RETRIES:
                return "alt_web_researcher"
            elif _retry_count(state, "alt_web_researcher") < MAX_RETRIES:
                return "alt_web_researcher"
            else:
                print("😢 Research failed after retries → END")
                return END

        # If no expected keyword → treat as abnormal, retry once
        if _retry_count(state, state["step_name"][-1]) < MAX_RETRIES:
            return state["step_name"][-1]  # retry same researcher
        else:
            return END
    except Exception as e:
        print(f"⚠️  Router-RESEARCH exception: {e}")
        return END


# --- 3.3  ROUTER AFTER INSTRUCTION PARSER ------------------------------------
def route_parse(state: WorkflowState) -> str:
    print("🔀 [Router-PARSE] deciding …")
    try:
        answers = state.get("answers", [])
        last = answers[-1] if answers else ""

        if "INSTRUCTION_PARSED" in last:
            state["success"].append(True)
            return "installer"

        if "INSTRUCTION_FAILURE" in last:
            # Not enough info → go back to alternative researcher
            print("🔄 Need more info – back to alt research")
            return "alt_web_researcher"

        # retry same parser a couple of times
        if _retry_count(state, "instruction_parser") < MAX_RETRIES:
            return "instruction_parser"

        return END
    except Exception as e:
        print(f"⚠️  Router-PARSE exception: {e}")
        return END


# --- 3.4  ROUTER AFTER INSTALL -----------------------------------------------
def route_install(state: WorkflowState) -> str:
    print("🔀 [Router-INSTALL] deciding …")
    try:
        answers = state.get("answers", [])
        last = answers[-1] if answers else ""

        if "INSTALL_SUCCESS" in last:
            state["success"].append(True)
            return "verifier"

        if "INSTALL_FAILURE" in last and _retry_count(state, "installer") < MAX_RETRIES:
            print("↻  Re-running installer")
            return "installer"

        # If install keeps failing, maybe parsing wrong → back to parser
        if _retry_count(state, "instruction_parser") < MAX_RETRIES:
            print("🔙  Going back to parser to refine commands")
            return "instruction_parser"

        return END
    except Exception as e:
        print(f"⚠️  Router-INSTALL exception: {e}")
        return END


# --- 3.5  ROUTER AFTER VERIFICATION ------------------------------------------
def route_verify(state: WorkflowState) -> str:
    print("🔀 [Router-VERIFY] deciding …")
    try:
        answers = state.get("answers", [])
        last = answers[-1] if answers else ""

        if "VERIFY_SUCCESS" in last:
            state["success"].append(True)
            return END

        if "VERIFY_FAILURE" in last and _retry_count(state, "installer") < MAX_RETRIES:
            print("🔄  Binary missing – re-install")
            return "installer"

        return END
    except Exception as e:
        print(f"⚠️  Router-VERIFY exception: {e}")
        return END


# -------- 4️⃣  EDGE DEFINITIONS ----------------------------------------------

# START → ARCH DETECTOR
workflow.add_edge(START, "arch_detector")

# ARCH DETECTOR conditional
workflow.add_conditional_edges(
    "arch_detector",
    route_arch,
    {
        "web_researcher": "web_researcher",
        "arch_detector": "arch_detector",  # retry
        END: END
    },
)

# PRIMARY WEB RESEARCHER conditional
workflow.add_conditional_edges(
    "web_researcher",
    route_research,
    {
        "instruction_parser": "instruction_parser",
        "alt_web_researcher": "alt_web_researcher",
        "web_researcher": "web_researcher",  # retry
        END: END,
    },
)

# ALT WEB RESEARCHER conditional (same router)
workflow.add_conditional_edges(
    "alt_web_researcher",
    route_research,
    {
        "instruction_parser": "instruction_parser",
        "alt_web_researcher": "alt_web_researcher",  # retry
        END: END,
    },
)

# INSTRUCTION PARSER conditional
workflow.add_conditional_edges(
    "instruction_parser",
    route_parse,
    {
        "installer": "installer",
        "alt_web_researcher": "alt_web_researcher",
        "instruction_parser": "instruction_parser",
        END: END,
    }
)

# INSTALLER conditional
workflow.add_conditional_edges(
    "installer",
    route_install,
    {
        "verifier": "verifier",
        "installer": "installer",
        "instruction_parser": "instruction_parser",
        END: END,
    }
)

# VERIFIER conditional
workflow.add_conditional_edges(
    "verifier",
    route_verify,
    {
        END: END,
        "installer": "installer",
    }
)

# -------- 5️⃣  COMPILE  -------------------------------------------------------

app = workflow.compile()
print("✅ Workflow compiled – ready to run.")