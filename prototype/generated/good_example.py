
from smolagents import CodeAgent, tool, HfApiModel
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Tuple, Any, Dict, Union, Optional, Callable
from functools import partial
import json

# pre-defined tools


'''
This module provides a set of tools for interacting with a web browser instance.
It is loaded in a sandbox python environment as part of the crafted workflow.
'''

from typing import List, Any
import requests

from smolagents import (
    Tool,
    DuckDuckGoSearchTool
)

API_BASE_URL = 'http://localhost:5000'

def build_formatted_output(action: str, observation: str, reward: float) -> str:
    action_formatted = action[:256].strip().replace('\n', ' - ')
    observation_formatted = observation[:2048].strip().replace('\n', ' - ')
    return f"""
action: {action_formatted}
observation: {observation_formatted}
reward: {reward}
"""

class SearchTool(Tool):
    name = "search_tool"
    description = "Perform a search using DuckDuckGo and return the results."
    inputs = {"query": {"type": "string", "description": "The search query."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        obs = ''
        action = "search:" + query
        try:
            search_tool = DuckDuckGoSearchTool()  # Instantiate the tool
            obs = search_tool(query)              # Call the instance
            reward = 1.0 if obs else 0.0
        except Exception:
            obs = "error"
            reward = 0.0
        return build_formatted_output(action, obs, reward)

class GoToUrlTool(Tool):
    name = "go_to_url_tool"
    description = "Navigate to a specified URL."
    inputs = {"url": {"type": "string", "description": "The URL to navigate to."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        action = "go_to_url: " + url
        obs = None
        try:
            route = API_BASE_URL + '/api/browser/navigate'
            response = requests.post(
                route,
                json={'url': url}
            )
            data = response.json()
            obs = f'navigated to ' + url if data.get('status') == 'success' else f'failed to navigate to ' + url
            reward = 1.0 if obs else 0.0
        except Exception:
            obs = obs if obs else 'failed to navigate to ' + url
            reward = 0.0
        return build_formatted_output(action, obs, reward)

class GetPageTextTool(Tool):
    name = "get_page_text_tool"
    description = "Retrieves the text content from the current web page using the browser instance."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        action = 'get_page_text()'
        try:
            route = API_BASE_URL + '/api/browser/content'
            response = requests.get(route)
            data = response.json()
            obs = data.get('content', 'No text found on the page.')
            reward = 1.0 if obs != 'No text found on the page.' else 0.0
        except Exception:
            obs = 'Error getting page text'
            reward = 0.0
        return build_formatted_output(action, obs, reward)

class GetNavigableLinksTool(Tool):
    name = "get_navigable_links_tool"
    description = "Retrieves a list of navigable links from the browser."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        action = 'get_navigable_links()'
        try:
            route = API_BASE_URL + '/api/browser/links'
            response = requests.get(route)
            data = response.json()
            obs = data.get('links', [])
            reward = 1.0 if obs else 0.0
        except Exception:
            obs = 'Error getting navigable links'
            reward = 0.0
        return build_formatted_output(action, obs, reward)

class IsLinkValidTool(Tool):
    name = "is_link_valid_tool"
    description = "Check if a link is valid for navigation."
    inputs = {"url": {"type": "string", "description": "The URL to check."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        action = "IsLinkValidTool: " + url
        try:
            route = API_BASE_URL + '/api/browser/link_valid'
            response = requests.post(
                route,
                json={'url': url}
            )
            data = response.json()
            obs = data.get('valid', False)
            reward = 1.0 if obs else 0.0
        except Exception:
            obs = 'Error checking link validity'
            reward = 0.0
        return build_formatted_output(action, str(obs), reward)

class ScreenshotTool(Tool):
    name = "screenshot_tool"
    description = "Take a screenshot of the current page."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        action = 'screenshot()'
        try:
            route = API_BASE_URL + '/api/browser/screenshot'
            response = requests.get(route)
            data = response.json()
            obs = data.get('filename', '')
            reward = 1.0 if obs else 0.0
        except Exception:
            obs = 'Error taking screenshot'
            reward = 0.0
        return build_formatted_output(action, obs, reward)

class TakeNoteTool(Tool):
    name = "take_note_tool"
    description = "Take a notes of the current page."
    inputs = {"note": {"type": "string", "description": "The note you want to take."}}
    output_type = "string"

    def forward(self, note: str) -> str:
        reward = 1.0
        action = "taking note."
        obs = note
        return build_formatted_output(action, obs, reward)

search_tool = SearchTool()
go_to_url_tool = GoToUrlTool()
get_page_text_tool = GetPageTextTool()
get_navigable_links_tool = GetNavigableLinksTool()
is_link_valid_tool = IsLinkValidTool()
take_note_tool = TakeNoteTool()
screenshot_tool = ScreenshotTool()

tools = [
    search_tool,
    go_to_url_tool,
    get_page_text_tool,
    get_navigable_links_tool,
    is_link_valid_tool,
    take_note_tool,
    screenshot_tool
]

# schema for the workflow state

from typing import TypedDict, List, Tuple, Any, Dict, Union, Optional, Callable

class Action(TypedDict):
    tool: str
    inputs: dict

class Observation(TypedDict):
    data: Any

class WorkflowState(TypedDict):
    goal: List[str]
    actions: List[Action]
    observations: List[Observation]
    rewards: List[float]

# smolagent factory

from smolagents import (
    CodeAgent,
    HfApiModel,
    ActionStep,
    TaskStep
)
import os

class SmolAgentFactory:
    def __init__(self, instruct_prompt, tools, model_id="Qwen/Qwen2.5-Coder-32B-Instruct", max_steps=2):
        self.model_id = model_id
        self.token = os.getenv("HF_TOKEN")
        self.tools = tools or []
        self.instruct_prompt = instruct_prompt

        if not self.token:
            raise ValueError("Hugging Face token is required. Please set the HF_TOKEN environment variable or pass a token.")
        try:
            self.engine = HfApiModel(
                model_id=model_id,
                token=self.token,
                max_tokens=4096,
            )

            self.agent = CodeAgent(
                tools=self.tools,
                model=self.engine,
                name="agent",
                max_steps=max_steps,
        )
        except Exception as e:
            raise ValueError(f"Error initializing SmolAgent: {e}") from e
        
    def build_worflow_step_prompt(self, state: WorkflowState) -> str:
        state_actions = state.get("actions", [])
        state_observations = state.get("observations", [])
        state_rewards = state.get("rewards", [])
        trajectories = zip(
            state_actions, 
            state_observations, 
            state_rewards
        )
        trajectories_prompt = "\n".join(
            f"Action: {action['tool']}, Observation: {observation['data'][:256]}"
            for action, observation, reward in trajectories
        )
        return f"""
        You previously performed the following actions:
        {trajectories_prompt}
        The end goal is to:
        {state["goal"][-1] if len(state["goal"]) > 0 else "complete the task"}.
        Your need to follow instructions:
        {self.instruct_prompt}
        """
    
    def parse_tool_output(self, output: str):
        actions = []
        observations = []
        rewards = []
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('action:'):
                action = {
                    "tool": line[7:].strip(),
                }
                actions.append(action)
            elif line.startswith('observation:'):
                obs_str = line[12:].strip()
                observation = {"data": obs_str}
                observations.append(observation)
            elif line.startswith('reward:'):
                reward_str = line[7:].strip()
                reward = float(reward_str)
                rewards.append(reward)
        return actions, observations, rewards
    
    def parse_memory_output(self):
        actions, observations, rewards = [], [], []
        for step in self.agent.memory.steps:
            if isinstance(step, ActionStep):
                error, feedback = step.error, step.observations
                step_output = error if error else feedback
                if not isinstance(step_output, str):
                    continue
                print(f"Step output:\n{step_output}")
                print("=" * 50)
                actions_step, obs_step, reward_step = self.parse_tool_output(step_output)
                actions.extend(actions_step)
                observations.extend(obs_step)
                rewards.extend(reward_step)
        return actions, observations, rewards
    
    def run(self, state: WorkflowState) -> dict:
        instructions = self.build_worflow_step_prompt(state)
        result = self.agent.run(instructions)
        actions, observations, rewards = self.parse_memory_output()
        action: Action = {
            "tool": actions[-1]["tool"] if actions else "unknown"
            #"inputs": actions[-1]["inputs"] if actions else {},
        }
        obs: Observation = {
            "data": observations
        }
        reward = sum(rewards) if rewards else 0.0
        return {
            **state,
            "goal": state["goal"],
            "actions": state["actions"] + [action],
            "observations": state["observations"] + [obs],
            "rewards": state["rewards"] + [reward],
            }


# LLM generated logical multi-agent graph
from langgraph.graph import StateGraph, START, END

# Define agent instructions
instruct_research = (
    "You are an expert research agent. "
    "Use your web search tools to find the most recent and relevant news articles "
    "about AI advancements. Extract key information including technological breakthroughs, "
    "company announcements, and research publications. Return comprehensive findings."
    " You should avoid being overwhelmed by too much information, load page by page, parse text of documents for informations your look for, be patient and and do step by step research."
)

instruct_summarizer = (
    "You are a summarization specialist. Given research findings in the observations list, "
    "synthesize a concise executive summary highlighting major advancements, trends, "
    "and significant implications. Structure your summary with clear sections and bullet points."
)


initial_state: WorkflowState = {
    "goal": ["Search the web for the latest news on AI advancements and summarize the findings."],
    "actions": [],
    "observations": [],
    "rewards": [],
}

EXISTING_TOOLS = tools

class WorkflowNodeFactory:
    @staticmethod
    def create_agent_node(agent_factory: SmolAgentFactory) -> Callable[[WorkflowState], dict]:
        def node_function(state: WorkflowState) -> dict:
            return agent_factory.run(state)
        return node_function

smolagent_test = SmolAgentFactory(
    instruct_research,
    EXISTING_TOOLS
)
result_test = smolagent_test.run(initial_state)
print("Test SmolAgent Result:")
print(result_test)
exit()

# Create agents with appropriate tools
smolagent_researcher = SmolAgentFactory(instruct_research, EXISTING_TOOLS)
smolagent_summarizer = SmolAgentFactory(instruct_summarizer, EXISTING_TOOLS)

# Build workflow graph
workflow = StateGraph(WorkflowState)
workflow.add_node("researcher", WorkflowNodeFactory.create_agent_node(smolagent_researcher))
workflow.add_node("summarizer", WorkflowNodeFactory.create_agent_node(smolagent_summarizer))

# Define routing logic
def research_success_router(state: WorkflowState) -> str:
    print(state)
    if state["observations"][-1] and state["observations"][-1]["data"]:
        return "to_summarizer"
    return "end"

# Set edges and conditional routing
workflow.set_entry_point("researcher")
workflow.add_conditional_edges(
    "researcher",
    research_success_router,
    {
        "to_summarizer": "summarizer",
        "end": END
    }
)
workflow.add_edge("summarizer", END)

# Compile workflow
app = workflow.compile()

result_state = app.invoke(initial_state)
print("Result:")
print(result_state)
    