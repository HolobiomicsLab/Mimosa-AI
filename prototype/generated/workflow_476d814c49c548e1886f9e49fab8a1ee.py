
from smolagents import CodeAgent, tool, HfApiModel
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Tuple, Any, Dict, Union, Optional, Callable
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

def print_action(args: List[Any]) -> None:
    '''
    Print the action being performed.
    Args:
        name (str): The name of the action.
        args (List[Any]): The arguments for the action.
    '''
    print("action:", args)

def print_observation(observation: str) -> None:
    '''
    Print the observation made.
    Args:
        observation (str): The observation to print.
    '''
    print("observation:", observation)

def print_reward(reward: float) -> None:
    '''
    Print the reward received.
    Args:
        reward (float): The reward value to print.
    '''
    print("reward:", reward)

class SearchTool(Tool):
    name = "search_tool"
    description = "Perform a search using DuckDuckGo and return the results."
    inputs = {"query": {"type": "string", "description": "The search query."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        obs = ''
        action = "search:" + query
        print_action([action])
        try:
            search_tool = DuckDuckGoSearchTool()  # Instantiate the tool
            obs = search_tool(query)              # Call the instance
            reward = 1.0 if obs else 0.0
        except Exception:
            obs = "error"
            reward = 0.0
        print_observation(obs)
        print_reward(reward)
        return obs

class GoToUrlTool(Tool):
    name = "go_to_url_tool"
    description = "Navigate to a specified URL."
    inputs = {"url": {"type": "string", "description": "The URL to navigate to."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        action = "go_to_url: " + url
        print_action([action])
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
        print_observation(obs)
        print_reward(reward)
        return obs

class GetPageTextTool(Tool):
    name = "get_page_text_tool"
    description = "Retrieves the text content from the current web page using the browser instance."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        action = 'get_page_text()'
        print_action([action])
        try:
            route = API_BASE_URL + '/api/browser/content'
            response = requests.get(route)
            data = response.json()
            obs = data.get('content', 'No text found on the page.')
            reward = 1.0 if obs != 'No text found on the page.' else 0.0
        except Exception:
            obs = 'Error getting page text'
            reward = 0.0
        print_observation(obs)
        print_reward(reward)
        return obs

class GetNavigableLinksTool(Tool):
    name = "get_navigable_links_tool"
    description = "Retrieves a list of navigable links from the browser."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        action = 'get_navigable_links()'
        print_action([action])
        try:
            route = API_BASE_URL + '/api/browser/links'
            response = requests.get(route)
            data = response.json()
            obs = data.get('links', [])
            reward = 1.0 if obs else 0.0
        except Exception:
            obs = 'Error getting navigable links'
            reward = 0.0
        print_observation(obs)
        print_reward(reward)
        return obs

class IsLinkValidTool(Tool):
    name = "is_link_valid_tool"
    description = "Check if a link is valid for navigation."
    inputs = {"url": {"type": "string", "description": "The URL to check."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        action = "IsLinkValidTool: " + url
        print_action([action])
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
        print_observation(obs)
        print_reward(reward)
        return obs

class ScreenshotTool(Tool):
    name = "screenshot_tool"
    description = "Take a screenshot of the current page."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        action = 'screenshot()'
        print_action([action])
        try:
            route = API_BASE_URL + '/api/browser/screenshot'
            response = requests.get(route)
            data = response.json()
            obs = data.get('filename', '')
            reward = 1.0 if obs else 0.0
        except Exception:
            obs = 'Error taking screenshot'
            reward = 0.0
        print_observation(obs)
        print_reward(reward)
        return obs

class TakeNoteTool(Tool):
    name = "take_note_tool"
    description = "Take a notes of the current page."
    inputs = {"note": {"type": "string", "description": "The note you want to take."}}
    output_type = "string"

    def forward(self, note: str) -> str:
        reward = 1.0
        action = "taking note."
        obs = note
        print_action([action])
        print_observation(obs)
        print_reward(reward)
        return obs

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
    goal: str
    actions: List[Action]
    observations: List[Observation]
    rewards: List[float]

# smolagent factory
from smolagents import (
    CodeAgent,
    HfApiModel
)
import os

from schema_factory import WorkflowState, Action, Observation

class SmolAgentFactory:
    def __init__(self, model_id="Qwen/Qwen2.5-Coder-32B-Instruct", instruct_prompt = "", tools=None, max_steps=5):
        self.model_id = model_id
        self.token = os.getenv("HF_TOKEN")
        self.tools = tools or []
        self.instruct_prompt = instruct_prompt

        if not self.token:
            raise ValueError("Hugging Face token is required. Please set the HF_TOKEN environment variable or pass a token.")
        
        self.engine = HfApiModel(
            model_id=model_id,
            token=self.token,
            max_tokens=8096,
        )
        
        self.agent = CodeAgent(
            tools=self.tools,
            model=self.engine,
            name="agent",
            max_steps=max_steps,
        )
    
    def parse_tool_output(self, output: str):
        actions = []
        observations = []
        rewards = []
        
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('action:'):
                try:
                    action_str = line[7:].strip()
                    action_args = eval(action_str)
                    action = {
                        "tool": action_args[0].split('(')[0],
                        "inputs": {"args": action_args}
                    }
                    actions.append(action)
                except:
                    pass
            elif line.startswith('observation:'):
                obs_str = line[12:].strip()
                observation = {"data": obs_str}
                observations.append(observation)
            elif line.startswith('reward:'):
                try:
                    reward_str = line[7:].strip()
                    reward = float(reward_str)
                    rewards.append(reward)
                except:
                    pass
        return actions, observations, rewards
    
    def run(self, state: WorkflowState) -> dict:
        instructions_template = f"""
{{self.instruct_prompt}}
Current task: {{goal}}.
        """
        try:
            instructions = instructions_template.format(
                goal=state["goal"][-1] if state["goal"] else "complete the task"
            )
            result = self.agent.run(instructions)
            actions, observations, rewards = self.parse_tool_output(result)

            action: Action = {
                "tool": actions[-1]["tool"] if actions else "unknown",
                "inputs": actions[-1]["inputs"] if actions else {},
            }
            obs: Observation = {
                "data": result
            }
            reward = sum(rewards) if rewards else 0.0
            return {
                **state,
                "goal": state["goal"],
                "actions": state["actions"] + [action],
                "observations": state["observations"] + [obs],
                "rewards": state["rewards"] + [reward],
            }
        except Exception as e:
            return {
                **state,
                "goal": state["goal"],
                "actions": state["actions"] + [None],
                "observations": state["observations"] + [None],
                "rewards": state["rewards"] + [0],
            }


# LLM generated logical multi-agent graph
from langgraph.graph import StateGraph, START, END

# Define agent instructions
instruct_research = (
    "You are an expert research agent. Your goal is {goal}. "
    "Use your web search tools to find the most recent and relevant news articles "
    "about AI advancements. Extract key information including technological breakthroughs, "
    "company announcements, and research publications. Return comprehensive findings."
)

instruct_summarizer = (
    "You are a summarization specialist. Given research findings in the observations list, "
    "synthesize a concise executive summary highlighting major advancements, trends, "
    "and significant implications. Structure your summary with clear sections and bullet points."
)

# Create agents with appropriate tools
smolagent_researcher = SmolAgentFactory(instruct_research, [web_search_tool])
smolagent_summarizer = SmolAgentFactory(instruct_summarizer, [])

# Build workflow graph
workflow = StateGraph(WorkflowState)
workflow.add_node("researcher", smolagent_researcher.run)
workflow.add_node("summarizer", smolagent_summarizer.run)

# Define routing logic
def research_success_router(state: WorkflowState) -> str:
    if state["observations"] and state["observations"][-1]["data"]:
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

initial_state: WorkflowState = {
    "goal": ["{goal_prompt}"],
    "actions": [],
    "observations": [],
    "rewards": [],
}

result_state = app.invoke(initial_state)
print(result_state)
    