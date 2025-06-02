from typing import List, Tuple, Type, Dict

import threading


class Interaction:
    """
    Interaction is a class that handles the interaction between the user and the agents.
    """
    def __init__(self, agents,
                 langs: List[str] = ["en", "zh"]
                ):
        self.is_active = True
        self.is_generating = False
        self.current_agent = None
        self.last_query = None
        self.last_answer = None
        self.last_reasoning = None
        self.agents = agents
        self.languages = langs
    
    def get_last_blocks_result(self) -> List[Dict]:
        """Get the last blocks result."""
        if self.current_agent is None:
            return []
        blks = []
        for agent in self.agents:
            blks.extend(agent.get_blocks_result())
        return blks
    
    def load_last_session(self):
        """Recover the last session."""
        for agent in self.agents:
            if agent.type == "planner_agent":
                continue
            agent.memory.load_memory(agent.type)
    
    def save_session(self):
        """Save the current session."""
        for agent in self.agents:
            agent.memory.save_memory(agent.type)

    def is_active(self) -> bool:
        return self.is_active
    
    def read_stdin(self) -> str:
        """Read the input from the user."""
        buffer = ""
        while not buffer:
            try:
                buffer = input("➤➤➤ ")
            except EOFError:
                return None
            if buffer == "exit" or buffer == "goodbye":
                return None
        return buffer
    
    def get_user(self) -> str:
        """Get the user input from the microphone or the keyboard."""
        query = self.read_stdin()
        if query is None:
            self.is_active = False
            self.last_query = None
            return None
        self.last_query = query
        return query
    
    def set_query(self, query: str) -> None:
        """Set the query"""
        self.is_active = True
        self.last_query = query
    
    def find_planner_agent(self) -> Type:
        """Find the planner agent in the list of agents."""
        for agent in self.agents:
            if agent.type == "planner_agent":
                return agent
        raise ValueError("No planner agent found")
    
    async def process_task(self) -> bool:
        """Request AI agents to process the user input."""
        if self.last_query is None or len(self.last_query) == 0:
            return False
        agent = self.find_planner_agent()
        self.is_generating = True
        self.current_agent = agent
        self.last_answer, self.last_reasoning = await agent.process(self.last_query)
        self.is_generating = False
        return True
    
    def get_updated_process_answer(self) -> str:
        """Get the answer from the last agent."""
        if self.current_agent is None:
            return None
        return self.current_agent.get_last_answer()
    
    def get_updated_block_answer(self) -> str:
        """Get the answer from the last agent."""
        if self.current_agent is None:
            return None
        return self.current_agent.get_last_block_answer()
    
    def show_answer(self) -> None:
        """Show the answer to the user."""
        if self.last_query is None:
            return
        if self.current_agent is not None:
            self.current_agent.show_answer()

