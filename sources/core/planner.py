from .dgm import GodelMachine
from typing import Optional


class Planner:
    """Planner class for long shot task planning"""

    def __init__(self, config) -> None:
        self.config = config
        self.dgm = GodelMachine(config)

    async def start_planner(
        self, goal_prompt: str, template_uuid: str | None, judge: bool
    ) -> None:
        """Start the planner with a given goal prompt and optional template UUID."""
        # NOTE for now the planner just wrap the GodelMachine start_dgm method
        # So the global goal and task specific goal are the same
        await self.dgm.start_dgm(
            goal_prompt=goal_prompt, template_uuid=template_uuid, judge=judge
        )
