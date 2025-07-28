from .dgm import GodelMachine




class Planner:
    """Planner class for long shot task planning"""

    def __init__(self, config) -> None:
        self.config = config
        self.dgm = GodelMachine(config)

    async def start_planner(
        self, goal: str, template_uuid: str | None, judge: bool, answer :str=None
    ):
        """Start the planner with a given goal prompt and optional template UUID."""
        # NOTE for now the planner just wrap the GodelMachine start_dgm method
        # So the global goal and task specific goal are the same
        uuid = await self.dgm.start_dgm(
            goal_prompt=goal, template_uuid=template_uuid, judge=judge, answer = answer
        )
        return uuid
