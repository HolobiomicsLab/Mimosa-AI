import json
import os
from pathlib import Path


class Benchmarker:
    def __init__(self, config, uuid: str):
        self.uuid = uuid
        self.path = Path("mimosa") / "memory" / self.uuid
        self.results = []

    def generate_text(self):
        step_dict = {}
        for root, _dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".json"):
                    steps = json.load(open(os.path.join(root, file)))
                    start_time = int(steps[0]["timing"].get("start_time", ""))
                    step_text = f"AGENT: {file.removeprefix('node_task_').removesuffix('.json')}\n"
                    step_text = f"USER TASK: {steps[0]['model_input_messages'][1]['content'][0].get('text', '')}\n"
                    for step in steps:
                        step_text += f"STEP {step.get('step_number', '')}\n"
                        step_text += f"\ACTION: {step.get('code_action', '')}\n"
                        error = step.get("error", "")
                        if error:
                            step_text += f"\ERROR: {error}\n"
                        else:
                            step_text += f"\RESULT: {step.get('observations', '')}\n"

                    step_dict[start_time] = step_text

        text = ""
        sorted_dict = dict(sorted(step_dict.items()))
        for key in sorted_dict:
            text += sorted_dict[key] + "\n"

        with open(self.path / "formated.txt", "w") as file:
            file.write(text)

    def get_text(self):
        return open(self.path / "formated.txt").read()

    def __str__(self):
        return f"Benchmarker text:\n\n{self.get_text()}"
