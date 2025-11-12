from sources.core.llm_provider import LLMProvider, LLMConfig, extract_model_pattern
from datetime import datetime

class PreCheck:
    def __init__(self, config):
        self.config = config

    def run(self):
        providers_ids = {
            "planner": self.config.planner_llm_model,
            "prompts": self.config.prompts_llm_model,
            "workflow": self.config.workflow_llm_model,
            "smolagent": self.config.smolagent_model_id,
        }

        prompt = "say hello to me. just one word not more."
        for name, model_id in providers_ids.items():
            if not model_id:
                raise ValueError(f"⚠️  No model configured for '{name}', skipping.")
            provider, model = extract_model_pattern(model_id)
            config = LLMConfig(provider=provider, model=model)
            try:
                llm = LLMProvider("test", system_msg="You are nice and concise.", config=config)
                _ = llm(prompt, use_cache=False)
            except Exception as e:
                print(f"❌ Provider for '{name}' failed: {e}")
                raise


