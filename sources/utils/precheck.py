import ast
import concurrent.futures
import os
import re
import time

import litellm

from sources.core.llm_provider import LLMConfig, LLMProvider, extract_model_pattern

CODE_PROMPT = (
    "Write a Python function `factorial(n: int) -> int` that returns n! using "
    "recursion. Output ONLY valid Python code, no markdown fences, no comments, "
    "no explanations, no example usage."
)

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    if not text:
        return ""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _compiles(code: str) -> bool:
    if not code:
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


class PreCheck:
    def __init__(self, config):
        self.config = config

    def _basic_check(self, name: str, model_id: str) -> bool:
        provider, model = extract_model_pattern(model_id)
        cfg = LLMConfig(provider=provider, model=model, max_tokens=512)
        try:
            llm = LLMProvider("test", system_msg="You are nice and concise.", config=cfg)
            _ = llm("say hello to me. just one word not more.", use_cache=False)
            return True
        except Exception as e:
            print(f"❌ Provider for '{name}' ({model_id}) failed: {e}")
            return False

    def _probe_openrouter_provider(
        self, model_id: str, or_provider: str, timeout: int = 90
    ) -> dict:
        """Probe a single (OpenRouter model, inference provider) pair.

        Pins the provider with allow_fallbacks=False and num_retries=0 so the result
        reflects that specific backend, not OpenRouter's automatic routing.

        Returns: {"provider", "elapsed", "compiles", "error"}
        """
        provider, model = extract_model_pattern(model_id)
        completion_params = {
            "model": f"{provider}/{model}",
            "messages": [
                {"role": "system", "content": "You write only valid Python code."},
                {"role": "user", "content": CODE_PROMPT},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
            "timeout": timeout,
            "api_key": os.getenv("OPENROUTER_API_KEY", ""),
            "num_retries": 0,
            "extra_body": {
                "provider": {
                    "order": [or_provider],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                }
            },
        }
        t0 = time.perf_counter()
        try:
            response = litellm.completion(**completion_params)
            elapsed = time.perf_counter() - t0
            text = (response.choices[0].message.content or "") if response.choices else ""
            code = _extract_code(text)
            return {
                "provider": or_provider,
                "elapsed": elapsed,
                "compiles": _compiles(code),
                "error": None,
            }
        except Exception as e:
            return {
                "provider": or_provider,
                "elapsed": time.perf_counter() - t0,
                "compiles": False,
                "error": str(e)[:200],
            }

    def _check_openrouter_providers(self, name: str, model_id: str) -> None:
        providers = self.config.openrouter_provider or []
        if not providers:
            return

        print(f"\n🔍 OpenRouter provider probe — {name} ({model_id})")
        max_workers = min(len(providers), 5)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._probe_openrouter_provider, model_id, p): p
                for p in providers
            }
            results = [fut.result() for fut in concurrent.futures.as_completed(futures)]

        order_idx = {p: i for i, p in enumerate(providers)}
        results.sort(key=lambda r: order_idx[r["provider"]])

        any_ok = False
        for r in results:
            status = "✅" if r["compiles"] else "❌"
            detail = f"error: {r['error']}" if r["error"] else f"compiles={r['compiles']}"
            print(
                f"  {status} {r['provider']:<22} time={r['elapsed']:6.2f}s   {detail}"
            )
            if r["compiles"]:
                any_ok = True

        if not any_ok:
            print(
                f"⚠️  No provider in the preference list produced compilable code for "
                f"'{name}' — runs will likely hallucinate. Consider revising "
                f"openrouter_provider in config."
            )

    def run(self) -> None:
        required = {
            "planner": self.config.planner_llm_model,
            "workflow": self.config.workflow_llm_model,
            "smolagent": self.config.smolagent_model_id,
        }
        optional = {
            "judge": getattr(self.config, "judge_model", None),
            "capsule_namer": getattr(self.config, "capsule_namer_model", None),
        }

        # 1) Basic reachability — required slots must succeed, optional ones
        # only warn on failure.
        for name, model_id in required.items():
            if not model_id:
                raise ValueError(f"⚠️  No model configured for '{name}'.")
            if not self._basic_check(name, model_id):
                raise RuntimeError(f"Required model '{name}' failed basic check.")

        for name, model_id in optional.items():
            if model_id:
                self._basic_check(name, model_id)

        providers_ids = {**required, **optional}

        # 2) Per-provider OpenRouter probe (latency + simple compile-check).
        if not self.config.openrouter_provider:
            return

        seen: set[str] = set()
        for name, model_id in providers_ids.items():
            if not model_id or model_id in seen:
                continue
            provider, _ = extract_model_pattern(model_id)
            if provider != "openrouter":
                continue
            seen.add(model_id)
            self._check_openrouter_providers(name, model_id)
