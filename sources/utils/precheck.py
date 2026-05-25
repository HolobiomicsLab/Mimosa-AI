"""OpenRouter provider precheck.

Goes beyond capability tests ("can it write factorial?") to probe for the
failure modes that quantization actually introduces:

  1. JSON-escape integrity — the model emits code inside a JSON envelope and
     must not double-escape backslashes. The canonical bug is `\\n` (two
     literal chars) where a real newline was meant.
  2. Determinism at temp=0 — three identical requests must produce identical
     outputs. FP16/BF16 reference is bit-stable; quantized FP8 stacks drift.
  3. Tiered quantization — try `bf16/fp16` first via OpenRouter's
     `quantizations` filter; fall back to `fp8` only if the provider has no
     higher-precision endpoint for this model. Drop providers that fail both.
"""

import ast
import concurrent.futures
import hashlib
import json
import os
import re
import time

import litellm

from sources.core.llm_provider import LLMConfig, LLMProvider, extract_model_pattern

TIER1_QUANTS = ["bf16", "fp16"]
TIER2_QUANTS = ["fp8"]
N_REPEAT = 3

JSON_CODE_SYSTEM = (
    'You answer with exactly one JSON object on a single line: '
    '{"code": "<python>"}. No prose, no markdown fences, no extra keys.'
)
JSON_CODE_PROMPT = (
    'Write Python that uses pandas to read "data.csv" into df, prints '
    'df.head(), and prints df.columns.tolist(). Include a comment containing '
    'the raw regex r"\\d+".'
)


def _strip_md_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    return s


def _validate_json_code(text: str) -> tuple[bool, str]:
    """Parsable JSON with a 'code' string that has no literal `\\n` (the
    quantization signature) and is valid Python at the AST level.
    """
    if not text:
        return False, "empty"
    try:
        obj = json.loads(_strip_md_fence(text))
    except json.JSONDecodeError as e:
        return False, f"json: {e.msg}"
    if not isinstance(obj, dict) or "code" not in obj:
        return False, "no 'code' field"
    code = obj["code"]
    if not isinstance(code, str):
        return False, "'code' not string"
    if "\\n" in code:
        return False, "double-escaped newline"
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax: {e.msg}"
    return True, "ok"


def _normalize_provider_name(name: str) -> str:
    """`parasail/fp8` -> `parasail`. The explicit quantizations filter
    handles tiering, so the suffix is redundant and gets in the way of the
    tier-1 probe."""
    return name.split("/", 1)[0]


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

    def _probe_with_quant(
        self,
        model_id: str,
        or_provider: str,
        quantizations: list[str],
        timeout: int = 90,
    ) -> dict:
        """Run N_REPEAT calls at temp=0 against (model, provider, quants).
        Aggregates JSON-code validity and bit-exact determinism across calls."""
        provider, model = extract_model_pattern(model_id)
        params = {
            "model": f"{provider}/{model}",
            "messages": [
                {"role": "system", "content": JSON_CODE_SYSTEM},
                {"role": "user", "content": JSON_CODE_PROMPT},
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
                    "quantizations": quantizations,
                }
            },
        }

        outputs: list[str] = []
        latencies: list[float] = []
        last_err: str | None = None
        for _ in range(N_REPEAT):
            t0 = time.perf_counter()
            try:
                resp = litellm.completion(**params)
                latencies.append(time.perf_counter() - t0)
                outputs.append(
                    (resp.choices[0].message.content or "") if resp.choices else ""
                )
            except Exception as e:
                latencies.append(time.perf_counter() - t0)
                last_err = str(e)[:200]

        validations = [_validate_json_code(o) for o in outputs]
        valid_count = sum(1 for ok, _ in validations)
        first_failure = next((reason for ok, reason in validations if not ok), None)
        non_empty = [o for o in outputs if o]
        hashes = {hashlib.sha256(o.encode()).hexdigest() for o in non_empty}
        deterministic = len(hashes) == 1 and len(non_empty) >= 2

        return {
            "provider": or_provider,
            "quantizations": quantizations,
            "n_calls": len(outputs),
            "valid_count": valid_count,
            "deterministic": deterministic,
            "mean_latency": (sum(latencies) / len(latencies)) if latencies else 0.0,
            "error": last_err,
            "failure_reason": first_failure,
        }

    @staticmethod
    def _passes(r: dict) -> bool:
        # All completed calls valid, deterministic across them, ≥2 completed
        # (one transient error is tolerated).
        return (
            r["n_calls"] >= 2
            and r["valid_count"] == r["n_calls"]
            and r["deterministic"]
        )

    def _probe_tiered(self, model_id: str, or_provider: str, timeout: int = 90) -> dict:
        clean = _normalize_provider_name(or_provider)
        r1 = self._probe_with_quant(model_id, clean, TIER1_QUANTS, timeout)
        if self._passes(r1):
            r1["tier"] = 1
            return r1
        r2 = self._probe_with_quant(model_id, clean, TIER2_QUANTS, timeout)
        if self._passes(r2):
            r2["tier"] = 2
            return r2
        worse = r2 if r2["valid_count"] >= r1["valid_count"] else r1
        worse["tier"] = 0
        return worse

    def _check_openrouter_providers(self, name: str, model_id: str) -> list[dict]:
        providers = self.config.openrouter_provider or []
        if not providers:
            return []

        print(f"\n🔍 OpenRouter provider probe — {name} ({model_id})")
        max_workers = min(len(providers), 5)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._probe_tiered, model_id, p): p
                for p in providers
            }
            results = [fut.result() for fut in concurrent.futures.as_completed(futures)]

        order_idx = {_normalize_provider_name(p): i for i, p in enumerate(providers)}
        results.sort(key=lambda r: order_idx.get(r["provider"], 999))

        for r in results:
            icon = {0: "❌", 1: "✅", 2: "⚠️ "}[r["tier"]]
            tier_label = {0: "fail   ", 1: "bf/fp16", 2: "fp8    "}[r["tier"]]
            det = "det" if r["deterministic"] else "drift"
            reason = r["error"] or r["failure_reason"] or ""
            line = (
                f"  {icon} {r['provider']:<16} tier={tier_label} "
                f"mean={r['mean_latency']:5.2f}s  "
                f"{r['valid_count']}/{r['n_calls']} valid, {det}"
            )
            if reason:
                line += f"  ({reason})"
            print(line)
        return results

    def _update_openrouter_provider_list(
        self, all_results: dict[str, list[dict]]
    ) -> None:
        """Reorder config.openrouter_provider by (worst-tier asc, mean latency
        asc). Drop any provider that failed any model's probe. Names get
        normalized — `parasail/fp8` becomes `parasail`, since the quantizations
        filter (not the suffix) now controls tiering.
        """
        original = list(self.config.openrouter_provider or [])
        if not original or not all_results:
            return

        agg: dict[str, dict] = {}
        for results in all_results.values():
            for r in results:
                p = r["provider"]
                a = agg.setdefault(p, {"tiers": [], "latencies": [], "ok": True})
                if r["tier"] == 0:
                    a["ok"] = False
                else:
                    a["tiers"].append(r["tier"])
                    a["latencies"].append(r["mean_latency"])

        kept = [
            (p, max(info["tiers"]), sum(info["latencies"]) / len(info["latencies"]))
            for p, info in agg.items()
            if info["ok"] and info["tiers"]
        ]
        kept.sort(key=lambda x: (x[1], x[2]))
        new_list = [p for p, _, _ in kept]
        removed = sorted(
            {_normalize_provider_name(p) for p in original} - set(new_list)
        )

        if not new_list:
            print(
                "\n⚠️  All providers failed at least one probe — leaving "
                "openrouter_provider unchanged. Runs will likely hallucinate."
            )
            return

        print("\n♻️  Updating openrouter_provider (in-memory, this run only):")
        print(f"   before:  {original}")
        print(f"   after:   {new_list}    (tier1 [bf16/fp16] first, then tier2 [fp8])")
        if removed:
            print(f"   removed: {removed}   (failed at least one probe)")
        self.config.openrouter_provider = new_list

    def run(self) -> None:
        print("🚦 Checking LLM providers...")
        required = {
            "smolagent": self.config.smolagent_model_id,
        }

        for name, model_id in required.items():
            if not model_id:
                raise ValueError(f"⚠️  No model configured for '{name}'.")
            if not self._basic_check(name, model_id):
                raise RuntimeError(f"Required model '{name}' failed basic check.")

        providers_ids = {**required}

        if not self.config.openrouter_provider:
            return

        all_results: dict[str, list[dict]] = {}
        seen: set[str] = set()
        for name, model_id in providers_ids.items():
            if not model_id or model_id in seen:
                continue
            provider, _ = extract_model_pattern(model_id)
            if provider != "openrouter":
                continue
            seen.add(model_id)
            all_results[model_id] = self._check_openrouter_providers(name, model_id)

        self._update_openrouter_provider_list(all_results)
