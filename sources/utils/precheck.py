"""OpenRouter provider precheck.

Goes beyond capability tests ("can it write factorial?") to probe for the
failure modes that quantization actually introduces:

  1. Endpoint discovery — query OpenRouter for which providers actually serve
     this model (and at what quantization), instead of probing a hand-curated
     list and getting 404s.
  2. JSON-escape integrity — the model emits code inside a JSON envelope and
     must not double-escape backslashes. The canonical bug is `\\n` (two
     literal chars) where a real newline was meant.
  3. Determinism at temp=0 — three identical requests should produce identical
     outputs. FP16/BF16 reference is bit-stable; quantized FP8 stacks drift.

Outcomes are classified, not pass/fail:
  - tier 1 (strict): content-valid AND deterministic across 3 calls
  - tier 2 (drift):  content-valid but drifts at temp=0  (accepted only if no
                     strict-pass providers exist for this model)
  - tier 0 (fail):   content-invalid, routing failure, or rate-limited
Final ordering: tier asc, discovered-quantization rank desc, latency asc.
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
from sources.utils.openrouter_endpoints import providers_for_model, quant_rank

N_REPEAT = 3
ALL_ACCEPTED_QUANTS = ["bf16", "fp16", "fp8"]

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
    quantization signature) and is valid Python at the AST level."""
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
    """`parasail/fp8` -> `parasail`. Discovery returns the bare provider
    name; the suffix in legacy configs is now redundant."""
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

    def _probe(
        self,
        model_id: str,
        or_provider: str,
        quantizations: list[str],
        timeout: int = 90,
    ) -> dict:
        """Run N_REPEAT temp=0 calls against (model, provider, quantizations).
        Returns content-validity and bit-exact determinism aggregates."""
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
    def _classify(r: dict) -> int:
        """0 = fail (drop), 1 = strict (valid + deterministic),
        2 = drift (valid but non-deterministic)."""
        if r["n_calls"] < 2:
            return 0
        if r["valid_count"] != r["n_calls"]:
            return 0
        return 1 if r["deterministic"] else 2

    def _probe_provider(
        self, model_id: str, or_provider: str, discovered_quant: str
    ) -> dict:
        """Single probe using the discovered quantization as the pinned filter."""
        if discovered_quant in {"bf16", "fp16", "fp8"}:
            quant_filter = [discovered_quant]
        else:
            # Unknown or lower-than-supported; allow the broad set and let the
            # outcome speak. We still tag the discovered value for sorting.
            quant_filter = ALL_ACCEPTED_QUANTS
        r = self._probe(model_id, or_provider, quant_filter)
        r["discovered_quant"] = discovered_quant
        r["tier"] = self._classify(r)
        return r

    def _resolve_candidates(self, model_id: str) -> list[tuple[str, str]]:
        """Returns [(provider_name, discovered_quant)] for `model_id`.

        - Discovers actual endpoints via OpenRouter API.
        - If `config.openrouter_provider` is set, intersects discovery with it
          (after normalizing `parasail/fp8` -> `parasail`).
        - If discovery fails, falls back to the configured list with unknown
          quants — so the precheck still runs, just blindly.
        """
        # Strip the leading 'openrouter/' provider prefix — the discovery API
        # wants the bare model slug (e.g. 'deepseek/deepseek-v3.2').
        _, slug = extract_model_pattern(model_id)
        discovered = providers_for_model(slug)
        configured = self.config.openrouter_provider or []
        configured_norm = {_normalize_provider_name(p) for p in configured}

        if not discovered:
            # Discovery failed; fall back to configured names with unknown quant.
            return [(p, "unknown") for p in sorted(configured_norm)]

        # Drop endpoints below fp8 — runtime's quantizations filter would
        # reject them anyway, no point burning probes.
        accepted = set(ALL_ACCEPTED_QUANTS) | {"unknown"}
        discovered = {p: q for p, q in discovered.items() if q in accepted}

        if configured_norm:
            return [(p, q) for p, q in discovered.items() if p in configured_norm]
        return list(discovered.items())

    def _check_openrouter_providers(self, name: str, model_id: str) -> list[dict]:
        candidates = self._resolve_candidates(model_id)
        if not candidates:
            print(f"\n⚠️  No OpenRouter endpoints found for {name} ({model_id}).")
            return []

        # Order candidates by discovered quant (best first) — affects probe
        # order under thread pool but not final ranking (we sort results below).
        candidates.sort(key=lambda x: -quant_rank(x[1]))

        summary = ", ".join(f"{p}({q})" for p, q in candidates)
        print(
            f"\n🔍 OpenRouter provider probe — {name} ({model_id})\n"
            f"   {len(candidates)} endpoint(s): {summary}"
        )

        max_workers = min(len(candidates), 5)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._probe_provider, model_id, prov, quant): (prov, quant)
                for prov, quant in candidates
            }
            results = [fut.result() for fut in concurrent.futures.as_completed(futures)]

        # Final sort: pass-strict first, then drift, then fail.
        # Within each tier, higher quant rank first, then lower latency.
        results.sort(
            key=lambda r: (
                r["tier"] if r["tier"] != 0 else 99,
                -quant_rank(r["discovered_quant"]),
                r["mean_latency"],
            )
        )

        icons = {0: "❌", 1: "✅", 2: "⚠️ "}
        tier_labels = {0: "fail  ", 1: "strict", 2: "drift "}
        for r in results:
            det = "det" if r["deterministic"] else "drift"
            reason = r["error"] or r["failure_reason"] or ""
            line = (
                f"  {icons[r['tier']]} {r['provider']:<16} "
                f"quant={r['discovered_quant']:<7} "
                f"{tier_labels[r['tier']]} "
                f"mean={r['mean_latency']:5.2f}s  "
                f"{r['valid_count']}/{r['n_calls']} valid, {det}"
            )
            if reason:
                line += f"  ({reason[:80]})"
            print(line)
        return results

    def _update_openrouter_provider_list(
        self, all_results: dict[str, list[dict]]
    ) -> None:
        """Reorder config.openrouter_provider by (worst-tier asc, best-quant
        across models desc, mean latency asc). Drop providers that failed any
        model's probe."""
        if not all_results:
            return

        agg: dict[str, dict] = {}
        for results in all_results.values():
            for r in results:
                p = r["provider"]
                a = agg.setdefault(
                    p, {"tiers": [], "quants": [], "latencies": [], "ok": True}
                )
                if r["tier"] == 0:
                    a["ok"] = False
                    continue
                a["tiers"].append(r["tier"])
                a["quants"].append(r["discovered_quant"])
                a["latencies"].append(r["mean_latency"])

        kept = [
            (
                p,
                max(info["tiers"]),                                       # worst tier
                max(quant_rank(q) for q in info["quants"]),               # best quant rank
                sum(info["latencies"]) / len(info["latencies"]),          # mean latency
            )
            for p, info in agg.items()
            if info["ok"] and info["tiers"]
        ]
        kept.sort(key=lambda x: (x[1], -x[2], x[3]))
        new_list = [p for p, _, _, _ in kept]
        original = list(self.config.openrouter_provider or [])
        original_norm = {_normalize_provider_name(p) for p in original}
        removed = sorted(original_norm - set(new_list)) if original_norm else []

        if not new_list:
            print(
                "\n⚠️  All providers failed at least one probe — leaving "
                "openrouter_provider unchanged. Runs will likely hallucinate."
            )
            return

        print("\n♻️  Updating openrouter_provider (in-memory, this run only):")
        print(f"   before:  {original or '(unset, used discovery)'}")
        print(f"   after:   {new_list}    (strict-pass first, then drift; higher precision wins ties)")
        if removed:
            print(f"   removed: {removed}   (failed at least one probe)")
        self.config.openrouter_provider = new_list

    def run(self) -> None:
        print("🚦 Checking LLM providers...")
        required = {
            "planner": self.config.planner_llm_model,
            "workflow": self.config.workflow_llm_model,
            "smolagent": self.config.smolagent_model_id,
        }
        optional = {
            "judge": getattr(self.config, "judge_model", None),
            "capsule_namer": getattr(self.config, "capsule_namer_model", None),
        }

        for name, model_id in required.items():
            if not model_id:
                raise ValueError(f"⚠️  No model configured for '{name}'.")
            if not self._basic_check(name, model_id):
                raise RuntimeError(f"Required model '{name}' failed basic check.")

        for name, model_id in optional.items():
            if model_id:
                self._basic_check(name, model_id)

        providers_ids = {**required, **optional}

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


if __name__ == "__main__":
    # Smoke test for the OpenRouter discovery + probe + classification path.
    # Skips the Anthropic basic_check so the test only needs OPENROUTER_API_KEY.
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))

    from config import Config

    cfg = Config()
    pc = PreCheck(cfg)

    print("🚦 OpenRouter precheck smoke test\n")
    all_results: dict[str, list[dict]] = {}
    seen: set[str] = set()
    candidates = {
        "smolagent": cfg.smolagent_model_id,
        "judge": getattr(cfg, "judge_model", None),
        "capsule_namer": getattr(cfg, "capsule_namer_model", None),
    }
    for label, model_id in candidates.items():
        if not model_id or model_id in seen:
            continue
        provider, _ = extract_model_pattern(model_id)
        if provider != "openrouter":
            continue
        seen.add(model_id)
        all_results[model_id] = pc._check_openrouter_providers(label, model_id)

    pc._update_openrouter_provider_list(all_results)
    print(f"\n✅ Final openrouter_provider: {cfg.openrouter_provider}")
