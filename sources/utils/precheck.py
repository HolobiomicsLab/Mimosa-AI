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
from sources.utils.openrouter_endpoints import (
    is_model_creator,
    providers_for_model,
    quant_rank,
)

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
        quantizations: list[str] | None,
        timeout: int = 90,
    ) -> dict:
        """Run N_REPEAT temp=0 calls against (model, provider, quantizations).
        Returns content-validity and bit-exact determinism aggregates.

        `quantizations=None` omits the OpenRouter filter entirely — required
        for first-party endpoints (google-vertex, google-ai-studio, etc.)
        that don't advertise a quant tag and would otherwise be excluded.
        """
        provider, model = extract_model_pattern(model_id)
        provider_routing: dict = {
            "order": [or_provider],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
        if quantizations:
            provider_routing["quantizations"] = quantizations
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
            "extra_body": {"provider": provider_routing},
        }
        if getattr(self.config, "save_logprobs", False):
            # Probe under the same params the runtime sends: with
            # require_parameters, providers lacking logprobs must fail
            # here rather than pass precheck and 404 at run time.
            params["logprobs"] = True
            params["top_logprobs"] = 5  # keep in sync with smolagent_factory.TOP_LOGPROBS

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
            quant_filter: list[str] | None = [discovered_quant]
        else:
            # 'unknown' — provider didn't tag any quantization. Typical for
            # first-party endpoints (google-vertex, google-ai-studio, openai,
            # anthropic) that serve full precision and don't expose a quant
            # tag. OpenRouter's `quantizations` filter is an exclusion list:
            # passing ["bf16","fp16","fp8"] would drop untagged endpoints
            # rather than treat the list as permissive. Omit the filter and
            # let probe content validity decide.
            quant_filter = None
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

        # Drop community endpoints below fp8 (runtime would reject them anyway).
        # First-party endpoints are kept at any quant: when the model creator
        # only self-hosts at int4/fp4 (e.g. moonshotai/kimi-k2.7-code), that's
        # still the authoritative serving of those weights, and for benchmark
        # reproducibility a pinned first-party endpoint beats a community
        # requantization. The probe below still gates them on JSON-validity +
        # determinism, so a misbehaving int4 endpoint is dropped on its merits.
        accepted = set(ALL_ACCEPTED_QUANTS) | {"unknown"}
        discovered = {
            p: q for p, q in discovered.items()
            if q in accepted or is_model_creator(p, slug)
        }

        if configured_norm:
            return [(p, q) for p, q in discovered.items() if p in configured_norm]
        return list(discovered.items())

    def _check_openrouter_providers(self, name: str, model_id: str) -> list[dict]:
        candidates = self._resolve_candidates(model_id)
        if not candidates:
            print(f"\n⚠️  No OpenRouter endpoints found for {name} ({model_id}).")
            return []

        # Bare model slug needed for model-creator checks below.
        _, model_slug = extract_model_pattern(model_id)

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
        # Within each tier: the model's own creator before community
        # resellers, then higher quant rank, then lower latency.  Only the
        # actual model creator (e.g. deepseek for deepseek/*) gets the
        # first-party boost — community resellers with quant=unknown are
        # demoted below fp8 providers.
        results.sort(
            key=lambda r: (
                r["tier"] if r["tier"] != 0 else 99,
                0 if is_model_creator(r["provider"], model_slug) else 1,
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

    def _populate_per_model_provider_lists(
        self,
        all_results: dict[str, list[dict]],
        required_model_ids: set[str],
    ) -> None:
        """Write a per-model sorted provider list into
        `config.openrouter_provider_by_model`. Each OpenRouter model gets its
        own list because endpoint coverage varies per model — a single shared
        list would either 404 at runtime or leave some use case empty.

        Sort within a model: tier asc (strict → drift), quant rank desc,
        latency asc. Fails loudly if any required model has no usable
        provider, since that's a runtime failure waiting to happen.
        """
        if not all_results:
            return

        print("\n♻️  Per-model provider selections (in-memory, this run only):")
        empty_required: list[str] = []

        for model_id, results in all_results.items():
            _, model_slug = extract_model_pattern(model_id)
            kept = [
                (
                    r["provider"],
                    r["tier"],
                    0 if is_model_creator(r["provider"], model_slug) else 1,
                    quant_rank(r["discovered_quant"]),
                    r["mean_latency"],
                    r["discovered_quant"],
                )
                for r in results
                if r["tier"] != 0
            ]
            # Stable sort: strict before drift; within each, the model's own
            # creator before community, then higher precision, then faster.
            kept.sort(key=lambda x: (x[1], x[2], -x[3], x[4]))
            new_list = [t[0] for t in kept]
            selected_quants = {t[5] for t in kept}

            if new_list:
                self.config.openrouter_provider_by_model[model_id] = new_list
                # Runtime `quantizations` is an exclusion filter — a provider
                # whose quant isn't in the list gets dropped even when pinned
                # by `order`. So the filter must include every quant the
                # precheck actually approved, plus the default-safe set as a
                # baseline. If any approved provider was untagged ("unknown",
                # typical for first-party endpoints that don't advertise a
                # tag), omit the filter entirely.
                if "unknown" in selected_quants:
                    self.config.openrouter_quantizations_by_model[model_id] = None
                else:
                    allowed = set(ALL_ACCEPTED_QUANTS) | selected_quants
                    self.config.openrouter_quantizations_by_model[model_id] = sorted(allowed)
                print(f"   {model_id}")
                print(f"     -> {new_list}")
            else:
                print(f"   {model_id}")
                print("     -> (no usable provider)")
                if model_id in required_model_ids:
                    empty_required.append(model_id)

        if empty_required:
            raise RuntimeError(
                "No usable OpenRouter provider for required model(s): "
                f"{empty_required}. Every probe failed (404, rate limit, or "
                "invalid content). Either widen `openrouter_provider` in "
                "config, pick a different model, or relax probe strictness."
            )

    def run(self) -> None:
        print("🚦 Checking LLM providers...")
        required = {
            "planner": self.config.planner_llm_model,
            "workflow": self.config.workflow_llm_model,
            "smolagent": self.config.smolagent_model_id,
            "judge": self.config.judge_model,
            "capsule_namer": self.config.capsule_namer_model,
        }

        for name, model_id in required.items():
            if not model_id:
                raise ValueError(f"⚠️  No model configured for '{name}'.")
            if not self._basic_check(name, model_id):
                raise RuntimeError(f"Required model '{name}' failed basic check.")

        providers_ids = {**required}

        all_results: dict[str, list[dict]] = {}
        required_openrouter: set[str] = set()
        seen: set[str] = set()
        for name, model_id in providers_ids.items():
            if not model_id or model_id in seen:
                continue
            provider, _ = extract_model_pattern(model_id)
            if provider != "openrouter":
                continue
            seen.add(model_id)
            if name in required:
                required_openrouter.add(model_id)
            all_results[model_id] = self._check_openrouter_providers(name, model_id)

        self._populate_per_model_provider_lists(all_results, required_openrouter)


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

    # Smoke test treats every probed model as required so the safety rail
    # is exercised end-to-end.
    pc._populate_per_model_provider_lists(all_results, set(all_results.keys()))
    print("\n✅ Final per-model provider lists:")
    for mid, plist in cfg.openrouter_provider_by_model.items():
        print(f"   {mid} -> {plist}")
