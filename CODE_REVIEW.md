# Mimosa-AI — Code Review & Improvement Suggestions

**Date:** 2026-04-09
**Reviewer:** Code Review via Cline
**Scope:** Full codebase review — architecture, code quality, testing, packaging, and documentation

---

## Executive Summary

Mimosa-AI is a well-architected research framework with a clear five-layer design, innovative Darwinian workflow evolution, and strong scientific grounding. The codebase is readable, the README is excellent, and the developer guide is thorough. Below I identify **bugs**, **architectural concerns**, **code quality issues**, and **concrete improvement opportunities** across the project.

**Severity legend:** 🔴 Bug/Critical · 🟡 Important · 🟢 Nice-to-have

---

## 1. Bugs & Correctness Issues

### 🔴 1.1 `recursive_self_evolution` calls non-existent `recursive_self_improvement`

In `sources/core/dgm.py` line 579, the method calls `self.recursive_self_improvement(...)` but no such method exists on `DarwinMachine`. The actual method is `self.recursive_self_evolution(...)`. This will produce an `AttributeError` at runtime on every iteration that attempts to continue beyond the first pass.

```python
# dgm.py:579 — BUG
runs = await self.recursive_self_improvement(...)  # ← should be recursive_self_evolution
```

### 🔴 1.2 `Config.__str__` references non-existent `workflow_llm_provider`

In `config.py` line 265, `__str__` references `self.workflow_llm_provider` which doesn't exist as an attribute. This will crash whenever the config is printed.

```python
f"workflow_llm_provider={self.workflow_llm_provider},\n"  # ← AttributeError
```

### 🔴 1.3 Unreachable code after `except` in `load_wf_state_result`

In `dgm.py` line 273, there's a `return None` after the `except Exception` block's `raise`, which is unreachable:

```python
except Exception as e:
    raise ValueError(f"❌ Error reading workflow state: {str(e)}") from e
return None  # ← unreachable
```

### 🟡 1.4 `ImprovementValidator.validate_improvement` receives list but accesses `.reward`

In `dgm.py` lines 486-487, `baseline_run=runs[-5:]` and `new_run=runs[-1:]` pass **lists** to `validate_improvement`, but the validator accesses `baseline_run.reward` (attribute of single object, not list). This will raise `AttributeError`.

### 🟡 1.5 `check_package.py` security logic is inverted

The `PackageCheck._check_version` method (line 37-43) sets `safe = False` initially, then checks if the version is within a "blocked" range. But the logic is inverted — it marks versions **within** the blocked range as `safe=True`, then exits if `not safe`. This means it blocks **safe** versions and allows the malicious one.

```python
# The blocked range should make safe=False, but the code does:
safe = threshold_down <= parsed <= threshold_up  # This marks the BLOCKED version as "safe"
if not safe:  # This exits on SAFE versions!
```

### 🟡 1.6 Shebang line typo in `main.py`

Line 1: `#!nne/usr/bin/env python3` — should be `#!/usr/bin/env python3`.

### 🟡 1.7 `workflow_factory.py` — `assemble_workflow` generates fragile f-string code

The assembled code at line 303 uses `raise(f"Could not save workflow data:" + str(e))` which calls `raise` on a string (not an exception), producing a `TypeError` at runtime.

---

## 2. Architecture & Design

### 🟡 2.1 Tight coupling via hardcoded paths

`Config.__init__` hardcodes a developer-specific path:
```python
self.workspace_dir = "/home/martin/Projects/CNRS/Toolomics/workspace"
```
This should default to a relative or environment-variable-based path, or require explicit configuration. New users cloning the repo will hit `AssertionError` from `validate_paths()`.

**Recommendation:** Default to `None` or a relative path like `"./workspace"` and require explicit setup via config file or environment variable.

### 🟡 2.2 `Config` class mixes concerns

`Config` handles file I/O (`load`, `dump`), path validation, directory creation, pricing client initialization, and serialization. Consider:
- Extract path management into a `PathManager`
- Extract serialization into `ConfigSerializer`
- Use a proper configuration library (e.g., `pydantic-settings`) for validation

### 🟡 2.3 Duplicate `get_flow_answers` method

`PromptGradient.get_flow_answers()` and `DarwinMachine.get_flow_answers()` are identical. Extract to a shared utility function or mixin.

### 🟡 2.4 `Factory.load_tools_code` has infinite retry loop

```python
while tool_setup == False:
    mcps = await tool_manager.discover_mcp_servers()
    tool_setup = await tool_manager.verify_tools()
```
If Toolomics is down, this loops forever without any backoff or user notification. Add a retry limit and exponential backoff.

### 🟡 2.5 No dependency injection / interface abstraction

Core classes (`DarwinMachine`, `Planner`, `WorkflowOrchestrator`) directly instantiate their dependencies in `__init__`. This makes unit testing very difficult. Consider passing dependencies as constructor arguments with sensible defaults.

### 🟢 2.6 Consider Protocol/ABC for evaluators

`GenericEvaluator` and `ScenarioEvaluator` share base class `BaseEvaluator` but there's no formal `Evaluator` protocol. Defining one would make the evaluator system more extensible and type-safe.

---

## 3. Code Quality

### 🟡 3.1 Inconsistent type annotations

- `Any` used where `dict` or concrete types would be clearer (e.g., `wf_state: any` in `dgm.py` — lowercase `any` isn't even a valid type)
- `WorkflowInfo.is_success` has return type `dict` but returns `bool`
- `WorkflowInfo.answers` has return type `dict` but returns `list`
- `WorkflowInfo.success` has return type `dict` but returns `list`

### 🟡 3.2 Inconsistent error handling patterns

Mixed approaches throughout:
- Some methods use `assert` for validation (fails silently when Python runs with `-O`)
- Some raise `ValueError`, others `RuntimeError`, others custom exceptions
- Some methods catch `Exception` broadly and re-raise, adding no value
- `planner.py:754` has `except Exception as e: raise e` which is a no-op

**Recommendation:** Establish a consistent exception hierarchy. Use `assert` only for invariants, never for input validation.

### 🟡 3.3 Magic strings and numbers

- `threshold_similary=0.8` (also a typo: "similary" → "similarity")
- `threshod_score=0.0` (typo: "threshod" → "threshold")
- Score thresholds (0.7, 0.8, 0.85, 0.9) scattered across files without named constants
- `time.sleep(10)` in planner.py:739 with comment "wait for files update" — fragile

### 🟡 3.4 Raw print statements mixed with structured logging

The codebase uses both `print()` with emoji and the `logging` module. Many critical messages use `print()` instead of `logger.error()`, making them invisible in log files. The `pretty_print` module is well-designed but underutilized.

### 🟡 3.5 `sys.path` manipulation

`workflow_selection.py` line 5: `sys.path.append(...)` is fragile. Use proper package structure with `__init__.py` files and relative imports instead.

### 🟢 3.6 Typos in parameter names (public API)

- `threshold_similary` → `threshold_similarity` (in `WorkflowSelector.select_best_workflows`)
- `threshod_score` → `threshold_score`
- `comparaison` → `comparison` (in main.py argparse help)
- `litterature` → `literature` (in evaluator.py)
- `detailled` → `detailed` (in planner.py)
- `criterions` → `criteria` (in main.py argparse help)
- `mearningful` → `meaningful" (in csv_mode.py comment)

---

## 4. Testing

### 🔴 4.1 No automated test suite

The `tests/` directory contains only manual test scripts (each with `if __name__ == "__main__"` blocks) that require:
- API keys
- Running Toolomics
- Specific workflow UUIDs

There are **zero unit tests** that can run in CI without external dependencies.

**Recommendation:**
1. Add `pytest` as a dev dependency
2. Create unit tests with mocked LLM responses for:
   - `Config` serialization/deserialization
   - `WorkflowFactory.extract_python_code`
   - `WorkflowFactory.validate_workflow_structure`
   - `PricingCalculator._normalize_model_name`
   - `ImprovementValidator` (already has inline test code — extract to pytest)
   - `Planner._extract_json_from_code_block`
   - `GenericEvaluator._extract_scores`
   - `WorkflowInfo` property loading
   - `AddressMCP` validation
   - `check_answer_success` / `evaluate_workflow_success`
3. Add integration tests that run with mocked MCP responses
4. Add `conftest.py` with reusable fixtures

### 🟡 4.2 Test file naming

`tests/__init.py` is missing a closing underscore — should be `__init__.py`. This may prevent test discovery.

---

## 5. Packaging & Dependencies

### 🟡 5.1 Duplicate/conflicting dependency specifications

Dependencies are declared in **three places** with inconsistent versions:
- `pyproject.toml`: `requires-python = ">=3.11,<3.14"`, uses `smolagents[all]`
- `sources/requirements.txt`: `smolagents[all]`, `fastmcp==2.8.1`, different pins
- `config.py` `runner_requirements`: `smolagents[litellm,mlx-lm,telemetry,mcp]`, `fastmcp==2.8.1`, `pandas==2.3.2`

The README says "Python 3.10+" but `pyproject.toml` requires ">=3.11". The `config.py` runner uses `python_version="3.10"`.

**Recommendation:** Consolidate to a single source of truth (`pyproject.toml`) and remove `sources/requirements.txt`. Use `pyproject.toml` optional dependencies for runner requirements.

### 🟡 5.2 Heavy dependencies for optional features

`pygame` is a required dependency (in `pyproject.toml`) but is only used for the optional planner visualization window. `torch` and `sentence-transformers` are heavyweight (~2GB+) but only used for workflow similarity matching.

**Recommendation:** Move these to optional dependency groups:
```toml
[project.optional-dependencies]
viz = ["pygame>=2.5.0"]
similarity = ["sentence-transformers==5.1.2", "torch>=1.11.0"]
```

### 🟡 5.3 Missing `__init__.py` files

Several packages lack proper `__init__.py`:
- `sources/evaluation/` — no `__init__.py`
- `sources/security/` — no `__init__.py`
- `sources/core/` — no `__init__.py`

While Python 3 supports namespace packages, explicit `__init__.py` files improve IDE support and make the package structure intentional.

### 🟢 5.4 `pyproject.toml` description is placeholder

```toml
description = "Add your description here"
```

---

## 6. Security

### 🟡 6.1 Generated workflow code execution

The `WorkflowRunner` executes LLM-generated Python code directly via `subprocess`. While this is inherent to the design, the current sandbox is minimal:
- No filesystem isolation (code runs with full user permissions)
- No network restrictions
- Memory limit (`max_memory_mb`) is declared but **never enforced** in `_run_command`
- CPU limit (`max_cpu_percent`) is declared but **never enforced**

**Recommendation:** At minimum, enforce the declared resource limits using `resource.setrlimit()` or `cgroups`. For stronger isolation, consider `bubblewrap`, `firejail`, or Docker containers.

### 🟡 6.2 Prompt injection surface

The `craft_instructions` parameter in `WorkflowFactory.llm_make_workflow` includes previous workflow output (agent answers, stderr) directly in the prompt without sanitization. A malicious MCP tool could inject instructions via its output.

### 🟢 6.3 API keys in memory

`LLMConfig` stores API keys as plain strings. Consider using `SecretStr` from pydantic to prevent accidental logging.

---

## 7. Performance

### 🟡 7.1 `WorkflowSelector` recomputes embeddings on every call

`cosine_similarity()` is called multiple times for the same workflow during sorting + filtering (once in `sort_similar_workflows` sort key, again in the list comprehension filter). Each call re-encodes both strings.

**Recommendation:** Compute embeddings once and cache them. Pre-compute workflow embeddings at discovery time.

### 🟡 7.2 `discover_mcp_at_address` scans ports sequentially

Port scanning from 5000-5200 one port at a time takes ~6+ minutes if most ports are closed (2s timeout each). Use `asyncio.gather` with concurrency limits:

```python
async def discover_mcp_at_address(self, address, port_min, port_max):
    tasks = [self._probe_port(address, port) for port in range(port_min, port_max + 1)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, MCP)]
```

### 🟢 7.3 Dependencies installed on every workflow run

`workflow_requirements_install()` is called for every workflow execution, even if dependencies haven't changed. Consider caching the installed state.

---

## 8. Documentation

### 🟢 8.1 README

The README is **excellent** — well-structured, with clear examples, benchmark results, and visual aids. Minor suggestions:
- The installation section says `cd mimosa` but the repo is `Mimosa-AI`
- Add a "Troubleshooting" section for common issues (Toolomics not found, API key errors)
- The `Configuration` section references `config_default.json` but it's not in the repo (generated by `config.py`)

### 🟢 8.2 Missing docstrings

Several public methods lack docstrings:
- `PromptGradient.get_hints()`
- `DarwinMachine.show_answers()`
- `DarwinMachine.get_craft_instructions()`
- Most `WorkflowInfo` properties

### 🟢 8.3 Outdated developer guide references

`docs/DEVELOPER_GUIDE.md` lists `csv_mode.py` under `sources/extensibility/` but it's actually in `sources/evaluation/`.

---

## 9. Prioritized Action Items

### Immediate (Bugs)
1. ~~Fix `recursive_self_improvement` → `recursive_self_evolution` in `dgm.py`~~ ✅ Fixed
2. ~~Fix `Config.__str__` reference to non-existent `workflow_llm_provider`~~ ✅ Fixed
3. ~~Fix security check logic inversion in `check_package.py`~~ ✅ Fixed (renamed to `safe_versions`)
4. ~~Fix shebang typo in `main.py`~~ ✅ Fixed
5. ~~Fix `raise(f-string)` in `assemble_workflow`~~ ✅ Fixed (now `raise(Exception(...))`)
6. ~~Fix `ImprovementValidator` receiving lists instead of single runs~~ ✅ Redesigned: now accepts both lists and single objects, with support for greedy/tournament/novelty/QD strategies
7. Fix `tests/__init.py` → `tests/__init__.py`

### Short-term (Quality)
8. Add unit tests with mocked dependencies (target: 60%+ coverage on core/)
9. Consolidate dependency specifications to `pyproject.toml` only
10. Fix all parameter name typos in public APIs
11. Remove hardcoded workspace path from `Config.__init__`
12. Add `__init__.py` to all source packages
13. Fix type annotations (`any` → proper types, return types on `WorkflowInfo`)

### Medium-term (Architecture)
14. Add dependency injection to core classes for testability
15. Parallelize MCP port scanning
16. Cache workflow embeddings in `WorkflowSelector`
17. Enforce resource limits in `WorkflowRunner`
18. Extract duplicate `get_flow_answers` into shared utility
19. Add retry limit to `Factory.load_tools_code` infinite loop
20. Move heavy optional dependencies to `[project.optional-dependencies]`

### Long-term (Robustness)
21. Add proper sandboxing for generated code execution
22. Implement CI pipeline with automated tests
23. Add structured logging throughout (replace `print()` calls)
24. Consider `pydantic` models for `Config` and all data schemas
25. Add OpenTelemetry spans for internal tracing (complement Langfuse)

---

*This review is based on static analysis of the codebase as of commit `911a853`. Some runtime behaviors may differ under specific configurations.*
