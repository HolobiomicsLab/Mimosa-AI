"""Tests for planner JSON extraction and make_plan retry behaviour.

Regression coverage for a failure where plan generation failed all retries:
the flat LLM cache replayed the same unparseable response on every attempt,
and the retry prompt was never rebuilt with the error feedback. Also covers
the more tolerant JSON extraction (plain fences, raw JSON, prose-wrapped).

Heavy dependencies (litellm etc. via sources.core.__init__) are avoided by
loading schema/planner directly and stubbing their sibling imports, so the
tests run offline and deterministically.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(_REPO_ROOT))


def _stub_package(name: str, path: Path, created: list[str]) -> types.ModuleType:
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    if name not in sys.modules:
        created.append(name)
        sys.modules[name] = pkg
    return sys.modules[name]


def _stub_module(created: list[str], name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    if name not in sys.modules:
        created.append(name)
        sys.modules[name] = mod
    return sys.modules[name]


def _load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: dataclass processing in schema.py resolves the
    # module through sys.modules while the class body is still executing.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Stub parent packages and sibling modules so importing planner.py never
# pulls in sources.core.__init__ (litellm/sentence_transformers). Only
# entries absent from sys.modules are stubbed; anything already imported by
# a previously collected test module is left untouched.
_created_stubs: list[str] = []
_stub_package("sources", _REPO_ROOT / "sources", _created_stubs)
_stub_package("sources.core", _REPO_ROOT / "sources" / "core", _created_stubs)
_stub_package("sources.cli", _REPO_ROOT / "sources" / "cli", _created_stubs)
_stub_package(
    "sources.extensibility", _REPO_ROOT / "sources" / "extensibility", _created_stubs
)
_stub_package("sources.utils", _REPO_ROOT / "sources" / "utils", _created_stubs)


def _noop(*args, **kwargs) -> None:
    """No-op stand-in for print/notify helpers."""
    return None


_stub_module(
    _created_stubs,
    "sources.cli.pretty_print",
    BOLD="",
    CYAN="",
    DIM="",
    RESET="",
    print_err=_noop,
    print_info=_noop,
    print_ok=_noop,
    print_phase=_noop,
    print_section=_noop,
    print_summary=_noop,
    print_warn=_noop,
)
_stub_module(
    _created_stubs,
    "sources.extensibility.text_to_speech",
    create_tts_service=lambda: None,
)
_stub_module(_created_stubs, "sources.utils.list_files", list_files=lambda *a, **k: "")
_stub_module(
    _created_stubs,
    "sources.utils.notify",
    PushNotifier=lambda *a, **k: SimpleNamespace(send_message=_noop),
)
_stub_module(
    _created_stubs,
    "sources.utils.perspicacite_client",
    query_perspicacite=lambda *a, **k: "",
)
_stub_module(
    _created_stubs, "sources.utils.planner_visualization", PlannerVisualizer=object
)
_stub_module(_created_stubs, "sources.core.evolution_engine", EvolutionEngine=object)
_stub_module(
    _created_stubs,
    "sources.core.llm_provider",
    LLMConfig=object,
    LLMProvider=None,  # replaced per-test
    extract_model_pattern=lambda model: ("stub", model),
)
_stub_module(_created_stubs, "sources.core.workflow_selection", WorkflowSelector=object)

_load("sources.core.schema", _REPO_ROOT / "sources" / "core" / "schema.py")
_planner = _load("sources.core.planner", _REPO_ROOT / "sources" / "core" / "planner.py")

# The stubs above only exist to satisfy planner.py's import-time bindings.
# Remove exactly the entries this module created so other test modules
# collected in the same pytest process import the real packages instead.
for _stubbed in _created_stubs:
    sys.modules.pop(_stubbed, None)

Planner = _planner.Planner
_extract = Planner._extract_json_from_code_block

_VALID_PLAN_JSON = json.dumps(
    {
        "goal": "test goal",
        "steps": [
            {
                "name": "step_one",
                "task": "do the thing",
                "depends_on": [],
                "required_inputs": [],
                "expected_outputs": ["out.csv"],
                "complexity": "medium",
            }
        ],
    }
)


def test_extract_fenced_json_block() -> None:
    text = f"Here is the plan:\n```json\n{_VALID_PLAN_JSON}\n```\nHope this helps."
    assert _extract(text)["steps"][0]["name"] == "step_one"


def test_extract_plain_fence() -> None:
    text = f"```\n{_VALID_PLAN_JSON}\n```"
    assert _extract(text)["goal"] == "test goal"


def test_extract_raw_json_without_fences() -> None:
    assert _extract(_VALID_PLAN_JSON)["goal"] == "test goal"


def test_extract_prose_wrapped_json() -> None:
    text = f"Sure! Here you go:\n{_VALID_PLAN_JSON}\nLet me know if you need changes."
    assert _extract(text)["steps"][0]["task"] == "do the thing"


def test_extract_skips_broken_block_for_parseable_span() -> None:
    text = f"```json\n{{broken json\n```\nFallback:\n{_VALID_PLAN_JSON}"
    assert _extract(text)["goal"] == "test goal"


def test_extract_no_json_returns_none() -> None:
    assert _extract("no braces here at all") is None
    assert _extract("") is None


def test_extract_invalid_json_raises_decode_error() -> None:
    text = '{"goal": "g", "steps": [{"name": "x",'
    try:
        _extract(text)
    except json.JSONDecodeError:
        return
    raise AssertionError("expected JSONDecodeError for unparseable JSON")


class _FakeLLMProvider:
    """Stand-in for LLMProvider that replays queued responses and records calls."""

    calls: list[dict] = []
    responses: list[str] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, prompt: str, timeout: int = 180, use_cache: bool = True) -> str:
        _FakeLLMProvider.calls.append({"prompt": prompt, "use_cache": use_cache})
        return _FakeLLMProvider.responses.pop(0)


def _make_planner(tmp_path: Path) -> Planner:
    """Build a Planner without running __init__ (no EvolutionEngine/LLM setup)."""
    planner = Planner.__new__(Planner)
    planner.config = SimpleNamespace(memory_dir=str(tmp_path), max_tokens=8192)
    planner.config_llm = None
    planner.notifier = SimpleNamespace(send_message=_noop)
    planner.make_scientific_grounded_prompt = lambda goal: goal
    return planner


def test_make_plan_retries_bypass_cache_and_use_error_feedback(
    tmp_path, monkeypatch
) -> None:
    """A parse failure must retry with cache off and an enhanced prompt."""
    monkeypatch.setattr(_planner, "LLMProvider", _FakeLLMProvider)
    monkeypatch.setattr(_planner.time, "sleep", _noop)
    _FakeLLMProvider.calls = []
    _FakeLLMProvider.responses = [
        "Sorry, I cannot produce JSON.",  # attempt 1: unparseable
        f"```json\n{_VALID_PLAN_JSON}\n```",  # attempt 2: valid
    ]

    planner = _make_planner(tmp_path)
    plan = planner.make_plan("system prompt", "test goal")

    assert len(plan.steps) == 1
    assert len(_FakeLLMProvider.calls) == 2
    first, second = _FakeLLMProvider.calls
    assert first["use_cache"] is True
    # Retry must not replay a cached (unparseable) response…
    assert second["use_cache"] is False
    # …and must carry the error feedback instead of the unchanged prompt.
    assert "Previous attempt failed with error" in second["prompt"]
    assert second["prompt"] != first["prompt"]


def test_make_plan_succeeds_without_retry_on_valid_response(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(_planner, "LLMProvider", _FakeLLMProvider)
    _FakeLLMProvider.calls = []
    _FakeLLMProvider.responses = [_VALID_PLAN_JSON]

    planner = _make_planner(tmp_path)
    plan = planner.make_plan("system prompt", "test goal")

    assert plan.steps[0].name == "step_one"
    assert len(_FakeLLMProvider.calls) == 1


if __name__ == "__main__":
    import tempfile

    class _MonkeyPatchShim:
        def setattr(self, target, name, value) -> None:
            setattr(target, name, value)

    for name, fn in list(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        if "tmp_path" in fn.__code__.co_varnames:
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d), _MonkeyPatchShim())
        else:
            fn()
        print(f"{name}: OK")
