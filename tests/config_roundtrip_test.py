#!/usr/bin/env python3
"""
Tests for Config JSON round-tripping.

A field assigned in ``Config.__init__`` but absent from ``jsonify``/``from_json``
is silently reverted whenever a config is dumped and reloaded, while ``__str__``
still prints it — so the config looks complete and is not.
"""

import ast
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as config_module
from config import AddressMCP, Config

# Fields deliberately excluded from the JSON round-trip.
NOT_PERSISTED = {
    # Secrets, re-read from the environment on every construction.
    "pushover_token",
    "pushover_user",
    # Derived at runtime by precheck rather than authored by the user.
    "openrouter_provider_by_model",
    "openrouter_quantizations_by_model",
}

# Known gaps, tracked in issue #182. Remove entries here as they are fixed;
# this set is an upper bound, so fixing one does not fail the test.
KNOWN_GAPS = {
    "literrature_grounding",
    "orchestrator_choose_model",
}


def _init_assigned_fields() -> list[str]:
    """Public ``self.<name>`` assignments in ``Config.__init__``, in order."""
    tree = ast.parse(open(config_module.__file__).read())
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Config"
    )
    init = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )

    fields: list[str] = []
    for node in ast.walk(init):
        target = None
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and not target.attr.startswith("_")
            and target.attr not in fields
        ):
            fields.append(target.attr)
    return fields


def test_every_init_field_is_persisted_by_jsonify():
    """Guard against a new field being added to __init__ but not to jsonify."""
    print("\n🧪 Testing jsonify covers every __init__ field...")

    fields = _init_assigned_fields()
    assert len(fields) > 40, f"AST scan found only {len(fields)} fields — scan is broken"

    persisted = set(Config().jsonify().keys())
    missing = set(fields) - persisted - NOT_PERSISTED

    assert missing <= KNOWN_GAPS, (
        "Config fields assigned in __init__ but absent from jsonify: "
        f"{sorted(missing - KNOWN_GAPS)}. Add them to jsonify and from_json, "
        "or to NOT_PERSISTED if they are secrets or runtime-derived."
    )
    print(f"✅ {len(persisted)} fields persisted; no new round-trip gaps")


def test_prompt_and_routing_fields_survive_a_round_trip():
    """prompt_smolagent, max_concurrent_eval_tasks and the quantization
    allow-list must survive dump/load.

    prompt_smolagent is the agent system prompt — the single most
    outcome-determining string in a run — and could not be set from a config
    file at all.
    """
    print("\n🧪 Testing round-trip of previously dropped fields...")

    original = Config()
    original.prompt_smolagent = os.path.abspath(original.prompt_planner)
    original.max_concurrent_eval_tasks = 7
    original.default_openrouter_quantizations = ["fp8"]

    restored = Config()
    restored.from_json(original.jsonify())

    assert restored.prompt_smolagent == original.prompt_smolagent, (
        f"prompt_smolagent not restored: {restored.prompt_smolagent}"
    )
    assert restored.max_concurrent_eval_tasks == 7, (
        f"max_concurrent_eval_tasks not restored: {restored.max_concurrent_eval_tasks}"
    )
    assert restored.default_openrouter_quantizations == ["fp8"], (
        f"quantizations not restored: {restored.default_openrouter_quantizations}"
    )
    print("✅ All three fields survive jsonify -> from_json")


def test_absent_discovery_addresses_keeps_the_default():
    """A config file without the key must not leave MCP discovery with no range."""
    print("\n🧪 Testing discovery_addresses default retention...")

    config = Config()
    default = list(config.discovery_addresses)
    assert default, "Config() should ship a default discovery range"

    config.from_json({"workspace_dir": config.workspace_dir})

    assert config.discovery_addresses == default, (
        f"discovery_addresses was wiped to {config.discovery_addresses}"
    )
    print("✅ Absent key keeps the default range")


def test_present_discovery_addresses_replaces_the_default():
    """An explicit list must still override the default."""
    print("\n🧪 Testing discovery_addresses override...")

    config = Config()
    config.from_json({
        "discovery_addresses": [{"ip": "127.0.0.1", "port_min": 5001, "port_max": 5200}]
    })

    assert config.discovery_addresses == [AddressMCP("127.0.0.1", 5001, 5200)], (
        f"explicit addresses not loaded: {config.discovery_addresses}"
    )
    print("✅ Explicit range overrides the default")


def run_all_tests():
    """Run all config round-trip tests."""
    print("Starting config round-trip tests...\n")

    try:
        test_every_init_field_is_persisted_by_jsonify()
        test_prompt_and_routing_fields_survive_a_round_trip()
        test_absent_discovery_addresses_keeps_the_default()
        test_present_discovery_addresses_replaces_the_default()

        print("\n🎉 All config round-trip tests passed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
