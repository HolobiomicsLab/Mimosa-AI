"""Shared native-harness budgets are durable, bounded, and fail closed."""

from __future__ import annotations

import fcntl
import json
import multiprocessing
import time

import pytest

from sources.core.harness_budget import (
    HarnessBudgetError,
    HarnessCallBudget,
    validate_native_harness_settings,
)


def _settings(tmp_path, **changes):
    bridge = tmp_path / "bridge.py"
    bridge.write_text("def complete(request): return {}\n")
    values = {
        "bridge_path": str(bridge),
        "bridge_sha256": "a" * 64,
        "ledger_path": str(tmp_path / "calls.jsonl"),
        "max_calls": 2,
        "total_timeout_seconds": 60.0,
        "call_timeout_seconds": 20,
        "reasoning_effort": "max",
        "max_observed_tokens": None,
    }
    values.update(changes)
    return values


def _call(label="astronomy"):
    return {
        "backend": "codex_cli",
        "auth_mode": "subscription",
        "model": "gpt-5.6-sol",
        "reasoning_effort": "max",
        "agent_name_sha256": "b" * 64,
        "transcript_sha256": "c" * 64,
        "transcript_bytes": len(label.encode()),
        "message_count": 1,
    }


def _receipt(status="completed", observed_total_tokens=7):
    output_tokens = min(2, observed_total_tokens or 0)
    usage = {
        "input_tokens": (observed_total_tokens or 0) - output_tokens,
        "output_tokens": output_tokens,
        "total_tokens": observed_total_tokens,
        "cached_input_tokens": None,
        "cache_creation_input_tokens": None,
    }
    if observed_total_tokens is None:
        usage = None
    return {
        "status": status,
        "model_identity": {
            "requested": {"model": "gpt-5.6-sol", "source": "request"},
            "configured": {
                "model": "gpt-5.6-sol",
                "source": "explicit_cli_argument",
            },
            "reported": None,
        },
        "actual_model": None,
        "observed_models": [],
        "usage": usage,
        "observed_total_tokens": observed_total_tokens,
        "usage_kind": "chatgpt_subscription",
        "cost_usd": None,
        "cost_kind": "unavailable",
        "cli_version": "codex-cli fixture",
        "diagnostic_count": 0,
        "elapsed_seconds": 0.25,
        "response_sha256": "d" * 64,
        "response_bytes": 4,
        "stop_applied": False,
        "stop_sequence_sha256": None,
        "text": "NEVER PERSIST RESPONSE TEXT",
        "error": "NEVER PERSIST RAW ERROR",
    }


def _reserve_in_process(configured, start, results):
    """Contend for one reservation from an isolated process."""
    start.wait(5)
    try:
        reservation = HarnessCallBudget(configured).reserve(_call())
    except Exception as exc:  # The parent asserts the public failure category.
        results.put(type(exc).__name__)
    else:
        results.put(f"reserved:{reservation.reservation_id}")


def test_settings_validation_is_pure_and_exact(tmp_path):
    configured = _settings(tmp_path)
    normalized = validate_native_harness_settings(configured)

    assert normalized == configured
    assert normalized is not configured
    assert not (tmp_path / "calls.jsonl").exists()
    assert not (tmp_path / "calls.jsonl.lock").exists()

    for key, value in {
        "bridge_path": "relative.py",
        "bridge_sha256": "not-a-digest",
        "ledger_path": "relative.jsonl",
        "max_calls": True,
        "total_timeout_seconds": float("inf"),
        "call_timeout_seconds": 0,
        "reasoning_effort": "automatic",
        "max_observed_tokens": 0,
    }.items():
        with pytest.raises(ValueError):
            validate_native_harness_settings(_settings(tmp_path, **{key: value}))

    with pytest.raises(ValueError, match="unsupported"):
        validate_native_harness_settings({**configured, "api_key": "forbidden"})


def test_shared_budget_reserves_and_completes_across_objects(tmp_path):
    configured = _settings(tmp_path, max_calls=2)
    first = HarnessCallBudget(configured)
    second = HarnessCallBudget(configured)

    reservation_1 = first.reserve(_call("proteomics"))
    first.complete(reservation_1, _receipt())
    reservation_2 = second.reserve(_call("geophysics"))
    second.complete(reservation_2, _receipt())

    with pytest.raises(HarnessBudgetError, match="call budget"):
        first.reserve(_call("ecology"))
    first.assert_healthy()

    records = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "policy", "reservation", "completion", "reservation", "completion"
    ]
    serialized = json.dumps(records)
    assert "NEVER PERSIST" not in serialized


def test_unfinished_or_failed_call_poison_future_dispatch(tmp_path):
    pending_settings = _settings(tmp_path, ledger_path=str(tmp_path / "pending.jsonl"))
    pending = HarnessCallBudget(pending_settings)
    pending.reserve(_call())

    with pytest.raises(HarnessBudgetError, match="unfinished"):
        HarnessCallBudget(pending_settings).assert_healthy()
    with pytest.raises(HarnessBudgetError, match="unfinished"):
        HarnessCallBudget(pending_settings).reserve(_call())

    failed_settings = _settings(tmp_path, ledger_path=str(tmp_path / "failed.jsonl"))
    failed = HarnessCallBudget(failed_settings)
    reservation = failed.reserve(_call())
    failed.complete(reservation, _receipt(status="failed", observed_total_tokens=None))
    with pytest.raises(HarnessBudgetError, match="failed"):
        HarnessCallBudget(failed_settings).assert_healthy()


def test_observed_token_ceiling_is_aggregate_and_missing_is_explicit(tmp_path):
    configured = _settings(tmp_path, max_calls=3, max_observed_tokens=10)
    budget = HarnessCallBudget(configured)
    first = budget.reserve(_call())
    budget.complete(first, _receipt(observed_total_tokens=7))
    second = budget.reserve(_call())
    budget.complete(second, _receipt(observed_total_tokens=4))

    with pytest.raises(HarnessBudgetError, match="observed token"):
        budget.assert_healthy()
    with pytest.raises(HarnessBudgetError, match="observed token"):
        budget.reserve(_call())

    missing_config = _settings(
        tmp_path,
        ledger_path=str(tmp_path / "missing.jsonl"),
        max_observed_tokens=10,
    )
    missing = HarnessCallBudget(missing_config)
    reservation = missing.reserve(_call())
    missing.complete(reservation, _receipt(observed_total_tokens=None))
    with pytest.raises(HarnessBudgetError, match="missing"):
        missing.assert_healthy()


def test_lock_wait_is_bounded_by_total_deadline(tmp_path):
    configured = _settings(tmp_path, total_timeout_seconds=0.08)
    budget = HarnessCallBudget(configured)
    lock_path = tmp_path / "calls.jsonl.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        with pytest.raises(HarnessBudgetError, match="lock"):
            budget.reserve(_call())
        elapsed = time.monotonic() - started

    assert 0.04 <= elapsed < 0.5


def test_policy_mismatch_cannot_weaken_existing_budget(tmp_path):
    configured = _settings(tmp_path, max_calls=2)
    budget = HarnessCallBudget(configured)
    reservation = budget.reserve(_call())
    budget.complete(reservation, _receipt())

    weaker = HarnessCallBudget({**configured, "max_calls": 20})
    with pytest.raises(HarnessBudgetError, match="policy"):
        weaker.reserve(_call())


def test_completion_after_monotonic_total_deadline_poisoned(tmp_path, monkeypatch):
    configured = _settings(tmp_path, total_timeout_seconds=2)
    budget = HarnessCallBudget(configured)
    reservation = budget.reserve(_call())
    monkeypatch.setattr(
        "sources.core.harness_budget.time.monotonic",
        lambda: reservation.deadline_monotonic + 0.01,
    )
    budget.complete(reservation, _receipt())

    with pytest.raises(HarnessBudgetError, match="total deadline"):
        budget.assert_healthy()


def test_one_shared_reservation_wins_across_processes(tmp_path):
    configured = _settings(tmp_path, max_calls=1)
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(target=_reserve_in_process, args=(configured, start, results))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    outcomes = [results.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        assert process.exitcode == 0

    assert sum(outcome.startswith("reserved:") for outcome in outcomes) == 1
    assert outcomes.count("HarnessBudgetError") == 1
    records = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [record["event"] for record in records] == ["policy", "reservation"]


def test_different_boot_origin_or_corrupt_tokens_fail_closed(tmp_path):
    boot_settings = _settings(tmp_path, ledger_path=str(tmp_path / "boot.jsonl"))
    boot_budget = HarnessCallBudget(boot_settings)
    reservation = boot_budget.reserve(_call())
    boot_budget.complete(reservation, _receipt())
    boot_records = [json.loads(line) for line in (tmp_path / "boot.jsonl").read_text().splitlines()]
    boot_records[0]["boot_time_epoch"] += 1
    (tmp_path / "boot.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in boot_records)
    )
    with pytest.raises(HarnessBudgetError, match="policy"):
        HarnessCallBudget(boot_settings).assert_healthy()

    token_settings = _settings(tmp_path, ledger_path=str(tmp_path / "tokens.jsonl"))
    token_budget = HarnessCallBudget(token_settings)
    reservation = token_budget.reserve(_call())
    token_budget.complete(reservation, _receipt())
    token_records = [json.loads(line) for line in (tmp_path / "tokens.jsonl").read_text().splitlines()]
    token_records[-1]["observed_total_tokens"] = -1
    (tmp_path / "tokens.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in token_records)
    )
    with pytest.raises(HarnessBudgetError, match="observed token"):
        HarnessCallBudget(token_settings).assert_healthy()
