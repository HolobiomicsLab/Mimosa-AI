#!/usr/bin/env python3
"""
Tests for transient network failure handling in CsvEvaluationMode.

A DNS/connectivity outage (e.g. `OpenrouterException - [Errno 8] nodename
nor servname provided, or not known`) must be waited out and the row retried,
not burned through the dataset and reported as a completed run.
"""

import asyncio
import os
import socket
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.benchmark_evaluation import csv_mode
from sources.benchmark_evaluation.csv_mode import (
    CsvEvaluationMode,
    NetworkUnavailableError,
    _is_transient_network_error,
)

# The exact message observed in production (logs/evaluation_csv_mode.log,
# 2026-08-04/05): 189 rows lost to this error across two runs.
PRODUCTION_ERRNO8 = (
    "❌ LLM API error: litellm.APIError: APIError: OpenrouterException - "
    "[Errno 8] nodename nor servname provided, or not known"
)


def _bare_evaluator() -> CsvEvaluationMode:
    """CsvEvaluationMode instance without running the heavy __init__."""
    import logging
    evaluator = object.__new__(CsvEvaluationMode)
    evaluator.logger = logging.getLogger("tests.csv_mode_network_retry")
    return evaluator


# --- _is_transient_network_error ---------------------------------------------

def test_production_errno8_message_is_detected():
    assert _is_transient_network_error(PRODUCTION_ERRNO8)


def test_linux_dns_messages_are_detected():
    assert _is_transient_network_error("[Errno -2] Name or service not known")
    assert _is_transient_network_error("[Errno -3] Temporary failure in name resolution")


def test_unreachable_and_transport_errors_are_detected():
    assert _is_transient_network_error("[Errno 51] Network is unreachable")
    assert _is_transient_network_error("[Errno 65] No route to host")
    assert _is_transient_network_error("litellm.APIConnectionError: ...")


def test_unrelated_errors_are_not_network_errors():
    assert not _is_transient_network_error("'bool' object has no attribute 'replace'")
    assert not _is_transient_network_error(
        "Dataset directory not found: datasets/ScienceAgentBench/datasets/clintox"
    )
    assert not _is_transient_network_error(
        "OpenrouterException - Unable to get json response - Expecting value"
    )
    assert not _is_transient_network_error("")


# --- _wait_for_network_recovery ------------------------------------------------

def test_wait_returns_immediately_when_dns_works(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))])
    evaluator = _bare_evaluator()
    asyncio.run(evaluator._wait_for_network_recovery())


def test_wait_returns_once_dns_recovers(monkeypatch):
    attempts = {"n": 0}
    real_gai = socket.getaddrinfo

    def flaky(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise socket.gaierror(8, "nodename nor servname provided, or not known")
        return real_gai(*args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", flaky)
    monkeypatch.setattr(csv_mode, "NETWORK_PROBE_INTERVAL_S", 0.01)
    evaluator = _bare_evaluator()
    asyncio.run(evaluator._wait_for_network_recovery())
    assert attempts["n"] == 3


def test_wait_raises_when_outage_persists(monkeypatch):
    def always_fail(*args, **kwargs):
        raise socket.gaierror(8, "nodename nor servname provided, or not known")

    monkeypatch.setattr(socket, "getaddrinfo", always_fail)
    monkeypatch.setattr(csv_mode, "NETWORK_PROBE_INTERVAL_S", 0.01)
    monkeypatch.setattr(csv_mode, "NETWORK_MAX_WAIT_S", 0.02)
    evaluator = _bare_evaluator()
    with pytest.raises(NetworkUnavailableError):
        asyncio.run(evaluator._wait_for_network_recovery())
