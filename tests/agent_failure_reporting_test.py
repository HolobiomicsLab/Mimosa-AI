"""An agent that crashes must not be reported as a timeout.

``run_cached`` runs the agent on a worker thread, joins with a deadline, and
treats ``completed is False`` as a timeout. The retry loop caught every
exception but the lines recording it were commented out, so an agent whose
three attempts all raised left ``completed=False`` and ``exception=None``. The
thread then exited, ``join`` returned immediately, and the caller raised

    TimeoutError: Agent 'reconstructor' execution timed out after 18000 seconds

roughly two minutes into a run — with the real exception discarded. Observed
against ``stealth/ox-alpha`` while running ASB capsules.

The discriminator is whether the worker is still alive after the join.
"""

import types

import pytest

from tests.smolagent_factory_logprobs_test import load_factory_module, make_bare_factory


@pytest.fixture(scope="module")
def factory_module() -> types.ModuleType:
    return load_factory_module()


class _Agent:
    """Stub smolagent: raises `fail_times` times, then returns `response`."""

    def __init__(self, fail_times=0, response="done", exc=None, block=False):
        self.fail_times = fail_times
        self.response = response
        self.exc = exc or RuntimeError("boom")
        self.block = block
        self.calls = 0

    def run(self, instructions):
        self.calls += 1
        if self.block:
            import time
            time.sleep(30)
        if self.calls <= self.fail_times:
            raise self.exc
        return self.response


def _factory(factory_module, agent, timeout=5):
    return make_bare_factory(
        factory_module,
        name="reconstructor",
        agent=agent,
        timeout=timeout,
        save_memories=lambda **kw: None,
        load_agent_memory=lambda *a, **kw: None,
    )


def test_successful_agent_returns_its_response(factory_module):
    f = _factory(factory_module, _Agent(fail_times=0, response="ok"))
    assert f.run_cached({}, "do the thing") == "ok"


def test_agent_that_recovers_within_its_retries_still_succeeds(factory_module):
    agent = _Agent(fail_times=2, response="ok")
    f = _factory(factory_module, agent)
    assert f.run_cached({}, "do the thing") == "ok"
    assert agent.calls == 3


def test_exhausted_retries_raise_the_real_exception_not_a_timeout(factory_module):
    boom = ValueError("Provider returned error")
    f = _factory(factory_module, _Agent(fail_times=99, exc=boom))

    with pytest.raises(ValueError) as excinfo:
        f.run_cached({}, "do the thing")

    assert excinfo.value is boom
    assert "timed out" not in str(excinfo.value)


def test_exhausted_retries_do_not_claim_the_configured_timeout_elapsed(factory_module):
    """The old message quoted an 18000s timeout on a run lasting milliseconds."""
    f = _factory(factory_module, _Agent(fail_times=99), timeout=18000)

    with pytest.raises(Exception) as excinfo:
        f.run_cached({}, "do the thing")

    assert not isinstance(excinfo.value, TimeoutError)
    assert "18000" not in str(excinfo.value)


def test_a_genuinely_slow_agent_still_raises_timeout(factory_module):
    """The worker is still alive past the deadline — a real timeout."""
    f = _factory(factory_module, _Agent(block=True), timeout=1)

    with pytest.raises(TimeoutError) as excinfo:
        f.run_cached({}, "do the thing")

    assert "timed out after 1 seconds" in str(excinfo.value)


def test_retry_count_is_bounded(factory_module):
    agent = _Agent(fail_times=99)
    f = _factory(factory_module, agent)
    with pytest.raises(Exception):
        f.run_cached({}, "do the thing")
    assert agent.calls == 3
