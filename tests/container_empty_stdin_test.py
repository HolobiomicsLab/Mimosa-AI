"""Fast-exiting controls need no stdin delivery; actual payloads stay checked."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from sources.core.container_runner import _exchange


def closed_control(error):
    """Represent closed stdin with completed, still-readable output streams."""
    stdout, stderr = asyncio.StreamReader(), asyncio.StreamReader()
    stdout.feed_data(b'control output')
    stdout.feed_eof()
    stderr.feed_data(b'control diagnostic')
    stderr.feed_eof()
    writer = Mock()
    writer.drain = AsyncMock(side_effect=error('closed control pipe'))
    return SimpleNamespace(stdin=writer, stdout=stdout, stderr=stderr,
                           wait=AsyncMock(return_value=0))


@pytest.mark.parametrize('error', [BrokenPipeError, ConnectionResetError])
def test_empty_payload_does_not_write_or_drain_closed_control(error):
    async def run():
        process = closed_control(error)
        result = await _exchange(process, b'', 1024)
        assert result == (b'control output', b'control diagnostic')
        process.stdin.write.assert_not_called()
        process.stdin.drain.assert_not_awaited()
        process.stdin.close.assert_called_once()
        process.wait.assert_awaited_once()
    asyncio.run(run())


@pytest.mark.parametrize('error', [BrokenPipeError, ConnectionResetError])
def test_nonempty_payload_delivery_failure_is_not_suppressed(error):
    async def run():
        process = closed_control(error)
        with pytest.raises(error, match='closed control pipe'):
            await _exchange(process, b'required request', 1024)
        process.stdin.write.assert_called_once_with(b'required request')
    asyncio.run(run())
