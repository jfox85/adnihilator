"""Synthetic transports only; no URLs, DNS or sockets are contacted."""

import socket
import threading
from unittest.mock import Mock

import pytest

from worker import recovery


class Reply:
    def __init__(self, status=200, headers=None, chunks=None):
        self.status, self.headers, self.chunks = status, headers or {}, list(chunks or [b'audio', b''])

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def read1(self, size):
        return self.chunks.pop(0) if self.chunks else b''


@pytest.fixture
def network(monkeypatch):
    replies, connections = [], []
    monkeypatch.setattr(recovery, 'resolve_public', lambda host, port, timeout: '93.184.216.34')
    def connection(*args):
        value = Mock()
        value.getresponse.return_value = replies.pop(0)
        connections.append((args, value))
        return value
    monkeypatch.setattr(recovery, 'PinnedConnection', connection)
    return replies, connections


@pytest.mark.parametrize('url', ['file:///tmp/file', 'http://user:pass@example.com/', 'https://example.com:8080/', 'http://example.com/\nheader', 'http://example.com/#fragment'])
def test_reject_invalid_url_without_network(url):
    with pytest.raises(recovery.RecoveryFetchError): recovery.validate_url(url)


@pytest.mark.parametrize('address', ['127.0.0.1', '10.0.0.1', '169.254.169.254', '192.168.1.1', '::1', 'fc00::1', '::ffff:127.0.0.1', '0.0.0.0', '224.0.0.1', '2002:7f00:1::'])
def test_private_and_mixed_dns_denied(monkeypatch, address):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: [(None, None, None, None, ('93.184.216.34', 443)), (None, None, None, None, (address, 443))])
    with pytest.raises(recovery.RecoveryFetchError): recovery.resolve_public('fixture.example', 443, .2)


def test_dns_deadline(monkeypatch):
    stopped = threading.Event()
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: stopped.wait(.1))
    with pytest.raises(recovery.RecoveryFetchError): recovery.resolve_public('fixture.example', 443, .001)
    stopped.set()


def test_pinned_tls_connect_validates_hostname(monkeypatch):
    sock, context = Mock(), Mock()
    monkeypatch.setattr(socket, 'create_connection', Mock(return_value=sock))
    monkeypatch.setattr(recovery.ssl, 'create_default_context', lambda: context)
    connection = recovery.PinnedConnection('fixture.example', 443, '93.184.216.34', True, 2)
    connection.connect()
    socket.create_connection.assert_called_once_with(('93.184.216.34', 443), 2)
    context.wrap_socket.assert_called_once_with(sock, server_hostname='fixture.example')


def test_stream_size_headers_and_heartbeat(network):
    replies, connections = network
    replies.append(Reply(chunks=[b'ab', b'cd', b'']))
    output, beat = bytearray(), Mock()
    recovery.fetch('https://fixture.example/audio', output.extend, 4, 10, beat)
    assert output == b'abcd' and beat.call_count >= 3
    assert connections[0][0][:4] == ('fixture.example', 443, '93.184.216.34', True)
    assert connections[0][1].request.call_args.kwargs['headers']['Accept-Encoding'] == 'identity'
    connections[0][1].close.assert_called_once()
    replies.append(Reply(chunks=[b'abcde']))
    with pytest.raises(recovery.RecoveryFetchError): recovery.fetch('https://fixture.example/audio', output.extend, 4, 10)
    replies.append(Reply(headers={'Content-Length': '5'}))
    with pytest.raises(recovery.RecoveryFetchError): recovery.fetch('https://fixture.example/audio', output.extend, 4, 10)
    replies.append(Reply(headers={'Content-Encoding': 'gzip'}))
    with pytest.raises(recovery.RecoveryFetchError): recovery.fetch('https://fixture.example/audio', output.extend, 4, 10)


def test_redirect_revalidates_destination_and_denies_downgrade(network, monkeypatch):
    replies, connections = network
    replies.append(Reply(302, {'Location': 'https://internal.example/audio'}))
    def resolve(host, *args):
        if host == 'internal.example': raise recovery.RecoveryFetchError('denied')
        return '93.184.216.34'
    monkeypatch.setattr(recovery, 'resolve_public', resolve)
    with pytest.raises(recovery.RecoveryFetchError): recovery.fetch('https://fixture.example/audio', lambda x: None, 10, 10)
    assert len(connections) == 1
    replies.append(Reply(302, {'Location': 'http://fixture.example/audio'}))
    with pytest.raises(recovery.RecoveryFetchError): recovery.fetch('https://fixture.example/audio', lambda x: None, 10, 10)


def test_redirect_limit_and_deadline(network, monkeypatch):
    replies, _ = network
    replies.extend(Reply(302, {'Location': 'https://fixture.example/' + str(n)}) for n in range(6))
    with pytest.raises(recovery.RecoveryFetchError): recovery.fetch('https://fixture.example/start', lambda x: None, 100, 10)
    replies.clear(); replies.append(Reply())
    clock = iter([0, 0, 0, 0, 11])
    monkeypatch.setattr(recovery.time, 'monotonic', lambda: next(clock))
    with pytest.raises(recovery.RecoveryFetchError): recovery.fetch('https://fixture.example/start', lambda x: None, 100, 10)


def test_external_fetch_reuses_bounded_path(monkeypatch):
    calls = []
    def fetch(url, write, maximum, seconds, heartbeat):
        calls.append((maximum, seconds)); heartbeat()
        write(b'<html>no supplied transcript</html>')
    monkeypatch.setattr(recovery, 'fetch', fetch)
    beat = Mock()
    assert recovery.fetch_external('https://fixture.example/page', beat) is None
    assert calls[0][0] == 8 * 1024 * 1024 and calls[0][1] <= 30
    beat.assert_called_once()


def test_lease_timer_renews_blocking_stage_and_stops():
    beat, observed = Mock(), threading.Event()
    beat.side_effect = lambda: observed.set() if beat.call_count >= 2 else None
    with recovery.renew_lease(beat, .005):
        assert observed.wait(.5)
    count = beat.call_count
    assert count >= 2
    # Context exit joins its own timer; no later renewals after return.
    assert beat.call_count == count
