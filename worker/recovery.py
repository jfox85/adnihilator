"""Bounded, destination-pinned fetches for explicit recovery only.

Legacy audio processing intentionally retains its existing downloader.
"""

import http.client
import ipaddress
import json
import queue
import socket
import ssl
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator
from urllib.parse import urljoin, urlsplit

from adnihilator.external_transcript import (extract_transcript_url, parse_lexfridman_transcript,
                                           parse_substack_transcript)
from adnihilator.models import TranscriptSegment

AUDIO_MAX_BYTES = 512 * 1024 * 1024
AUDIO_MAX_SECONDS = 900.0
EXTERNAL_MAX_BYTES = 8 * 1024 * 1024
EXTERNAL_MAX_SECONDS = 30.0
MAX_REDIRECTS = 5
_DNS_SLOTS = threading.BoundedSemaphore(2)


class RecoveryFetchError(ValueError):
    """Safe failure without source URLs or response content."""


def validate_url(url: str) -> tuple[str, str, int]:
    try:
        parsed = urlsplit(url)
        if (len(url) > 8000 or any(ord(c) < 33 for c in url) or parsed.scheme not in ('http', 'https')
                or not parsed.hostname or parsed.username is not None or parsed.password is not None
                or parsed.fragment or parsed.port not in (None, 80, 443)):
            raise ValueError()
        return parsed.scheme, parsed.hostname.encode('idna').decode(), parsed.port or (443 if parsed.scheme == 'https' else 80)
    except (ValueError, UnicodeError):
        raise RecoveryFetchError('Recovery URL denied') from None


def resolve_public(host: str, port: int, timeout: float) -> str:
    """Bound DNS wait and reject every non-public answer; pin the chosen address."""
    if not _DNS_SLOTS.acquire(blocking=False):
        raise RecoveryFetchError("Recovery resolver capacity exceeded")
    result: queue.Queue = queue.Queue(maxsize=1)
    def resolve() -> None:
        try:
            result.put(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except Exception:
            result.put(None)
        finally:
            _DNS_SLOTS.release()
    threading.Thread(target=resolve, daemon=True).start()
    try:
        answers = result.get(timeout=min(10.0, timeout))
    except queue.Empty:
        raise RecoveryFetchError('Recovery DNS timeout') from None
    if not answers:
        raise RecoveryFetchError('Recovery destination unavailable')
    addresses = [entry[4][0] for entry in answers]
    def public(address: str) -> bool:
        ip = ipaddress.ip_address(address)
        return ip.is_global and not ip.is_multicast and not ip.is_reserved and not (isinstance(ip, ipaddress.IPv6Address) and (ip.sixtofour or ip.teredo))
    if any(not public(ip) for ip in addresses):
        raise RecoveryFetchError('Recovery destination denied')
    return addresses[0]


class PinnedConnection(http.client.HTTPConnection):
    """Connect to the validated address, retaining Host and TLS name validation."""

    def __init__(self, host: str, port: int, address: str, secure: bool, timeout: float) -> None:
        super().__init__(host, port, timeout=timeout)
        self.address, self.secure = address, secure

    def connect(self) -> None:
        self.sock = socket.create_connection((self.address, self.port), self.timeout)
        if self.secure:
            self.sock = ssl.create_default_context().wrap_socket(self.sock, server_hostname=self.host)


def fetch(url: str, write: Callable[[bytes], object], max_bytes: int, seconds: float,
          heartbeat: Callable[[], None] = lambda: None) -> None:
    """No proxy/env credentials, decompression or automatic redirects; total deadline."""
    deadline = time.monotonic() + seconds
    used = 0
    for redirect in range(MAX_REDIRECTS + 1):
        heartbeat()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RecoveryFetchError('Recovery deadline exceeded')
        scheme, host, port = validate_url(url)
        address = resolve_public(host, port, remaining)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RecoveryFetchError('Recovery deadline exceeded')
        connection = PinnedConnection(host, port, address, scheme == 'https', min(10.0, remaining))
        try:
            parsed = urlsplit(url)
            connection.request('GET', (parsed.path or '/') + ('?' + parsed.query if parsed.query else ''),
                               headers={'Accept-Encoding': 'identity', 'User-Agent': 'Adnihilator-Transcript-Recovery/1'})
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RecoveryFetchError('Recovery deadline exceeded')
            if connection.sock:
                connection.sock.settimeout(min(10.0, remaining))
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                destination = urljoin(url, response.getheader('Location') or '')
                next_scheme, _, _ = validate_url(destination)
                if redirect == MAX_REDIRECTS or destination == url or (scheme == 'https' and next_scheme != 'https'):
                    raise RecoveryFetchError('Recovery redirect denied')
                url = destination
                continue
            if response.status != 200 or response.getheader('Content-Encoding', 'identity').lower() != 'identity':
                raise RecoveryFetchError('Recovery response denied')
            length = response.getheader('Content-Length')
            if length and (not length.isdigit() or int(length) > max_bytes):
                raise RecoveryFetchError('Recovery size exceeded')
            while True:
                heartbeat()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RecoveryFetchError('Recovery deadline exceeded')
                # read1 avoids indefinitely waiting to fill a chunk on trickle streams.
                if connection.sock:
                    connection.sock.settimeout(min(10.0, remaining))
                chunk = response.read1(min(65536, max_bytes - used + 1))
                if not chunk:
                    return
                used += len(chunk)
                if used > max_bytes:
                    raise RecoveryFetchError('Recovery size exceeded')
                write(chunk)
        except (OSError, http.client.HTTPException):
            raise RecoveryFetchError('Recovery transport failed') from None
        finally:
            connection.close()
    raise RecoveryFetchError('Recovery redirect denied')


def download_audio(url: str, path: Path, heartbeat: Callable[[], None]) -> None:
    with path.open('wb') as output:
        fetch(url, output.write, AUDIO_MAX_BYTES, AUDIO_MAX_SECONDS, heartbeat)


def fetch_external(url: str, heartbeat: Callable[[], None]) -> list[TranscriptSegment] | None:
    """Bound both page and extracted transcript fetch under one 30-second budget."""
    deadline = time.monotonic() + EXTERNAL_MAX_SECONDS
    def read(target: str) -> bytes:
        content = bytearray()
        fetch(target, content.extend, EXTERNAL_MAX_BYTES, max(.001, deadline - time.monotonic()), heartbeat)
        return bytes(content)
    try:
        _, host, _ = validate_url(url)
        if host == 'lexfridman.com' or host == 'www.lexfridman.com':
            target = url.rstrip('/')
            return parse_lexfridman_transcript(read(target if target.endswith('-transcript') else target + '-transcript').decode('utf-8'))
        page = read(url).decode('utf-8')
        target = extract_transcript_url(page)
        if target:
            # Fetch validates/pins the extracted destination independently.
            data = json.loads(read(target))
            return parse_substack_transcript(data) if isinstance(data, list) else None
    except (RecoveryFetchError, ValueError, UnicodeError, KeyError, TypeError):
        return None  # Existing local ASR fallback, never unbounded legacy fetching.
    return None


@contextmanager
def renew_lease(heartbeat: Callable[[], None], interval: float = 30.0) -> Iterator[None]:
    """Keep the lease alive across download, external fetch and model stages."""
    stopped = threading.Event()
    def safe_heartbeat() -> None:
        try:
            heartbeat()
        except Exception:
            pass
    def run() -> None:
        while not stopped.wait(interval):
            safe_heartbeat()
    safe_heartbeat()
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=1.0)
