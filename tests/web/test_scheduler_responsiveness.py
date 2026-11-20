"""Scheduled feed sync must never block request handling."""

import asyncio
import threading
import time
from unittest.mock import patch

import pytest

from web.services import scheduler as scheduler_module
from web.services.scheduler import FeedSyncScheduler


class _Session:
    def close(self) -> None:
        pass


def test_sync_runs_off_the_event_loop(tmp_path, monkeypatch) -> None:
    """The event loop keeps serving while a slow (blocking) sync runs."""
    monkeypatch.setenv("FEED_SYNC_LOCK_PATH", str(tmp_path / "feed-sync.lock"))

    def slow_sync(db):
        time.sleep(0.5)  # feedparser and SQLAlchemy calls are blocking
        return {}

    async def scenario() -> int:
        ticks = 0
        done = False

        async def heartbeat() -> None:
            nonlocal ticks
            while not done:
                await asyncio.sleep(0.01)
                ticks += 1

        sched = FeedSyncScheduler(_Session)
        beat = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)
        with patch("web.services.feed_sync.sync_all_podcasts", side_effect=slow_sync):
            await sched._sync_all_feeds()
        done = True
        await beat
        return ticks

    # A blocked loop gets ~1 tick in 0.5s; a free loop gets dozens.
    assert asyncio.run(scenario()) >= 10


def test_only_one_process_syncs_at_a_time(tmp_path, monkeypatch) -> None:
    """A second worker skips the sync while another holds the lock."""
    monkeypatch.setenv("FEED_SYNC_LOCK_PATH", str(tmp_path / "feed-sync.lock"))
    calls = []
    first_running = threading.Event()
    release = threading.Event()

    def sync(db):
        calls.append(threading.get_ident())
        first_running.set()
        release.wait(5)
        return {}

    with patch("web.services.feed_sync.sync_all_podcasts", side_effect=sync):
        first = threading.Thread(target=scheduler_module._run_sync_once, args=(_Session,))
        first.start()
        assert first_running.wait(5)
        assert scheduler_module._run_sync_once(_Session) is False  # skipped: lock held
        release.set()
        first.join(5)
    assert len(calls) == 1


def test_sync_failure_is_contained(tmp_path, monkeypatch) -> None:
    """A failing sync is logged, not raised, and releases the lock."""
    monkeypatch.setenv("FEED_SYNC_LOCK_PATH", str(tmp_path / "feed-sync.lock"))
    with patch("web.services.feed_sync.sync_all_podcasts", side_effect=RuntimeError("boom")):
        assert scheduler_module._run_sync_once(_Session) is True
    with patch("web.services.feed_sync.sync_all_podcasts", return_value={}):
        assert scheduler_module._run_sync_once(_Session) is True  # lock was released


def test_feed_fetches_have_a_network_timeout() -> None:
    """A stalled feed server cannot hang a sync forever."""
    import socket

    from web.services import feed_sync

    seen = []

    def fake_parse(url, **kwargs):
        seen.append(socket.getdefaulttimeout())
        raise RuntimeError("stop")

    class _Podcast:
        source_rss_url = "https://example.invalid/feed.xml"

    with patch.object(feed_sync.feedparser, "parse", side_effect=fake_parse):
        with pytest.raises(RuntimeError):
            feed_sync._sync_podcast_entries(None, _Podcast())
    assert seen and seen[0] is not None and seen[0] <= 60
    assert socket.getdefaulttimeout() is None  # restored afterwards
