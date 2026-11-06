"""Real SQLite/HTTP regressions for stale snapshots and historical numbers."""

import socket
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import feedparser
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session

from web.models import Episode, Podcast

HEADERS = {"X-Worker-API-Key": "synthetic-worker-key"}
AUTH = ("synthetic-admin", "synthetic-password")


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[tuple]:
    """Use a fresh WAL database and forbid all outbound TCP traffic."""
    connect = socket.socket.connect

    def local_only(sock: socket.socket, address: Any) -> Any:
        if sock.family == socket.AF_UNIX:
            return connect(sock, address)
        raise AssertionError("Network forbidden in recovery/read-path regressions")

    monkeypatch.setattr(socket.socket, "connect", local_only)
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "safety.db"))
    monkeypatch.setenv("ADMIN_USERNAME", AUTH[0])
    monkeypatch.setenv("ADMIN_PASSWORD", AUTH[1])
    monkeypatch.setenv("WORKER_API_KEY", HEADERS["X-Worker-API-Key"])
    monkeypatch.setenv("FEED_SYNC_ENABLED", "false")
    monkeypatch.setenv("R2_PUBLIC_URL", "https://media.example.invalid")
    from web.app import app
    from web import dependencies

    with TestClient(app, raise_server_exceptions=False) as client:
        factory = dependencies.SessionFactory
        with factory() as db:
            db.add(Podcast(id="podcast", feed_token="synthetic-feed", title="Synthetic", source_rss_url="https://example.invalid/feed"))
            db.add(Episode(id="episode", podcast_id="podcast", guid="synthetic-episode", title="Synthetic",
                           status="processing", claimed_at=datetime.utcnow() - timedelta(hours=3),
                           retry_count=0, original_audio_url="https://example.invalid/audio.mp3"))
            db.commit()
        yield client, factory


def claim(client: TestClient) -> Any:
    """Use an independent ASGI portal with the existing isolated app lifespan."""
    other = TestClient(client.app, raise_server_exceptions=False)
    try:
        response = other.post("/api/queue/claim", headers=HEADERS)
        assert response.status_code == 200, response.text
        return response.json()
    finally:
        other.close()


class TestConditionalStaleRecovery:
    """Only the still-expired snapshot can consume a recovery attempt."""

    @pytest.mark.parametrize("repetition", range(2))
    def test_two_stale_recoveries_cannot_return_the_same_claim(self, harness: tuple, repetition: int) -> None:
        """A stale second loader cannot reset a freshly re-claimed episode."""
        client, factory = harness
        both_loaded = threading.Barrier(2)
        first_response = threading.Event()
        lock = threading.Lock()
        loads = 0

        def pause_stale_load(session: Session, instance: Any) -> None:
            nonlocal loads
            if not isinstance(instance, Episode) or instance.id != "episode" or instance.claimed_at is None:
                return
            if instance.claimed_at > datetime.utcnow() - timedelta(hours=2):
                return
            with lock:
                loads += 1
                position = loads
            if position <= 2:
                both_loaded.wait(timeout=10)
                if position == 2:
                    assert first_response.wait(timeout=10)

        def request() -> Any:
            try:
                return claim(client)
            finally:
                first_response.set()

        event.listen(Session, "loaded_as_persistent", pause_stale_load)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(request) for _ in range(2)]
                results = [future.result(timeout=15) for future in futures]
        finally:
            first_response.set()
            event.remove(Session, "loaded_as_persistent", pause_stale_load)
        assert loads == 2
        assert sum(result is not None for result in results) == 1
        with factory() as db:
            episode = db.get(Episode, "episode")
            assert episode.status == "processing"
            assert episode.retry_count == 1
            assert episode.claimed_at > datetime.utcnow() - timedelta(minutes=1)

    @pytest.mark.parametrize("repetition", range(2))
    def test_progress_renewal_survives_an_already_loaded_recovery(self, harness: tuple, repetition: int) -> None:
        """Heartbeat renewal wins over a recovery holding an old ORM snapshot."""
        client, factory = harness
        loaded = threading.Event()
        renewed = threading.Event()
        lock = threading.Lock()
        paused = False

        def pause_once(session: Session, instance: Any) -> None:
            nonlocal paused
            if not isinstance(instance, Episode) or instance.id != "episode":
                return
            with lock:
                if paused:
                    return
                paused = True
            loaded.set()
            assert renewed.wait(timeout=10)

        event.listen(Session, "loaded_as_persistent", pause_once)
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(claim, client)
                assert loaded.wait(timeout=10)
                response = client.post("/api/queue/episode/progress", headers=HEADERS,
                                       json={"step": "alive", "percent": 0})
                assert response.status_code == 200
                renewed.set()
                assert future.result(timeout=15) is None
        finally:
            renewed.set()
            event.remove(Session, "loaded_as_persistent", pause_once)
        with factory() as db:
            episode = db.get(Episode, "episode")
            assert (episode.status, episode.retry_count, episode.progress_step, episode.progress_percent) == ("processing", 0, "alive", 0)
            assert episode.claimed_at > datetime.utcnow() - timedelta(minutes=1)

    @pytest.mark.parametrize("retry,expected_status,expected_retry", [(0, "pending", 1), (1, "pending", 2), (2, "failed", 2)])
    def test_normal_recovery_keeps_existing_retry_budget(
        self, harness: tuple, retry: int, expected_status: str, expected_retry: int,
    ) -> None:
        """Atomic recovery does not grant extra attempts or change exhaustion."""
        from web.routes.api import requeue_stuck_episodes
        _, factory = harness
        with factory() as db:
            episode = db.get(Episode, "episode")
            episode.retry_count = retry
            db.commit()
            assert requeue_stuck_episodes(db) == 1
            db.expire_all()
            episode = db.get(Episode, "episode")
            assert (episode.status, episode.retry_count, episode.claimed_at) == (expected_status, expected_retry, None)
            assert requeue_stuck_episodes(db) == 0


def poison(factory: Any, processed: float | None, original: float | None = 80) -> tuple:
    """Simulate historical rows without using the now-validated write API."""
    with factory.kw["bind"].begin() as connection:
        connection.exec_driver_sql(
            "UPDATE episodes SET status='complete', processed_audio_key='synthetic.mp3', "
            "processed_duration=?, duration=?, ads_removed_seconds=?, llm_cost_usd=?, progress_percent=? WHERE id='episode'",
            (processed, original, processed, processed, processed),
        )
    return stored_numbers(factory)


def stored_numbers(factory: Any) -> tuple:
    """Read raw values so ORM conversions cannot conceal accidental cleanup."""
    with factory.kw["bind"].connect() as connection:
        return tuple(connection.exec_driver_sql(
            "SELECT processed_duration,duration,ads_removed_seconds,llm_cost_usd,progress_percent FROM episodes WHERE id='episode'"
        ).one())


class TestHistoricalNumericReads:
    """Invalid legacy values are unavailable at read time without DB writes."""

    @pytest.mark.parametrize("value,expected", [(float("inf"), None), (float("-inf"), None),
                                               (-1, None), (None, None), (0, 0), (12.5, 12.5)])
    def test_status_remains_json_safe_and_preserves_zero(self, harness: tuple, value: float | None, expected: float | None) -> None:
        """Old values cannot take the polling endpoint down with a JSON 500."""
        client, factory = harness
        before = poison(factory, value)
        response = client.get("/podcasts/podcast/episodes/status", auth=AUTH)
        assert response.status_code == 200, response.text
        episode = response.json()["episodes"]["episode"]
        for field in ["processed_duration", "ads_removed_seconds", "llm_cost_usd", "progress_percent"]:
            assert episode[field] == expected
        assert stored_numbers(factory) == before

    @pytest.mark.parametrize("processed,original,expected", [
        (float("inf"), 80, "1:20"), (float("-inf"), 80, "1:20"), (-10, 80, "1:20"),
        (None, 80, "1:20"), (0, 80, "0:00"), (float("inf"), float("inf"), None),
        (None, -5, None), (None, 0, "0:00"), (float("nan"), 90, "1:30"),
    ])
    def test_feed_uses_only_valid_duration_without_changing_rows(
        self, harness: tuple, processed: float | None, original: float | None, expected: str | None,
    ) -> None:
        """Local feedparser validates output, including known zero and no duration."""
        client, factory = harness
        before = poison(factory, processed, original)
        response = client.get("/feed/synthetic-feed.xml")
        assert response.status_code == 200, response.text
        parsed = feedparser.parse(response.content)
        assert not parsed.bozo
        assert len(parsed.entries) == 1
        duration = parsed.entries[0].get("itunes_duration")
        if expected is None:
            assert duration is None
        else:
            # This test checks numeric read safety; both standard RSS clock
            # forms and the shared UI's padded form preserve the same seconds.
            def seconds(value: str) -> int:
                total = 0
                for part in value.split(":"):
                    total = total * 60 + int(part)
                return total
            assert duration is not None
            assert seconds(duration) == seconds(expected)
        assert stored_numbers(factory) == before
