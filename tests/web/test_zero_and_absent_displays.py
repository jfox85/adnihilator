"""Server and feed displays distinguish known zero from unavailable values."""

from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from bs4 import BeautifulSoup
from fastapi.testclient import TestClient
import feedparser
import pytest

from web.models import Episode, Podcast


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    """Render only disposable local data, with no remote feeds or providers."""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "displays.db"))
    monkeypatch.setenv("FEED_SYNC_ENABLED", "false")
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "synthetic")
    monkeypatch.setenv("WORKER_API_KEY", "synthetic-worker")
    monkeypatch.setenv("R2_PUBLIC_URL", "https://media.example.invalid")
    from web.app import app
    from web import dependencies

    with TestClient(app) as result:
        result.auth = ("admin", "synthetic")
        with dependencies.SessionFactory() as db:
            db.add(Podcast(id="display", feed_token="display-feed", title="Display controls", source_rss_url="https://example.invalid/feed"))
            for name, duration, processed, removed, cost, status in [
                ("zero", 0, 0, 0, 0, "complete"),
                ("absent", None, None, None, None, "complete"),
                ("fraction", 3725, 3602, 0.5, Decimal("0.00005"), "complete"),
                ("active", 3602, None, None, None, "processing"),
            ]:
                db.add(Episode(id=name, podcast_id="display", guid=name, title=name, status=status,
                               original_audio_url="https://example.invalid/audio.mp3", duration=duration,
                               processed_duration=processed, ads_removed_seconds=removed, llm_cost_usd=cost,
                               processed_audio_key=f"synthetic/{name}.mp3" if status == "complete" else None,
                               claimed_at=datetime.utcnow() if status == "processing" else None))
            db.commit()
        yield result


class TestZeroAndAbsentDisplays:
    """The same known/unknown facts have consistent meanings on every surface."""

    def test_original_zero_duration_is_visible_and_absent_duration_is_omitted(self, client: TestClient) -> None:
        """The metadata guard checks availability rather than truthiness."""
        page = BeautifulSoup(client.get("/podcasts/display/episodes").text, "html.parser")
        assert "00:00:00" in page.select_one('[data-id="zero"] .episode-meta').get_text()
        assert "00:00:00" not in page.select_one('[data-id="absent"] .episode-meta').get_text()

    def test_absent_removed_seconds_are_unknown_on_both_pages(self, client: TestClient) -> None:
        """Missing removal data cannot claim that detection found no ads."""
        episodes = BeautifulSoup(client.get("/podcasts/display/episodes").text, "html.parser")
        queue = BeautifulSoup(client.get("/queue").text, "html.parser")
        assert episodes.select_one('[data-id="absent"] .stats-ads').get_text(strip=True) == "—"
        recent = next(row for row in queue.select('.recent-item') if row.select_one('.recent-item-title').get_text(strip=True) == "absent")
        assert "Removed —" in recent.get_text(" ", strip=True)
        assert "No ads detected" not in recent.get_text()

    def test_known_zero_removal_is_explicit_on_both_pages(self, client: TestClient) -> None:
        """Zero means a completed no-ad result, while unknown remains distinct."""
        episodes = BeautifulSoup(client.get("/podcasts/display/episodes").text, "html.parser")
        queue = BeautifulSoup(client.get("/queue").text, "html.parser")
        assert episodes.select_one('[data-id="zero"] .stats-ads').get_text(strip=True) == "No ads detected"
        recent = next(row for row in queue.select('.recent-item') if row.select_one('.recent-item-title').get_text(strip=True) == "zero")
        assert "No ads detected" in recent.get_text()

    def test_feed_uses_shared_format_and_preserves_zero_and_missing(self, client: TestClient) -> None:
        """RSS and UI use the same HH:MM:SS contract for known durations."""
        response = client.get("/feed/display-feed.xml")
        assert response.status_code == 200
        parsed = feedparser.parse(response.content)
        assert not parsed.bozo
        entries = {entry.id: entry for entry in parsed.entries}
        assert entries["zero"].itunes_duration == "00:00:00"
        assert entries["fraction"].itunes_duration == "01:00:02"
        assert "itunes_duration" not in entries["absent"]

    def test_real_callback_zero_values_survive_status_html_and_feed(self, client: TestClient) -> None:
        """A worker's zero values retain meaning after persistence and readback."""
        response = client.post("/api/queue/active/complete", headers={"X-Worker-API-Key": "synthetic-worker"}, json={
            "processed_audio_key": "synthetic/active.mp3", "processed_duration": 0,
            "ads_removed_seconds": 0, "llm_cost_usd": 0,
        })
        assert response.status_code == 200
        status = client.get("/podcasts/display/episodes/status").json()["episodes"]["active"]
        assert (status["processed_duration"], status["ads_removed_seconds"], status["llm_cost_usd"]) == (0, 0, 0)
        page = BeautifulSoup(client.get("/podcasts/display/episodes").text, "html.parser")
        row = page.select_one('[data-id="active"]')
        assert row.select_one('.stats-ads').get_text(strip=True) == "No ads detected"
        assert "00:00:00" in row.select_one('.stats-duration').get_text()
        assert "$0.0000" in row.select_one('.stats-cost').get_text()
        feed = feedparser.parse(client.get("/feed/display-feed.xml").content)
        active = next(entry for entry in feed.entries if entry.id == "active")
        assert active.itunes_duration == "00:00:00"
