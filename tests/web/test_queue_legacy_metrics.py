"""Legacy metrics cannot make the authenticated queue unavailable or lie."""

import sqlite3
import socket
from collections.abc import Iterator
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def queue_client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[tuple[TestClient, str]]:
    """Use real login and a fresh explicit SQLite file, never existing state."""
    database = str(tmp_path / "queue-legacy.sqlite")
    for name, value in {
        "DATABASE_PATH": database, "ADMIN_USERNAME": "queue-test", "ADMIN_PASSWORD": "synthetic-password",
        "WORKER_API_KEY": "synthetic-worker", "FEED_SYNC_ENABLED": "false",
    }.items():
        monkeypatch.setenv(name, value)

    original_socket = socket.socket

    def isolated_socket(family=socket.AF_INET, *args, **kwargs):
        if family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError("IP networking forbidden in queue tests")
        return original_socket(family, *args, **kwargs)

    monkeypatch.setattr(socket, "socket", isolated_socket)
    from web.app import app
    from web.models import Episode, Podcast
    from web import dependencies

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/login", data={"username": "queue-test", "password": "synthetic-password"}, follow_redirects=False)
        assert response.status_code == 303
        with dependencies.SessionFactory() as db:
            db.add(Podcast(id="podcast", source_rss_url="https://example.invalid/feed", title="Synthetic Podcast"))
            db.flush()
            db.add(Episode(id="completed", podcast_id="podcast", guid="completed", title="Completed metric", status="complete", llm_cost_usd=Decimal("0.123")))
            db.add(Episode(id="pending", podcast_id="podcast", guid="pending", title="Healthy pending", status="pending"))
            db.add(Episode(id="failed", podcast_id="podcast", guid="failed", title="Healthy failed", status="failed"))
            db.commit()
        yield client, database


def stored_snapshot(database: str) -> str:
    """A logical dump checks every stored value, including SQLite infinity."""
    with sqlite3.connect(database) as db:
        return "\n".join(db.iterdump())


class TestQueueLegacyMetrics:
    @pytest.mark.parametrize("value,expected", [
        (float("inf"), "—"), (float("-inf"), "—"), (float("nan"), "—"),
        (-1, "—"), (None, "—"), ("not-a-number", "—"),
        (0, "No ads detected"), (125.5, "Removed 2min 5s"), (3600, "Removed 60min 0s"),
    ])
    def test_real_queue_request_preserves_stored_metrics_and_known_values(self, queue_client, value, expected) -> None:
        """Invalid data is unknown; known zero and finite MMSS stay unchanged."""
        client, database = queue_client
        with sqlite3.connect(database) as db:
            db.execute("UPDATE episodes SET ads_removed_seconds=? WHERE id='completed'", (value,))
        before = stored_snapshot(database)
        response = client.get("/queue")
        assert response.status_code == 200
        assert "Completed metric" in response.text
        assert "Healthy pending" in response.text and "Healthy failed" in response.text
        from bs4 import BeautifulSoup
        row = BeautifulSoup(response.text, "html.parser").select_one(".recent-item")
        metric = row.select_one(".time-saved, .no-ads")
        assert metric and metric.get_text(strip=True) == expected
        assert row.select_one(".llm-cost-badge").get_text(strip=True) == "$0.123"
        assert stored_snapshot(database) == before

    def test_invalid_completed_metric_does_not_block_healthy_queue_actions(self, queue_client) -> None:
        """Poisoned history must not prevent skip/retry and subsequent rendering."""
        client, database = queue_client
        with sqlite3.connect(database) as db:
            db.execute("UPDATE episodes SET ads_removed_seconds=? WHERE id='completed'", (float("inf"),))
        assert client.get("/queue").status_code == 200
        assert client.post("/podcasts/podcast/episodes/pending/skip").json() == {"status": "skipped"}
        assert client.post("/podcasts/podcast/episodes/failed/queue").json() == {"status": "queued"}
        assert client.get("/queue").status_code == 200
        with sqlite3.connect(database) as db:
            assert db.execute("SELECT status FROM episodes WHERE id='pending'").fetchone()[0] == "skipped"
            assert db.execute("SELECT status FROM episodes WHERE id='failed'").fetchone()[0] == "pending"
            assert db.execute("SELECT ads_removed_seconds FROM episodes WHERE id='completed'").fetchone()[0] == float("inf")
