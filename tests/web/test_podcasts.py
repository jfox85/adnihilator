"""Tests for podcast management routes."""

import pytest
from fastapi.testclient import TestClient

from web.models import Podcast


@pytest.fixture
def client(monkeypatch):
    """Create test client with mocked dependencies."""
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "testpass")
    monkeypatch.setenv("WORKER_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_PATH", ":memory:")

    from web.app import app
    with TestClient(app) as client:
        yield client


def test_list_podcasts_empty(client):
    """Test listing podcasts when empty."""
    response = client.get("/podcasts", auth=("admin", "testpass"))
    assert response.status_code == 200
    assert response.json() == []


def test_add_podcast(client):
    """Test adding a podcast."""
    response = client.post(
        "/podcasts",
        data={"source_rss_url": "https://example.com/feed.xml"},
        auth=("admin", "testpass"),
        follow_redirects=False,
    )
    # Should redirect after successful add
    assert response.status_code in (302, 303)

    # Verify it appears in the list
    response = client.get("/podcasts", auth=("admin", "testpass"))
    assert response.status_code == 200
    podcasts = response.json()
    assert len(podcasts) == 1
    assert podcasts[0]["source_rss_url"] == "https://example.com/feed.xml"


def test_delete_podcast(client):
    """Test deleting a podcast."""
    # Create a podcast first
    response = client.post(
        "/podcasts",
        data={"source_rss_url": "https://example.com/feed.xml"},
        auth=("admin", "testpass"),
        follow_redirects=False,
    )
    assert response.status_code in (302, 303)

    # Get the podcast ID
    response = client.get("/podcasts", auth=("admin", "testpass"))
    podcasts = response.json()
    assert len(podcasts) == 1
    podcast_id = podcasts[0]["id"]

    # Delete it
    response = client.delete(
        f"/podcasts/{podcast_id}",
        auth=("admin", "testpass"),
    )
    assert response.status_code == 200

    # Verify deleted
    response = client.get("/podcasts", auth=("admin", "testpass"))
    assert response.json() == []
