"""Tests for RSS feed routes."""

import pytest
from fastapi.testclient import TestClient

from web.models import Episode, EpisodeStatus


@pytest.fixture(scope="function")
def client(monkeypatch, tmp_path):
    """Create test client."""
    # Use a temporary file for the database so all connections see the same data
    db_path = str(tmp_path / "test.db")

    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "testpass")
    monkeypatch.setenv("WORKER_API_KEY", "test-key")
    monkeypatch.setenv("R2_PUBLIC_URL", "https://test-bucket.r2.dev")
    monkeypatch.setenv("DATABASE_PATH", db_path)

    from web.app import app
    with TestClient(app) as client:
        yield client


def create_test_podcast(client):
    """Helper to create a test podcast via API."""
    response = client.post(
        "/podcasts",
        data={"source_rss_url": "https://example.com/feed.xml"},
        auth=("admin", "testpass"),
        follow_redirects=False,
    )
    assert response.status_code in (302, 303)

    # Get the podcast
    response = client.get("/podcasts", auth=("admin", "testpass"))
    podcasts = response.json()
    assert len(podcasts) >= 1
    return podcasts[0]


def create_test_episode(client, podcast_id, status, title="Episode 1", processed_audio_key=None):
    """Helper to create a test episode directly in the database."""
    from web.dependencies import SessionFactory
    from web.models import Podcast

    db = SessionFactory()
    try:
        episode = Episode(
            podcast_id=podcast_id,
            guid=f"ep-{title}",
            title=title,
            original_audio_url=f"https://example.com/{title}.mp3",
            status=status.value,
            processed_audio_key=processed_audio_key,
            processed_duration=3600.0 if status == EpisodeStatus.COMPLETE else None,
        )
        db.add(episode)
        db.commit()
        episode_id = episode.id
        return episode_id
    finally:
        db.close()


def test_feed_not_found(client):
    """Test requesting non-existent feed."""
    response = client.get("/feed/nonexistent-token.xml")
    assert response.status_code == 404


def test_feed_returns_xml(client):
    """Test that feed returns valid XML."""
    podcast = create_test_podcast(client)
    create_test_episode(
        client,
        podcast["id"],
        EpisodeStatus.COMPLETE,
        title="Episode 1",
        processed_audio_key=f"{podcast['id']}/ep1.mp3"
    )

    response = client.get(f"/feed/{podcast['feed_token']}.xml")

    assert response.status_code == 200
    assert "application/xml" in response.headers["content-type"]
    assert "<rss" in response.text
    # Podcast title might be None if not fetched from feed yet
    if podcast["title"]:
        assert podcast["title"] in response.text


def test_feed_contains_complete_episodes(client):
    """Test that feed includes only complete episodes."""
    podcast = create_test_podcast(client)

    # Add a complete episode
    create_test_episode(
        client,
        podcast["id"],
        EpisodeStatus.COMPLETE,
        title="Episode 1",
        processed_audio_key=f"{podcast['id']}/ep1.mp3"
    )

    # Add a pending episode (should not appear)
    create_test_episode(
        client,
        podcast["id"],
        EpisodeStatus.PENDING,
        title="Pending Episode"
    )

    response = client.get(f"/feed/{podcast['feed_token']}.xml")
    assert "Episode 1" in response.text
    assert "Pending Episode" not in response.text


def test_feed_audio_urls_point_to_r2(client):
    """Test that audio URLs point to R2."""
    podcast = create_test_podcast(client)
    create_test_episode(
        client,
        podcast["id"],
        EpisodeStatus.COMPLETE,
        title="Episode 1",
        processed_audio_key=f"{podcast['id']}/ep1.mp3"
    )

    response = client.get(f"/feed/{podcast['feed_token']}.xml")

    assert "https://test-bucket.r2.dev" in response.text
