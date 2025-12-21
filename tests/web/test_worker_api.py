"""Tests for worker API routes."""

import pytest
from fastapi.testclient import TestClient

from web.models import EpisodeStatus


@pytest.fixture(scope="function")
def client(monkeypatch, tmp_path):
    """Create test client with mocked dependencies."""
    import os

    # Use a temporary file for the database so all connections see the same data
    db_path = str(tmp_path / "test.db")

    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "testpass")
    monkeypatch.setenv("WORKER_API_KEY", "test-worker-key")
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

    # Get the podcast ID
    response = client.get("/podcasts", auth=("admin", "testpass"))
    podcasts = response.json()
    assert len(podcasts) >= 1
    return podcasts[0]["id"]


def create_test_episode(client, podcast_id, status=EpisodeStatus.PENDING):
    """Helper to create a test episode directly in the database."""
    from web.dependencies import SessionFactory
    from web.models import Episode

    db = SessionFactory()
    try:
        episode = Episode(
            podcast_id=podcast_id,
            guid="test-episode-1",
            title="Test Episode",
            original_audio_url="https://example.com/test.mp3",
            status=status.value,
        )
        db.add(episode)
        db.commit()
        db.refresh(episode)
        return episode.id
    finally:
        db.close()


def get_episode_from_db(episode_id):
    """Helper to fetch an episode from the database."""
    from web.dependencies import SessionFactory
    from web.models import Episode

    db = SessionFactory()
    try:
        episode = db.query(Episode).filter_by(id=episode_id).first()
        # Eagerly load all attributes before closing session
        data = {
            "id": episode.id,
            "status": episode.status,
            "retry_count": episode.retry_count,
            "error_message": episode.error_message,
            "processed_audio_key": episode.processed_audio_key,
            "claimed_at": episode.claimed_at,
        }
        return data
    finally:
        db.close()


def test_claim_requires_auth(client):
    """Test that claim endpoint requires API key."""
    response = client.post("/api/queue/claim")
    assert response.status_code == 422  # Missing required header


def test_claim_no_pending_jobs(client):
    """Test claiming when no jobs are pending."""
    response = client.post(
        "/api/queue/claim",
        headers={"X-Worker-API-Key": "test-worker-key"},
    )
    assert response.status_code == 200
    assert response.json() is None


def test_claim_returns_episode(client):
    """Test claiming returns a pending episode."""
    podcast_id = create_test_podcast(client)
    create_test_episode(client, podcast_id, status=EpisodeStatus.PENDING)

    response = client.post(
        "/api/queue/claim",
        headers={"X-Worker-API-Key": "test-worker-key"},
    )
    assert response.status_code == 200

    data = response.json()
    assert data is not None
    assert "id" in data
    assert "original_audio_url" in data
    assert data["status"] == "processing"


def test_claim_is_atomic(client):
    """Test that claiming marks episode as processing."""
    podcast_id = create_test_podcast(client)

    # Create multiple episodes
    episode_ids = []
    for i in range(3):
        from web.dependencies import SessionFactory
        from web.models import Episode

        db = SessionFactory()
        try:
            episode = Episode(
                podcast_id=podcast_id,
                guid=f"test-episode-{i}",
                title=f"Test Episode {i}",
                original_audio_url=f"https://example.com/test{i}.mp3",
                status=EpisodeStatus.PENDING.value,
            )
            db.add(episode)
            db.commit()
            db.refresh(episode)
            episode_ids.append(episode.id)
        finally:
            db.close()

    response = client.post(
        "/api/queue/claim",
        headers={"X-Worker-API-Key": "test-worker-key"},
    )
    claimed_id = response.json()["id"]

    # Verify status changed in DB
    episode_data = get_episode_from_db(claimed_id)
    assert episode_data["status"] == EpisodeStatus.PROCESSING.value
    assert episode_data["claimed_at"] is not None


def test_complete_episode(client):
    """Test marking an episode as complete."""
    podcast_id = create_test_podcast(client)
    episode_id = create_test_episode(client, podcast_id, status=EpisodeStatus.PROCESSING)

    response = client.post(
        f"/api/queue/{episode_id}/complete",
        headers={"X-Worker-API-Key": "test-worker-key"},
        json={
            "processed_audio_key": f"{podcast_id}/{episode_id}.mp3",
            "processed_duration": 3600.0,
            "ads_removed_seconds": 120.5,
        },
    )
    assert response.status_code == 200

    episode_data = get_episode_from_db(episode_id)
    assert episode_data["status"] == EpisodeStatus.COMPLETE.value
    assert episode_data["processed_audio_key"] == f"{podcast_id}/{episode_id}.mp3"


def test_fail_episode(client):
    """Test marking an episode as failed."""
    podcast_id = create_test_podcast(client)
    episode_id = create_test_episode(client, podcast_id, status=EpisodeStatus.PROCESSING)

    response = client.post(
        f"/api/queue/{episode_id}/fail",
        headers={"X-Worker-API-Key": "test-worker-key"},
        json={"error": "Transcription failed"},
    )
    assert response.status_code == 200

    episode_data = get_episode_from_db(episode_id)
    # First failure should retry (back to pending)
    assert episode_data["status"] == EpisodeStatus.PENDING.value
    assert episode_data["retry_count"] == 1


def test_fail_episode_max_retries(client):
    """Test that max retries marks episode as permanently failed."""
    podcast_id = create_test_podcast(client)

    # Create an episode with retry_count already at max
    from web.dependencies import SessionFactory
    from web.models import Episode

    db = SessionFactory()
    try:
        episode = Episode(
            podcast_id=podcast_id,
            guid="test-episode-max-retry",
            title="Test Episode Max Retry",
            original_audio_url="https://example.com/test-max.mp3",
            status=EpisodeStatus.PROCESSING.value,
            retry_count=2,  # Already at max
        )
        db.add(episode)
        db.commit()
        db.refresh(episode)
        episode_id = episode.id
    finally:
        db.close()

    response = client.post(
        f"/api/queue/{episode_id}/fail",
        headers={"X-Worker-API-Key": "test-worker-key"},
        json={"error": "Final failure"},
    )
    assert response.status_code == 200

    episode_data = get_episode_from_db(episode_id)
    assert episode_data["status"] == EpisodeStatus.FAILED.value
    assert episode_data["error_message"] == "Final failure"
