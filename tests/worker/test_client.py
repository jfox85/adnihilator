"""Tests for the worker API client."""

import pytest
from unittest.mock import MagicMock, patch

from worker.client import WorkerClient, EpisodeJob


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.Client - the actual class used by WorkerClient."""
    with patch("worker.client.httpx.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance
        yield mock_instance


def test_claim_returns_job(mock_httpx_client):
    """Test claiming returns an EpisodeJob."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "ep-123",
        "podcast_id": "pod-456",
        "guid": "guid-789",
        "title": "Test Episode",
        "original_audio_url": "https://example.com/ep.mp3",
        "status": "processing",
    }
    mock_httpx_client.post.return_value = mock_response

    client = WorkerClient("https://api.example.com", "test-key")
    job = client.claim()

    assert job is not None
    assert job.id == "ep-123"
    assert job.original_audio_url == "https://example.com/ep.mp3"


def test_claim_returns_none_when_empty(mock_httpx_client):
    """Test claiming returns None when queue is empty."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = None
    mock_httpx_client.post.return_value = mock_response

    client = WorkerClient("https://api.example.com", "test-key")
    job = client.claim()

    assert job is None


def test_complete_sends_metadata(mock_httpx_client):
    """Test completing sends the right metadata."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "complete"}
    mock_httpx_client.post.return_value = mock_response

    client = WorkerClient("https://api.example.com", "test-key")
    client.complete(
        episode_id="ep-123",
        audio_key="pod/ep.mp3",
        duration=3600.0,
        ads_removed=120.5,
    )

    mock_httpx_client.post.assert_called_once()
    call_args = mock_httpx_client.post.call_args
    assert "ep-123" in call_args[0][0]
    assert call_args[1]["json"]["processed_audio_key"] == "pod/ep.mp3"


def test_fail_sends_error(mock_httpx_client):
    """Test failing sends the error message."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "retrying"}
    mock_httpx_client.post.return_value = mock_response

    client = WorkerClient("https://api.example.com", "test-key")
    client.fail("ep-123", "Something went wrong")

    mock_httpx_client.post.assert_called_once()
    call_args = mock_httpx_client.post.call_args
    assert call_args[1]["json"]["error"] == "Something went wrong"
