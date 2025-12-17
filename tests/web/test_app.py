"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Create a test client with mocked auth."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "testpass")
    monkeypatch.setenv("WORKER_API_KEY", "test-worker-key")
    monkeypatch.setenv("DATABASE_PATH", str(db_path))

    from web.app import app
    # Use context manager to trigger lifespan events
    with TestClient(app) as client:
        yield client


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_requires_auth(client):
    """Test that root requires authentication."""
    response = client.get("/")
    assert response.status_code == 401


def test_root_with_auth(client):
    """Test root with valid auth returns HTML."""
    response = client.get("/", auth=("admin", "testpass"))
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
