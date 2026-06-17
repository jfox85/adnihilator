"""Tests for authentication."""

import pytest
from fastapi import HTTPException, Request
from fastapi.security import HTTPBasicCredentials

from web.auth import verify_admin, verify_worker_api_key


def _api_request(cookies: dict | None = None) -> Request:
    """Build a minimal non-browser (API) Request for verify_admin.

    No HTML accept header, so unauthenticated calls raise 401 rather than
    redirecting to the login page.
    """
    headers = []
    if cookies:
        cookie_header = "; ".join(f"{k}={v}" for k, v in cookies.items())
        headers.append((b"cookie", cookie_header.encode()))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers,
    }
    return Request(scope)


def test_verify_admin_valid(monkeypatch):
    """Test valid admin credentials."""
    monkeypatch.setenv("ADMIN_USERNAME", "testuser")
    monkeypatch.setenv("ADMIN_PASSWORD", "testpass")

    credentials = HTTPBasicCredentials(username="testuser", password="testpass")
    # Should not raise
    verify_admin(_api_request(), credentials)


def test_verify_admin_invalid_password(monkeypatch):
    """Test invalid admin password."""
    monkeypatch.setenv("ADMIN_USERNAME", "testuser")
    monkeypatch.setenv("ADMIN_PASSWORD", "testpass")

    credentials = HTTPBasicCredentials(username="testuser", password="wrongpass")
    with pytest.raises(HTTPException) as exc_info:
        verify_admin(_api_request(), credentials)
    assert exc_info.value.status_code == 401


def test_verify_admin_invalid_username(monkeypatch):
    """Test invalid admin username."""
    monkeypatch.setenv("ADMIN_USERNAME", "testuser")
    monkeypatch.setenv("ADMIN_PASSWORD", "testpass")

    credentials = HTTPBasicCredentials(username="wronguser", password="testpass")
    with pytest.raises(HTTPException) as exc_info:
        verify_admin(_api_request(), credentials)
    assert exc_info.value.status_code == 401


def test_verify_admin_valid_session_cookie(monkeypatch):
    """A valid session cookie authenticates without Basic credentials."""
    monkeypatch.setenv("ADMIN_USERNAME", "testuser")
    monkeypatch.setenv("ADMIN_PASSWORD", "testpass")

    from web.auth import SESSION_COOKIE, create_session_token

    token = create_session_token()
    # Should not raise even though no Basic credentials are supplied.
    verify_admin(_api_request(cookies={SESSION_COOKIE: token}), credentials=None)


def test_verify_worker_api_key_valid(monkeypatch):
    """Test valid worker API key."""
    monkeypatch.setenv("WORKER_API_KEY", "secret-key-123")

    # Should not raise
    verify_worker_api_key("secret-key-123")


def test_verify_worker_api_key_invalid(monkeypatch):
    """Test invalid worker API key."""
    monkeypatch.setenv("WORKER_API_KEY", "secret-key-123")

    with pytest.raises(HTTPException) as exc_info:
        verify_worker_api_key("wrong-key")
    assert exc_info.value.status_code == 401
