"""Tests for authentication."""

import os
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials

from web.auth import verify_admin, verify_worker_api_key


def test_verify_admin_valid(monkeypatch):
    """Test valid admin credentials."""
    monkeypatch.setenv("ADMIN_USERNAME", "testuser")
    monkeypatch.setenv("ADMIN_PASSWORD", "testpass")

    credentials = HTTPBasicCredentials(username="testuser", password="testpass")
    # Should not raise
    verify_admin(credentials)


def test_verify_admin_invalid_password(monkeypatch):
    """Test invalid admin password."""
    monkeypatch.setenv("ADMIN_USERNAME", "testuser")
    monkeypatch.setenv("ADMIN_PASSWORD", "testpass")

    credentials = HTTPBasicCredentials(username="testuser", password="wrongpass")
    with pytest.raises(HTTPException) as exc_info:
        verify_admin(credentials)
    assert exc_info.value.status_code == 401


def test_verify_admin_invalid_username(monkeypatch):
    """Test invalid admin username."""
    monkeypatch.setenv("ADMIN_USERNAME", "testuser")
    monkeypatch.setenv("ADMIN_PASSWORD", "testpass")

    credentials = HTTPBasicCredentials(username="wronguser", password="testpass")
    with pytest.raises(HTTPException) as exc_info:
        verify_admin(credentials)
    assert exc_info.value.status_code == 401


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
