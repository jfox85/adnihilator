"""Authentication helpers for the web service."""

import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()


def get_admin_username() -> str:
    """Get admin username from environment."""
    return os.environ.get("ADMIN_USERNAME", "admin")


def get_admin_password() -> str:
    """Get admin password from environment."""
    password = os.environ.get("ADMIN_PASSWORD")
    if not password:
        raise ValueError("ADMIN_PASSWORD environment variable not set")
    return password


def get_worker_api_key() -> str:
    """Get worker API key from environment."""
    key = os.environ.get("WORKER_API_KEY")
    if not key:
        raise ValueError("WORKER_API_KEY environment variable not set")
    return key


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)) -> None:
    """Verify admin credentials via HTTP Basic Auth.

    Raises HTTPException with 401 if credentials are invalid.
    """
    correct_username = secrets.compare_digest(
        credentials.username, get_admin_username()
    )
    correct_password = secrets.compare_digest(
        credentials.password, get_admin_password()
    )

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


def verify_worker_api_key(api_key: str) -> None:
    """Verify worker API key.

    Raises HTTPException with 401 if key is invalid.
    """
    if not secrets.compare_digest(api_key, get_worker_api_key()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
