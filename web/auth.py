"""Authentication helpers for the web service."""

import hashlib
import hmac
import os
import secrets
import time

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.security import HTTPBasic, HTTPBasicCredentials


class LoginRequiredError(Exception):
    def __init__(self, next_url: str = "/"):
        self.next_url = next_url

security = HTTPBasic(auto_error=False)

SESSION_COOKIE = "adnihilator_session"
SESSION_MAX_AGE = 30 * 24 * 3600  # 30 days


def get_admin_username() -> str:
    return os.environ.get("ADMIN_USERNAME", "admin")


def get_admin_password() -> str:
    password = os.environ.get("ADMIN_PASSWORD")
    if not password:
        raise ValueError("ADMIN_PASSWORD environment variable not set")
    return password


def get_worker_api_key() -> str:
    key = os.environ.get("WORKER_API_KEY")
    if not key:
        raise ValueError("WORKER_API_KEY environment variable not set")
    return key


def _session_secret() -> str:
    return get_admin_password()


def create_session_token() -> str:
    expires = int(time.time()) + SESSION_MAX_AGE
    payload = f"{get_admin_username()}:{expires}"
    sig = hmac.new(
        _session_secret().encode(), payload.encode(), hashlib.sha256
    ).hexdigest()
    return f"{payload}:{sig}"


def verify_session_token(token: str) -> bool:
    parts = token.split(":")
    if len(parts) != 3:
        return False
    username, expires_str, sig = parts
    try:
        expires = int(expires_str)
    except ValueError:
        return False
    if time.time() > expires:
        return False
    payload = f"{username}:{expires_str}"
    expected = hmac.new(
        _session_secret().encode(), payload.encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return False
    return secrets.compare_digest(username, get_admin_username())


def check_credentials(username: str, password: str) -> bool:
    return secrets.compare_digest(
        username, get_admin_username()
    ) and secrets.compare_digest(password, get_admin_password())


def verify_admin(
    request: Request,
    credentials: HTTPBasicCredentials | None = Depends(security),
) -> None:
    """Verify admin via session cookie or HTTP Basic Auth.

    Cookie is checked first. If absent/invalid, falls back to Basic Auth.
    If neither works, redirects to login page for browser requests or
    returns 401 for API requests.
    """
    token = request.cookies.get(SESSION_COOKIE)
    if token and verify_session_token(token):
        return

    if credentials and check_credentials(credentials.username, credentials.password):
        return

    if "text/html" in request.headers.get("accept", ""):
        raise LoginRequiredError(next_url=request.url.path)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Basic"},
    )


def verify_worker_api_key(api_key: str) -> None:
    if not secrets.compare_digest(api_key, get_worker_api_key()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
