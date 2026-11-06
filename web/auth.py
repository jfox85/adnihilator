"""Authentication helpers for the web service."""

import base64
import binascii
import hashlib
import hmac
import os
import secrets
import time

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials


class LoginRequiredError(Exception):
    def __init__(self, next_url: str = "/"):
        self.next_url = next_url

security = HTTPBasic(auto_error=False)

SESSION_COOKIE = "adnihilator_session"
SESSION_MAX_AGE = 30 * 24 * 3600  # 30 days


def safe_login_destination(destination: str) -> str:
    """Allow local absolute paths without browser URL-normalization escapes."""
    if (
        not destination.startswith("/")
        or destination.startswith("//")
        or "\\" in destination
        or any(ord(character) < 32 or ord(character) == 127 for character in destination)
    ):
        return "/"
    return destination


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
    """Sign an ASCII-safe session while preserving the configured subject bytes."""
    expires = int(time.time()) + SESSION_MAX_AGE
    subject = base64.urlsafe_b64encode(get_admin_username().encode("utf-8")).decode("ascii").rstrip("=")
    payload = f"v2:{subject}:{expires}"
    sig = hmac.new(
        _session_secret().encode(), payload.encode(), hashlib.sha256
    ).hexdigest()
    return f"{payload}:{sig}"


def verify_session_token(token: str) -> bool:
    """Verify current tokens and still-valid legacy sessions without a rollout logout."""
    parts = token.split(":")
    versioned = len(parts) == 4 and parts[0] == "v2"
    if versioned:
        _, subject, expires_str, sig = parts
        payload = f"v2:{subject}:{expires_str}"
    elif len(parts) == 3:
        username, expires_str, sig = parts
        payload = f"{username}:{expires_str}"
    else:
        return False
    try:
        expires = int(expires_str)
    except ValueError:
        return False
    if time.time() > expires:
        return False
    expected = hmac.new(
        _session_secret().encode(), payload.encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(sig.encode("utf-8"), expected.encode("utf-8")):
        return False
    if versioned:
        try:
            username = base64.b64decode(
                subject + "=" * (-len(subject) % 4), altchars=b"-_", validate=True
            ).decode("utf-8")
        except (ValueError, UnicodeError, binascii.Error):
            return False
    return secrets.compare_digest(username.encode("utf-8"), get_admin_username().encode("utf-8"))


def check_credentials(username: str, password: str) -> bool:
    return secrets.compare_digest(
        username.encode("utf-8"), get_admin_username().encode("utf-8")
    ) and secrets.compare_digest(password.encode("utf-8"), get_admin_password().encode("utf-8"))


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
    if not secrets.compare_digest(api_key.encode("utf-8"), get_worker_api_key().encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
