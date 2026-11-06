"""Regressions for untrusted login destinations and Unicode auth inputs."""

from collections.abc import Iterator
import hashlib
import hmac
import time
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from web.auth import SESSION_COOKIE, verify_session_token, verify_worker_api_key


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    """Run the real app against a disposable database and synthetic credentials."""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "login.db"))
    monkeypatch.setenv("ADMIN_USERNAME", "test-admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "test-password")
    monkeypatch.setenv("WORKER_API_KEY", "test-worker-key")
    monkeypatch.setenv("FEED_SYNC_ENABLED", "false")
    from web.app import app

    with TestClient(app) as test_client:
        yield test_client


class TestLoginDestinations:
    @pytest.mark.parametrize(
        "destination",
        [
            "https://example.invalid/phish",
            "//example.invalid/phish",
            "/\\example.invalid/phish",
            "javascript:alert(1)",
            "/\t/example.invalid/phish",
            "relative/path",
            "",
        ],
    )
    def test_untrusted_destination_stays_on_dashboard(self, client: TestClient, destination: str) -> None:
        """A successful login cannot redirect the admin to another origin."""
        response = client.post(
            "/login",
            data={"username": "test-admin", "password": "test-password", "next": destination},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == "/"
        assert SESSION_COOKIE in response.cookies

    @pytest.mark.parametrize("destination", ["/queue", "/podcasts/example/episodes", "/queue?status=pending"])
    def test_local_destination_is_preserved(self, client: TestClient, destination: str) -> None:
        """Safe app paths and their query strings still work after signing in."""
        response = client.post(
            "/login",
            data={"username": "test-admin", "password": "test-password", "next": destination},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == destination


class TestUnicodeAuthentication:
    @pytest.mark.parametrize(
        "username,password", [("ü", "wrong"), ("🎧", "wrong"), ("test-admin", "🎧")]
    )
    def test_wrong_unicode_credentials_show_login_error(
        self, client: TestClient, username: str, password: str
    ) -> None:
        """An ordinary failed login must not become a server error."""
        response = client.post("/login", data={"username": username, "password": password})
        assert response.status_code == 401
        assert "Invalid credentials" in response.text

    def test_configured_unicode_password_can_authenticate(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An accented passphrase works through login and the signed session."""
        monkeypatch.setenv("ADMIN_PASSWORD", "pässwörd-🎧")
        response = client.post(
            "/login", data={"username": "test-admin", "password": "pässwörd-🎧"}, follow_redirects=False
        )
        assert response.status_code == 303
        assert client.get("/").status_code == 200

    def test_non_ascii_cookie_signature_is_rejected(self, client: TestClient) -> None:
        """A tampered cookie cannot crash browser or API authentication."""
        cookie = f'{SESSION_COOKIE}="test-admin:9999999999:é"'.encode("latin-1")
        response = client.get("/", headers=[(b"cookie", cookie), (b"accept", b"text/html")], follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"].startswith("/login?")
        assert client.get("/", headers=[(b"cookie", cookie)]).status_code == 401

    def test_unicode_session_subject_is_untrusted(self, client: TestClient) -> None:
        """A Unicode forgery returns false without authorizing or throwing."""
        assert not verify_session_token("🎧:9999999999:invalid")

    def test_wrong_unicode_worker_key_is_unauthorized(self, client: TestClient) -> None:
        """The worker credential boundary also treats foreign text as a denial."""
        with pytest.raises(HTTPException) as error:
            verify_worker_api_key("é-invalid")
        assert error.value.status_code == 401


class TestSessionSubjectEncoding:
    @pytest.mark.parametrize("username", ["admin", "ü", "🎧", "用户", "admin:ops", "a:b:c"])
    def test_configured_username_creates_a_usable_cookie(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, username: str
    ) -> None:
        """Configured Unicode and delimiter-containing names survive cookie transport."""
        monkeypatch.setenv("ADMIN_USERNAME", username)
        response = client.post(
            "/login", data={"username": username, "password": "test-password"}, follow_redirects=False
        )
        assert response.status_code == 303
        response.headers["set-cookie"].encode("ascii")
        assert client.get("/").status_code == 200

    def test_existing_legacy_cookie_still_authenticates(self, client: TestClient) -> None:
        """The rollout does not sign out existing valid ASCII sessions."""
        payload = f"test-admin:{int(time.time()) + 3600}"
        signature = hmac.new(b"test-password", payload.encode(), hashlib.sha256).hexdigest()
        client.cookies.set(SESSION_COOKIE, f"{payload}:{signature}")
        assert client.get("/").status_code == 200

    @pytest.mark.parametrize("token", [
        "v2:bad!:9999999999:invalid", "v2:é:9999999999:invalid", "v3:abc:9999999999:invalid",
        "v2::9999999999:invalid", "v2:abc:invalid:invalid",
    ])
    def test_invalid_versioned_subjects_are_denied(self, client: TestClient, token: str) -> None:
        """Malformed new-format cookies remain untrusted instead of raising."""
        assert not verify_session_token(token)

    @pytest.mark.parametrize("subject", ["bad!", "_w", "é", ""])
    def test_signed_invalid_subject_is_still_denied(self, client: TestClient, subject: str) -> None:
        """Malformed base64/UTF-8 must be rejected even past signature validation."""
        payload = f"v2:{subject}:{int(time.time()) + 3600}"
        signature = hmac.new(b"test-password", payload.encode(), hashlib.sha256).hexdigest()
        assert not verify_session_token(f"{payload}:{signature}")
