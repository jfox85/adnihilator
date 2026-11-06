"""Consumer checks for canonical wrapped and camel-case diagnostic redaction."""

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from web.models import Episode, Podcast


@pytest.fixture
def diagnostic_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    """Render only an isolated temporary database with synthetic credentials."""
    monkeypatch.setenv('DATABASE_PATH', str(tmp_path / 'diagnostics.db'))
    monkeypatch.setenv('ADMIN_USERNAME', 'synthetic-admin')
    monkeypatch.setenv('ADMIN_PASSWORD', 'synthetic-password')
    monkeypatch.setenv('FEED_SYNC_ENABLED', 'false')
    from web.app import app
    from web import dependencies
    with TestClient(app) as client:
        client.auth = ('synthetic-admin', 'synthetic-password')
        with dependencies.SessionFactory() as db:
            db.add(Podcast(id='diagnostic-podcast', title='Synthetic diagnostics', source_rss_url='http://127.0.0.1/fixture'))
            db.flush()
            db.add(Episode(id='diagnostic-episode', podcast_id='diagnostic-podcast', guid='diagnostic', status='failed'))
            db.commit()
        yield client


class TestCanonicalDiagnosticConsumers:
    @pytest.mark.parametrize('diagnostic,canary', [
        ('error: access_token=WRAPPED-ACCESS-CANARY', 'WRAPPED-ACCESS-CANARY'),
        ('upstream_error: {"client_secret":"WRAPPED-CLIENT-CANARY"}', 'WRAPPED-CLIENT-CANARY'),
        ('exception: --api-key=WRAPPED-FLAG-CANARY', 'WRAPPED-FLAG-CANARY'),
        ('error: Authorization: RAW-AUTHORIZATION-CANARY', 'RAW-AUTHORIZATION-CANARY'),
        ('accessKey=CAMEL-ACCESS-CANARY', 'CAMEL-ACCESS-CANARY'),
        ('secretKey=CAMEL-SECRET-CANARY', 'CAMEL-SECRET-CANARY'),
        ('clientSecret=CAMEL-CLIENT-CANARY', 'CAMEL-CLIENT-CANARY'),
        ('privateKey=CAMEL-PRIVATE-CANARY', 'CAMEL-PRIVATE-CANARY'),
        ('sessionToken=CAMEL-SESSION-CANARY', 'CAMEL-SESSION-CANARY'),
        ('idToken=CAMEL-ID-CANARY', 'CAMEL-ID-CANARY'),
        ('accessToken=CAMEL-TOKEN-CANARY', 'CAMEL-TOKEN-CANARY'),
    ])
    @pytest.mark.parametrize('surface', ['/queue', '/podcasts/diagnostic-podcast/episodes', '/podcasts/diagnostic-podcast/episodes/status'])
    def test_wrappers_and_camel_credentials_do_not_leak_in_html_or_status(
        self, diagnostic_client: TestClient, diagnostic: str, canary: str, surface: str
    ) -> None:
        """The actual consumer response must redact values absent environment."""
        from web import dependencies
        with dependencies.SessionFactory() as db:
            db.get(Episode, 'diagnostic-episode').error_message = diagnostic
            db.commit()
        response = diagnostic_client.get(surface)
        assert response.status_code == 200
        assert canary not in response.text
        assert '[REDACTED]' in response.text
