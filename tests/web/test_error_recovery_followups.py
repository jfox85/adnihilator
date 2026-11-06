"""Cross-review regressions for isolated recovery and negotiated error pages."""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from web.models import Episode, Podcast

AUTH = ('synthetic-admin', 'synthetic-password')
HISTORICAL = {
    'access_token': 'HISTORIC-CANARY-ACCESS',
    'client_secret': 'HISTORIC-CANARY-CLIENT',
    'OPENAI_API_KEY': 'HISTORIC-CANARY-NAMED',
    'GEMINI_API_KEY': 'HISTORIC-CANARY-GEMINI',
    'R2_SECRET_KEY': 'HISTORIC-CANARY-STORAGE',
    'Authorization': 'Token HISTORIC-CANARY-AUTHORIZATION',
    'password': 'HISTORIC-CANARY-PARTONE HISTORIC-CANARY-PARTTWO',
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    """Use only disposable SQLite state and private synthetic credentials."""
    monkeypatch.setenv('DATABASE_PATH', str(tmp_path / 'followups.db'))
    monkeypatch.setenv('ADMIN_USERNAME', AUTH[0])
    monkeypatch.setenv('ADMIN_PASSWORD', AUTH[1])
    monkeypatch.setenv('FEED_SYNC_ENABLED', 'false')
    from web.app import app
    from web import dependencies
    with TestClient(app, raise_server_exceptions=False) as result:
        result.auth = AUTH
        with dependencies.SessionFactory() as db:
            for name in ['bad', 'healthy']:
                db.add(Podcast(id=name, title=name, source_rss_url=f'http://127.0.0.1/{name}.xml'))
                db.flush()
                db.add(Episode(id=name + '-episode', podcast_id=name, guid=name, title=name, status='pending'))
            db.add(Episode(id='historical', podcast_id='healthy', guid='historical', status='failed', error_message=json.dumps(HISTORICAL)))
            db.commit()
        yield result


class TestNegotiatedBrowserErrors:
    @pytest.mark.parametrize('accept,html', [
        ('application/json, text/html;q=0', False),
        ('text/html;q=0.2, application/json;q=0.9', False),
        ('application/json;q=0.9, text/html;q=0.2', False),
        ('text/html;q=0, */*;q=1', False),
        ('text/html;q=0.9, application/json;q=0.2', True),
        ('text/html, application/json;q=0.9', True),
        ('text/html', True),
        ('application/json', False),
        ('*/*', False),
        ('text/html;q=0.5, */*;q=0.9', False),
        ('text/*;q=0.8, application/json;q=0.7', True),
        ('text/html;q=nan, application/json', False),
        ('text/html;q=2', False),
        ('text/html;q=0.8, application/json;q=0.8', True),
        ('application/json;q=0.8, text/html;q=0.8', False),
    ])
    def test_http_error_honors_acceptable_preference(self, client: TestClient, accept: str, html: bool) -> None:
        """Explicitly forbidden/less-preferred HTML cannot replace JSON errors."""
        response = client.get('/podcasts/missing/episodes/status', headers={'Accept': accept})
        assert response.status_code == 404
        assert response.headers['content-type'].startswith('text/html' if html else 'application/json')
        if not html:
            assert response.json() == {'detail': 'Podcast not found'}

    def test_generic_500_honors_json_preference(self, client: TestClient) -> None:
        """The same predicate applies to unexpected errors without details."""
        from web.app import app
        async def fail() -> None:
            raise RuntimeError('synthetic private exception')
        app.add_api_route('/__followup_error__', fail, methods=['GET'])
        try:
            response = client.get('/__followup_error__', headers={'Accept': 'application/json, text/html;q=0'})
            assert response.status_code == 500
            assert response.headers['content-type'].startswith('text/plain')
            assert response.text == 'Internal Server Error'
        finally:
            app.router.routes[:] = [route for route in app.router.routes if getattr(route, 'path', None) != '/__followup_error__']


class TestHistoricalNamedCredentials:
    @pytest.mark.parametrize('surface', ['/queue', '/podcasts/healthy/episodes', '/podcasts/healthy/episodes/status'])
    def test_rotated_named_values_are_absent_from_html_and_json(self, client: TestClient, surface: str) -> None:
        """Historical values absent from the current environment still redact."""
        response = client.get(surface)
        assert response.status_code == 200
        for canary in HISTORICAL.values():
            assert canary not in response.text
            for part in canary.split():
                if len(part) > 12:
                    assert part not in response.text
        assert '[REDACTED]' in response.text


class TestFailureClassificationContext:
    @pytest.mark.parametrize('diagnostic', [
        'Gemini detection failed: Expecting value: line 1 column 403 (char 402)',
        "UnicodeDecodeError: codec cannot decode byte in position 403",
        'Episode 401 of the series could not be decoded',
        'JSON parse failed at char 503',
        'curl audio download failed: returned error: 403',
        'audio host download failed: HTTP 503 Service Unavailable',
    ])
    def test_nonprovider_numbers_and_download_failures_keep_generic_summary(self, client: TestClient, diagnostic: str) -> None:
        """Only an actual provider failure should tell the owner to buy credits."""
        from web import dependencies
        with dependencies.SessionFactory() as db:
            db.get(Episode, 'historical').error_message = diagnostic
            db.commit()
        error = client.get('/podcasts/healthy/episodes/status').json()['episodes']['historical']
        assert error['error_summary'] == 'Processing failed'
        for surface in ['/queue', '/podcasts/healthy/episodes']:
            assert 'AI provider credits are used up' not in client.get(surface).text


@pytest.mark.skipif(not shutil.which('node'), reason='Node unavailable')
class TestRenderedRecovery:
    def test_notice_can_be_dismissed_without_hiding_later_recovery(self, client: TestClient) -> None:
        """A persistent fault cannot force the same dismissed notice back open."""
        scripts = re.findall(r'<script>(.*?)</script>', client.get('/').text, re.S)
        base = next(script for script in scripts if 'function importOPML' in script)
        program = Path(__file__).with_name('ui_notice_dismissal_probe.cjs').read_text()
        result = subprocess.run(['node', '-e', program], input=json.dumps({'base': base}), capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

    def test_queue_failure_isolated_and_retry_keeps_healthy_groups_live(self, client: TestClient) -> None:
        """A real removed podcast's 404 must not starve another podcast."""
        from web import dependencies
        html = client.get('/queue').text
        scripts = re.findall(r'<script>(.*?)</script>', html, re.S)
        base = next(script for script in scripts if 'function importOPML' in script)
        queue = next(script for script in scripts if 'function updateStatuses' in script)
        assert client.delete('/podcasts/bad').status_code == 200
        missing = client.get('/podcasts/bad/episodes/status')
        assert missing.status_code == 404
        with dependencies.SessionFactory() as db:
            episode = db.get(Episode, 'healthy-episode')
            episode.status = 'processing'
            episode.progress_step = 'transcribing'
            episode.progress_percent = 23
            db.commit()
        healthy = client.get('/podcasts/healthy/episodes/status').json()
        program = Path(__file__).with_name('ui_queue_isolation_probe.cjs').read_text()
        result = subprocess.run(['node', '-e', program], input=json.dumps({'base': base, 'queue': queue, 'healthy': healthy, 'bad_status': missing.status_code}), capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

    def test_successful_zero_import_retry_clears_only_its_own_notice(self, client: TestClient) -> None:
        """Recovery clears OPML errors while preserving unrelated stale status."""
        scripts = re.findall(r'<script>(.*?)</script>', client.get('/').text, re.S)
        base = next(script for script in scripts if 'function importOPML' in script)
        program = Path(__file__).with_name('ui_opml_recovery_probe.cjs').read_text()
        result = subprocess.run(['node', '-e', program], input=json.dumps({'base': base}), capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
