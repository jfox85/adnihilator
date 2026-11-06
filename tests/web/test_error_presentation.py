"""HTTP and rendered-script regressions for safe, recoverable error states."""

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
CANARY = 'SYNTHHISTORICALSECRET'
RAW_ERROR = f'Gemini 402 prepayment credits are depleted key={CANARY} https://name:SYNTHURLPASSWORD@fixture.invalid/upload?signature=SYNTHSIGNATURE'


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[tuple]:
    """Serve only synthetic records in a temporary SQLite database."""
    monkeypatch.setenv('DATABASE_PATH', str(tmp_path / 'errors.db'))
    monkeypatch.setenv('ADMIN_USERNAME', AUTH[0])
    monkeypatch.setenv('ADMIN_PASSWORD', AUTH[1])
    monkeypatch.setenv('WORKER_API_KEY', 'synthetic-worker-key')
    monkeypatch.setenv('FEED_SYNC_ENABLED', 'false')
    from web.app import app
    from web import dependencies
    with TestClient(app, raise_server_exceptions=False) as client:
        with dependencies.SessionFactory() as db:
            podcast = Podcast(title='Synthetic', source_rss_url='http://127.0.0.1/fixture')
            db.add(podcast)
            db.flush()
            episode = Episode(podcast_id=podcast.id, guid='failed', title='Synthetic failure', status='failed', error_message=RAW_ERROR)
            db.add(episode)
            db.commit()
            ids = (podcast.id, episode.id)
        yield client, dependencies.SessionFactory, ids


class TestErrorPresentation:
    @pytest.mark.skipif(not shutil.which('node'), reason='Node unavailable')
    @pytest.mark.parametrize('count', [150, 12000])
    def test_poll_uses_indexed_rows_and_skips_unchanged_dom(self, harness: tuple, count: int) -> None:
        """A large poll must not search the document once per returned episode."""
        client, _, (podcast_id, _) = harness
        scripts = re.findall(r'<script>(.*?)</script>', client.get(f'/podcasts/{podcast_id}/episodes', auth=AUTH).text, re.S)
        source = next(script for script in scripts if 'function pollStatuses' in script)
        program = Path(__file__).with_name('ui_poll_scale_probe.cjs').read_text()
        result = subprocess.run(['node', '-e', program], input=json.dumps({'source': source, 'count': count}), capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

    @pytest.mark.parametrize('surface', ['episodes', 'queue', 'status'])
    def test_historical_diagnostics_are_redacted_everywhere(self, harness: tuple, surface: str) -> None:
        """Old database errors must not expose credentials in HTML or JSON."""
        client, _, (podcast_id, episode_id) = harness
        url = '/queue' if surface == 'queue' else f'/podcasts/{podcast_id}/episodes' + ('/status' if surface == 'status' else '')
        response = client.get(url, auth=AUTH)
        assert response.status_code == 200
        for secret in [CANARY, 'SYNTHURLPASSWORD', 'SYNTHSIGNATURE']:
            assert secret not in response.text
        if surface == 'status':
            error = response.json()['episodes'][episode_id]
            assert 'credits are used up' in error['error_summary']
            assert '[REDACTED]' in error['error_message']
        else:
            assert 'credits are used up' in response.text
            assert 'Diagnostic details' in response.text
            assert '<details' in response.text
            assert '<details open' not in response.text

    @pytest.mark.parametrize('raw,summary', [
        ('Gemini 402 payment required', 'credits are used up'),
        ('Gemini 429 prepayment credits depleted', 'credits are used up'),
        ('401 invalid API key', 'credentials need attention'),
        ('403 permission denied', 'credentials need attention'),
        ('400 API_KEY_INVALID', 'credentials need attention'),
        ('404 model not found', 'model is unavailable'),
        ('503 service unavailable', 'temporarily unavailable'),
        ('ffmpeg failed', 'Processing failed'),
    ])
    def test_consistent_provider_summary(self, harness: tuple, raw: str, summary: str) -> None:
        """Polling and both pages use the same kind summary and next action."""
        client, factory, (podcast_id, episode_id) = harness
        with factory() as db:
            db.get(Episode, episode_id).error_message = raw
            db.commit()
        result = client.get(f'/podcasts/{podcast_id}/episodes/status', auth=AUTH).json()['episodes'][episode_id]
        assert summary in result['error_summary']
        assert result['error_action']
        for url in ['/queue', f'/podcasts/{podcast_id}/episodes']:
            page = client.get(url, auth=AUTH)
            assert result['error_summary'] in page.text
            assert result['error_action'] in page.text

    @pytest.mark.parametrize('url', ['/missing-page', '/podcasts/missing/episodes'])
    def test_browser_404_has_app_chrome_and_way_back(self, harness: tuple, url: str) -> None:
        """A browser error is a useful page rather than a raw JSON object."""
        client, _, _ = harness
        response = client.get(url, auth=AUTH, headers={'Accept': 'text/html'})
        assert response.status_code == 404
        assert response.headers['content-type'].startswith('text/html')
        assert 'AdNihilator' in response.text
        assert 'Page not found' in response.text
        assert 'Back to Dashboard' in response.text
        assert '{"detail"' not in response.text

    def test_json_api_404_is_unchanged(self, harness: tuple) -> None:
        client, _, _ = harness
        response = client.post('/api/queue/missing/fail', headers={'X-Worker-API-Key': 'not-a-real-key'}, json={'error': 'x'})
        # Invalid worker authentication remains JSON with its original status.
        assert response.status_code == 401
        assert response.headers['content-type'].startswith('application/json')
        response = client.get('/api/missing', headers={'Accept': 'text/html'})
        assert response.status_code == 404
        assert response.json() == {'detail': 'Not Found'}

    def test_html_500_never_exposes_exception_details(self, harness: tuple) -> None:
        """Even unexpected failures keep private diagnostics out of the page."""
        client, _, _ = harness
        from web.app import app
        async def fail() -> None:
            raise RuntimeError('private failure key=' + CANARY)
        app.add_api_route('/__synthetic_error_test__', fail, methods=['GET'])
        try:
            response = client.get('/__synthetic_error_test__', headers={'Accept': 'text/html'})
            assert response.status_code == 500
            assert response.headers['content-type'].startswith('text/html')
            assert 'Something went wrong' in response.text
            assert CANARY not in response.text
            assert 'RuntimeError' not in response.text
            assert 'Back to Dashboard' in response.text
        finally:
            app.router.routes[:] = [route for route in app.router.routes if getattr(route, 'path', None) != '/__synthetic_error_test__']

    @pytest.mark.skipif(not shutil.which('node'), reason='Node unavailable')
    def test_rendered_polling_and_import_recover_from_errors(self, harness: tuple) -> None:
        """Execute the real scripts with delayed fetch, failures and auth expiry."""
        client, factory, (podcast_id, episode_id) = harness
        with factory() as db:
            pending = Episode(podcast_id=podcast_id, guid='pending', title='Pending', status='pending')
            db.add(pending)
            db.commit()
            pending_id = pending.id
        html = client.get(f'/podcasts/{podcast_id}/episodes', auth=AUTH).text
        scripts = re.findall(r'<script>(.*?)</script>', html, re.S)
        base = next(script for script in scripts if 'function importOPML' in script)
        episodes = next(script for script in scripts if 'function pollStatuses' in script)
        queue = next(script for script in re.findall(r'<script>(.*?)</script>', client.get('/queue', auth=AUTH).text, re.S) if 'function updateStatuses' in script)
        index = next(script for script in re.findall(r'<script>(.*?)</script>', client.get('/', auth=AUTH).text, re.S) if 'function refreshAll' in script)
        data = client.get(f'/podcasts/{podcast_id}/episodes/status', auth=AUTH).json()
        data['episodes'][episode_id]['status'] = 'pending'
        program = Path(__file__).with_name('ui_error_probe.cjs').read_text()
        result = subprocess.run(['node', '-e', program], input=json.dumps({'base': base, 'episodes': episodes, 'queue': queue, 'index': index, 'data': data, 'episode_id': episode_id, 'pending_id': pending_id}), capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
