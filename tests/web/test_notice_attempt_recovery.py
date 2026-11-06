"""New explicit attempts must produce visible, accessible error feedback."""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def pages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[tuple[TestClient, str, str]]:
    """Render actual scripts against an isolated synthetic session database."""
    monkeypatch.setenv('DATABASE_PATH', str(tmp_path / 'attempts.db'))
    monkeypatch.setenv('ADMIN_USERNAME', 'synthetic-admin')
    monkeypatch.setenv('ADMIN_PASSWORD', 'synthetic-password')
    monkeypatch.setenv('FEED_SYNC_ENABLED', 'false')
    from web.app import app
    with TestClient(app) as client:
        client.auth = ('synthetic-admin', 'synthetic-password')
        scripts = re.findall(r'<script>(.*?)</script>', client.get('/').text, re.S)
        base = next(script for script in scripts if 'function importOPML' in script)
        dashboard = next(script for script in scripts if 'function refreshAll' in script)
        yield client, base, dashboard


@pytest.mark.skipif(not shutil.which('node'), reason='Node is required for rendered-script verification')
class TestExplicitAttemptFeedback:
    @pytest.mark.parametrize('mode', ['action', 'opml', 'offscreen', 'refresh-all-partial'])
    def test_new_failures_are_visible_and_dismissed_background_fault_stays_muted(
        self, pages: tuple, mode: str
    ) -> None:
        """Run actual request helpers and callers, preserving background muting."""
        client, base, dashboard = pages
        malformed = client.post('/opml/import', files={'file': ('malformed.opml', b'<opml><outline', 'text/xml')})
        assert malformed.status_code == 400
        program = Path(__file__).with_name('ui_explicit_attempt_probe.cjs').read_text()
        result = subprocess.run(['node', '-e', program], input=json.dumps({'base': base, 'dashboard': dashboard, 'mode': mode, 'opml_status': malformed.status_code}), capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
