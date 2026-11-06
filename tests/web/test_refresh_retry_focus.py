"""Deferred real partial responses preserve the notice's keyboard focus."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.web.test_notice_attempt_recovery import pages


@pytest.mark.skipif(not shutil.which('node'), reason='Node is required for rendered-script verification')
def test_refresh_retry_keeps_notice_focus_until_real_response_and_recovers(pages: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture actual mixed sync-all output, then delay delivery to actual scripts."""
    client, base, dashboard = pages
    from web import dependencies
    from web.models import Podcast
    from web.services import feed_sync
    import feedparser

    with dependencies.SessionFactory() as db:
        db.add_all([Podcast(id='bad', title='Bad fixture', source_rss_url='http://127.0.0.1/bad'),
                    Podcast(id='good', title='Healthy fixture', source_rss_url='http://127.0.0.1/good')])
        db.commit()
    parsed = feedparser.parse('<rss><channel><title>Healthy</title></channel></rss>')
    failed = True

    def parse_fixture(url: str) -> object:
        if url.endswith('/bad') and failed:
            raise RuntimeError('synthetic feed failure')
        return parsed

    monkeypatch.setattr(feed_sync.feedparser, 'parse', parse_fixture)
    partial = client.post('/podcasts/sync-all')
    assert partial.status_code == 200
    assert partial.json()['results']['bad']['error']
    assert partial.json()['results']['good']['error'] is None
    failed = False
    success = client.post('/podcasts/sync-all')
    assert success.status_code == 200
    assert all(row['error'] is None for row in success.json()['results'].values())
    program = Path(__file__).with_name('ui_refresh_retry_focus_probe.cjs').read_text()
    result = subprocess.run(['node', '-e', program], input=json.dumps(dict(base=base, dashboard=dashboard, partial=partial.json(), success=success.json())), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
