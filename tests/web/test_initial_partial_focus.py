"""Initial explicit partial results need a usable keyboard recovery context."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.web.test_notice_attempt_recovery import pages


@pytest.mark.skipif(not shutil.which('node'), reason='Node is required for rendered-script verification')
@pytest.mark.parametrize('move_focus', [False, True])
def test_partial_failure_recovers_stranded_focus_without_stealing_new_focus(
    pages: tuple, monkeypatch: pytest.MonkeyPatch, move_focus: bool
) -> None:
    """Defer real mixed HTTP200 results to actual rendered request scripts."""
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
    program = Path(__file__).with_name('ui_initial_partial_focus_probe.cjs').read_text()
    result = subprocess.run(['node', '-e', program], input=json.dumps(dict(base=base, dashboard=dashboard, partial=partial.json(), success=success.json(), moveFocus=move_focus)), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
