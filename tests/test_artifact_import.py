"""Offline import recovery; original artifact bytes and unrelated DBs stay intact."""

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import select, text

from adnihilator.artifact_import import read_artifact
from scripts.import_transcripts import import_local
from web.database import get_engine, get_session_factory
from web.models import Episode, TranscriptVersion
import scripts.import_transcripts as module


@pytest.fixture
def inputs(tmp_path):
    feed = tmp_path / 'feed.xml'
    feed.write_text('<rss><channel><title>Fixture</title><item><guid>g1</guid><title>One</title><pubDate>Wed, 30 Sep 2026 10:00:00 GMT</pubDate><enclosure url="https://example.com/1.mp3"/></item><item><guid>g2</guid><title>Two</title><pubDate>Tue, 29 Sep 2026 10:00:00 GMT</pubDate><enclosure url="https://example.com/2.mp3"/></item></channel></rss>')
    root = tmp_path / 'artifacts'
    (root / 'p1').mkdir(parents=True)
    for n in (1, 2):
        (root / 'p1' / f'worker{n}.json').write_text(json.dumps(dict(audio_path=f'https://example.com/{n}.mp3', segments=[dict(index=0, start=.123, end=4.99, text=f'Memoir {n}')], model_info=dict(engine='fixture'))))
    return feed, root


def test_interrupted_import_resumes_without_duplicates(inputs, tmp_path, monkeypatch):
    feed, root = inputs
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / 'p1').glob('*')}
    original = module.read_artifact
    calls = 0
    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 2: raise RuntimeError('simulated interruption')
        return original(*args)
    monkeypatch.setattr(module, 'read_artifact', interrupted)
    output = tmp_path / 'local.sqlite'
    with pytest.raises(RuntimeError): import_local(output, feed, root, 'p1', 'https://example.com/feed')
    monkeypatch.setattr(module, 'read_artifact', original)
    assert import_local(output, feed, root, 'p1', 'https://example.com/feed')['available'] == 2
    assert import_local(output, feed, root, 'p1', 'https://example.com/feed')['available'] == 2
    engine = get_engine('sqlite:///' + str(output))
    with get_session_factory(engine)() as db:
        assert len(db.scalars(select(Episode)).all()) == 2
        assert len(db.scalars(select(TranscriptVersion)).all()) == 2
    engine.dispose()
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / 'p1').glob('*')}


def test_import_refuses_unmarked_database_without_journal_mutation(inputs, tmp_path):
    feed, root = inputs
    path = tmp_path / 'existing.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('create table original(value text)')
        db.execute("insert into original values ('preserved')")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError): import_local(path, feed, root, 'p1', 'https://example.com/feed')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum


def test_partial_artifacts_are_rejected(inputs):
    _, root = inputs
    path = root / 'p1' / 'worker1.json'
    path.write_text('{"segments":')
    with pytest.raises(ValueError): read_artifact(path, 'e1', 'p1', 'g1', 'https://example.com/1.mp3')
    path.write_text(json.dumps(dict(audio_path='https://example.com/1.mp3', duration=3600, segments=[dict(index=0, start=0, end=10, text='Partial prefix')])))
    with pytest.raises(ValueError): read_artifact(path, 'e1', 'p1', 'g1', 'https://example.com/1.mp3')
