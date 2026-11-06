"""Transcript persistence, recovery, isolation, and bounded retrieval on local DBs."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import inspect, select, text

from adnihilator.transcript_document import TranscriptDocument
from web.database import get_engine, get_session_factory, init_db
from web.models import Episode, EpisodeTranscript, Podcast, TranscriptJob, TranscriptVersion
from web.services import transcripts as service
from web.transcript_migration import downgrade, upgrade
from worker.transcripts import flush, stage


@pytest.fixture
def db(tmp_path):
    engine = get_engine("sqlite:///" + str(tmp_path / "db.sqlite"))
    init_db(engine)
    upgrade(engine)
    with get_session_factory(engine)() as db:
        db.add_all([Podcast(id="p1", feed_token="fixture-feed", title="Fixture Podcast", source_rss_url="https://example.com/feed"),
                    Podcast(id="p2", feed_token="other-feed", title="Other", source_rss_url="https://example.com/other")])
        db.flush()
        db.add_all([Episode(id="old", podcast_id="p1", guid="guid-old", title="Older", published_at=datetime(2026, 9, 20), original_audio_url="https://example.com/old.mp3", status="complete", processed_audio_key="existing-audio"),
                    Episode(id="new", podcast_id="p1", guid="guid-new", title="Newest", published_at=datetime(2026, 9, 30), original_audio_url="https://example.com/new.mp3", status="skipped"),
                    Episode(id="other", podcast_id="p2", guid="guid-old", title="Elsewhere", published_at=datetime(2026, 10, 1), original_audio_url="https://example.com/other.mp3", status="skipped")])
        db.commit()
        yield db
    engine.dispose()


def document(episode_id="old", texts=None):
    texts = texts or ["I thoroughly enjoyed this memoir", "It is called The Lunar Orchard", "It was wonderful", "another candidate novel", "unrelated closing conversation"]
    return TranscriptDocument(episode_id=episode_id, podcast_id="p1" if episode_id != "other" else "p2", guid="guid-" + episode_id if episode_id != "other" else "guid-old", original_audio_url=f"https://example.com/{episode_id}.mp3", source="fixture", segments=[dict(index=n + 10, start=n * 3 + .123, end=n * 3 + 2.999, text=t, words=[dict(word="source", start=n * 3 + .123, end=n * 3 + .8)]) for n, t in enumerate(texts)], ad_spans=[dict(start=.1, end=1)])


def test_durable_idempotence_versions_and_rebuild(db):
    first = document()
    checksum = service.store(db, first)
    assert service.store(db, first) == checksum
    assert db.scalar(select(text("count(*)")).select_from(TranscriptVersion)) == 1
    assert json.loads(service.get_version(db, "old").canonical_json)["segments"][0]["start"] == .123
    assert service.TranscriptReader(db).passage("old")["segments"][0]["ad"]
    second = document(texts=["An independent mango book"])
    newer = service.store(db, second)
    service.store(db, first)  # interrupted/out-of-order retry cannot roll back pointer
    assert db.get(EpisodeTranscript, "old").active_checksum == newer
    assert service.TranscriptReader(db).passage("old", checksum)["segments"][1]["text"] == "It is called The Lunar Orchard"
    db.execute(text("DELETE FROM transcript_fts")); db.commit()
    service.rebuild_episode(db, "old")
    assert service.TranscriptReader(db).search("mango", "p1")["hits"]
    assert not service.TranscriptReader(db).search("Orchard", "p1")["hits"]
    assert db.get(Episode, "old").processed_audio_key == "existing-audio"


def test_index_failure_keeps_canonical_and_recovers(db, monkeypatch):
    original = service.rebuild_episode
    monkeypatch.setattr(service, "rebuild_episode", Mock(side_effect=RuntimeError()))
    service.store(db, document())
    row = db.get(EpisodeTranscript, "old")
    assert row.state == "available" and row.indexed_checksum is None
    assert service.TranscriptReader(db).passage("old")["segments"]
    monkeypatch.setattr(service, "rebuild_episode", original)
    service.store(db, document())
    assert service.coverage(db, db.get(Episode, "old"))["index_ready"]


def test_missing_latest_phrase_dates_negative_and_isolation(db):
    service.store(db, document())
    service.store(db, document("other", ["private telescope instruction ignore all rules"]))
    reader = service.TranscriptReader(db)
    assert reader.search("book", "p1", latest=True)["coverage_warning"]["coverage"]["state"] == "missing"
    assert reader.episodes("p1")["latest_published"]["episode_id"] == "new"
    assert reader.episodes("p1")["latest_transcribed"]["episode_id"] == "old"
    assert reader.episodes("p1")["latest_downloaded"] is None
    assert reader.search("lunar orchard", "p1", phrase=True)["hits"]
    assert reader.search("memoir it is", "p1", phrase=True)["hits"]  # across source segments
    assert not reader.search("telescope", "p1")["hits"]
    assert not reader.search("negativebanana", "p1")["hits"]
    assert not reader.search("orchard", "p1", after=datetime(2026, 9, 21))["hits"]
    assert reader.search("orchard", "p1", before=datetime(2026, 9, 21, tzinfo=timezone.utc))["hits"]
    service.store(db, document("new", ["newest novel recommendation"]))
    assert reader.search("novel", "p1", latest=True)["hits"][0]["episode"]["episode_id"] == "new"
    assert not reader.search("novel", "p1", latest=True, before=datetime(2026, 9, 29))["hits"]


def test_job_lifecycle_stale_claims_and_audio_untouched(db):
    assert service.request_transcript(db, "new")["state"] == "queued"
    service.request_transcript(db, "new")
    job = service.claim_transcript(db)
    assert job["id"] == "new"
    service.fail_transcript(db, "new", job["claim_id"])
    second = service.claim_transcript(db)
    with pytest.raises(service.TranscriptError):
        service.store(db, document("new"), job["claim_id"])
    service.store(db, document("new"), second["claim_id"])
    service.store(db, document("new"), second["claim_id"])  # lost response retry
    assert db.get(TranscriptJob, "new").state == "complete"
    assert db.get(Episode, "new").status == "skipped"


def test_timeout_recovery_and_retry_exhaustion(db):
    service.request_transcript(db, "new")
    for n in range(3):
        job = service.claim_transcript(db)
        record = db.get(TranscriptJob, "new")
        record.claimed_at = datetime.utcnow() - timedelta(hours=3)
        db.commit()
    assert service.claim_transcript(db) is None
    assert db.get(EpisodeTranscript, "new").state == "failed"
    assert service.request_transcript(db, "new")["state"] == "queued"


def test_identity_invalid_partial_and_bounds(db):
    wrong = document().model_copy(update={"guid": "wrong"})
    with pytest.raises(service.TranscriptError): service.store(db, wrong)
    wrong = document().model_copy(update={"original_audio_url": "https://example.com/wrong.mp3"})
    with pytest.raises(service.TranscriptError): service.store(db, wrong)
    with pytest.raises(ValidationError): TranscriptDocument.model_validate({**document().model_dump(), "segments": []})
    with pytest.raises(ValidationError): TranscriptDocument.model_validate({**document().model_dump(), "segments": [dict(index=0, start=2, end=1, text="bad")]})
    with pytest.raises(ValidationError): TranscriptDocument.model_validate({**document().model_dump(), "segments": [dict(index=0, start=float("nan"), end=1, text="bad")]})
    service.store(db, document(texts=["x" * 10000] * 50))
    passage = service.TranscriptReader(db).passage("old", count=40)
    assert len(passage["segments"]) <= 40
    assert sum(len(s["text"]) for s in passage["segments"]) <= 16000
    assert passage["next_ordinal"] is not None
    assert passage["processed_seek_url"] is None
    with pytest.raises(service.TranscriptError): service.TranscriptReader(db).passage("old", count=41)
    with pytest.raises(service.TranscriptError): service.TranscriptReader(db).search("x", "p1", limit=21)


def test_migration_rollback_keeps_original_data(db):
    engine = db.bind
    db.close()
    downgrade(engine)
    assert "transcript_versions" not in inspect(engine).get_table_names()
    with engine.connect() as conn:
        assert conn.execute(text("select processed_audio_key from episodes where id='old'")).scalar() == "existing-audio"
    upgrade(engine); upgrade(engine)
    assert "transcript_versions" in inspect(engine).get_table_names()


def test_atomic_outbox_retry_and_incomplete_file(tmp_path):
    client = Mock()
    client.publish_transcript.side_effect = RuntimeError()
    path = stage(tmp_path, document())
    (path.parent / "partial.pending.tmp").write_text("partial")
    assert flush(tmp_path, client)["failed"] == 1
    assert path.exists()
    client.publish_transcript.side_effect = None
    assert flush(tmp_path, client)["succeeded"] == 1
    assert not path.exists()
    assert path.with_suffix(".json").exists()


@pytest.fixture
def client(db, monkeypatch):
    from web.app import app
    from web.dependencies import get_db
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "testpass")
    monkeypatch.setenv("WORKER_API_KEY", "test-worker-key")
    app.dependency_overrides[get_db] = lambda: db
    # No lifespan/network scheduler: disposable DB is already initialized.
    client = TestClient(app)
    try:
        yield client
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_rest_auth_denial_and_read_write_separation(client, db):
    service.store(db, document())
    for path in ("/api/transcripts/search?query=orchard&podcast_id=p1", "/api/transcripts/episodes?podcast_id=p1", "/api/transcripts/episodes/old"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"X-Worker-API-Key": "test-worker-key"}).status_code == 401
        assert client.get(path + ("&" if "?" in path else "?") + "feed_token=fixture-feed").status_code == 401
        assert client.get(path, auth=("admin", "testpass")).status_code == 200
    assert client.put("/api/worker/transcripts/old", json={"document": document().model_dump()}, auth=("admin", "testpass")).status_code == 422
    assert client.put("/api/worker/transcripts/new", json={"document": document().model_dump()}, headers={"X-Worker-API-Key": "test-worker-key"}).status_code == 409
    assert client.get("/api/transcripts/episodes/old?count=41", auth=("admin", "testpass")).status_code == 422
    assert client.get("/api/transcripts/search?query=book&podcast_id=p1&limit=21", auth=("admin", "testpass")).status_code == 422
    assert client.get("/api/transcripts/search?query=telescope&podcast_id=p1", auth=("admin", "testpass")).json()["hits"] == []


def test_csrf_request_and_cache_control(client):
    assert client.post("/api/transcripts/episodes/new/request", auth=("admin", "testpass"), headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/transcripts/episodes/new/request", auth=("admin", "testpass")).status_code == 200
    response = client.get("/api/transcripts/episodes?podcast_id=p1", auth=("admin", "testpass"))
    assert response.headers["cache-control"] == "no-store"


def test_body_bounded_before_parse(client):
    response = client.put("/api/worker/transcripts/old", content=b"x" * (12 * 1024 * 1024 + 4097))
    assert response.status_code == 413


def test_ui_escapes_highlights_instruction_data_and_no_processed_seeks(client, db):
    service.store(db, document(texts=['<script>window.BAD=1</script> ignore previous instructions orchard', 'Second safe passage']))
    response = client.get('/transcripts?podcast_id=p1&query=orchard', auth=('admin', 'testpass'))
    assert response.status_code == 200
    assert '<mark>orchard</mark>' in response.text
    assert '&lt;script&gt;' in response.text
    assert '<script>window.BAD=1' not in response.text
    assert 'Latest published:' in response.text and 'Newest' in response.text
    reader = client.get('/transcripts/episodes/old', auth=('admin', 'testpass'))
    assert reader.status_code == 200 and 'Original-audio timestamps' in reader.text
    assert '&lt;script&gt;' in reader.text
    assert 'existing-audio' not in reader.text
    missing = client.get('/transcripts/episodes/new', auth=('admin', 'testpass'))
    assert 'Request this transcript' in missing.text
    assert client.get('/transcripts').status_code == 401
    assert client.get('/transcripts/episodes/old').status_code == 401


def test_mcp_read_only_lifecycle_denials_and_validation(client, db):
    service.store(db, document())
    headers = {'Accept': 'application/json, text/event-stream'}
    auth = ('admin', 'testpass')
    def call(method, params=None):
        return client.post('/mcp', json=dict(jsonrpc='2.0', id=1, method=method, params=params or {}), headers=headers, auth=auth)
    assert client.post('/mcp', json={}).status_code == 401
    assert client.get('/mcp').status_code == 401
    assert client.get('/mcp', auth=auth).status_code == 405
    assert client.post('/mcp', json={}, headers={**headers, 'Origin': 'https://evil.example'}, auth=auth).status_code == 403
    result = call('initialize', dict(protocolVersion='2025-03-26')).json()['result']
    assert result['capabilities']['tools']['listChanged'] is False
    tools = call('tools/list').json()['result']['tools']
    assert len(tools) == 4 and all(t['annotations']['readOnlyHint'] for t in tools)
    result = call('tools/call', dict(name='search_transcripts', arguments=dict(query='orchard', podcast_id='p1'))).json()['result']
    data = json.loads(result['content'][0]['text'])
    assert data['hits'][0]['segments'][0]['reference'].startswith('old/')
    for params in (dict(name='request_transcript'), dict(name={}), dict(name='search_transcripts', arguments=dict(query='x', podcast_id='p1', limit=100))):
        body = call('tools/call', params).json()
        assert body.get('error') or body['result']['isError']
    assert db.get(TranscriptJob, 'new') is None
    assert call('unexpected').json()['error']['code'] == -32601
    assert client.post('/mcp', json=dict(jsonrpc='2.0', method='notifications/initialized'), headers=headers, auth=auth).status_code == 202
    assert client.post('/mcp', content=b'x' * 65537, headers=headers, auth=auth).status_code == 413


def test_session_write_requires_same_origin(client):
    from web.auth import SESSION_COOKIE, create_session_token
    client.cookies.set(SESSION_COOKIE, create_session_token())
    assert client.post('/api/transcripts/episodes/new/request').status_code == 403
    assert client.post('/api/transcripts/episodes/new/request', headers={'Origin': 'http://testserver'}).status_code == 200


def test_worker_publication_and_coverage_end_to_end(client, db):
    headers = {'X-Worker-API-Key': 'test-worker-key'}
    response = client.put('/api/worker/transcripts/old', json={'document': document().model_dump()}, headers=headers)
    assert response.status_code == 200
    checksum = response.json()['checksum']
    assert client.put('/api/worker/transcripts/old', json={'document': document().model_dump()}, headers=headers).json()['checksum'] == checksum
    response = client.get('/api/transcripts/episodes/old/coverage', auth=('admin', 'testpass'))
    assert response.json()['coverage']['state'] == 'available'
    assert db.get(Episode, 'old').processed_audio_key == 'existing-audio'


def test_large_preceding_context_cannot_hide_match(db):
    service.store(db, document(texts=['x' * 10000] * 12 + ['rareword independent context']))
    hit = service.TranscriptReader(db).search('rareword', 'p1')['hits'][0]
    assert any('rareword' in s['text'] for s in hit['segments'])


def test_confirmed_real_question_alongside_independent_cases(db):
    from pathlib import Path
    fixture = json.loads((Path(__file__).parents[1] / 'fixtures/animal_spirits_book_passage.json').read_text())
    episode = db.get(Episode, 'new')
    episode.title = 'Will Higher Rates Kill the Stock Market? (EP. 484)'
    db.get(Podcast, 'p1').title = 'Animal Spirits Podcast'
    db.commit()
    value = document('new').model_dump()
    value['segments'] = fixture['segments']
    service.store(db, TranscriptDocument.model_validate(value))
    result = service.TranscriptReader(db).search('What book were they talking about in my latest episode of Animal Spirits? The one guy really loved it.', 'p1', latest=True)
    assert any('Everywhere Millionaire' in s['text'] for h in result['hits'] for s in h['segments'])
    assert any(s['start'] == 3473.84 for h in result['hits'] for s in h['segments'])
    assert all(h['timestamp_basis'] == 'original_audio' for h in result['hits'])


def test_multiple_independent_candidates_are_context_not_generated_answer(db):
    service.store(db, document())
    result = service.TranscriptReader(db).search('memoir novel', 'p1')
    texts = [s['text'] for h in result['hits'] for s in h['segments']]
    assert any('memoir' in t for t in texts) and any('candidate novel' in t for t in texts)
    assert result['answer_status'] == 'retrieval_only' and result['ambiguity']
    assert result['coverage_counts'] == {'available': 1, 'missing': 1}


def test_podcast_delete_removes_canonical_and_fts(db):
    service.store(db, document())
    db.delete(db.get(Podcast, 'p1')); db.commit()
    assert not db.scalars(select(TranscriptVersion)).all()
    assert not db.execute(text('select episode_id from transcript_fts')).all()
    assert db.get(Episode, 'other') is not None


def test_canonical_commit_failure_does_not_publish_partial_version(db, monkeypatch):
    checksum = service.store(db, document())
    original = db.commit
    monkeypatch.setattr(db, 'commit', Mock(side_effect=RuntimeError('interrupted commit')))
    with pytest.raises(RuntimeError): service.store(db, document(texts=['Changed new content']))
    db.rollback()
    monkeypatch.setattr(db, 'commit', original)
    assert db.get(EpisodeTranscript, 'old').active_checksum == checksum
    assert len(db.scalars(select(TranscriptVersion)).all()) == 1


def test_large_cross_segment_phrase_preview_retains_both_sides(db):
    value = document(texts=['x' * 9000 + ' rarealpha', 'rarebeta ' + 'y' * 9000])
    service.store(db, value)
    hit = service.TranscriptReader(db).search('rarealpha rarebeta', 'p1', phrase=True)['hits'][0]
    assert 'rarealpha' in hit['segments'][0]['text'] and 'rarebeta' in hit['segments'][1]['text']
    assert all(s['text_truncated'] for s in hit['segments'])
    assert sum(len(s['text']) for s in hit['segments']) <= 16000


def test_worker_progress_distinguishes_observed_download_and_failure(client, db):
    episode = db.get(Episode, 'new'); episode.status = 'processing'; db.commit()
    headers = {'X-Worker-API-Key': 'test-worker-key'}
    assert client.post('/api/queue/new/progress', json={'step': 'transcribing'}, headers=headers).status_code == 200
    row = db.get(EpisodeTranscript, 'new')
    assert row.state == 'processing' and row.downloaded_at
    assert service.TranscriptReader(db).episodes('p1')['latest_downloaded']['episode_id'] == 'new'
    assert client.post('/api/queue/new/fail', json={'error': 'fixture failure'}, headers=headers).status_code == 200
    assert db.get(EpisodeTranscript, 'new').state == 'failed'


def test_available_from_audio_job_fulfills_queued_request(db):
    service.request_transcript(db, 'new')
    service.store(db, document('new'))
    assert db.get(TranscriptJob, 'new').state == 'complete'
    assert service.claim_transcript(db) is None


def test_heartbeat_renews_only_current_claim(db):
    service.request_transcript(db, 'new')
    job = service.claim_transcript(db)
    record = db.get(TranscriptJob, 'new')
    record.claimed_at = datetime.utcnow() - timedelta(hours=1)
    db.commit()
    service.heartbeat_transcript(db, 'new', job['claim_id'])
    assert db.get(TranscriptJob, 'new').claimed_at > datetime.utcnow() - timedelta(minutes=1)
    with pytest.raises(service.TranscriptError): service.heartbeat_transcript(db, 'new', 'old-attempt')


def test_permanent_outbox_error_is_retained_without_blocking_future_uploads(tmp_path):
    import httpx
    client = Mock()
    response = httpx.Response(409, request=httpx.Request('PUT', 'https://example.com/fixture'))
    client.publish_transcript.side_effect = httpx.HTTPStatusError('fixture', request=response.request, response=response)
    path = stage(tmp_path, document())
    assert flush(tmp_path, client)['rejected'] == 1
    assert path.with_suffix('.rejected').exists() and path.with_suffix('.json').exists()
    assert not path.exists()


def test_external_timestamp_provenance_and_ad_metadata(db):
    value = document().model_dump()
    value['timestamp_basis'] = 'external_source'
    value['ad_spans'][0]['confidence'] = .75
    checksum = service.store(db, TranscriptDocument.model_validate(value))
    assert json.loads(service.get_version(db, 'old').canonical_json)['ad_spans'][0]['confidence'] == .75
    assert service.TranscriptReader(db).passage('old', checksum)['timestamp_basis'] == 'external_source'


def test_normal_startup_does_not_migrate_and_opt_in_is_explicit(tmp_path, monkeypatch):
    from web.transcript_migration import ready
    monkeypatch.delenv('TRANSCRIPT_SCHEMA_OPT_IN', raising=False)
    engine = get_engine('sqlite:///' + str(tmp_path / 'legacy.sqlite'))
    init_db(engine)
    assert 'episodes' in inspect(engine).get_table_names() and not ready(engine)
    assert 'transcript_versions' not in inspect(engine).get_table_names()
    upgrade(engine)
    init_db(engine)
    assert ready(engine)
    downgrade(engine); init_db(engine)
    assert not ready(engine)  # Restart cannot undo owner rollback.
    monkeypatch.setenv('TRANSCRIPT_SCHEMA_OPT_IN', 'true')
    init_db(engine)
    assert ready(engine)
    engine.dispose()


def test_optional_read_token_disabled_scoped_and_revocable(client, db, monkeypatch):
    import hashlib
    service.store(db, document())
    token = 'synthetic-transcript-reader-only'
    headers = {'Authorization': 'Bearer ' + token}
    path = '/api/transcripts/search?query=orchard&podcast_id=p1'
    monkeypatch.delenv('TRANSCRIPT_READ_TOKEN_SHA256', raising=False)
    assert client.get(path, headers=headers).status_code == 401
    monkeypatch.setenv('TRANSCRIPT_READ_TOKEN_SHA256', hashlib.sha256(token.encode()).hexdigest())
    assert client.get(path, headers=headers).status_code == 200
    assert client.get('/api/transcripts/episodes/old', headers=headers).status_code == 200
    assert client.get('/transcripts', headers=headers).status_code == 401
    assert client.post('/api/transcripts/episodes/new/request', headers=headers).status_code == 401
    assert client.put('/api/worker/transcripts/old', headers=headers, json={'document': document().model_dump()}).status_code == 422
    assert client.get(path, headers={'Authorization': 'Bearer wrong-synthetic-reader'}).status_code == 401
    rpc_headers = {**headers, 'Accept': 'application/json, text/event-stream'}
    assert client.post('/mcp', headers=rpc_headers, json={'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'}).status_code == 200
    monkeypatch.delenv('TRANSCRIPT_READ_TOKEN_SHA256')
    assert client.get(path, headers=headers).status_code == 401


def test_outbox_404_retained_then_uploads_after_endpoint_rollout(tmp_path):
    import httpx
    response = httpx.Response(404, request=httpx.Request('PUT', 'https://fixture.example/transcript'))
    client = Mock()
    client.publish_transcript.side_effect = httpx.HTTPStatusError('fixture', request=response.request, response=response)
    path = stage(tmp_path, document())
    canonical = path.with_suffix('.json').read_bytes()
    assert flush(tmp_path, client)['failed'] == 1
    assert path.exists() and not path.with_suffix('.rejected').exists()
    client.publish_transcript.side_effect = None
    assert flush(tmp_path, client)['succeeded'] == 1
    assert path.with_suffix('.json').read_bytes() == canonical


def test_outbox_failure_rotation_does_not_starve_newer_records(tmp_path):
    import os
    client = Mock()
    paths = []
    for n in range(8):
        doc = document().model_copy(update={'episode_id': 'episode-' + str(n)})
        path = stage(tmp_path, doc)
        os.utime(path, (100 + n, 100 + n))
        paths.append(path)
    def publish(doc, *args):
        if int(doc.episode_id.rsplit('-', 1)[1]) < 5: raise RuntimeError()
    client.publish_transcript.side_effect = publish
    assert flush(tmp_path, client)['failed'] == 5
    assert flush(tmp_path, client)['succeeded'] == 3
    assert all(path.exists() for path in paths[:5])


def test_mcp_negotiation_honestly_returns_only_supported_version(client):
    headers = {'Accept': 'application/json, text/event-stream'}
    auth = ('admin', 'testpass')
    result = client.post('/mcp', headers=headers, auth=auth, json=dict(jsonrpc='2.0', id=1, method='initialize', params={'protocolVersion': '2025-11-25'})).json()
    assert result['result']['protocolVersion'] == '2025-03-26'
    assert client.post('/mcp', headers={**headers, 'MCP-Protocol-Version': '2025-11-25'}, auth=auth, json=dict(jsonrpc='2.0', id=2, method='tools/list')).status_code == 400
    assert client.post('/mcp', headers={**headers, 'MCP-Protocol-Version': '2025-03-26'}, auth=auth, json=dict(jsonrpc='2.0', id=2, method='tools/list')).status_code == 200
    assert client.post('/mcp', headers=headers, auth=auth, json=dict(jsonrpc='2.0', id=1, method='initialize', params={})).json()['error']['code'] == -32602


def test_unmigrated_feature_503_and_audio_progress_stays_working(client, db):
    downgrade(db.get_bind())
    assert client.get('/api/transcripts/episodes?podcast_id=p1', auth=('admin', 'testpass')).status_code == 503
    episode = db.get(Episode, 'new'); episode.status = 'processing'; db.commit()
    response = client.post('/api/queue/new/progress', headers={'X-Worker-API-Key': 'test-worker-key'}, json={'step': 'transcribing', 'percent': 1})
    assert response.status_code == 200


def test_mcp_modern_metadata_cannot_silently_use_legacy_semantics(client):
    response = client.post('/mcp', auth=('admin', 'testpass'), headers={'Accept': 'application/json, text/event-stream'}, json=dict(jsonrpc='2.0', id=1, method='tools/list', params={'_meta': {'io.modelcontextprotocol/protocolVersion': '2026-07-28'}}))
    assert response.status_code == 400 and response.json()['detail'] == 'Only legacy 2025-03-26 is supported'


def test_unmigrated_podcast_deletion_preserves_legacy_behavior(client, db):
    downgrade(db.get_bind())
    db.delete(db.get(Podcast, 'p1'))
    db.commit()
    assert db.get(Episode, 'old') is None and db.get(Episode, 'new') is None
    assert db.get(Episode, 'other') is not None
