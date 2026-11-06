"""Real SQLite/ASGI schedules for confirmed transcript write races."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta
import threading
from typing import Any

import pytest
from sqlalchemy import event, func, select, text
from sqlalchemy.orm import Session

from adnihilator.transcript_document import TranscriptDocument
from tests.web.test_transcript_ui_regressions import transcript_ui_client
from web.models import Episode, EpisodeTranscript, TranscriptJob, TranscriptVersion

HEADERS = {"X-Worker-API-Key": "synthetic-worker"}


@pytest.fixture
def race_case(transcript_ui_client: tuple) -> tuple:
    client, factory = transcript_ui_client
    with factory() as db:
        db.add(Episode(id="race", podcast_id="ui-podcast", guid="race-guid", title="Synthetic race",
                       status="skipped", original_audio_url="https://example.invalid/race.mp3"))
        db.commit()
    return client, factory


def race_document(word: str) -> dict[str, Any]:
    return TranscriptDocument(episode_id="race", podcast_id="ui-podcast", guid="race-guid",
                              original_audio_url="https://example.invalid/race.mp3", source="synthetic-race",
                              segments=[dict(index=0, start=0, end=2, text=word)]).model_dump()


def claim(client: Any) -> str:
    assert client.post("/api/transcripts/episodes/race/request").status_code == 200
    response = client.post("/api/worker/transcript-jobs/claim", headers=HEADERS)
    assert response.status_code == 200
    return response.json()["claim_id"]


@contextmanager
def before_job_write(engine: Any, competing: Any, expiry: bool = False):
    """A second real request wins before a stale observation's write.

    Inject before an explicit write lock/CAS if present, otherwise after the
    vulnerable SELECT has executed. This permits either valid serialization
    strategy without requiring the old implementation's statement sequence.
    """
    state = {"fired": False}

    def win() -> None:
        if not state["fired"]:
            state["fired"] = True
            competing()

    def before(conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, many: bool) -> None:
        lowered = statement.lower().strip()
        if not state["fired"] and (lowered == "begin immediate" or lowered.startswith("update transcript_jobs")):
            win()

    def after(conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, many: bool) -> None:
        lowered = statement.lower()
        if not state["fired"] and lowered.lstrip().startswith("select") and "from transcript_jobs" in lowered:
            win()

    event.listen(engine, "before_cursor_execute", before)
    event.listen(engine, "after_cursor_execute", after)
    try:
        yield state
    finally:
        event.remove(engine, "before_cursor_execute", before)
        event.remove(engine, "after_cursor_execute", after)


class TestExistingClaimFences:
    @pytest.mark.parametrize("operation", ["fail", "publish"])
    def test_late_operation_cannot_overwrite_new_claim(self, race_case: tuple, operation: str) -> None:
        client, factory = race_case
        old = claim(client)
        newer = []

        def replace_claim() -> None:
            assert client.post("/api/worker/transcript-jobs/race/fail", headers=HEADERS, json={"claim_id": old}).status_code == 200
            response = client.post("/api/worker/transcript-jobs/claim", headers=HEADERS)
            assert response.status_code == 200
            newer.append(response.json()["claim_id"])

        with before_job_write(factory.kw["bind"], replace_claim) as schedule:
            if operation == "fail":
                response = client.post("/api/worker/transcript-jobs/race/fail", headers=HEADERS, json={"claim_id": old})
            else:
                response = client.put("/api/worker/transcripts/race", headers=HEADERS,
                                      json={"claim_id": old, "document": race_document("stale bytes")})
        assert schedule["fired"]
        assert response.status_code == 409
        assert client.post("/api/worker/transcript-jobs/race/heartbeat", headers=HEADERS, json={"claim_id": newer[0]}).status_code == 200
        with factory() as db:
            job = db.get(TranscriptJob, "race")
            assert (job.state, job.claim_id, job.attempt) == ("processing", newer[0], 2)
            assert db.scalar(select(func.count()).select_from(TranscriptVersion).where(TranscriptVersion.episode_id == "race")) == 0

    def test_successful_renewal_is_not_revoked_from_expired_snapshot(self, race_case: tuple) -> None:
        client, factory = race_case
        token = claim(client)
        with factory() as db:
            db.get(TranscriptJob, "race").claimed_at = datetime.utcnow() - timedelta(hours=3)
            db.commit()

        def renew() -> None:
            response = client.post("/api/worker/transcript-jobs/race/heartbeat", headers=HEADERS, json={"claim_id": token})
            assert response.status_code == 200

        with before_job_write(factory.kw["bind"], renew, expiry=True) as schedule:
            response = client.post("/api/worker/transcript-jobs/claim", headers=HEADERS)
        assert schedule["fired"]
        assert response.status_code == 200 and response.json() is None
        with factory() as db:
            job = db.get(TranscriptJob, "race")
            assert (job.state, job.claim_id, job.attempt) == ("processing", token, 1)


class TestConcurrentFirstRequest:
    def test_two_first_requests_are_idempotent_without_unique_500(self, race_case: tuple) -> None:
        client, factory = race_case
        barrier, guard = threading.Barrier(2), threading.Lock()
        seen = 0

        def pause_unlocked_read(conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, many: bool) -> None:
            nonlocal seen
            if "from episode_transcripts" not in statement.lower() or "race" not in parameters:
                return
            if conn.connection.driver_connection.in_transaction:
                return  # A real write transaction already serializes this read.
            with guard:
                seen += 1
                pause = seen <= 2
            if pause:
                barrier.wait(timeout=5)

        event.listen(factory.kw["bind"], "after_cursor_execute", pause_unlocked_read)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _: client.post("/api/transcripts/episodes/race/request"), range(2)))
        finally:
            event.remove(factory.kw["bind"], "after_cursor_execute", pause_unlocked_read)
        assert [result.status_code for result in results] == [200, 200]
        with factory() as db:
            assert db.scalar(select(func.count()).select_from(TranscriptJob).where(TranscriptJob.episode_id == "race")) == 1
            assert db.scalar(select(func.count()).select_from(EpisodeTranscript).where(EpisodeTranscript.episode_id == "race")) == 1
            assert (db.get(TranscriptJob, "race").state, db.get(TranscriptJob, "race").attempt) == ("queued", 0)


class TestIndexPublication:
    def test_old_rebuild_cannot_remove_new_active_index(self, race_case: tuple) -> None:
        client, factory = race_case
        first = client.put("/api/worker/transcripts/race", headers=HEADERS, json={"document": race_document("oldalpha")})
        assert first.status_code == 200
        state = {"armed": False, "fired": False, "new_checksum": None}

        def canonical_committed(session: Session) -> None:
            if not state["fired"] and session.get_bind() is factory.kw["bind"]:
                state["armed"] = True

        def publish_new() -> None:
            if state["armed"] and not state["fired"]:
                state["fired"] = True
                response = client.put("/api/worker/transcripts/race", headers=HEADERS, json={"document": race_document("newbeta")})
                assert response.status_code == 200
                state["new_checksum"] = response.json()["checksum"]

        def before(conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, many: bool) -> None:
            lowered = statement.lower().strip()
            if state["armed"] and (lowered == "begin immediate" or lowered.startswith("update episode_transcripts")):
                publish_new()

        def after(conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, many: bool) -> None:
            if state["armed"] and "from transcript_versions" in statement.lower():
                publish_new()

        event.listen(Session, "after_commit", canonical_committed)
        event.listen(factory.kw["bind"], "before_cursor_execute", before)
        event.listen(factory.kw["bind"], "after_cursor_execute", after)
        try:
            response = client.put("/api/worker/transcripts/race", headers=HEADERS, json={"document": race_document("oldalpha")})
        finally:
            event.remove(Session, "after_commit", canonical_committed)
            event.remove(factory.kw["bind"], "before_cursor_execute", before)
            event.remove(factory.kw["bind"], "after_cursor_execute", after)
        assert response.status_code == 200 and state["fired"]
        coverage = client.get("/api/transcripts/episodes/race/coverage").json()["coverage"]
        assert coverage["checksum"] == state["new_checksum"] and coverage["index_ready"]
        assert coverage["error"] is None
        assert client.get("/api/transcripts/search", params={"query": "newbeta", "podcast_id": "ui-podcast"}).json()["hits"]
        with factory() as db:
            assert set(db.execute(text("SELECT checksum FROM transcript_fts WHERE episode_id='race'")).scalars()) == {state["new_checksum"]}
