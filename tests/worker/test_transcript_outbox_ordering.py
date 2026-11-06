"""Local durable outbox ordering, using actual transcript HTTP persistence."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
from pathlib import Path
import threading
from unittest.mock import Mock

import httpx
import pytest

from adnihilator.transcript_document import TranscriptDocument
from tests.web.test_transcript_ui_regressions import transcript_ui_client
from worker.transcripts import flush, stage


def revision(text: str, tagged: bool = False, episode_id: str = "outbox") -> TranscriptDocument:
    """Build distinct valid producer revisions without paid transcription."""
    return TranscriptDocument(episode_id=episode_id, podcast_id="ui-podcast", guid="guid-" + episode_id,
                              original_audio_url="https://example.invalid/" + episode_id + ".mp3", source="synthetic-outbox",
                              segments=[dict(index=0, start=0, end=2, text=text)],
                              ad_spans=[dict(start=0, end=1)] if tagged else [])


@pytest.fixture
def outbox_http(transcript_ui_client: tuple) -> tuple:
    """Publish through real worker auth/ASGI/SQLite, keeping secrets synthetic."""
    client, factory = transcript_ui_client
    from web.models import Episode
    with factory() as db:
        for name in ("outbox", "unrelated"):
            db.add(Episode(id=name, podcast_id="ui-podcast", guid="guid-" + name, title=name,
                           status="skipped", original_audio_url="https://example.invalid/" + name + ".mp3"))
        db.commit()

    class Publisher:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.fail_next = False

        def publish_transcript(self, document: TranscriptDocument, claim_id: str | None, downloaded_at: str | None) -> None:
            self.calls.append(document.checksum())
            if self.fail_next:
                self.fail_next = False
                raise httpx.ConnectError("synthetic pre-delivery failure")
            response = client.put("/api/worker/transcripts/" + document.episode_id,
                                  headers={"X-Worker-API-Key": "synthetic-worker"},
                                  json=dict(document=document.model_dump(), claim_id=claim_id, downloaded_at=downloaded_at))
            response.raise_for_status()

    return Publisher(), client, factory


class TestLocalRevisionOrdering:
    @pytest.mark.parametrize("trial", [1, 2])
    def test_failed_old_delivery_cannot_roll_back_a_new_local_stage(self, tmp_path: Path, outbox_http: tuple, trial: int) -> None:
        publisher, client, _ = outbox_http
        old, final = revision("old interim"), revision("final correction", tagged=True)
        stage(tmp_path, old)
        publisher.fail_next = True
        assert flush(tmp_path, publisher)["failed"] == 1
        stage(tmp_path, final)
        before = len(publisher.calls)
        flush(tmp_path, publisher)
        flush(tmp_path, publisher)  # restart/lost-retry cycle cannot resurrect old pending
        assert publisher.calls[before:] == [final.checksum()]
        passage = client.get("/api/transcripts/episodes/outbox").json()
        assert passage["checksum"] == final.checksum()
        assert passage["segments"][0]["ad"] is True
        assert all((tmp_path / "outbox" / (doc.checksum() + ".json")).exists() for doc in (old, final))

    @pytest.mark.parametrize("boundary", ["marker", "pending", "cleanup"])
    def test_staging_crash_retains_archives_and_recovery_never_publishes_superseded_bytes(
        self, tmp_path: Path, outbox_http: tuple, monkeypatch: pytest.MonkeyPatch, boundary: str
    ) -> None:
        """Inject actual filesystem replacement failures at commit boundaries."""
        publisher, client, _ = outbox_http
        old, final = revision("old interim"), revision("final correction", tagged=True)
        stage(tmp_path, old)
        import worker.transcripts as outbox
        replace = outbox.os.replace
        fired = False

        def crash(source: Path, target: Path) -> None:
            nonlocal fired
            name = Path(target).name
            chosen = (boundary == "marker" and name.startswith(".latest-")) or (
                boundary == "pending" and name.endswith(".pending") and final.checksum() in name
            ) or (boundary == "cleanup" and name.endswith(".superseded") and old.checksum() in name)
            if chosen and not fired:
                fired = True
                raise OSError("synthetic interrupted replace")
            replace(source, target)

        with monkeypatch.context() as scoped:
            scoped.setattr(outbox.os, "replace", crash)
            with pytest.raises(OSError):
                stage(tmp_path, final)
        assert fired
        assert (tmp_path / "outbox" / (final.checksum() + ".json")).exists()
        if boundary == "marker":
            # The latest marker is the stage commit point. A pre-commit failure
            # retains B's archive; producer retry commits B before publishing it.
            stage(tmp_path, final)
        flush(tmp_path, publisher)
        assert publisher.calls == [final.checksum()]
        assert client.get("/api/transcripts/episodes/outbox").json()["checksum"] == final.checksum()
        assert (tmp_path / "outbox" / (old.checksum() + ".json")).exists()

    def test_duplicate_stages_and_unrelated_episode_do_not_change_server_idempotency(self, tmp_path: Path, outbox_http: tuple) -> None:
        publisher, client, factory = outbox_http
        final, other = revision("final", tagged=True), revision("other", episode_id="unrelated")
        stage(tmp_path, final)
        stage(tmp_path, final)
        stage(tmp_path, other)
        assert flush(tmp_path, publisher, limit=5)["succeeded"] == 2
        stage(tmp_path, final)
        assert flush(tmp_path, publisher)["succeeded"] == 1
        from sqlalchemy import func, select
        from web.models import TranscriptVersion
        with factory() as db:
            assert db.scalar(select(func.count()).select_from(TranscriptVersion).where(TranscriptVersion.episode_id == "outbox")) == 1
        assert client.get("/api/transcripts/episodes/unrelated").json()["checksum"] == other.checksum()

    def test_claim_scopes_do_not_coalesce_each_other(self, tmp_path: Path) -> None:
        """The same document under another existing claim is a distinct local key."""
        document = revision("same canonical content")
        stage(tmp_path, document, None)
        stage(tmp_path, document, "existing-recovery-claim")
        publisher = Mock()
        assert flush(tmp_path, publisher)["succeeded"] == 2
        assert {call.args[1] for call in publisher.publish_transcript.call_args_list} == {None, "existing-recovery-claim"}

    def test_inflight_old_publish_finishes_before_new_stage_can_publish(self, tmp_path: Path, outbox_http: tuple) -> None:
        publisher, _, _ = outbox_http
        old, final = revision("old interim"), revision("final", tagged=True)
        stage(tmp_path, old)
        started, release, staging = threading.Event(), threading.Event(), threading.Event()
        publish = publisher.publish_transcript

        def paused(document: TranscriptDocument, claim_id: str | None, downloaded_at: str | None) -> None:
            started.set()
            assert release.wait(5)
            publish(document, claim_id, downloaded_at)

        publisher.publish_transcript = paused
        with ThreadPoolExecutor(max_workers=2) as pool:
            publishing = pool.submit(flush, tmp_path, publisher)
            assert started.wait(5)

            def later_stage() -> None:
                staging.set()
                stage(tmp_path, final)

            new_stage = pool.submit(later_stage)
            assert staging.wait(5)
            try:
                with pytest.raises(TimeoutError):
                    new_stage.result(timeout=0.3)
            finally:
                release.set()
            publishing.result(timeout=5)
            new_stage.result(timeout=5)
        publisher.publish_transcript = publish
        flush(tmp_path, publisher)
        assert publisher.calls == [old.checksum(), final.checksum()]

    def test_ambiguous_legacy_pending_is_retained_without_blocking_other_episodes(self, tmp_path: Path) -> None:
        """Mutable old mtimes are not promoted into an invented revision order."""
        old, final = revision("legacy old"), revision("legacy final", tagged=True)
        directory = tmp_path / "outbox"
        directory.mkdir()
        for document in (old, final):
            (directory / (document.checksum() + ".json")).write_bytes(document.canonical_bytes())
            (directory / (document.checksum() + ".pending")).write_text(json.dumps(dict(document=document.model_dump(), claim_id=None, downloaded_at=None)))
        other = revision("healthy unrelated", episode_id="unrelated")
        stage(tmp_path, other)
        publisher = Mock()
        result = flush(tmp_path, publisher)
        assert result["failed"] == 2
        assert publisher.publish_transcript.call_count == 1
        assert publisher.publish_transcript.call_args.args[0].episode_id == "unrelated"
        assert len(list(directory.glob("*.pending"))) == 2
        assert all((directory / (document.checksum() + ".json")).exists() for document in (old, final))
