"""Regressions for repeated GUIDs and failed feed transactions."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock, get_ident
from unittest.mock import patch

import feedparser
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from web.database import Base, get_engine, init_db
from web.models import Episode, Podcast
from web.services.feed_sync import FeedSyncError, sync_all_podcasts, sync_podcast


@pytest.fixture
def db() -> Iterator[Session]:
    """Use an isolated database with real uniqueness constraints."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def parsed_feed(duplicate: bool = False) -> feedparser.FeedParserDict:
    """Parse local RSS bytes instead of fetching a real podcast."""
    first = '<item><guid>episode-1</guid><title>First title</title><enclosure url="https://example.invalid/one.mp3" type="audio/mpeg"/></item>'
    second = '<item><guid>episode-1</guid><title>Repeated title</title></item>' if duplicate else ""
    return feedparser.parse(f'<rss version="2.0"><channel><title>Local feed</title>{first}{second}</channel></rss>')


def force_flush_conflict(session: Session, context: object, instances: object) -> None:
    """Simulate a conflicting writer with a real constraint failure at flush."""
    for row in list(session.new):
        if isinstance(row, Episode) and row.podcast_id == "broken":
            session.add(Episode(podcast_id=row.podcast_id, guid=row.guid))


class TestFeedSyncIsolation:
    def test_two_concurrent_refreshes_converge_without_error(self, tmp_path: Path) -> None:
        """Two processes' equivalent snapshots converge on one durable episode."""
        engine = init_db(get_engine(f"sqlite:///{tmp_path / 'race.db'}"))
        with Session(engine) as session:
            session.add(Podcast(id="race", title="Local feed", source_rss_url="https://example.invalid/feed"))
            session.commit()
        barrier = Barrier(2)
        seen: set[int] = set()
        lock = Lock()

        def synchronize_snapshots(
            connection: object, cursor: object, statement: str, parameters: object,
            context: object, executemany: bool,
        ) -> None:
            """Make only each thread's first existing-GUID snapshot overlap."""
            if "SELECT episodes.guid" not in statement:
                return
            with lock:
                first_snapshot = get_ident() not in seen
                seen.add(get_ident())
            if first_snapshot:
                barrier.wait(timeout=5)

        def refresh() -> int:
            """Use independent real SQLAlchemy sessions, as multiple app workers do."""
            with Session(engine) as session:
                count = sync_podcast(session, session.get(Podcast, "race"))
                assert session.query(Episode).count() == 1
                return count

        event.listen(engine, "after_cursor_execute", synchronize_snapshots)
        try:
            with patch("web.services.feed_sync.feedparser.parse", return_value=parsed_feed()):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = [future.result(timeout=10) for future in [pool.submit(refresh), pool.submit(refresh)]]
        finally:
            event.remove(engine, "after_cursor_execute", synchronize_snapshots)
        assert sorted(results) == [0, 1]
        with Session(engine) as session:
            assert session.query(Episode).count() == 1
            assert session.get(Podcast, "race").last_synced_at is not None
        engine.dispose()

    def test_unrelated_check_constraint_is_not_retried(self, db: Session) -> None:
        """A real non-GUID constraint failure is rolled back without refetching."""
        podcast = Podcast(id="check-failure", source_rss_url="https://example.invalid/feed")
        db.add(podcast)
        db.commit()

        def invalidate_status(session: Session, context: object, instances: object) -> None:
            """Fail the existing SQLite status CHECK during the episode flush."""
            for row in session.new:
                if isinstance(row, Episode):
                    row.status = "invalid-status"

        event.listen(db, "before_flush", invalidate_status)
        try:
            with patch("web.services.feed_sync.feedparser.parse", return_value=parsed_feed()) as parse:
                with pytest.raises(FeedSyncError, match="CHECK constraint failed: ck_episode_status"):
                    sync_podcast(db, podcast)
                parse.assert_called_once()
                assert parse.call_args.args == ("https://example.invalid/feed",)
        finally:
            event.remove(db, "before_flush", invalidate_status)

        assert db.query(Episode).count() == 0
        assert db.get(Podcast, "check-failure").last_synced_at is None
        # A subsequent successful transaction proves rollback restored usability.
        with patch("web.services.feed_sync.feedparser.parse", return_value=parsed_feed()):
            assert sync_podcast(db, podcast) == 1

    def test_bulk_diagnostics_mask_credentials(self, db: Session) -> None:
        """A failed fetch can report its cause without returning URL credentials."""
        db.add(Podcast(id="secret", source_rss_url="https://example.invalid/feed"))
        db.commit()
        error = RuntimeError("fetch failed key=SYNTHCANARY https://name:SYNTHPASSWORD@example.invalid/feed")
        with patch("web.services.feed_sync.feedparser.parse", side_effect=error):
            results = sync_all_podcasts(db)
        assert "SYNTHCANARY" not in results["secret"]["error"]
        assert "SYNTHPASSWORD" not in results["secret"]["error"]
        assert "fetch failed" in results["secret"]["error"]

    def test_repeated_feed_guid_is_added_once(self, db: Session) -> None:
        """A publisher repeating a GUID cannot break the subscription."""
        podcast = Podcast(id="duplicates", source_rss_url="https://example.invalid/feed")
        db.add(podcast)
        db.commit()
        with patch("web.services.feed_sync.feedparser.parse", return_value=parsed_feed(duplicate=True)):
            assert sync_podcast(db, podcast) == 1
            assert sync_podcast(db, podcast) == 0
        episode = db.query(Episode).one()
        assert episode.title == "First title"
        assert podcast.title == "Local feed"
        assert podcast.last_synced_at is not None

    def test_failed_single_sync_rolls_back_before_returning(self, db: Session) -> None:
        """A real failed flush leaves the caller able to query its session."""
        podcast = Podcast(id="broken", source_rss_url="https://example.invalid/broken")
        db.add(podcast)
        db.commit()
        event.listen(db, "before_flush", force_flush_conflict)
        try:
            with patch("web.services.feed_sync.feedparser.parse", return_value=parsed_feed()):
                with pytest.raises(FeedSyncError, match="IntegrityError"):
                    sync_podcast(db, podcast)
        finally:
            event.remove(db, "before_flush", force_flush_conflict)
        assert db.query(Episode).count() == 0
        assert db.get(Podcast, "broken").last_synced_at is None

    def test_failed_flush_does_not_abort_other_subscriptions(self, db: Session) -> None:
        """Refresh All retains good results around a bad feed transaction."""
        for identity in ["good-before", "broken", "good-after"]:
            db.add(Podcast(id=identity, source_rss_url=f"https://example.invalid/{identity}"))
        db.commit()
        event.listen(db, "before_flush", force_flush_conflict)
        try:
            with patch("web.services.feed_sync.feedparser.parse", return_value=parsed_feed()):
                results = sync_all_podcasts(db)
        finally:
            event.remove(db, "before_flush", force_flush_conflict)
        assert results["broken"]["added"] == 0
        assert "UNIQUE constraint failed" in results["broken"]["error"]
        for identity in ["good-before", "good-after"]:
            assert results[identity] == {"added": 1, "error": None}
            assert db.get(Podcast, identity).last_synced_at is not None
        assert db.query(Episode).count() == 2
