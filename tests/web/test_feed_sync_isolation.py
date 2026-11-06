"""Regressions for repeated GUIDs and failed feed transactions."""

from collections.abc import Iterator
from unittest.mock import patch

import feedparser
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from web.database import Base
from web.models import Episode, Podcast
from web.services.feed_sync import sync_all_podcasts, sync_podcast


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
                with pytest.raises(IntegrityError):
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
