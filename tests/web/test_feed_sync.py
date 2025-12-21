"""Tests for feed sync service."""

import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from web.database import Base
from web.models import Episode, Podcast


@pytest.fixture
def db_session():
    """Create test database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def mock_feedparser():
    """Mock feedparser."""
    with patch("web.services.feed_sync.feedparser") as mock:
        yield mock


def test_sync_podcast_adds_new_episodes(db_session, mock_feedparser):
    """Test syncing adds new episodes."""
    from web.services.feed_sync import sync_podcast

    # Create podcast
    podcast = Podcast(
        source_rss_url="https://example.com/feed.xml",
        title="Test Podcast",
    )
    db_session.add(podcast)
    db_session.commit()

    # Mock feed response - use spec to prevent MagicMock auto-attributes
    mock_feed = MagicMock()
    mock_feed.title = "Test Podcast"
    # Explicitly set no summary/subtitle so hasattr returns False
    del mock_feed.summary
    del mock_feed.subtitle
    del mock_feed.itunes_image
    del mock_feed.image

    # Create entry mocks with only needed attributes
    entry1 = MagicMock()
    entry1.id = "ep-1"
    entry1.title = "Episode 1"
    entry1.enclosures = [MagicMock(href="https://example.com/ep1.mp3")]
    entry1.published_parsed = (2024, 1, 1, 0, 0, 0, 0, 0, 0)
    # Remove auto-attributes that would cause MagicMock to be stored
    del entry1.summary
    del entry1.description
    del entry1.link
    del entry1.itunes_duration

    entry2 = MagicMock()
    entry2.id = "ep-2"
    entry2.title = "Episode 2"
    entry2.enclosures = [MagicMock(href="https://example.com/ep2.mp3")]
    entry2.published_parsed = (2024, 1, 2, 0, 0, 0, 0, 0, 0)
    del entry2.summary
    del entry2.description
    del entry2.link
    del entry2.itunes_duration

    mock_feedparser.parse.return_value = MagicMock(
        feed=mock_feed,
        entries=[entry1, entry2],
    )

    # Sync
    added = sync_podcast(db_session, podcast)

    assert added == 2
    episodes = db_session.query(Episode).filter_by(podcast_id=podcast.id).all()
    assert len(episodes) == 2


def test_sync_podcast_skips_existing(db_session, mock_feedparser):
    """Test syncing skips already-known episodes."""
    from web.services.feed_sync import sync_podcast

    # Create podcast with existing episode
    podcast = Podcast(
        source_rss_url="https://example.com/feed.xml",
        title="Test Podcast",
    )
    db_session.add(podcast)
    db_session.commit()

    episode = Episode(
        podcast_id=podcast.id,
        guid="ep-1",
        title="Episode 1",
    )
    db_session.add(episode)
    db_session.commit()

    # Mock feed response - use spec to prevent MagicMock auto-attributes
    mock_feed = MagicMock()
    mock_feed.title = "Test Podcast"
    # Explicitly set no summary/subtitle so hasattr returns False
    del mock_feed.summary
    del mock_feed.subtitle
    del mock_feed.itunes_image
    del mock_feed.image

    # Create entry mock with only needed attributes
    entry1 = MagicMock()
    entry1.id = "ep-1"
    entry1.title = "Episode 1"
    entry1.enclosures = [MagicMock(href="https://example.com/ep1.mp3")]
    entry1.published_parsed = (2024, 1, 1, 0, 0, 0, 0, 0, 0)
    del entry1.summary
    del entry1.description
    del entry1.link
    del entry1.itunes_duration

    mock_feedparser.parse.return_value = MagicMock(
        feed=mock_feed,
        entries=[entry1],
    )

    # Sync
    added = sync_podcast(db_session, podcast)

    assert added == 0
    episodes = db_session.query(Episode).filter_by(podcast_id=podcast.id).all()
    assert len(episodes) == 1
