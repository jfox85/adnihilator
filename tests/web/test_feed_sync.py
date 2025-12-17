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

    # Mock feed response
    mock_feedparser.parse.return_value = MagicMock(
        feed=MagicMock(title="Test Podcast"),
        entries=[
            MagicMock(
                id="ep-1",
                title="Episode 1",
                enclosures=[MagicMock(href="https://example.com/ep1.mp3")],
                published_parsed=(2024, 1, 1, 0, 0, 0, 0, 0, 0),
            ),
            MagicMock(
                id="ep-2",
                title="Episode 2",
                enclosures=[MagicMock(href="https://example.com/ep2.mp3")],
                published_parsed=(2024, 1, 2, 0, 0, 0, 0, 0, 0),
            ),
        ],
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

    # Mock feed response with same episode
    mock_feedparser.parse.return_value = MagicMock(
        feed=MagicMock(title="Test Podcast"),
        entries=[
            MagicMock(
                id="ep-1",
                title="Episode 1",
                enclosures=[MagicMock(href="https://example.com/ep1.mp3")],
                published_parsed=(2024, 1, 1, 0, 0, 0, 0, 0, 0),
            ),
        ],
    )

    # Sync
    added = sync_podcast(db_session, podcast)

    assert added == 0
    episodes = db_session.query(Episode).filter_by(podcast_id=podcast.id).all()
    assert len(episodes) == 1
