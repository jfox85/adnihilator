"""Tests for database models."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from web.database import Base
from web.models import Podcast, Episode, EpisodeStatus


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_create_podcast(db_session):
    """Test creating a podcast."""
    podcast = Podcast(
        source_rss_url="https://example.com/feed.xml",
        title="Test Podcast",
    )
    db_session.add(podcast)
    db_session.commit()

    assert podcast.id is not None
    assert podcast.feed_token is not None
    assert len(podcast.feed_token) >= 32
    assert podcast.auto_process is True


def test_create_episode(db_session):
    """Test creating an episode linked to a podcast."""
    podcast = Podcast(
        source_rss_url="https://example.com/feed.xml",
        title="Test Podcast",
    )
    db_session.add(podcast)
    db_session.commit()

    episode = Episode(
        podcast_id=podcast.id,
        guid="episode-123",
        title="Test Episode",
        original_audio_url="https://example.com/episode.mp3",
    )
    db_session.add(episode)
    db_session.commit()

    assert episode.id is not None
    assert episode.status == EpisodeStatus.PENDING
    assert episode.retry_count == 0


def test_episode_status_enum(db_session):
    """Test that episode status uses the enum correctly."""
    podcast = Podcast(
        source_rss_url="https://example.com/feed.xml",
        title="Test Podcast",
    )
    db_session.add(podcast)
    db_session.commit()

    episode = Episode(
        podcast_id=podcast.id,
        guid="episode-456",
        title="Another Episode",
        original_audio_url="https://example.com/ep2.mp3",
        status=EpisodeStatus.PROCESSING,
    )
    db_session.add(episode)
    db_session.commit()

    assert episode.status == EpisodeStatus.PROCESSING
