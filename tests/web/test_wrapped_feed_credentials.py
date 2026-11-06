"""Wrapped credentials stay masked through actual feed transaction boundaries."""

import traceback
from pathlib import Path

import feedparser
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from web.models import Base, Podcast
from web.services.feed_sync import FeedSyncError, sync_all_podcasts, sync_podcast


@pytest.mark.parametrize("wrapper", [
    'stderr="password={secret}"',
    '{{"detail":"api_key={secret}"}}',
    "Error: --token={secret}",
    "error=OPENAI_API_KEY={secret}",
    "Authorization: {secret} rejected by upstream",
])
def test_wrapped_credentials_survive_no_public_error_boundary(
    wrapper: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Single/bulk sync, traceback formatting and healthy continuation agree."""
    canary = "SYNTHETIC-WRAPPED-UNKNOWN-CREDENTIAL"
    payload = "feed refused\n" + wrapper.format(secret=canary)

    def parse(url: str) -> feedparser.FeedParserDict:
        if url.endswith("/bad"):
            raise RuntimeError(payload)
        return feedparser.FeedParserDict(feed={}, entries=[])

    monkeypatch.setattr("web.services.feed_sync.feedparser.parse", parse)
    engine = create_engine(f"sqlite:///{tmp_path / 'feeds.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        bad = Podcast(id="bad", source_rss_url="https://example.invalid/bad")
        good = Podcast(id="good", source_rss_url="https://example.invalid/good")
        db.add_all([bad, good])
        db.commit()
        with pytest.raises(FeedSyncError) as captured:
            sync_podcast(db, bad)
        trace = "".join(traceback.format_exception(captured.value))
        assert canary not in trace
        assert "feed refused" in trace
        assert "[REDACTED]" in trace
        assert db.query(Podcast).count() == 2
        results = sync_all_podcasts(db)
        assert canary not in results["bad"]["error"]
        assert results["good"]["error"] is None
        assert db.get(Podcast, "good").last_synced_at is not None
    engine.dispose()
