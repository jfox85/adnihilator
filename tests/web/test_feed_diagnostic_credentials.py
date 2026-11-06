"""Feed errors redact URL and header credentials before diagnostic boundaries."""

import traceback
from unittest.mock import patch

import feedparser
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from web.database import Base
from web.models import Episode, Podcast
from web.services.feed_sync import FeedSyncError, sync_all_podcasts, sync_podcast


class TestFeedDiagnosticCredentials:
    @pytest.mark.parametrize("diagnostic,canary", [
        ("HTTPS://user:FEED-UPPER-CREDENTIAL@example.invalid/feed", "FEED-UPPER-CREDENTIAL"),
        ("https://FEED-USERNAME-CREDENTIAL@example.invalid/feed", "FEED-USERNAME-CREDENTIAL"),
        ("Authorization: Basic dXNlcjpGRUVELUJBU0lDLVNFQ1JFVA==", "dXNlcjpGRUVELUJBU0lDLVNFQ1JFVA=="),
    ])
    def test_credentials_are_absent_from_returned_errors_and_tracebacks(
        self, diagnostic: str, canary: str
    ) -> None:
        """Direct failures and bulk responses remain safe without blocking later feeds."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        good_feed = feedparser.parse(
            '<rss version="2.0"><channel><title>Healthy</title><item><guid>one</guid></item></channel></rss>'
        )

        def parse(url: str) -> feedparser.FeedParserDict:
            """Keep all feed I/O mocked while exercising real transactions."""
            if url.endswith("/bad"):
                raise RuntimeError(f"feed fetch failed: {diagnostic}")
            return good_feed

        try:
            with Session(engine) as db:
                podcast = Podcast(id="bad", source_rss_url="https://example.invalid/bad")
                db.add_all([podcast, Podcast(id="good", source_rss_url="https://example.invalid/good")])
                db.commit()
                with patch("web.services.feed_sync.feedparser.parse", side_effect=parse):
                    with pytest.raises(FeedSyncError) as failure:
                        sync_podcast(db, podcast)
                    formatted = "".join(traceback.format_exception(failure.value))
                    assert canary not in formatted
                    assert "feed fetch failed" in formatted
                    assert "[REDACTED]" in formatted
                    results = sync_all_podcasts(db)
                assert canary not in results["bad"]["error"]
                assert "feed fetch failed" in results["bad"]["error"]
                assert results["good"] == {"added": 1, "error": None}
                assert db.query(Episode).count() == 1
        finally:
            engine.dispose()
