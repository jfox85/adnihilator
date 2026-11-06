"""Historical diagnostic assignments cannot depend on current environment keys."""

import json
import traceback
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from adnihilator.redaction import redact_secrets
from web.models import Base, Podcast
from web.services.feed_sync import FeedSyncError, sync_podcast


class TestHistoricalNamedCredentials:
    """Mask historical assignments while keeping ordinary diagnostics useful."""

    @pytest.mark.parametrize("name", [
        "access_token", "refresh_token", "client_secret", "OPENAI_API_KEY",
        "GEMINI_API_KEY", "R2_SECRET_KEY", "AWS_SECRET_ACCESS_KEY", "private_key",
        "x-api-key", "X-Goog-Api-Key", "password", "passwd", "secret", "Signature",
        "Credential", "session_token",
    ])
    def test_historical_json_field_is_masked(self, name: str) -> None:
        """Rotated or worker-only credentials are masked without env lookup."""
        value = "HISTORICAL-SYNTHETIC-CREDENTIAL"
        text = json.dumps({name: value, "status": 503, "token_count": 17})
        safe = redact_secrets(text)
        assert value not in safe
        decoded = json.loads(safe)
        assert decoded[name] == "[REDACTED]"
        assert decoded["status"] == 503
        assert decoded["token_count"] == 17
        assert redact_secrets(safe) == safe

    @pytest.mark.parametrize("text", [
        'password = "FIRST MIDDLE LAST"',
        "client_secret='FIRST MIDDLE LAST'",
        '{"OPENAI_API_KEY":"FIRST MIDDLE LAST"}',
        "GEMINI_API_KEY=FIRST-MIDDLE-LAST",
        "Authorization: Token FIRST-MIDDLE-LAST",
        '{"Authorization":"Token FIRST MIDDLE LAST"}',
        "Authorization: AWS4-HMAC-SHA256 Credential=FIRST, Signature=LAST",
        'Authorization: Digest username="FIRST", response="LAST"',
        'OPENAI_API_KEY="FIRST MIDDLE LAST',
        "client_secret='FIRST MIDDLE LAST",
    ])
    def test_whole_quoted_or_authorization_value_is_masked(self, text: str) -> None:
        """A partial marker must not leave the remaining credential visible."""
        safe = redact_secrets(text)
        for word in ("FIRST", "MIDDLE", "LAST"):
            assert word not in safe
        assert "[REDACTED]" in safe
        assert redact_secrets(safe) == safe

    def test_aws_access_key_identifier_shape_is_masked(self) -> None:
        """Use the public documentation example, never a real account key."""
        assert "AKIAIOSFODNN7EXAMPLE" not in redact_secrets("credential AKIAIOSFODNN7EXAMPLE rejected")

    def test_basic_header_does_not_consume_next_diagnostic_line(self) -> None:
        """Invalid header folding must not hide the next line's cause."""
        safe = redact_secrets("Authorization: Basic\nordinary failure on the following line")
        assert "ordinary failure on the following line" in safe

    @pytest.mark.parametrize("text", [
        '{"token_count":17,"key_count":2,"keyboard":"letters","monkey":"animal"}',
        "Basic authentication failed: server returned 401 Unauthorized",
        "400 API_KEY_INVALID: API key not valid. Please pass a valid API key.",
        "ordinary failure; status=503; duration=42; input_tokens=120",
    ])
    def test_noncredential_fields_and_cause_are_preserved(self, text: str) -> None:
        """Field names describing counters are not credential assignments."""
        assert redact_secrets(text) == text

    def test_real_feed_error_and_traceback_mask_historical_fields(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """Exercise a caller boundary without relying on known env values."""
        engine = create_engine(f"sqlite:///{tmp_path / 'feed.db'}")
        Base.metadata.create_all(engine)
        canaries = ["HISTORIC-SYNTH-ACCESS", "HISTORIC-SYNTH-CLIENT", "HISTORIC-SYNTH-PROVIDER"]
        payload = "feed refused " + json.dumps(dict(zip(
            ["access_token", "client_secret", "OPENAI_API_KEY"], canaries,
        )))

        def fail_parse(url: str) -> None:
            raise RuntimeError(payload)

        monkeypatch.setattr("web.services.feed_sync.feedparser.parse", fail_parse)
        with Session(engine) as db:
            podcast = Podcast(source_rss_url="https://example.invalid/rss")
            db.add(podcast)
            db.commit()
            with pytest.raises(FeedSyncError) as captured:
                sync_podcast(db, podcast)
            formatted = "".join(traceback.format_exception(captured.value))
            for canary in canaries:
                assert canary not in formatted
            assert "feed refused" in formatted
            assert "[REDACTED]" in formatted
        engine.dispose()
