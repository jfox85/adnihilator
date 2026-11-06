"""Credential diagnostics retain no URI userinfo or Basic authorization value."""

import pytest

from adnihilator.redaction import redact_secrets


class TestResidualCredentialRedaction:
    @pytest.mark.parametrize("message,canary", [
        ("fetch HTTPS://user:UPPER-SYNTH-CREDENTIAL@example.invalid/feed", "UPPER-SYNTH-CREDENTIAL"),
        ("fetch Http://user:MIXED-SYNTH-CREDENTIAL@example.invalid/feed", "MIXED-SYNTH-CREDENTIAL"),
        ("fetch hTtPs://user:CASE-SYNTH-CREDENTIAL@example.invalid/feed", "CASE-SYNTH-CREDENTIAL"),
        ("fetch https://USERONLY-SYNTH-CREDENTIAL@example.invalid/feed", "USERONLY-SYNTH-CREDENTIAL"),
        ("fetch https://EMPTY-PASSWORD-SYNTH:@example.invalid/feed", "EMPTY-PASSWORD-SYNTH"),
        ("fetch https://ENCODED%40USER-SYNTH@example.invalid/feed", "ENCODED%40USER-SYNTH"),
        ("Authorization: Basic dXNlcjpTWU5USC1CQVNJQy1TRUNSRVQ=", "dXNlcjpTWU5USC1CQVNJQy1TRUNSRVQ="),
        ("authorization=basic dXNlcjpMT1dFUi1CQVNJQy1TRUNSRVQ=", "dXNlcjpMT1dFUi1CQVNJQy1TRUNSRVQ="),
        ('{"Authorization":"Basic dXNlcjpKU09OLUJBU0lDLVNFQ1JFVA=="}', "dXNlcjpKU09OLUJBU0lDLVNFQ1JFVA=="),
    ])
    def test_userinfo_and_basic_credentials_are_removed(self, message: str, canary: str) -> None:
        """Case and absent URI passwords cannot bypass the diagnostic boundary."""
        safe = redact_secrets(message)
        assert canary not in safe
        assert "[REDACTED]" in safe
        assert redact_secrets(safe) == safe

    @pytest.mark.parametrize("message", [
        "Basic authentication failed: server returned 401 Unauthorized",
        "fetch HTTPS://example.invalid/path?contact=person@example.invalid failed",
    ])
    def test_ordinary_diagnostics_and_url_query_text_are_preserved(self, message: str) -> None:
        """Do not mistake authentication terminology or URL query email for secrets."""
        assert redact_secrets(message) == message
