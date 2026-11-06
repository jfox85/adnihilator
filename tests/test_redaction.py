"""Diagnostic redaction is provider-independent and preserves ordinary wording."""

import pytest

from adnihilator.redaction import redact_secrets


class TestDiagnosticRedaction:
    @pytest.mark.parametrize("message,canary", [
        ("https://example.invalid/?key=FAKE-QUERY-secret&alt=json", "FAKE-QUERY-secret"),
        ("https://example.invalid/?X-Amz-Signature=FAKE-SIGNATURE-secret", "FAKE-SIGNATURE-secret"),
        ("https://user:FAKE-USERINFO-secret@example.invalid/audio", "FAKE-USERINFO-secret"),
        ("Authorization: Bearer FAKE-BEARER-secret", "FAKE-BEARER-secret"),
        ('{"api_key":"FAKE-JSON-secret"}', "FAKE-JSON-secret"),
        ("key=FAKE-PLAIN-secret", "FAKE-PLAIN-secret"),
        ("provider failed with sk-unit_secret_canary", "sk-unit_secret_canary"),
        ("provider failed with AIzaUNIT_SECRET_CANARY", "AIzaUNIT_SECRET_CANARY"),
    ])
    def test_secret_shapes_are_masked(self, message: str, canary: str) -> None:
        """Assignments, signed URLs and key patterns do not survive diagnostics."""
        safe = redact_secrets(message)
        assert canary not in safe
        assert "[REDACTED]" in safe
        assert redact_secrets(safe) == safe

    def test_known_opaque_values_are_removed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment values and explicit configured values need no prefix."""
        monkeypatch.setenv("GEMINI_API_KEY", "opaque-unit-environment")
        safe = redact_secrets("opaque-unit-environment and opaque-unit-extra", ("opaque-unit-extra",))
        assert "opaque-unit" not in safe

    def test_short_known_values_are_masked_as_tokens(self) -> None:
        """Unusually short credentials still disappear as delimited tokens."""
        assert "tiny" not in redact_secrets("prefix-tiny-suffix", ("tiny",))

    def test_safe_error_reason_is_preserved(self) -> None:
        """Do not mistake the phrase 'API key not valid' for a key assignment."""
        message = "400 API_KEY_INVALID: API key not valid. Please pass a valid API key."
        assert redact_secrets(message) == message
