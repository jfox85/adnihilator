"""Ordinary diagnostic labels must not hide credential assignments."""

import pytest

from adnihilator.redaction import redact_secrets


class TestWrappedDiagnosticCredentials:
    """Exercise the same fields in nested, labelled and command-line contexts."""

    @pytest.mark.parametrize("name", ["password", "api_key", "token", "OPENAI_API_KEY", "client_secret", "access_token"])
    @pytest.mark.parametrize("wrapper", [
        "Error: {name}={secret}",
        'stderr="{name}={secret}"',
        '{{"detail": "{name}={secret}"}}',
        "error={name}={secret}",
        "--{name}={secret}",
    ])
    def test_labels_do_not_suppress_redaction(self, name: str, wrapper: str) -> None:
        """Outer noncredential names cannot consume the inner secret."""
        secret = "WRAPPED-SYNTHETIC-OPAQUE-CREDENTIAL"
        text = wrapper.format(name=name, secret=secret)
        safe = redact_secrets(text)
        assert secret not in safe
        assert "[REDACTED]" in safe
        assert redact_secrets(safe) == safe

    @pytest.mark.parametrize("text", [
        "Authorization: SYNTHRAWKEY rejected by upstream",
        '{"Authorization":"SYNTHRAWKEY extra"}',
        "Authorization: SYNTHRAWKEY ",
        "Error: Authorization: SYNTHRAWKEY rejected",
    ])
    def test_unknown_first_word_is_not_treated_as_auth_scheme(self, text: str) -> None:
        """A raw key is secret even if followed by diagnostic text."""
        safe = redact_secrets(text)
        assert "SYNTHRAWKEY" not in safe
        assert redact_secrets(safe) == safe

    def test_nested_secret_values_are_masked_once(self) -> None:
        """An already-protected quoted span does not need recursive parsing."""
        text = 'password="password=' * 2000 + 'DEEP-SYNTH-CREDENTIAL' + '"'
        assert "DEEP-SYNTH-CREDENTIAL" not in redact_secrets(text)

    def test_nonsecret_labels_preserve_ordinary_fields(self) -> None:
        """Keep useful error context when no recognized credential exists."""
        text = 'Error: detail="status=503 retry_count=2 keyboard=letters"'
        assert redact_secrets(text) == text
