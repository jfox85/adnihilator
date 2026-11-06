"""Historical camelCase credential names retain the same diagnostic protection."""

import pytest

from adnihilator.redaction import redact_secrets


class TestCamelCaseCredentials:
    """Credential values must remain private across common SDK field spellings."""

    @pytest.mark.parametrize("name", [
        "accessKey", "secretKey", "clientSecret", "privateKey", "sessionToken",
        "idToken", "accessToken", "refreshToken", "apiKey", "APIKey",
        "customClientSecret", "r2AccessKey", "awsSecretAccessKey",
        "pASSWORD", "PaSsWoRd", "tOKEN", "sECRET", "cREDENTIAL",
        "aUTHORIZATION", "AuThOrIzAtIoN", "PREFIX_pASSWORD",
        "PrEfIx-PaSsWoRd", "aPi_KeY", "x_CrEdEnTiAl",
    ])
    @pytest.mark.parametrize("wrapper", [
        "{name}={secret}",
        '{{"{name}":"{secret}"}}',
        'stderr="{name}={secret}"',
        "Error: --{name}='{secret} extra'",
    ])
    def test_camel_case_secret_is_masked(self, name: str, wrapper: str) -> None:
        """Mask the complete value without depending on configured credentials."""
        secret = "CAMEL-SYNTHETIC-HISTORICAL-CREDENTIAL"
        safe = redact_secrets(wrapper.format(name=name, secret=secret))
        assert secret not in safe
        assert "[REDACTED]" in safe
        assert redact_secrets(safe) == safe

    @pytest.mark.parametrize("message", [
        "retryCount=2 statusCode=503 keyboard=letters",
        'detail="keyCount=12 tokenCount=40 secretCount=0"',
        "accessKeyCount=3 secretKeyCount=1 sessionTokenCount=9",
    ])
    def test_counters_and_ordinary_fields_are_preserved(self, message: str) -> None:
        """A credential word inside a counter name does not make it a secret."""
        assert redact_secrets(message) == message
