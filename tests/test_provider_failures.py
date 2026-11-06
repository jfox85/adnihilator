"""Replay permanent provider failures and secret-bearing error boundaries."""

import logging
import socket
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from google.genai.errors import ClientError

from adnihilator.config import Config
from adnihilator.gemini_audio import GeminiAudioClient
from adnihilator.gemini_chunked import GeminiChunkedClient, detect_ads_in_chunk
from worker.client import WorkerClient
from worker.daemon import WorkerDaemon

PREPAY = "429 Your prepayment credits are depleted. Please go to AI Studio to manage your project and billing."
INVALID_KEY = "400 API_KEY_INVALID: API key not valid. https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta&key=FAKE-CANARY-unit-secret"
PERMANENT = [
    ValueError(PREPAY),
    ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Your prepayment credits are depleted."}}),
    ClientError(402, {"error": {"code": 402, "status": "RESOURCE_EXHAUSTED", "message": "Your prepayment credits are depleted."}}),
    ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "API key not valid.", "details": [{"reason": "API_KEY_INVALID"}]}}),
    ClientError(404, {"error": {"code": 404, "status": "NOT_FOUND", "message": "models/gemini-2.0-flash-exp is not found for API version v1beta"}}),
]


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider exception construction is offline; no network is permitted."""
    def denied(*args: object, **kwargs: object) -> None:
        raise AssertionError("Network forbidden in failure replay tests")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def sdk() -> Mock:
    """Return only a mocked upload/generate surface, without a real client."""
    client = Mock()
    client.files.upload.return_value = SimpleNamespace(name="files/synthetic", uri="synthetic", state="ACTIVE")
    client.models.generate_content.return_value = SimpleNamespace(text='{"ads":[]}', usage_metadata=None, candidates=[SimpleNamespace(finish_reason="STOP")])
    return client


class TestPermanentProviderFailures:
    @pytest.mark.parametrize("error", PERMANENT)
    @pytest.mark.parametrize("stage", ["upload", "generate"])
    def test_permanent_failure_never_uses_throttle_retries(self, error: Exception, stage: str) -> None:
        """Credit, auth and removed-model errors do not spend repeated requests."""
        client = sdk()
        target = client.files.upload if stage == "upload" else client.models.generate_content
        target.side_effect = error
        with patch("time.sleep") as sleep:
            with pytest.raises(Exception):
                detect_ads_in_chunk(client, Path("chunk.mp3"), 0, 300, "gemini-2.5-flash")
        assert target.call_count == 1
        sleep.assert_not_called()

    @pytest.mark.parametrize("error", PERMANENT)
    def test_global_failure_stops_queued_chunks_and_keeps_cause(self, error: Exception, tmp_path: Path) -> None:
        """Only the first request starts in a serial permanent-outage replay."""
        client = sdk()
        client.models.generate_content.side_effect = error
        detector = GeminiChunkedClient("synthetic-key", model="gemini-2.5-flash")
        detector._client = client
        chunks = [(tmp_path / f"chunk-{n}.mp3", n * 300, (n + 1) * 300) for n in range(8)]
        with patch("adnihilator.gemini_chunked.split_audio_into_chunks", return_value=chunks), patch("time.sleep"):
            with pytest.raises(RuntimeError, match="refusing to return incomplete ad coverage") as failure:
                detector.detect_ads(tmp_path / "episode.mp3", max_parallel=1)
        assert client.files.upload.call_count == 1
        assert client.models.generate_content.call_count == 1
        assert str(error) in str(failure.value)

    def test_ordinary_throttle_can_still_retry(self) -> None:
        """A normal 429 is distinct from a permanent 429 prepayment failure."""
        client = sdk()
        success = client.models.generate_content.return_value
        client.models.generate_content.side_effect = [ClientError(429, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "Rate limit exceeded; retry later"}}), success]
        with patch("time.sleep") as sleep:
            assert detect_ads_in_chunk(client, Path("chunk.mp3"), 0, 300, "gemini-2.5-flash") == []
        assert client.models.generate_content.call_count == 2
        sleep.assert_called_once_with(30)

    def test_owner_authorized_transient_retry_budget(self) -> None:
        """Owner D3 authorizes three total transient attempts, not more."""
        client = sdk()
        client.models.generate_content.side_effect = ValueError("503 The service is currently unavailable.")
        with patch("time.sleep") as sleep:
            with pytest.raises(Exception):
                detect_ads_in_chunk(client, Path("chunk.mp3"), 0, 300, "gemini-2.5-flash")
        assert client.models.generate_content.call_count == 3
        assert [call.args[0] for call in sleep.call_args_list] == [30, 60]

    def test_parallel_outage_only_allows_already_inflight_work(self, tmp_path: Path) -> None:
        """Queued chunks cannot start provider work after a global failure."""
        client = sdk()
        client.models.generate_content.side_effect = ValueError(PREPAY)
        detector = GeminiChunkedClient("synthetic-key", model="gemini-2.5-flash")
        detector._client = client
        chunks = [(tmp_path / f"chunk-{n}.mp3", n * 300, (n + 1) * 300) for n in range(20)]
        with patch("adnihilator.gemini_chunked.split_audio_into_chunks", return_value=chunks), patch("time.sleep"):
            with pytest.raises(RuntimeError):
                detector.detect_ads(tmp_path / "episode.mp3", max_parallel=2)
        assert client.files.upload.call_count <= 2
        assert client.models.generate_content.call_count <= 2

    def test_global_cause_takes_precedence_over_earlier_transient(self, tmp_path: Path) -> None:
        """A short-lived error cannot hide the actionable billing outage."""
        client = sdk()
        client.models.generate_content.side_effect = [ValueError("503 The service is currently unavailable."), ValueError(PREPAY)]
        detector = GeminiChunkedClient("synthetic-key", model="gemini-2.5-flash")
        detector._client = client
        chunks = [(tmp_path / f"chunk-{n}.mp3", n * 300, (n + 1) * 300) for n in range(2)]
        with (
            patch("adnihilator.gemini_chunked.split_audio_into_chunks", return_value=chunks),
            patch("concurrent.futures.as_completed", side_effect=lambda futures: list(futures)),
            patch("time.sleep"),
        ):
            with pytest.raises(RuntimeError) as failure:
                detector.detect_ads(tmp_path / "episode.mp3", max_parallel=1)
        assert "credits are depleted" in str(failure.value)

    def test_cancelled_chunk_makes_no_upload(self) -> None:
        """Cancellation is checked before entering the SDK."""
        client = sdk()
        cancelled = Event()
        cancelled.set()
        with pytest.raises(CancelledError):
            detect_ads_in_chunk(client, Path("chunk.mp3"), 0, 300, "gemini-2.5-flash", cancel_event=cancelled)
        client.files.upload.assert_not_called()


class TestErrorRedaction:
    @pytest.mark.parametrize("message,canary", [
        (INVALID_KEY, "FAKE-CANARY-unit-secret"),
        ("Authorization: Bearer FAKE-BEARER-unit-secret", "FAKE-BEARER-unit-secret"),
        ('{"x-goog-api-key":"FAKE-HEADER-unit-secret"}', "FAKE-HEADER-unit-secret"),
        ("request failed with sk-unit_secret_canary", "sk-unit_secret_canary"),
        ("request failed with AIzaUNIT_SECRET_CANARY", "AIzaUNIT_SECRET_CANARY"),
    ])
    def test_worker_redacts_before_output_and_persistence(
        self, message: str, canary: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Known secret shapes are removed even when absent from the environment."""
        worker = WorkerDaemon.__new__(WorkerDaemon)
        worker.api_client = Mock()
        worker.api_client.claim.return_value = SimpleNamespace(id="synthetic")
        worker.process_job = Mock(side_effect=ValueError(message))
        assert worker.run_once()
        logged = capsys.readouterr().out
        persisted = worker.api_client.fail.call_args.args[1]
        assert canary not in logged and canary not in persisted
        assert "[REDACTED]" in persisted
        if message == INVALID_KEY:
            assert "400" in persisted and "API_KEY_INVALID" in persisted and "API key not valid" in persisted

    def test_custom_configured_key_is_redacted(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Nonstandard provider environment names still get exact-value redaction."""
        canary = "synthetic-opaque-credential"
        monkeypatch.setenv("CUSTOM_PROVIDER_CREDENTIAL", canary)
        worker = WorkerDaemon.__new__(WorkerDaemon)
        worker.config = Config()
        worker.config.llm.api_key_env = "CUSTOM_PROVIDER_CREDENTIAL"
        worker.api_client = Mock()
        worker.api_client.claim.return_value = SimpleNamespace(id="synthetic")
        worker.process_job = Mock(side_effect=ValueError(f"opaque failure {canary}"))
        worker.run_once()
        assert canary not in capsys.readouterr().out
        assert canary not in worker.api_client.fail.call_args.args[1]

    def test_single_shot_error_does_not_expose_query_key(self) -> None:
        """The direct client error is safe even before a worker catches it."""
        client = sdk()
        client.files.upload.side_effect = ValueError(INVALID_KEY)
        detector = GeminiAudioClient("synthetic-key", model="gemini-2.5-flash")
        detector._client = client
        from adnihilator.provider_errors import ProviderOutageError
        with pytest.raises(ProviderOutageError) as error:
            detector.detect_ads(Path("synthetic.mp3"))
        assert "FAKE-CANARY-unit-secret" not in str(error.value)
        assert "API_KEY_INVALID" in str(error.value)
        assert error.value.__cause__ is None and error.value.__suppress_context__

    def test_chunk_logs_and_aggregated_error_are_safe(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Keeping the cause must not reintroduce the historical log exposure."""
        client = sdk()
        client.models.generate_content.side_effect = ValueError(INVALID_KEY)
        detector = GeminiChunkedClient("synthetic-key", model="gemini-2.5-flash")
        detector._client = client
        with patch("adnihilator.gemini_chunked.split_audio_into_chunks", return_value=[(tmp_path / "chunk.mp3", 0, 300)]):
            with pytest.raises(RuntimeError) as error:
                detector.detect_ads(tmp_path / "episode.mp3")
        assert "FAKE-CANARY-unit-secret" not in caplog.text
        assert "FAKE-CANARY-unit-secret" not in str(error.value)
        assert "API_KEY_INVALID" in str(error.value)

    def test_api_client_redacts_direct_failure_calls(self, caplog: pytest.LogCaptureFixture) -> None:
        """The shared persistence boundary protects callers other than run_once."""
        client = WorkerClient.__new__(WorkerClient)
        client.api_url = "https://example.invalid"
        client.api_key = "synthetic-worker-key"
        client.client = Mock()
        caplog.set_level(logging.INFO)
        client.fail("episode", INVALID_KEY)
        assert "FAKE-CANARY-unit-secret" not in client.client.post.call_args.kwargs["json"]["error"]
        assert "FAKE-CANARY-unit-secret" not in caplog.text

    def test_claim_exception_trace_does_not_reintroduce_secret(self, caplog: pytest.LogCaptureFixture) -> None:
        """Logging a sanitized message must not append the raw exception again."""
        client = WorkerClient.__new__(WorkerClient)
        client.api_url = "https://example.invalid"
        client.api_key = "synthetic-worker-key"
        client.client = Mock()
        client.client.post.side_effect = RuntimeError(INVALID_KEY)
        assert client.claim() is None
        assert "FAKE-CANARY-unit-secret" not in caplog.text
