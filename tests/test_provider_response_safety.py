"""Provider-free regressions for semantic detection and review responses."""

import json
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from adnihilator.config import Config
from adnihilator.gemini_audio import GeminiAudioClient
from adnihilator.gemini_chunked import GeminiChunkedClient, detect_ads_in_chunk
from adnihilator.models import AdSpan, TranscriptSegment
from worker.daemon import DetectionSafetyError, WorkerDaemon


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any accidental network call instead of using paid providers."""
    def denied(*args: object, **kwargs: object) -> None:
        raise AssertionError("Network forbidden in provider response regressions")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def sdk_response(payload: object) -> Mock:
    """Create a mocked upload and model response without an SDK client."""
    sdk = Mock()
    sdk.files.upload.return_value = SimpleNamespace(name="files/synthetic", uri="synthetic", state="ACTIVE")
    sdk.models.generate_content.return_value = SimpleNamespace(text=json.dumps(payload), usage_metadata=None)
    return sdk


VALID_AD = {"start_time": 10, "end_time": 40, "sponsor": "Synthetic", "confidence": 0.9, "ad_type": "host_read"}
INVALID_GEMINI = [
    {}, {"status": "error"}, {"ads": {}}, {"ads": None}, [],
    {"ads": [{"sponsor": "Synthetic"}]}, {"ads": [None]},
    {"ads": [dict(VALID_AD, start_time=float("nan"), end_time=float("nan"))]},
    {"ads": [dict(VALID_AD, end_time=float("inf"))]},
    {"ads": [dict(VALID_AD, start_time=True)]},
    {"ads": [dict(VALID_AD, start_time="10")]},
    {"ads": [dict(VALID_AD, start_time=40, end_time=10)]},
    {"ads": [dict(VALID_AD, confidence=float("nan"))]},
    {"ads": [dict(VALID_AD, confidence=2)]},
    {"ads": [dict(VALID_AD, sponsor=" ")]},
]


class TestGeminiResponseSafety:
    @pytest.mark.parametrize("payload", INVALID_GEMINI)
    @pytest.mark.parametrize("mode", ["single", "chunk"])
    def test_invalid_response_cannot_establish_ad_coverage(self, payload: object, mode: str) -> None:
        """Invalid structure and non-finite values must fail instead of approving cuts."""
        sdk = sdk_response(payload)
        with pytest.raises(ValueError):
            if mode == "single":
                client = GeminiAudioClient("synthetic-key", model="gemini-2.5-flash")
                client._client = sdk
                client.detect_ads(Path("synthetic.mp3"), duration=300)
            else:
                detect_ads_in_chunk(sdk, Path("chunk.mp3"), 300, 600, "gemini-2.5-flash", max_retries=1)

    def test_whole_chunked_client_rejects_invalid_coverage(self, tmp_path: Path) -> None:
        """A syntactically valid empty object cannot bypass the whole-episode gate."""
        client = GeminiChunkedClient("synthetic-key", model="gemini-2.5-flash")
        client._client = sdk_response({})
        with patch("adnihilator.gemini_chunked.split_audio_into_chunks", return_value=[(tmp_path / "chunk.mp3", 0, 300)]):
            with pytest.raises(RuntimeError, match="refusing to return incomplete ad coverage"):
                client.detect_ads(tmp_path / "episode.mp3", duration=300)

    @pytest.mark.parametrize("mode", ["single", "chunk"])
    @pytest.mark.parametrize("payload", [{"ads": []}, {"ads": [VALID_AD]}])
    def test_explicit_valid_response_still_works(self, mode: str, payload: dict) -> None:
        """Both explicit no-ads coverage and a valid detection remain supported."""
        sdk = sdk_response(payload)
        if mode == "single":
            client = GeminiAudioClient("synthetic-key", model="gemini-2.5-flash")
            client._client = sdk
            ads, _ = client.detect_ads(Path("synthetic.mp3"), duration=300)
        else:
            ads = detect_ads_in_chunk(sdk, Path("chunk.mp3"), 300, 600, "gemini-2.5-flash", max_retries=1)
        assert len(ads) == len(payload["ads"])

    def test_existing_default_confidences_are_preserved(self) -> None:
        """Schema validation does not alter the existing two clients' defaults."""
        payload = {"ads": [{"start_time": 10, "end_time": 40, "sponsor": "Synthetic"}]}
        sdk = sdk_response(payload)
        single = GeminiAudioClient("synthetic-key", model="gemini-2.5-flash")
        single._client = sdk
        spans, _ = single.detect_ads(Path("synthetic.mp3"), duration=300)
        assert spans[0].confidence == 0.95
        ads = detect_ads_in_chunk(sdk, Path("chunk.mp3"), 0, 300, "gemini-2.5-flash", max_retries=1)
        assert ads[0]["confidence"] == 0.9


@pytest.fixture
def reviewer(monkeypatch: pytest.MonkeyPatch) -> WorkerDaemon:
    """Build a bare worker, without startup, with a synthetic provider setting."""
    monkeypatch.setenv("SAFETY_TEST_KEY", "synthetic-unit-key")
    worker = WorkerDaemon.__new__(WorkerDaemon)
    worker.config = Config()
    worker.config.llm.provider = "openai"
    worker.config.llm.api_key_env = "SAFETY_TEST_KEY"
    return worker


def review(worker: WorkerDaemon, content: str | None, spans: list[AdSpan] | None = None) -> list[AdSpan]:
    """Call the real review gate with only the OpenAI transport mocked."""
    sdk = Mock()
    sdk.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=None
    )
    spans = spans if spans is not None else [AdSpan(start=10, end=100, confidence=0.9, reason="Questionable", sources=["gemini"])]
    with patch("openai.OpenAI", return_value=sdk):
        result, _ = worker._final_llm_review(
            spans, [TranscriptSegment(index=0, start=0, end=200, text="Editorial discussion")],
            None, 200, {"anomaly": "review required"},
        )
    return result


class TestFinalReviewSafety:
    @pytest.mark.parametrize("content", [
        None, "", "{}", "[]", '{"concerns":["uncertain"]}',
        '{"approved_ads":[-1]}', '{"approved_ads":[1]}', '{"approved_ads":[true]}',
        '{"approved_ads":[0.0]}', '{"approved_ads":"0"}',
        '{"approved_ads":[0],"rejected_ads":[{"index":-1}]}',
        '{"approved_ads":[0],"merge_suggestions":[{"indices":[0,1]}]}',
    ])
    def test_invalid_review_cannot_authorize_cuts(self, reviewer: WorkerDaemon, content: str | None) -> None:
        """A missing or malformed approval cannot silently authorize audio removal."""
        with pytest.raises(DetectionSafetyError, match="Required final LLM review failed"):
            review(reviewer, content)

    @pytest.mark.parametrize("approvals,expected_count", [([], 0), ([0], 1)])
    def test_explicit_approval_or_rejection_is_honored(
        self, reviewer: WorkerDaemon, approvals: list[int], expected_count: int
    ) -> None:
        """A valid explicit empty approval remains a successful decision not to cut."""
        assert len(review(reviewer, json.dumps({"approved_ads": approvals}))) == expected_count

    def test_missing_confidence_is_not_reported_as_eighty_percent(
        self, reviewer: WorkerDaemon, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Logs cannot manufacture a confidence value the reviewer never returned."""
        review(reviewer, '{"approved_ads":[0]}')
        assert "80%" not in capsys.readouterr().out

    def test_explicit_transcript_evidence_keeps_existing_precedence(self, reviewer: WorkerDaemon) -> None:
        """Valid reviews preserve the existing explicit transcript evidence policy."""
        spans = [AdSpan(start=10, end=40, confidence=0.9, reason="Explicit", sources=["transcript_explicit"])]
        assert review(reviewer, '{"approved_ads":[0],"rejected_ads":[{"index":0}]}', spans) == spans

    def test_valid_merge_uses_original_input_indices(self, reviewer: WorkerDaemon) -> None:
        """Typed review data still supports a valid multi-span merge."""
        spans = [
            AdSpan(start=10, end=40, confidence=0.9, reason="First"),
            AdSpan(start=50, end=70, confidence=0.8, reason="Second"),
        ]
        result = review(reviewer, '{"approved_ads":[0,1],"merge_suggestions":[{"indices":[0,1]}]}', spans)
        assert [(span.start, span.end) for span in result] == [(10, 70)]

    def test_observed_json_mode_provider_error_still_fails_closed(self, reviewer: WorkerDaemon) -> None:
        """Replay the historical 400 without making an OpenAI request."""
        sdk = Mock()
        sdk.chat.completions.create.side_effect = ValueError(
            "Error code: 400 - 'messages' must contain the word 'json' in some form, to use 'response_format' of type 'json_object'."
        )
        with patch("openai.OpenAI", return_value=sdk):
            with pytest.raises(DetectionSafetyError):
                reviewer._final_llm_review(
                    [AdSpan(start=10, end=100, confidence=0.9, reason="Questionable")], [], None, 200, {}
                )
