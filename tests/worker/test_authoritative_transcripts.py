"""Required transcript decisions cannot silently publish missing coverage."""

import json
import socket
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from adnihilator.ad_keywords import find_ad_candidates
from adnihilator.ad_llm import OpenAIClient
from adnihilator.config import Config
from adnihilator.gemini_audio import GeminiAudioClient
from adnihilator.models import AdSpan, TranscriptSegment
from tests.test_provider_completion_safety import google_response, google_sdk, openai_response
from worker.client import EpisodeJob
from worker.daemon import DetectionSafetyError, WorkerDaemon


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Actual SDK objects use only in-process mocked transports."""
    def denied(*args: object, **kwargs: object) -> None:
        raise AssertionError("Network forbidden in authoritative transcript regressions")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket.socket, "bind", denied)
    monkeypatch.setattr(socket.socket, "sendto", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


def transcript(ad: bool = True) -> list[TranscriptSegment]:
    """One real bounded read gives the refiner one legitimate line index."""
    return [TranscriptSegment(index=0, start=10, end=50, text=(
        "Thanks to our sponsor Squarespace. Visit squarespace.com and use promo code SAVE for twenty percent off."
        if ad else "Astronomers explain the motion of distant stars and galaxies."
    ))]


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> WorkerDaemon:
    """Build a bare worker, never instantiate a daemon or external clients."""
    monkeypatch.setenv("AUTHORITATIVE_CRED", "synthetic-provider")
    result = WorkerDaemon.__new__(WorkerDaemon)
    result.config = Config()
    result.config.llm.provider = "openai"
    result.config.llm.api_key_env = "AUTHORITATIVE_CRED"
    result.api_client = Mock()
    result.r2_client = Mock()
    result.r2_client.get_file_size.return_value = 5
    result._download_audio = Mock()
    result._get_gemini_client = Mock(return_value=None)
    result._get_openai_client = Mock(return_value=None)
    result.artifacts_dir = None
    result.whisper_model = "synthetic"
    result.device = "cpu"
    result.engine = "whisper"
    return result


def job(route: str = "whisper", description: str | None = None) -> EpisodeJob:
    """A synthetic source can never download outside the mocked boundary."""
    return EpisodeJob(id="episode", podcast_id="podcast", guid="synthetic", title="Synthetic",
                      original_audio_url="https://example.invalid/audio.mp3",
                      source_url="https://example.invalid/episode" if route == "external" else None,
                      description=description)


def fake_splice(input_path: str, output_path: str, spans: list[AdSpan], duration: float, **kwargs: object) -> dict:
    """Only a scratch placeholder is uploaded by the mocked storage client."""
    Path(output_path).write_bytes(b"clean")
    removed = sum(span.end - span.start for span in spans if span.confidence >= 0.35)
    return {"time_removed": removed, "new_duration": duration - removed}


@contextmanager
def pipeline(worker: WorkerDaemon, route: str, segments: list[TranscriptSegment]) -> Iterator[Mock]:
    """Use the actual orchestration, heuristic detector, two-pass and refiner."""
    worker.engine = "parakeet" if route == "parakeet" else "whisper"
    with (
        patch("worker.daemon.get_duration", return_value=120),
        patch("worker.daemon.fetch_external_transcript", return_value=segments if route == "external" else None),
        patch("worker.daemon.transcribe_audio", return_value=segments),
        patch("worker.daemon.transcribe_audio_parakeet", return_value=segments),
        patch("adnihilator.two_pass.transcribe_audio", return_value=segments),
        patch("adnihilator.two_pass.transcribe_audio_parakeet", return_value=segments),
        patch("adnihilator.two_pass.extract_audio_regions", return_value=[]),
        patch("worker.daemon.splice_audio", side_effect=fake_splice) as splice,
    ):
        yield splice


INVALID_DECISIONS = [
    ("{}", "stop", None), ('{"ads":[', "stop", None),
    ('{"ads":[]} trailing unfinished {', "stop", None),
    ('{"ads":[]}', "length", None),
    ('{"ads":[{"start_line":0,"end_line":0,"confidence":0.95}]}', "length", None),
    ('{"ads":[]}', None, None), ('{"ads":[]}', "stop", "Request refused"),
]


class TestRequiredTranscriptDecisions:
    """A malformed or unfinished decision cannot become cuts or no ads."""

    @pytest.mark.parametrize("route", ["external", "whisper", "parakeet"])
    @pytest.mark.parametrize("content,finish,refusal", INVALID_DECISIONS)
    def test_invalid_refinement_stops_real_publication_path(
        self, worker: WorkerDaemon, route: str, content: str, finish: str | None, refusal: str | None,
    ) -> None:
        """Actual SDK responses traverse the real refiner and worker choice."""
        with openai_response(content, finish, refusal) as sdk, patch("openai.OpenAI", return_value=sdk), pipeline(worker, route, transcript()) as splice:
            with pytest.raises((ValueError, DetectionSafetyError)):
                worker.process_job(job(route))
        splice.assert_not_called()
        worker.r2_client.upload_file.assert_not_called()
        worker.api_client.complete.assert_not_called()

    @pytest.mark.parametrize("route", ["external", "whisper", "parakeet"])
    @pytest.mark.parametrize("ads,expected", [([], 0), ([{"start_line": 0, "end_line": 0, "confidence": 0.95}], 40)])
    def test_completed_empty_and_valid_line_decisions_remain_valid(
        self, worker: WorkerDaemon, route: str, ads: list[dict], expected: int,
    ) -> None:
        """Explicit no-ad coverage and actual bounded ad decisions both work."""
        with openai_response(json.dumps({"ads": ads})) as sdk, patch("openai.OpenAI", return_value=sdk), pipeline(worker, route, transcript()):
            worker.process_job(job(route))
        assert worker.api_client.complete.call_args.kwargs["ads_removed"] == expected

    @pytest.mark.parametrize("record", [
        {}, {"start_line": 0}, {"start_line": 0, "end_line": 1},
        {"start_line": -1, "end_line": 0}, {"start_line": True, "end_line": 0},
        {"start_line": 0.0, "end_line": 0}, {"start_line": 1, "end_line": 0},
        {"start_line": 0, "end_line": 0, "confidence": float("nan")},
        {"ad_start": 10, "ad_end": float("inf")}, {"ad_start": 50, "ad_end": 10},
        {"ad_start": 130, "ad_end": 150},
    ])
    def test_invalid_record_cannot_be_skipped_or_heuristically_approved(self, record: dict) -> None:
        """Every nonempty record is validated, including direct-time compatibility."""
        config = Config()
        candidates = find_ad_candidates(transcript(), 120)
        assert candidates
        with openai_response(json.dumps({"ads": [record]})) as sdk, patch("openai.OpenAI", return_value=sdk):
            with pytest.raises(ValueError):
                OpenAIClient("synthetic-provider").refine_candidates(transcript(), candidates, config)

    @pytest.mark.parametrize("route", ["external", "whisper", "parakeet"])
    def test_required_worker_api_error_cannot_use_optional_heuristic_fallback(self, worker: WorkerDaemon, route: str) -> None:
        """Required worker refinement differs from explicitly optional CLI fallback."""
        sdk = Mock()
        sdk.chat.completions.create.side_effect = RuntimeError("simulated provider unavailable")
        with patch("openai.OpenAI", return_value=sdk), pipeline(worker, route, transcript()) as splice:
            with pytest.raises((ValueError, DetectionSafetyError)):
                worker.process_job(job(route))
        splice.assert_not_called()
        worker.api_client.complete.assert_not_called()

    def test_optional_client_api_error_retains_existing_fallback(self) -> None:
        """The non-required client keeps its documented API-error heuristic mode."""
        sdk = Mock()
        sdk.chat.completions.create.side_effect = RuntimeError("simulated provider unavailable")
        config = Config()
        with patch("openai.OpenAI", return_value=sdk):
            spans = OpenAIClient("synthetic-provider").refine_candidates(
                transcript(), find_ad_candidates(transcript(), 120), config,
            )
        assert spans and "heuristic_fallback" in spans[0].reason


class TestMissingRequiredTranscripts:
    """Required text lanes reject missing evidence without breaking fast paths."""

    @pytest.mark.parametrize("route", ["whisper", "parakeet"])
    @pytest.mark.parametrize("segments", [[], [TranscriptSegment(index=0, start=0, end=120, text="   ")]])
    def test_empty_local_two_pass_never_publishes(self, worker: WorkerDaemon, route: str, segments: list) -> None:
        """Empty arrays and empty placeholder text cannot establish coverage."""
        with pipeline(worker, route, segments) as splice:
            with pytest.raises(DetectionSafetyError, match="transcript"):
                worker.process_job(job(route))
        splice.assert_not_called()
        worker.r2_client.upload_file.assert_not_called()
        worker.api_client.complete.assert_not_called()

    @pytest.mark.parametrize("route", ["whisper", "parakeet"])
    @pytest.mark.parametrize("gemini_ad", [False, True])
    @pytest.mark.parametrize("segments", [[], [TranscriptSegment(index=0, start=0, end=120, text="")]])
    def test_empty_parallel_text_lane_never_publishes(self, worker: WorkerDaemon, route: str, gemini_ad: bool, segments: list) -> None:
        """A valid Gemini sibling cannot substitute for the required text lane."""
        worker.config.llm.provider = "none"
        worker.config.gemini.enabled = True
        worker.config.detection.parallel_enabled = True
        detector = GeminiAudioClient("synthetic-provider", model="gemini-2.5-flash")
        detector._client = google_sdk(google_response({"ads": [{"start_time": 10, "end_time": 50, "sponsor": "Synthetic", "ad_type": "host_read"}] if gemini_ad else []}))
        worker._get_gemini_client.return_value = detector
        with pipeline(worker, route, segments) as splice:
            with pytest.raises(DetectionSafetyError, match="transcript"):
                worker.process_job(job(route))
        splice.assert_not_called()
        worker.r2_client.upload_file.assert_not_called()
        worker.api_client.complete.assert_not_called()

    @pytest.mark.parametrize("route", ["whisper", "parakeet"])
    def test_nonempty_ad_free_two_pass_is_still_valid(self, worker: WorkerDaemon, route: str) -> None:
        """No keyword candidates in a real transcript is legitimate no-ad evidence."""
        with pipeline(worker, route, transcript(ad=False)):
            worker.process_job(job(route))
        assert worker.api_client.complete.call_args.kwargs["ads_removed"] == 0

    def test_description_timestamp_fast_path_does_not_require_asr(self, worker: WorkerDaemon) -> None:
        """An explicitly accepted timestamp path can legitimately have no transcript."""
        stamp = SimpleNamespace(start=10, end=50, confidence=0.95, label="Sponsor", extraction_method="synthetic")
        with pipeline(worker, "whisper", []) as splice, patch("worker.daemon.extract_ad_timestamps", return_value=[stamp]):
            worker.process_job(job())
        splice.assert_called_once()
        assert worker.api_client.complete.call_args.kwargs["detection_source"] == "timestamps"

    @pytest.mark.parametrize("missing", [[], [TranscriptSegment(index=0, start=0, end=120, text=" ")]])
    def test_empty_optional_external_lookup_uses_real_local_fallback(self, worker: WorkerDaemon, missing: list) -> None:
        """An optional lookup without text is not a failed required ASR lane."""
        with pipeline(worker, "whisper", transcript(ad=False)), patch("worker.daemon.fetch_external_transcript", return_value=missing):
            worker.process_job(job("external"))
        assert worker.api_client.complete.call_args.kwargs["ads_removed"] == 0

    @pytest.mark.parametrize("route", ["whisper", "parakeet"])
    def test_nonempty_parallel_text_and_completed_empty_audio_are_valid(self, worker: WorkerDaemon, route: str) -> None:
        """Both required lanes cover a legitimate ad-free episode."""
        worker.config.llm.provider = "none"
        worker.config.gemini.enabled = True
        worker.config.detection.parallel_enabled = True
        detector = GeminiAudioClient("synthetic-provider", model="gemini-2.5-flash")
        detector._client = google_sdk(google_response({"ads": []}))
        worker._get_gemini_client.return_value = detector
        with pipeline(worker, route, transcript(ad=False)):
            worker.process_job(job(route))
        assert worker.api_client.complete.call_args.kwargs["ads_removed"] == 0


class TestOptionalTranscriptNeighbors:
    """Optional decisions retain their original conservative fallback behavior."""

    @pytest.mark.parametrize("finish,expected", [("stop", [(10, 50)]), ("length", [])])
    def test_optional_hunt_requires_completed_decision(self, worker: WorkerDaemon, finish: str, expected: list) -> None:
        """Unfinished optional hunt output cannot add cuts; STOP stays valid."""
        body = json.dumps({"found": True, "ad_start": 10, "ad_end": 50, "confidence": 0.9})
        with openai_response(body, finish) as sdk, patch("openai.OpenAI", return_value=sdk):
            spans, _ = worker._llm_hunt_in_window(transcript(), "Synthetic", 0, 120, 120)
        assert [(span.start, span.end) for span in spans] == expected

    @pytest.mark.parametrize("finish,ranges,expected", [
        ("stop", [{"start": 10, "end": 50}], [(10, 50)]),
        ("length", [{"start": 10, "end": 50}], [(0, 900)]),
        ("stop", [{"start": float("nan"), "end": 50}], [(0, 900)]),
        ("stop", [{"start": 950, "end": 1000}], [(0, 900)]),
    ])
    def test_optional_mega_refine_keeps_original_on_invalid_decision(
        self, worker: WorkerDaemon, finish: str, ranges: list, expected: list,
    ) -> None:
        """Optional malformed refinement retains all previously detected coverage."""
        original = AdSpan(start=0, end=900, confidence=0.9, reason="existing")
        with openai_response(json.dumps({"ad_ranges": ranges}), finish) as sdk, patch("openai.OpenAI", return_value=sdk):
            spans, _ = worker._refine_mega_spans([original], transcript(), None)
        assert [(span.start, span.end) for span in spans] == expected

    @pytest.mark.parametrize("record", [
        {"ad_start": 0, "ad_end": 60},
        {"start_line": 0, "end_line": 0},
    ])
    def test_valid_legacy_time_clamps_and_line_decisions_agree(self, record: dict) -> None:
        """Legacy finite overlapping timestamps remain compatible with line mode."""
        with openai_response(json.dumps({"ads": [record]})) as sdk, patch("openai.OpenAI", return_value=sdk):
            spans = OpenAIClient("synthetic-provider").refine_candidates(
                transcript(), find_ad_candidates(transcript(), 120), Config(),
            )
        assert [(span.start, span.end) for span in spans] == [(10, 50)]


    @pytest.mark.parametrize("content", ['{"ads":null}', '{"ads":{}}', '{"ads":[{}, {"start_line":0,"end_line":0}]}'])
    def test_invalid_container_or_mixed_record_is_not_partial_success(self, content: str) -> None:
        """Any invalid record invalidates the complete required decision."""
        with openai_response(content) as sdk, patch("openai.OpenAI", return_value=sdk):
            with pytest.raises(ValueError):
                OpenAIClient("synthetic-provider").refine_candidates(
                    transcript(), find_ad_candidates(transcript(), 120), Config(),
                )

    def test_complete_json_fence_retains_compatibility(self) -> None:
        """A complete conventional fence still contains one authoritative object."""
        with openai_response('```json\n{"ads": []}\n```') as sdk, patch("openai.OpenAI", return_value=sdk):
            spans = OpenAIClient("synthetic-provider").refine_candidates(
                transcript(), find_ad_candidates(transcript(), 120), Config(),
            )
        assert spans == []

    def test_validation_error_redacts_actual_opaque_key_and_trace(self) -> None:
        """New schema errors cannot echo an opaque key through Pydantic input."""
        canary = "opaqueSyntheticCredentialForRefinerOnly"
        body = json.dumps({"ads": [{"start_line": 0, "end_line": 0, "confidence": canary}]})
        with openai_response(body) as sdk, patch("openai.OpenAI", return_value=sdk):
            with pytest.raises(ValueError) as caught:
                OpenAIClient(canary).refine_candidates(
                    transcript(), find_ad_candidates(transcript(), 120), Config(),
                )
        assert canary not in str(caught.value)
        assert canary not in "".join(traceback.format_exception(caught.value))
