"""Offline regressions for complete required provider decisions and diagnostics."""

import json
import socket
import traceback
from concurrent.futures import CancelledError
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from google.genai import types
from openai import OpenAI as SDKOpenAI
import pytest

from adnihilator.ad_llm import OpenAIClient
from adnihilator.config import Config
from adnihilator.gemini_audio import GeminiAudioClient
from adnihilator.gemini_chunked import GeminiChunkedClient, detect_ads_in_chunk
from adnihilator.models import AdSpan, TranscriptSegment
from worker.client import EpisodeJob
from worker.daemon import DetectionSafetyError, WorkerDaemon


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """SDK models and MockTransport work without any real network."""
    def denied(*args: object, **kwargs: object) -> None:
        raise AssertionError("Network forbidden in completion safety tests")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def openai_response(content: str, finish: str | None = "stop", refusal: str | None = None) -> SDKOpenAI:
    """Parse actual OpenAI ChatCompletion objects through an offline transport."""
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "provider.invalid"
        return httpx.Response(200, json={
            "id": "synthetic", "object": "chat.completion", "created": 0, "model": "synthetic",
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content, "refusal": refusal}}],
        })

    return SDKOpenAI(api_key="synthetic-provider", base_url="https://provider.invalid/v1",
                     http_client=httpx.Client(transport=httpx.MockTransport(respond)))


def google_response(
    payload: object, finish: types.FinishReason | None = types.FinishReason.STOP,
    block: types.BlockedReason | None = None, safety_blocked: bool = False,
) -> types.GenerateContentResponse:
    """Use actual google-genai candidate, finish and feedback models."""
    candidate = types.Candidate(
        content=types.Content(role="model", parts=[types.Part.from_text(text=json.dumps(payload))]),
        finish_reason=finish,
        safety_ratings=[types.SafetyRating(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                          probability=types.HarmProbability.HIGH, blocked=True)] if safety_blocked else None,
    )
    return types.GenerateContentResponse(
        candidates=[candidate],
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=block) if block else None,
    )


def google_sdk(response: types.GenerateContentResponse) -> Mock:
    """Replace only Google's transport surface, keeping actual response models."""
    client = Mock()
    client.files.upload.return_value = types.File(name="files/synthetic", uri="synthetic", state=types.FileState.ACTIVE)
    client.models.generate_content.return_value = response
    return client


def google_detect(response: types.GenerateContentResponse, mode: str) -> list[AdSpan] | list[dict]:
    """Exercise either required Gemini decision path without reading audio."""
    sdk = google_sdk(response)
    if mode == "chunk":
        return detect_ads_in_chunk(sdk, Path("synthetic.mp3"), 0, 120, "gemini-2.5-flash", max_retries=1)
    client = GeminiAudioClient("synthetic-provider", model="gemini-2.5-flash")
    client._client = sdk
    return client.detect_ads(Path("synthetic.mp3"), duration=120)[0]


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> WorkerDaemon:
    """A bare worker object, with a fake key under a non-key-shaped env name."""
    monkeypatch.setenv("CREDENTIAL", "opaque-provider-value-8675309")
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    daemon.config = Config()
    daemon.config.llm.provider = "openai"
    daemon.config.llm.api_key_env = "CREDENTIAL"
    return daemon


def worker_review(worker: WorkerDaemon) -> list[AdSpan]:
    """Invoke the required final approval gate with one proposed cut."""
    return worker._final_llm_review(
        [AdSpan(start=10, end=40, confidence=0.9, reason="Sponsor")],
        [TranscriptSegment(index=0, start=0, end=120, text="Sponsor advertisement")], None, 120, {},
    )[0]


class TestCompleteProviderDecisions:
    """A parseable JSON fragment does not establish a complete decision."""

    @pytest.mark.parametrize("finish,refusal", [
        ("length", None), ("content_filter", None), ("tool_calls", None),
        ("function_call", None), (None, None), ("stop", "Cannot provide this response"),
    ])
    @pytest.mark.parametrize("stage", ["merge", "review"])
    def test_openai_incomplete_or_refused_decision_fails_closed(
        self, finish: str | None, refusal: str | None, stage: str, worker: WorkerDaemon,
    ) -> None:
        """Actual parsed SDK metadata takes precedence over valid JSON."""
        content = '{"ads":[]}' if stage == "merge" else '{"approved_ads":[0]}'
        with openai_response(content, finish, refusal) as sdk, patch("openai.OpenAI", return_value=sdk):
            with pytest.raises((ValueError, DetectionSafetyError)):
                if stage == "merge":
                    OpenAIClient("synthetic-provider").merge_and_refine({"duration": 120}, Config())
                else:
                    worker_review(worker)

    @pytest.mark.parametrize("stage", ["merge", "review"])
    def test_openai_explicit_empty_completed_decision_remains_valid(self, stage: str, worker: WorkerDaemon) -> None:
        """A STOP response can explicitly decide to make no cuts."""
        content = '{"ads":[]}' if stage == "merge" else '{"approved_ads":[]}'
        with openai_response(content) as sdk, patch("openai.OpenAI", return_value=sdk):
            if stage == "merge":
                assert OpenAIClient("synthetic-provider").merge_and_refine({"duration": 120}, Config())[0] == []
            else:
                assert worker_review(worker) == []

    @pytest.mark.parametrize("finish", [types.FinishReason.MAX_TOKENS, types.FinishReason.SAFETY,
                                       types.FinishReason.RECITATION, types.FinishReason.FINISH_REASON_UNSPECIFIED, None])
    @pytest.mark.parametrize("mode", ["single", "chunk"])
    def test_google_non_stop_cannot_establish_coverage(self, finish: types.FinishReason | None, mode: str) -> None:
        """Valid JSON cannot hide a partial, blocked or unfinished candidate."""
        with pytest.raises(ValueError):
            google_detect(google_response({"ads": []}, finish), mode)

    @pytest.mark.parametrize("mode", ["single", "chunk"])
    @pytest.mark.parametrize("block,safety", [(types.BlockedReason.SAFETY, False), (None, True)])
    def test_google_block_flags_override_stop_text(
        self, mode: str, block: types.BlockedReason | None, safety: bool,
    ) -> None:
        """Explicit prompt/candidate blocks cannot authorize an empty result."""
        with pytest.raises(ValueError):
            google_detect(google_response({"ads": []}, block=block, safety_blocked=safety), mode)

    @pytest.mark.parametrize("mode", ["single", "chunk"])
    def test_google_completed_explicit_empty_remains_valid(self, mode: str) -> None:
        """A genuine STOP/no-block response can return a complete empty list."""
        assert google_detect(google_response({"ads": []}), mode) == []

    @pytest.mark.parametrize("failure", ["outside", "incomplete", "blocked"])
    def test_invalid_required_gemini_decision_never_reaches_publication(
        self, worker: WorkerDaemon, failure: str,
    ) -> None:
        """Exercise the real tiered job boundary, not only the response parser."""
        payload = {"ads": [{"start_time": 130, "end_time": 150, "sponsor": "Synthetic"}]} if failure == "outside" else {"ads": []}
        response = google_response(payload, types.FinishReason.MAX_TOKENS if failure == "incomplete" else types.FinishReason.STOP,
                                   types.BlockedReason.SAFETY if failure == "blocked" else None)
        detector = GeminiAudioClient("synthetic-provider", model="gemini-2.5-flash")
        detector._client = google_sdk(response)
        worker.config.llm.provider = "none"
        worker.config.gemini.enabled = True
        worker._get_gemini_client = Mock(return_value=detector)
        worker._download_audio = Mock()
        worker.api_client = Mock()
        worker.r2_client = Mock()
        worker.artifacts_dir = None
        worker.engine = "whisper"
        worker.whisper_model = "synthetic"
        worker.device = "cpu"
        job = EpisodeJob(id="synthetic", podcast_id="synthetic", guid="synthetic", title="Synthetic",
                         original_audio_url="https://example.invalid/audio.mp3")
        with patch("worker.daemon.get_duration", return_value=120), patch("worker.daemon.splice_audio") as splice:
            with pytest.raises(DetectionSafetyError, match="Required Gemini"):
                worker.process_job(job)
        splice.assert_not_called()
        worker.r2_client.upload_file.assert_not_called()
        worker.api_client.complete.assert_not_called()


class TestRequiredCoverageBounds:
    """Invalid nonempty spans cannot disappear into an empty success."""

    @pytest.mark.parametrize("start,end", [(130, 150), (-50, -10), (120, 150), (-10, 0)])
    @pytest.mark.parametrize("mixed", [False, True])
    def test_merge_rejects_any_wholly_outside_span(self, start: int, end: int, mixed: bool) -> None:
        """Even a valid sibling cannot excuse an invalid authoritative range."""
        ads = [{"start": start, "end": end}]
        if mixed:
            ads.insert(0, {"start": 10, "end": 40})
        with openai_response(json.dumps({"ads": ads})) as sdk, patch("openai.OpenAI", return_value=sdk):
            with pytest.raises(ValueError, match="overlap"):
                OpenAIClient("synthetic-provider").merge_and_refine({"duration": 120}, Config())

    @pytest.mark.parametrize("start,end", [(130, 150), (120, 150)])
    @pytest.mark.parametrize("mode", ["single", "chunk"])
    @pytest.mark.parametrize("mixed", [False, True])
    def test_google_rejects_wholly_outside_ranges(self, start: int, end: int, mode: str, mixed: bool) -> None:
        """Single-shot and chunked detections enforce the same audio coverage."""
        ads = [{"start_time": start, "end_time": end, "sponsor": "Synthetic"}]
        if mixed:
            ads.insert(0, {"start_time": 10, "end_time": 40, "sponsor": "Synthetic"})
        with pytest.raises(ValueError, match="overlap"):
            google_detect(google_response({"ads": ads}), mode)

    @pytest.mark.parametrize("mode", ["single", "chunk"])
    def test_google_partial_overlap_still_clamps(self, mode: str) -> None:
        """Single-shot and chunked wrappers preserve partly overlapping cuts."""
        ads = google_detect(google_response({"ads": [{"start_time": 100, "end_time": 150, "sponsor": "Synthetic"}]}), mode)
        if mode == "single":
            assert [(ad.start, ad.end) for ad in ads] == [(100, 120)]
        else:
            assert [(ad["start"], ad["end"]) for ad in ads] == [(100, 120)]

    @pytest.mark.parametrize("start,end,expected", [(-10, 20, (0, 20)), (100, 150, (100, 120))])
    def test_partial_overlap_still_clamps(self, start: int, end: int, expected: tuple[int, int]) -> None:
        """Finite partly overlapping boundaries retain their existing useful cuts."""
        with openai_response(json.dumps({"ads": [{"start": start, "end": end}]})) as sdk, patch("openai.OpenAI", return_value=sdk):
            ads, _ = OpenAIClient("synthetic-provider").merge_and_refine({"duration": 120}, Config())
        assert [(ad.start, ad.end) for ad in ads] == [expected]


class TestUnexpectedCancellationAndInnerSecrets:
    """Loss of coverage and custom-key diagnostics are independent safeguards."""

    @pytest.mark.parametrize("all_cancelled", [False, True])
    def test_unexpected_chunk_cancellation_cannot_be_success(self, tmp_path: Path, all_cancelled: bool) -> None:
        """Cancelled futures without a recorded outage are required failures."""
        sdk = google_sdk(google_response({"ads": []}))
        sdk.models.generate_content.side_effect = (
            [CancelledError("upstream cancellation")] * 3 if all_cancelled else
            [google_response({"ads": []}), CancelledError("upstream cancellation"), google_response({"ads": []})]
        )
        detector = GeminiChunkedClient("synthetic-provider", model="gemini-2.5-flash")
        detector._client = sdk
        chunks = [(tmp_path / f"chunk-{n}.mp3", n * 120, (n + 1) * 120) for n in range(3)]
        with patch("adnihilator.gemini_chunked.split_audio_into_chunks", return_value=chunks):
            with pytest.raises(RuntimeError, match="incomplete ad coverage"):
                detector.detect_ads(tmp_path / "episode.mp3", max_parallel=1)

    @pytest.mark.parametrize("stage", ["review", "hunt", "mega"])
    def test_inner_error_masks_key_from_custom_environment_name(
        self, worker: WorkerDaemon, stage: str, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Sanitize the first exception/log boundary, before run_once handles it."""
        secret = "opaque-provider-value-8675309"
        sdk = Mock()
        sdk.chat.completions.create.side_effect = RuntimeError("Opaque credential rejected: " + secret)
        segments = [TranscriptSegment(index=0, start=0, end=600, text="Sponsor advertisement")]
        with patch("openai.OpenAI", return_value=sdk):
            if stage == "review":
                with pytest.raises(DetectionSafetyError) as failure:
                    worker_review(worker)
                assert secret not in "".join(traceback.format_exception(failure.value))
            elif stage == "hunt":
                worker._llm_hunt_in_window(segments, "Synthetic", 0, 120, 600)
            else:
                worker._refine_mega_spans([AdSpan(start=10, end=500, confidence=0.9, reason="Sponsor")], segments, None, threshold=300)
        sdk.chat.completions.create.assert_called_once()
        assert secret not in capsys.readouterr().out
