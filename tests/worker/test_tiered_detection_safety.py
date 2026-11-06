"""Tiered workers must finish detection before publishing processed audio."""

import socket
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from adnihilator.ad_keywords import find_ad_candidates
from adnihilator.config import Config
from adnihilator.models import AdSpan, SponsorInfo, TranscriptSegment
from worker.client import EpisodeJob
from worker.daemon import DetectionSafetyError, WorkerDaemon


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent all real provider, download and object-storage requests."""
    def denied(*args: object, **kwargs: object) -> None:
        raise AssertionError("Network forbidden in tiered worker regressions")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def transcript(ad: bool = True) -> list[TranscriptSegment]:
    """Provide realistic context around one bounded sponsor read."""
    return [
        TranscriptSegment(
            index=index, start=index * 10, end=(index + 1) * 10,
            text=("This episode is brought to you by Squarespace. Visit squarespace.com/audio and use code AUDIO for twenty percent off."
                  if ad and 15 <= index <= 18 else "The astronomer explains the motion of stars and distant galaxies."),
        )
        for index in range(60)
    ]


@pytest.fixture
def worker() -> WorkerDaemon:
    """Use a bare object with fake storage/API clients, never a running worker."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    daemon.config = Config()
    daemon.api_client = Mock()
    daemon.r2_client = Mock()
    daemon.r2_client.get_file_size.return_value = len(b"clean")
    daemon._download_audio = Mock()
    daemon._get_gemini_client = Mock(return_value=None)
    daemon.artifacts_dir = None
    daemon.whisper_model = "synthetic"
    daemon.device = "cpu"
    daemon.engine = "whisper"
    return daemon


def job(source_url: str | None = None) -> EpisodeJob:
    """Build a synthetic job without an externally reachable audio source."""
    return EpisodeJob(id="episode", podcast_id="podcast", guid="synthetic", title="Synthetic", original_audio_url="https://example.invalid/audio", source_url=source_url)


def fake_splice(input_path: str, output_path: str, spans: list[AdSpan], duration: float, **kwargs: object) -> dict:
    """Write a stand-in upload file; observe only cuts above the real threshold."""
    Path(output_path).write_bytes(b"clean")
    removed = sum(span.end - span.start for span in spans if span.confidence >= 0.35)
    return {"time_removed": removed, "new_duration": duration - removed}


class TestTieredDetectionSafety:
    @pytest.mark.parametrize("engine", ["whisper", "parakeet"])
    def test_disabled_gemini_still_detects_ads(self, worker: WorkerDaemon, engine: str) -> None:
        """Transcription alone cannot stand in for the heuristic detection stage."""
        worker.engine = engine
        with (
            patch("worker.daemon.get_duration", return_value=600),
            patch("worker.daemon.transcribe_audio", return_value=transcript()),
            patch("worker.daemon.transcribe_audio_parakeet", return_value=transcript()),
            patch("worker.daemon.find_ad_candidates", wraps=find_ad_candidates) as detect,
            patch("worker.daemon.splice_audio", side_effect=fake_splice) as splice,
        ):
            worker.process_job(job())
        detect.assert_called_once()
        spans = splice.call_args.args[2]
        assert any(span.start <= 150 and 190 <= span.end < 400 and span.confidence >= 0.35 for span in spans)
        assert worker.api_client.complete.call_args.kwargs["ads_removed"] > 0

    def test_external_transcript_is_detected_before_completion(self, worker: WorkerDaemon) -> None:
        """An external source label also cannot suppress actual detection."""
        with (
            patch("worker.daemon.get_duration", return_value=600),
            patch("worker.daemon.fetch_external_transcript", return_value=transcript()),
            patch("worker.daemon.transcribe_audio") as asr,
            patch("worker.daemon.find_ad_candidates", wraps=find_ad_candidates) as detect,
            patch("worker.daemon.splice_audio", side_effect=fake_splice),
        ):
            worker.process_job(job("https://example.invalid/episode"))
        asr.assert_not_called()
        detect.assert_called_once()
        assert worker.api_client.complete.call_args.kwargs["detection_source"] == "external"
        assert worker.api_client.complete.call_args.kwargs["ads_removed"] > 0

    @pytest.mark.parametrize("unavailable", [False, True])
    def test_required_gemini_failure_never_publishes(self, worker: WorkerDaemon, unavailable: bool) -> None:
        """The real 503 shape cannot downgrade required Gemini coverage."""
        worker.config.gemini.enabled = True
        if not unavailable:
            client = Mock()
            client.detect_ads.side_effect = ValueError("503 The service is currently unavailable.")
            worker._get_gemini_client.return_value = client
        with (
            patch("worker.daemon.get_duration", return_value=600),
            patch("worker.daemon.transcribe_audio", return_value=transcript()),
            patch("worker.daemon.splice_audio", side_effect=fake_splice) as splice,
        ):
            with pytest.raises(DetectionSafetyError, match="Required Gemini"):
                worker.process_job(job())
        splice.assert_not_called()
        worker.r2_client.upload_file.assert_not_called()
        worker.api_client.complete.assert_not_called()

    def test_ad_free_transcript_can_complete(self, worker: WorkerDaemon) -> None:
        """A completed heuristic pass with no removable ads remains successful."""
        with (
            patch("worker.daemon.get_duration", return_value=600),
            patch("worker.daemon.transcribe_audio", return_value=transcript(ad=False)),
            patch("worker.daemon.find_ad_candidates", wraps=find_ad_candidates) as detect,
            patch("worker.daemon.splice_audio", side_effect=fake_splice),
        ):
            worker.process_job(job())
        detect.assert_called_once()
        assert worker.api_client.complete.call_args.kwargs["ads_removed"] == 0

    def test_successful_empty_gemini_response_does_not_run_asr(self, worker: WorkerDaemon) -> None:
        """Explicit successful empty coverage is different from a failed provider."""
        worker.config.gemini.enabled = True
        worker._get_gemini_client.return_value = Mock(detect_ads=Mock(return_value=([], {})))
        with (
            patch("worker.daemon.get_duration", return_value=600),
            patch("worker.daemon.transcribe_audio") as asr,
            patch("worker.daemon.splice_audio", side_effect=fake_splice),
        ):
            worker.process_job(job())
        asr.assert_not_called()
        assert worker.api_client.complete.call_args.kwargs["detection_source"] == "gemini"

    def test_empty_local_transcript_cannot_establish_coverage(self, worker: WorkerDaemon) -> None:
        """Missing transcript evidence is a failure rather than proof of no ads."""
        with (
            patch("worker.daemon.get_duration", return_value=600),
            patch("worker.daemon.transcribe_audio", return_value=[]),
            patch("worker.daemon.splice_audio") as splice,
        ):
            with pytest.raises(DetectionSafetyError, match="Required transcript detection"):
                worker.process_job(job())
        splice.assert_not_called()
        worker.api_client.complete.assert_not_called()

    def test_parallel_provider_failure_still_fails_closed(self, worker: WorkerDaemon) -> None:
        """Tiered repairs cannot weaken the existing required parallel lane."""
        worker.config.detection.parallel_enabled = True
        worker.config.gemini.enabled = True
        client = Mock()
        client.detect_ads.side_effect = ValueError("503 The service is currently unavailable.")
        worker._get_gemini_client.return_value = client
        with (
            patch("worker.daemon.get_duration", return_value=600),
            patch("worker.daemon.transcribe_audio", return_value=transcript()),
            patch("worker.daemon.splice_audio") as splice,
        ):
            with pytest.raises(DetectionSafetyError, match="Required Gemini"):
                worker.process_job(job())
        splice.assert_not_called()
        worker.r2_client.upload_file.assert_not_called()
        worker.api_client.complete.assert_not_called()

    def test_two_pass_result_is_not_detected_twice(self, worker: WorkerDaemon, monkeypatch: pytest.MonkeyPatch) -> None:
        """The already-completed OpenAI two-pass path keeps its existing behavior."""
        monkeypatch.setenv("TIERED_TEST_KEY", "synthetic-key")
        worker.config.llm.provider = "openai"
        worker.config.llm.api_key_env = "TIERED_TEST_KEY"
        with (
            patch("worker.daemon.extract_sponsors", return_value=SponsorInfo(sponsors=[], extraction_method="none")),
            patch("worker.daemon.get_duration", return_value=600),
            patch("worker.daemon.two_pass_detect", return_value=(transcript(ad=False), [])),
            patch("worker.daemon.find_ad_candidates") as detect,
            patch("worker.daemon.splice_audio", side_effect=fake_splice),
        ):
            worker.process_job(job())
        detect.assert_not_called()
        worker.api_client.complete.assert_called_once()
