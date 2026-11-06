"""Exercise real worker branches with local mocked audio/providers only."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from adnihilator.models import AdTimestamp, SponsorInfo, TranscriptSegment
from worker.client import EpisodeJob
from worker.daemon import WorkerDaemon
import worker.daemon as module


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    worker = WorkerDaemon.__new__(WorkerDaemon)
    worker.api_client, worker.r2_client = Mock(), Mock()
    worker.r2_client.get_file_size.return_value = 3
    worker.artifacts_dir = None
    worker.transcripts_dir = tmp_path / 'outbox'
    worker.engine, worker.whisper_model, worker.device = 'parakeet', 'small', 'cpu'
    worker.config = SimpleNamespace(llm=SimpleNamespace(provider='openai', api_key='synthetic-local-only'), detection=SimpleNamespace(parallel_enabled=False))
    # This fixture exercises successful required detection followed by an
    # independent transcript-storage failure, never a provider outage.
    refiner = Mock()
    refiner.refine_candidates.return_value = []  # Valid no-ad result.
    monkeypatch.setattr(module, 'create_llm_client', Mock(return_value=refiner))
    worker._download_audio = lambda url, path: path.write_bytes(b'raw')
    monkeypatch.setattr(module, 'download_recovery_audio', lambda url, path, heartbeat: path.write_bytes(b'raw'))
    monkeypatch.setattr(module, 'fetch_recovery_external', lambda url, heartbeat: None)
    monkeypatch.setattr(module, 'get_duration', lambda path: 100)
    monkeypatch.setattr(module, 'extract_sponsors', lambda *a, **k: SponsorInfo(sponsors=[], extraction_method='none'))
    def splice(input_path, output_path, *a, **k):
        Path(output_path).write_bytes(b'out')
        return dict(new_duration=90, time_removed=10)
    monkeypatch.setattr(module, 'splice_audio', splice)
    return worker


def test_timestamp_fast_path_does_not_create_transcript(daemon, monkeypatch):
    monkeypatch.setattr(module, 'extract_ad_timestamps', lambda value: [AdTimestamp(start=20, end=30, confidence=.99, extraction_method='fixture')])
    monkeypatch.setattr(module, 'transcribe_audio_parakeet', Mock(side_effect=AssertionError('Must not transcribe fast path')))
    daemon.process_job(EpisodeJob('e1', 'p1', 'guid', 'Fixture', 'https://example.com/audio'))
    daemon.api_client.complete.assert_called_once()
    daemon.api_client.publish_transcript.assert_not_called()
    assert not list(daemon.transcripts_dir.glob('*/*.json'))


def test_transcript_upload_failure_keeps_audio_completion_and_outbox(daemon, monkeypatch):
    monkeypatch.setattr(module, 'extract_ad_timestamps', lambda value: [])
    monkeypatch.setattr(module, 'fetch_external_transcript', lambda url: [TranscriptSegment(index=0, start=1.123, end=5.678, text='Independent remembered conversation')])
    daemon.api_client.publish_transcript.side_effect = RuntimeError()
    daemon.process_job(EpisodeJob('e1', 'p1', 'guid', 'Fixture', 'https://example.com/audio', source_url='https://example.com/page'))
    module.create_llm_client.assert_called_once_with(daemon.config, require_success=True)
    module.create_llm_client.return_value.refine_candidates.assert_called_once()
    daemon.api_client.complete.assert_called_once()
    assert list(daemon.transcripts_dir.glob('*/*.pending'))
    assert list(daemon.transcripts_dir.glob('*/*.json'))


def test_transcript_disk_failure_does_not_fail_audio(daemon, monkeypatch):
    monkeypatch.setattr(module, 'extract_ad_timestamps', lambda value: [])
    monkeypatch.setattr(module, 'fetch_external_transcript', lambda url: [TranscriptSegment(index=0, start=1, end=5, text='memoir')])
    monkeypatch.setattr(module, 'stage', Mock(side_effect=OSError()))
    daemon.process_job(EpisodeJob('e1', 'p1', 'guid', 'Fixture', 'https://example.com/audio', source_url='https://example.com/page'))
    module.create_llm_client.assert_called_once_with(daemon.config, require_success=True)
    module.create_llm_client.return_value.refine_candidates.assert_called_once()
    daemon.api_client.complete.assert_called_once()


def test_explicit_transcription_job_has_no_ad_or_paid_processing(daemon, monkeypatch):
    monkeypatch.setattr(module, 'fetch_external_transcript', lambda url: None)
    transcribe = Mock(return_value=[TranscriptSegment(index=0, start=0, end=10, text='Local transcript')])
    monkeypatch.setattr(module, 'transcribe_audio_parakeet', transcribe)
    monkeypatch.setattr(module, 'extract_sponsors', Mock(side_effect=AssertionError('No ad detection')))
    daemon.process_transcript_job(dict(id='e1', podcast_id='p1', guid='guid', original_audio_url='https://example.com/audio', claim_id='attempt-1'))
    transcribe.assert_called_once()
    daemon.api_client.publish_transcript.assert_called_once()
    daemon.api_client.complete.assert_not_called()
    daemon.r2_client.upload_file.assert_not_called()


def test_gemini_fast_path_keeps_missing_transcript(daemon, monkeypatch):
    from adnihilator.models import AdSpan
    monkeypatch.setattr(module, 'extract_ad_timestamps', lambda value: [])
    gemini = Mock()
    gemini.detect_ads.return_value = ([AdSpan(start=20, end=30, confidence=.9, reason='fixture')], None)
    daemon._get_gemini_client = lambda **kwargs: gemini
    monkeypatch.setattr(module, 'transcribe_audio_parakeet', Mock(side_effect=AssertionError('No fast-path transcription')))
    daemon.process_job(EpisodeJob('e1', 'p1', 'guid', 'Fixture', 'https://example.com/audio'))
    daemon.api_client.complete.assert_called_once()
    daemon.api_client.publish_transcript.assert_not_called()


def test_run_once_dispatches_explicit_request_without_audio_job(daemon, monkeypatch):
    daemon.api_client.claim.return_value = None
    daemon.api_client.claim_transcript.return_value = dict(id='e1', claim_id='attempt')
    daemon.process_transcript_job = Mock()
    assert daemon.run_once() is True
    daemon.process_transcript_job.assert_called_once()
    daemon.api_client.complete.assert_not_called()


def test_fair_recovery_under_continuous_audio_load(daemon):
    audio = EpisodeJob('audio', 'p1', 'g', 'Audio', 'https://example.com/audio')
    daemon.api_client.claim.return_value = audio
    daemon.api_client.claim_transcript.return_value = dict(id='recovery', claim_id='attempt')
    daemon.process_job = Mock()
    daemon.process_transcript_job = Mock()
    for _ in range(4):
        assert daemon.run_once()
    assert daemon.process_job.call_count == 3
    assert daemon.process_transcript_job.call_count == 1
    assert daemon.api_client.claim.call_count == 3  # Recovery does not abandon a claimed audio job.


def test_recovery_unavailable_preserves_audio_processing(daemon):
    daemon._audio_since_recovery = 3
    daemon.api_client.claim_transcript.return_value = None
    daemon.api_client.claim.return_value = EpisodeJob('audio', 'p1', 'g', 'Audio', 'https://example.com/audio')
    daemon.process_job = Mock()
    assert daemon.run_once()
    daemon.process_job.assert_called_once()


def test_old_server_claim_endpoint_absent_returns_no_recovery():
    import httpx
    from worker.client import WorkerClient
    client = WorkerClient('https://fixture.example', 'synthetic-worker')
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404)))
    try:
        assert client.claim_transcript() is None
    finally:
        client.close()


def test_recovery_lease_renews_inside_download_and_external_fetch(daemon, monkeypatch):
    import threading
    from worker.recovery import renew_lease
    renewed = threading.Event()
    daemon.api_client.heartbeat_transcript.side_effect = lambda *args: renewed.set()
    monkeypatch.setattr(module, 'renew_lease', lambda heartbeat: renew_lease(heartbeat, .005))
    def download(url, path, heartbeat):
        renewed.clear()
        assert renewed.wait(.5)  # No ASR progress callback exists in this stage.
        path.write_bytes(b'fixture')
    def external(url, heartbeat):
        renewed.clear()
        assert renewed.wait(.5)
        return None
    monkeypatch.setattr(module, 'download_recovery_audio', download)
    monkeypatch.setattr(module, 'fetch_recovery_external', external)
    monkeypatch.setattr(module, 'transcribe_audio_parakeet', Mock(return_value=[TranscriptSegment(index=0, start=0, end=10, text='fixture')]))
    daemon.process_transcript_job(dict(id='e1', podcast_id='p1', guid='guid', original_audio_url='https://fixture.example/audio', source_url='https://fixture.example/page', claim_id='attempt'))
    assert daemon.api_client.heartbeat_transcript.call_count >= 4
    daemon.api_client.publish_transcript.assert_called_once()
