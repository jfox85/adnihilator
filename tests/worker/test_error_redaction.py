"""Worker errors must not persist authentication values."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from worker.daemon import WorkerDaemon


def test_job_failure_redacts_credentials_before_logging_and_reporting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-private-key")
    monkeypatch.setattr("worker.daemon.flush", lambda *args: None)
    worker = WorkerDaemon.__new__(WorkerDaemon)
    worker.api_client = Mock()
    worker.api_client.claim.return_value = SimpleNamespace(id="episode-test")
    worker.transcripts_dir = "unused"
    worker.process_job = Mock(side_effect=ValueError("request key=synthetic-private-key failed"))
    assert worker.run_once()
    assert "synthetic-private-key" not in capsys.readouterr().out
    worker.api_client.fail.assert_called_once_with(
        "episode-test", "request key=[REDACTED] failed"
    )
