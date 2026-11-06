"""Regression coverage for new-key uploads and safe upload failures."""

import io
import json
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from adnihilator.gemini_files import GeminiFileClient


def test_upload_does_not_use_discovery_and_returns_sdk_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "episode.mp3"
    media.write_bytes(b"audio")
    calls = []

    def send(request: Request, timeout: int) -> io.BytesIO:
        calls.append(request)
        response = io.BytesIO(json.dumps({"file": {"name": "files/test"}}).encode())
        response.headers = {"X-Goog-Upload-URL": "https://generativelanguage.googleapis.com/upload/session"}
        return response

    monkeypatch.setattr("adnihilator.gemini_files.urlopen", send)
    legacy = Mock()
    result = GeminiFileClient(legacy, "synthetic-key").upload_file(str(media))
    assert all("discovery" not in request.full_url for request in calls)
    assert calls[0].get_header("X-goog-api-key") == "synthetic-key"
    assert all("synthetic-key" not in request.full_url for request in calls)
    assert calls[1].data == b"audio"
    legacy.get_file.assert_called_once_with("files/test")
    assert result is legacy.get_file.return_value


def test_upload_failure_does_not_disclose_key_or_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "episode.mp3"
    media.write_bytes(b"audio")

    def fail(request: Request, timeout: int) -> None:
        raise HTTPError("https://example.test/?key=synthetic-key", 429,
                        "synthetic-key session-secret", {}, None)

    monkeypatch.setattr("adnihilator.gemini_files.urlopen", fail)
    with pytest.raises(ValueError, match="HTTP 429") as result:
        GeminiFileClient(Mock(), "synthetic-key").upload_file(str(media))
    assert "synthetic-key" not in str(result.value)
    assert "session-secret" not in str(result.value)
    assert result.value.__suppress_context__


def test_upload_refuses_unexpected_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "episode.mp3"
    media.write_bytes(b"audio")
    response = io.BytesIO(b"")
    response.headers = {"X-Goog-Upload-URL": "https://other.example/upload"}
    send = Mock(return_value=response)
    monkeypatch.setattr("adnihilator.gemini_files.urlopen", send)
    with pytest.raises(ValueError, match="Unexpected Gemini upload destination"):
        GeminiFileClient(Mock(), "synthetic-key").upload_file(str(media))
    assert send.call_count == 1
