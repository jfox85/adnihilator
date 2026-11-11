"""Safety tests for chunked Gemini detection."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from adnihilator.gemini_chunked import detect_ads_in_chunk


def test_empty_chunk_response_fails_closed(tmp_path: Path) -> None:
    """An empty model response must not be interpreted as no advertisements."""
    chunk_path = tmp_path / "chunk.mp3"
    chunk_path.touch()
    client = Mock()
    client.files.upload.return_value = SimpleNamespace(
        name="uploaded-chunk",
        state=SimpleNamespace(name="ACTIVE"),
    )
    client.models.generate_content.return_value = SimpleNamespace(text="", candidates=[SimpleNamespace(finish_reason="STOP")])

    with pytest.raises(ValueError, match="Empty.*Gemini"):
        detect_ads_in_chunk(
            client,
            chunk_path,
            chunk_start=0.0,
            chunk_end=300.0,
            model_name="gemini-test",
            max_retries=1,
        )


def _run_chunk(tmp_path: Path, ads: list[dict], chunk_start: float, chunk_end: float) -> list[dict]:
    """Run detection for one chunk with a canned Gemini response."""
    chunk_path = tmp_path / "chunk.mp3"
    chunk_path.touch()
    client = Mock()
    client.files.upload.return_value = SimpleNamespace(
        name="uploaded-chunk",
        state=SimpleNamespace(name="ACTIVE"),
    )
    client.models.generate_content.return_value = SimpleNamespace(
        text=json.dumps({"ads": ads}),
        candidates=[SimpleNamespace(finish_reason="STOP")],
    )
    return detect_ads_in_chunk(
        client,
        chunk_path,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        model_name="gemini-test",
        max_retries=1,
    )


def test_chunk_relative_timestamps_are_converted_to_absolute(tmp_path: Path) -> None:
    """Normal chunk-relative detections are offset by the chunk start."""
    ads = _run_chunk(
        tmp_path,
        [{"start_time": 10.0, "end_time": 70.0, "sponsor": "Acme"}],
        chunk_start=4200.0,
        chunk_end=4409.0,
    )

    assert [(ad["start"], ad["end"]) for ad in ads] == [(4210.0, 4270.0)]


def test_episode_absolute_timestamps_are_recovered(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Timestamps that only fit as episode-absolute are reinterpreted, not fatal.

    Regression: Animal Spirits EP. 490's final 209s chunk returned
    episode-absolute times, which failed the whole episode on every retry.
    """
    with caplog.at_level(logging.WARNING, logger="adnihilator.gemini_chunked"):
        ads = _run_chunk(
            tmp_path,
            [{"start_time": 4250.0, "end_time": 4310.0, "sponsor": "Acme"}],
            chunk_start=4200.0,
            chunk_end=4409.0,
        )

    assert [(ad["start"], ad["end"]) for ad in ads] == [(4250.0, 4310.0)]
    assert "episode-absolute" in caplog.text
    assert "4250" in caplog.text  # raw response is logged for diagnosis


def test_detection_outside_chunk_in_both_interpretations_is_dropped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Unplaceable detections are dropped with a warning; valid ones are kept."""
    with caplog.at_level(logging.WARNING, logger="adnihilator.gemini_chunked"):
        ads = _run_chunk(
            tmp_path,
            [
                {"start_time": 900.0, "end_time": 960.0, "sponsor": "Ghost"},
                {"start_time": 20.0, "end_time": 50.0, "sponsor": "Acme"},
            ],
            chunk_start=4200.0,
            chunk_end=4409.0,
        )

    assert [(ad["start"], ad["end"], ad["sponsor"]) for ad in ads] == [(4220.0, 4250.0, "Acme")]
    assert "Dropping Gemini detection" in caplog.text
    assert "Ghost" in caplog.text  # raw response is logged for diagnosis


def test_malformed_chunk_response_fails_closed(tmp_path: Path) -> None:
    """Malformed JSON must propagate as a detection failure."""
    chunk_path = tmp_path / "chunk.mp3"
    chunk_path.touch()
    client = Mock()
    client.files.upload.return_value = SimpleNamespace(
        name="uploaded-chunk",
        state=SimpleNamespace(name="ACTIVE"),
    )
    client.models.generate_content.return_value = SimpleNamespace(text="not JSON", candidates=[SimpleNamespace(finish_reason="STOP")])

    with pytest.raises(ValueError, match="Failed to parse Gemini"):
        detect_ads_in_chunk(
            client,
            chunk_path,
            chunk_start=0.0,
            chunk_end=300.0,
            model_name="gemini-test",
            max_retries=1,
        )
