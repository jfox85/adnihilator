"""Safety tests for chunked Gemini detection."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from adnihilator.gemini_chunked import detect_ads_in_chunk


def test_empty_chunk_response_fails_closed(tmp_path: Path) -> None:
    """An empty model response must not be interpreted as no advertisements."""
    chunk_path = tmp_path / "chunk.mp3"
    chunk_path.touch()
    genai = Mock()
    genai.upload_file.return_value = SimpleNamespace(
        name="uploaded-chunk",
        state=SimpleNamespace(name="ACTIVE"),
    )
    model = Mock()
    model.generate_content.return_value = SimpleNamespace(text="")
    genai.GenerativeModel.return_value = model

    with pytest.raises(ValueError, match="Empty Gemini response"):
        detect_ads_in_chunk(
            genai,
            chunk_path,
            chunk_start=0.0,
            chunk_end=300.0,
            model_name="gemini-test",
            max_retries=1,
        )


def test_malformed_chunk_response_fails_closed(tmp_path: Path) -> None:
    """Malformed JSON must propagate as a detection failure."""
    chunk_path = tmp_path / "chunk.mp3"
    chunk_path.touch()
    genai = Mock()
    genai.upload_file.return_value = SimpleNamespace(
        name="uploaded-chunk",
        state=SimpleNamespace(name="ACTIVE"),
    )
    model = Mock()
    model.generate_content.return_value = SimpleNamespace(text="not JSON")
    genai.GenerativeModel.return_value = model

    with pytest.raises(ValueError, match="Failed to parse Gemini"):
        detect_ads_in_chunk(
            genai,
            chunk_path,
            chunk_start=0.0,
            chunk_end=300.0,
            model_name="gemini-test",
            max_retries=1,
        )
