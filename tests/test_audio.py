"""Tests for audio utilities."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from adnihilator.audio import AudioError, get_duration, validate_audio_file


class TestGetDuration:
    def test_file_not_found(self, tmp_path):
        """Should raise AudioError for non-existent file."""
        fake_path = tmp_path / "nonexistent.mp3"
        with pytest.raises(AudioError, match="File not found"):
            get_duration(str(fake_path))

    def test_not_a_file(self, tmp_path):
        """Should raise AudioError for directories."""
        with pytest.raises(AudioError, match="Not a file"):
            get_duration(str(tmp_path))

    @patch("subprocess.run")
    def test_ffprobe_not_found(self, mock_run, tmp_path):
        """Should raise AudioError if ffprobe is not installed."""
        # Create a dummy file
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        mock_run.side_effect = FileNotFoundError()

        with pytest.raises(AudioError, match="ffprobe not found"):
            get_duration(str(test_file))

    @patch("subprocess.run")
    def test_ffprobe_failure(self, mock_run, tmp_path):
        """Should raise AudioError if ffprobe fails."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        mock_run.side_effect = subprocess.CalledProcessError(1, "ffprobe", stderr="error")

        with pytest.raises(AudioError, match="ffprobe failed"):
            get_duration(str(test_file))

    @patch("subprocess.run")
    def test_successful_duration_extraction(self, mock_run, tmp_path):
        """Should return duration from ffprobe output."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        # Mock ffprobe output
        ffprobe_output = {"format": {"duration": "123.456"}}
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(ffprobe_output)
        mock_run.return_value = mock_result

        duration = get_duration(str(test_file))
        assert duration == 123.456

    @patch("subprocess.run")
    def test_malformed_ffprobe_output(self, mock_run, tmp_path):
        """Should raise AudioError for malformed ffprobe output."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        mock_result = MagicMock()
        mock_result.stdout = "not json"
        mock_run.return_value = mock_result

        with pytest.raises(AudioError, match="Failed to parse"):
            get_duration(str(test_file))

    @patch("subprocess.run")
    def test_missing_duration_field(self, mock_run, tmp_path):
        """Should raise AudioError if duration field is missing."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        mock_result = MagicMock()
        mock_result.stdout = json.dumps({"format": {}})
        mock_run.return_value = mock_result

        with pytest.raises(AudioError, match="Failed to parse"):
            get_duration(str(test_file))


class TestValidateAudioFile:
    def test_file_not_found(self, tmp_path):
        """Should raise AudioError for non-existent file."""
        fake_path = tmp_path / "nonexistent.mp3"
        with pytest.raises(AudioError, match="File not found"):
            validate_audio_file(str(fake_path))

    def test_not_a_file(self, tmp_path):
        """Should raise AudioError for directories."""
        with pytest.raises(AudioError, match="Not a file"):
            validate_audio_file(str(tmp_path))

    def test_unsupported_extension(self, tmp_path):
        """Should raise AudioError for unsupported file types."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("not audio")

        with pytest.raises(AudioError, match="Unsupported audio format"):
            validate_audio_file(str(test_file))

    @pytest.mark.parametrize("extension", [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"])
    def test_supported_extensions(self, tmp_path, extension):
        """Should accept supported audio extensions."""
        test_file = tmp_path / f"test{extension}"
        test_file.write_bytes(b"fake audio data")

        # Should not raise
        validate_audio_file(str(test_file))

    def test_case_insensitive_extension(self, tmp_path):
        """Should accept extensions regardless of case."""
        test_file = tmp_path / "test.MP3"
        test_file.write_bytes(b"fake audio data")

        # Should not raise
        validate_audio_file(str(test_file))
