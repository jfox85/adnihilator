"""External ffmpeg diagnostics must not bypass splicing error handling."""

import shutil
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from adnihilator.audio import get_duration
from adnihilator.models import AdSpan
from adnihilator.splice import SpliceError, _splice_with_concat, splice_audio

INVALID_DIAGNOSTIC = b"x" * 2998 + b"\xe2 invalid ffmpeg diagnostic"


def ffmpeg_failure() -> subprocess.CalledProcessError:
    """Reconstruct the invalid-byte offset observed in the production log."""
    return subprocess.CalledProcessError(1, ["ffmpeg"], stderr=INVALID_DIAGNOSTIC)


class TestSpliceDiagnostics:
    def test_invalid_filter_diagnostic_still_attempts_fallback(self) -> None:
        """An unreadable diagnostic cannot suppress the existing fallback."""
        span = AdSpan(start=1, end=2, confidence=0.9, reason="Synthetic cut")
        with (
            patch("adnihilator.splice.subprocess.run", side_effect=ffmpeg_failure()),
            patch("adnihilator.splice._splice_with_concat") as fallback,
        ):
            result = splice_audio("input.mp3", "output.mp3", [span], duration=6)
        fallback.assert_called_once_with("input.mp3", "output.mp3", [(0.0, 1.0), (2.0, 6)])
        assert result["new_duration"] == 5

    def test_invalid_copy_diagnostic_remains_a_splice_error(self) -> None:
        """No-ad copy errors retain a readable domain error."""
        with patch("adnihilator.splice.subprocess.run", side_effect=ffmpeg_failure()):
            with pytest.raises(SpliceError, match="Failed to copy audio") as error:
                splice_audio("input.mp3", "output.mp3", [], duration=6)
        assert "\ufffd invalid ffmpeg diagnostic" in str(error.value)

    @pytest.mark.parametrize("fail_at", ["extract", "concat"])
    def test_invalid_fallback_diagnostic_remains_a_splice_error(self, fail_at: str) -> None:
        """Both segment extraction and concat report the actual ffmpeg failure."""
        effects = [ffmpeg_failure()] if fail_at == "extract" else [subprocess.CompletedProcess([], 0), ffmpeg_failure()]
        with patch("adnihilator.splice.subprocess.run", side_effect=effects):
            with pytest.raises(SpliceError) as error:
                _splice_with_concat("input.mp3", "output.mp3", [(0, 1)])
        assert "\ufffd invalid ffmpeg diagnostic" in str(error.value)
        assert ("extract segment" if fail_at == "extract" else "concatenate segments") in str(error.value)

    @pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg tools unavailable")
    @pytest.mark.parametrize("channels,sample_rate", [(1, 22050), (2, 44100)])
    def test_real_audio_fallback_finishes_after_undecodable_filter_error(
        self, tmp_path: Path, channels: int, sample_rate: int
    ) -> None:
        """The fallback produces playable mono and stereo VBR audio."""
        source = tmp_path / "source.mp3"
        output = tmp_path / "clean.mp3"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             "sine=frequency=440:duration=6", "-ac", str(channels), "-ar", str(sample_rate),
             "-codec:a", "libmp3lame", "-q:a", "4", str(source)],
            capture_output=True, check=True,
        )
        run = subprocess.run
        failed_filters: list[list[str]] = []

        def fail_filter_only(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            """Keep the fallback real while injecting the observed filter failure."""
            if "-filter_complex" in args:
                failed_filters.append(args)
                raise ffmpeg_failure()
            return run(args, **kwargs)

        with patch("adnihilator.splice.subprocess.run", side_effect=fail_filter_only):
            stats = splice_audio(
                str(source), str(output), [AdSpan(start=1, end=2, confidence=0.9, reason="Synthetic cut")], 6
            )
        assert len(failed_filters) == 1
        assert stats["new_duration"] == 5
        assert get_duration(str(output)) == pytest.approx(5, abs=0.2)
