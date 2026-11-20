"""Embedded ID3 chapters must be rewritten to match the spliced audio."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from adnihilator.models import AdSpan
from adnihilator.splice import splice_audio

pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg tools unavailable"
)


def _chaptered_mp3(path: Path, chapters: list[tuple[float, float, str]], duration: float) -> None:
    """Encode a sine MP3 with ID3 chapters and a title tag."""
    meta = path.with_suffix(".ffmeta")
    lines = [";FFMETADATA1", "title=Synthetic Episode"]
    for start, end, title in chapters:
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(start * 1000)}", f"END={int(end * 1000)}", f"title={title}"]
    meta.write_text("\n".join(lines) + "\n")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         f"sine=frequency=440:duration={duration}", "-i", str(meta), "-map_metadata", "1",
         "-map_chapters", "1", "-codec:a", "libmp3lame", "-q:a", "6", "-id3v2_version", "3", str(path)],
        capture_output=True, check=True,
    )


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_chapters", "-show_format", "-of", "json", str(path)],
        capture_output=True, check=True, text=True,
    )
    return json.loads(out.stdout)


class TestSpliceChapters:
    def test_chapters_are_remapped_and_removed_chapter_dropped(self, tmp_path: Path) -> None:
        source, output = tmp_path / "in.mp3", tmp_path / "out.mp3"
        _chaptered_mp3(source, [(0, 4, "Intro"), (4, 10, "Sponsors"), (10, 16, "Topic A"), (16, 20, "Topic B")], 20)

        stats = splice_audio(str(source), str(output), [AdSpan(start=4, end=10, confidence=0.95, reason="ad")], 20)

        probed = _probe(output)
        chapters = [(round(float(c["start_time"])), round(float(c["end_time"])), c["tags"]["title"])
                    for c in probed["chapters"]]
        assert chapters == [(0, 4, "Intro"), (4, 10, "Topic A"), (10, 14, "Topic B")]
        assert probed["format"]["tags"]["title"] == "Synthetic Episode"
        assert stats["new_duration"] == pytest.approx(14)

    def test_keep_segments_are_reported(self, tmp_path: Path) -> None:
        """Callers get the kept ranges so descriptions can be remapped too."""
        source, output = tmp_path / "in.mp3", tmp_path / "out.mp3"
        _chaptered_mp3(source, [(0, 6, "Only")], 6)
        stats = splice_audio(str(source), str(output), [AdSpan(start=1, end=2, confidence=0.9, reason="ad")], 6)
        assert stats["keep_segments"] == [(0.0, 1), (2, 6)]

    def test_no_chapters_in_source_is_fine(self, tmp_path: Path) -> None:
        source, output = tmp_path / "in.mp3", tmp_path / "out.mp3"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             "sine=frequency=440:duration=6", "-codec:a", "libmp3lame", "-q:a", "6", str(source)],
            capture_output=True, check=True,
        )
        splice_audio(str(source), str(output), [AdSpan(start=1, end=2, confidence=0.9, reason="ad")], 6)
        assert _probe(output)["chapters"] == []
