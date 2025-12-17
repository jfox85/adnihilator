"""Audio splicing module using ffmpeg."""

import subprocess
import tempfile
from pathlib import Path

from .models import AdSpan


class SpliceError(Exception):
    """Error during audio splicing."""

    pass


def splice_audio(
    input_path: str,
    output_path: str,
    ad_spans: list[AdSpan],
    duration: float,
    min_confidence: float = 0.35,
) -> dict:
    """Remove ad segments from an audio file.

    Args:
        input_path: Path to the input audio file.
        output_path: Path for the output audio file.
        ad_spans: List of ad spans to remove.
        duration: Total duration of the audio in seconds.
        min_confidence: Minimum confidence threshold for ad removal.

    Returns:
        Dictionary with splice statistics.

    Raises:
        SpliceError: If splicing fails.
    """
    # Filter ad spans by confidence
    spans_to_remove = [s for s in ad_spans if s.confidence >= min_confidence]

    if not spans_to_remove:
        # No ads to remove, just copy the file
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_path, "-c", "copy", output_path],
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise SpliceError(f"Failed to copy audio: {e.stderr.decode()}")

        return {
            "ads_removed": 0,
            "time_removed": 0.0,
            "original_duration": duration,
            "new_duration": duration,
        }

    # Sort ad spans by start time
    sorted_spans = sorted(spans_to_remove, key=lambda s: s.start)

    # Calculate segments to KEEP (inverse of ad spans)
    keep_segments: list[tuple[float, float]] = []
    current_pos = 0.0

    for span in sorted_spans:
        if span.start > current_pos:
            keep_segments.append((current_pos, span.start))
        current_pos = max(current_pos, span.end)

    # Add final segment if there's content after last ad
    if current_pos < duration:
        keep_segments.append((current_pos, duration))

    if not keep_segments:
        raise SpliceError("No content would remain after removing ads")

    # Use ffmpeg filter_complex to concatenate segments
    time_removed = sum(s.end - s.start for s in sorted_spans)

    try:
        _splice_with_filter_complex(input_path, output_path, keep_segments)
    except SpliceError:
        # Fallback to segment-by-segment approach
        _splice_with_concat(input_path, output_path, keep_segments)

    return {
        "ads_removed": len(sorted_spans),
        "time_removed": time_removed,
        "original_duration": duration,
        "new_duration": duration - time_removed,
        "segments_kept": len(keep_segments),
    }


def _splice_with_filter_complex(
    input_path: str,
    output_path: str,
    keep_segments: list[tuple[float, float]],
) -> None:
    """Splice audio using ffmpeg filter_complex.

    This approach is faster and more accurate for multiple segments.
    """
    # Build filter_complex string
    # Format: [0:a]atrim=start:end,asetpts=PTS-STARTPTS[a0];...;[a0][a1]..concat=n=N:v=0:a=1[out]
    filter_parts = []
    concat_inputs = []

    for i, (start, end) in enumerate(keep_segments):
        label = f"a{i}"
        filter_parts.append(f"[0:a]atrim={start}:{end},asetpts=PTS-STARTPTS[{label}]")
        concat_inputs.append(f"[{label}]")

    # Build concat filter
    concat_filter = f"{''.join(concat_inputs)}concat=n={len(keep_segments)}:v=0:a=1[out]"
    filter_complex = ";".join(filter_parts) + ";" + concat_filter

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise SpliceError(f"ffmpeg filter_complex failed: {e.stderr.decode()}")


def _splice_with_concat(
    input_path: str,
    output_path: str,
    keep_segments: list[tuple[float, float]],
) -> None:
    """Splice audio using segment extraction and concat demuxer.

    Fallback approach if filter_complex fails.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        segment_files: list[Path] = []

        # Extract each segment
        for i, (start, end) in enumerate(keep_segments):
            segment_path = tmpdir_path / f"segment_{i:04d}.mp3"
            segment_files.append(segment_path)

            cmd = [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-ss", str(start),
                "-to", str(end),
                "-c", "copy",
                str(segment_path),
            ]

            try:
                subprocess.run(cmd, capture_output=True, check=True)
            except subprocess.CalledProcessError as e:
                raise SpliceError(f"Failed to extract segment {i}: {e.stderr.decode()}")

        # Create concat list file
        concat_list = tmpdir_path / "concat_list.txt"
        with open(concat_list, "w") as f:
            for seg_path in segment_files:
                f.write(f"file '{seg_path}'\n")

        # Concatenate segments
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            output_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            raise SpliceError(f"Failed to concatenate segments: {e.stderr.decode()}")
