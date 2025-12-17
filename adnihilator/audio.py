"""Audio utilities for AdNihilator."""

import json
import subprocess
from pathlib import Path


class AudioError(Exception):
    """Error related to audio processing."""

    pass


def get_duration(path: str) -> float:
    """Get the duration of an audio file in seconds using ffprobe.

    Args:
        path: Path to the audio file.

    Returns:
        Duration in seconds.

    Raises:
        AudioError: If the file doesn't exist, isn't readable, or ffprobe fails.
    """
    audio_path = Path(path)

    if not audio_path.exists():
        raise AudioError(f"File not found: {path}")

    if not audio_path.is_file():
        raise AudioError(f"Not a file: {path}")

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise AudioError(
            "ffprobe not found. Please install ffmpeg: https://ffmpeg.org/download.html"
        )
    except subprocess.CalledProcessError as e:
        raise AudioError(f"ffprobe failed: {e.stderr}")

    try:
        data = json.loads(result.stdout)
        duration = float(data["format"]["duration"])
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise AudioError(f"Failed to parse ffprobe output: {e}")

    return duration


def validate_audio_file(path: str) -> None:
    """Validate that a file exists and appears to be a valid audio file.

    Args:
        path: Path to the audio file.

    Raises:
        AudioError: If validation fails.
    """
    audio_path = Path(path)

    if not audio_path.exists():
        raise AudioError(f"File not found: {path}")

    if not audio_path.is_file():
        raise AudioError(f"Not a file: {path}")

    # Check for common audio extensions
    valid_extensions = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
    if audio_path.suffix.lower() not in valid_extensions:
        raise AudioError(
            f"Unsupported audio format: {audio_path.suffix}. "
            f"Supported formats: {', '.join(sorted(valid_extensions))}"
        )


def extract_audio_segment(
    input_path: str,
    output_path: str,
    start: float,
    end: float,
) -> None:
    """Extract a segment of audio from a file.

    Args:
        input_path: Path to the input audio file.
        output_path: Path for the output segment.
        start: Start time in seconds.
        end: End time in seconds.

    Raises:
        AudioError: If extraction fails.
    """
    duration = end - start
    if duration <= 0:
        raise AudioError(f"Invalid segment: end ({end}) must be greater than start ({start})")

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",  # Overwrite output
                "-i", input_path,
                "-ss", str(start),
                "-t", str(duration),
                "-acodec", "copy",  # Copy codec for speed
                output_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise AudioError(
            "ffmpeg not found. Please install ffmpeg: https://ffmpeg.org/download.html"
        )
    except subprocess.CalledProcessError as e:
        raise AudioError(f"ffmpeg extraction failed: {e.stderr}")


def extract_audio_regions(
    input_path: str,
    output_dir: str,
    regions: list[tuple[float, float]],
    buffer: float = 30.0,
) -> list[tuple[str, float, float]]:
    """Extract multiple audio regions to separate files.

    Merges overlapping regions and adds a buffer before/after each region.

    Args:
        input_path: Path to the input audio file.
        output_dir: Directory to write segment files.
        regions: List of (start, end) tuples in seconds.
        buffer: Seconds of buffer to add before/after each region.

    Returns:
        List of (output_path, original_start, buffered_start) tuples.
        The buffered_start is the actual start time in the original audio,
        which is needed to adjust timestamps.

    Raises:
        AudioError: If extraction fails.
    """
    if not regions:
        return []

    # Get total duration to cap buffer at end
    total_duration = get_duration(input_path)

    # Sort and merge overlapping regions (with buffer)
    sorted_regions = sorted(regions, key=lambda r: r[0])
    merged: list[tuple[float, float]] = []

    for start, end in sorted_regions:
        buffered_start = max(0, start - buffer)
        buffered_end = min(total_duration, end + buffer)

        if merged and buffered_start <= merged[-1][1]:
            # Overlaps with previous, extend it
            merged[-1] = (merged[-1][0], max(merged[-1][1], buffered_end))
        else:
            merged.append((buffered_start, buffered_end))

    # Extract each merged region
    results: list[tuple[str, float, float]] = []
    output_path_obj = Path(output_dir)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    for idx, (buffered_start, buffered_end) in enumerate(merged):
        segment_path = str(output_path_obj / f"region_{idx}.mp3")
        extract_audio_segment(input_path, segment_path, buffered_start, buffered_end)
        # Return: path, the original region start (for matching), buffered start (for timestamp adjustment)
        # Find the original region start that this merged region covers
        original_start = buffered_start + buffer  # Approximate original start
        for orig_start, orig_end in sorted_regions:
            if orig_start >= buffered_start and orig_end <= buffered_end:
                original_start = orig_start
                break
        results.append((segment_path, original_start, buffered_start))

    return results
