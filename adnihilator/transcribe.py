"""Transcription module using faster-whisper."""

from pathlib import Path

from .models import TranscriptSegment, WordTimestamp


class TranscriptionError(Exception):
    """Error during transcription."""

    pass


def get_model_path(model_name: str) -> Path:
    """Get the expected path for a Whisper model.

    Args:
        model_name: Name of the model (e.g., "small", "medium", "large").

    Returns:
        Path where the model should be cached.
    """
    # faster-whisper uses HuggingFace cache by default
    # Models are stored in ~/.cache/huggingface/hub/
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def is_model_downloaded(model_name: str) -> bool:
    """Check if a Whisper model is already downloaded.

    Args:
        model_name: Name of the model to check.

    Returns:
        True if the model appears to be downloaded.
    """
    try:
        from faster_whisper import WhisperModel

        # Try to load the model without downloading
        # This will fail fast if the model isn't cached
        WhisperModel(model_name, device="cpu", compute_type="int8", download_root=None)
        return True
    except Exception:
        return False


def download_model(model_name: str, device: str = "cpu") -> None:
    """Download a Whisper model.

    Args:
        model_name: Name of the model to download (e.g., "small", "medium", "large").
        device: Device to use ("cpu" or "cuda").

    Raises:
        TranscriptionError: If download fails.
    """
    try:
        from faster_whisper import WhisperModel

        compute_type = "float16" if device == "cuda" else "int8"
        print(f"Downloading Whisper model '{model_name}'...")
        WhisperModel(model_name, device=device, compute_type=compute_type)
        print(f"Model '{model_name}' downloaded successfully.")
    except ImportError:
        raise TranscriptionError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        )
    except Exception as e:
        raise TranscriptionError(f"Failed to download model '{model_name}': {e}")


from typing import Callable, Optional

ProgressCallback = Callable[[int], None]  # Called with percent 0-100


def transcribe_audio(
    path: str,
    model_name: str = "small",
    device: str = "cpu",
    word_timestamps: bool = True,
    duration: Optional[float] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> list[TranscriptSegment]:
    """Transcribe an audio file using faster-whisper.

    Args:
        path: Path to the audio file.
        model_name: Whisper model to use (e.g., "tiny", "small", "medium", "large").
        device: Device to use ("cpu" or "cuda").
        word_timestamps: Whether to include word-level timestamps. Disabling this
            provides ~3x speedup but only gives segment-level timing.
        duration: Total audio duration in seconds (for progress estimation).
        progress_callback: Optional callback called with progress percentage (0-100).

    Returns:
        List of TranscriptSegment objects.

    Raises:
        TranscriptionError: If transcription fails.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise TranscriptionError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        )

    audio_path = Path(path)
    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {path}")

    try:
        # Configure compute type based on device
        compute_type = "float16" if device == "cuda" else "int8"

        model = WhisperModel(model_name, device=device, compute_type=compute_type)

        # Transcribe with VAD filter; word timestamps optional for speed
        segments_iter, info = model.transcribe(
            str(audio_path),
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
            word_timestamps=word_timestamps,
        )

        # Use info.duration if available and duration not provided
        total_duration = duration or getattr(info, 'duration', None)

        # Convert to our segment format
        segments: list[TranscriptSegment] = []
        last_progress = -1
        for idx, segment in enumerate(segments_iter):
            # Extract word timestamps if available and requested
            words = None
            if word_timestamps and segment.words:
                words = [
                    WordTimestamp(
                        word=w.word.strip(),
                        start=w.start,
                        end=w.end,
                        probability=w.probability,
                    )
                    for w in segment.words
                    if w.word.strip()  # Skip empty words
                ]

            segments.append(
                TranscriptSegment(
                    index=idx,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                    words=words,
                )
            )

            # Report progress based on segment end time vs total duration
            if progress_callback and total_duration and total_duration > 0:
                progress = min(99, int((segment.end / total_duration) * 100))
                if progress > last_progress:
                    progress_callback(progress)
                    last_progress = progress

        # Final progress
        if progress_callback:
            progress_callback(100)

        return segments

    except Exception as e:
        if "model" in str(e).lower() and "not found" in str(e).lower():
            raise TranscriptionError(
                f"Model '{model_name}' not found. "
                f"Run: adnihilator download-model {model_name}"
            )
        raise TranscriptionError(f"Transcription failed: {e}")


def transcribe_audio_regions(
    path: str,
    regions: list[tuple[float, float]],
    model_name: str = "small",
    device: str = "cpu",
    time_offset: float = 0.0,
) -> list[TranscriptSegment]:
    """Transcribe specific regions of an audio file with word timestamps.

    This is used for the two-pass optimization where we only need word-level
    timestamps for identified ad regions.

    Args:
        path: Path to the audio file (can be a segment extracted from original).
        regions: List of (start, end) tuples in seconds (relative to the file).
        model_name: Whisper model to use.
        device: Device to use ("cpu" or "cuda").
        time_offset: Offset to add to all timestamps (for when transcribing
            an extracted segment that doesn't start at 0).

    Returns:
        List of TranscriptSegment objects with word-level timestamps,
        adjusted by time_offset.

    Raises:
        TranscriptionError: If transcription fails.
    """
    # Transcribe with word timestamps enabled
    segments = transcribe_audio(
        path,
        model_name=model_name,
        device=device,
        word_timestamps=True,
    )

    # Adjust timestamps if offset provided
    if time_offset != 0.0:
        adjusted_segments = []
        for seg in segments:
            adjusted_words = None
            if seg.words:
                adjusted_words = [
                    WordTimestamp(
                        word=w.word,
                        start=w.start + time_offset,
                        end=w.end + time_offset,
                        probability=w.probability,
                    )
                    for w in seg.words
                ]
            adjusted_segments.append(
                TranscriptSegment(
                    index=seg.index,
                    start=seg.start + time_offset,
                    end=seg.end + time_offset,
                    text=seg.text,
                    words=adjusted_words,
                )
            )
        segments = adjusted_segments

    # Filter to only segments overlapping with requested regions
    if regions:
        filtered = []
        for seg in segments:
            for region_start, region_end in regions:
                if seg.end >= region_start and seg.start <= region_end:
                    filtered.append(seg)
                    break
        return filtered

    return segments
