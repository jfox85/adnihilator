"""Two-pass transcription optimization for faster ad detection.

This module implements a two-pass approach:
1. Pass 1: Fast segment-level transcription (no word timestamps)
2. Keyword detection + LLM refinement to identify ad regions
3. Extract audio for identified ad regions only
4. Pass 2: Word-level transcription only for ad regions (slower but targeted)
5. Merge word timestamps back into the segment-level transcript

This provides ~3x speedup for typical podcasts where ads are 5-10% of content.
"""

import tempfile
from pathlib import Path
from typing import Callable

from .ad_keywords import find_ad_candidates
from .ad_llm import AdLLMClient
from .audio import extract_audio_regions
from .config import Config
from .models import AdSpan, TranscriptSegment, WordTimestamp
from .transcribe import transcribe_audio


def two_pass_detect(
    audio_path: str,
    llm_client: AdLLMClient,
    config: Config,
    duration: float,
    model_name: str = "small",
    device: str = "cpu",
    buffer_seconds: float = 30.0,
    progress_callback: Callable[[int], None] | None = None,
    sponsors: "SponsorInfo | None" = None,
    podcast_name: str | None = None,
) -> tuple[list[TranscriptSegment], list[AdSpan]]:
    """Run two-pass ad detection for optimal speed/precision tradeoff.

    Args:
        audio_path: Path to the audio file.
        llm_client: LLM client for refinement.
        config: Configuration object.
        duration: Audio duration in seconds.
        model_name: Whisper model to use.
        device: Device to use ("cpu" or "cuda").
        buffer_seconds: Buffer to add around ad regions for word-level pass.
        progress_callback: Optional callback(percent: int) for progress updates.
        sponsors: Optional sponsor information for enhanced detection.
        podcast_name: Optional podcast name for sponsor detection.

    Returns:
        Tuple of (segments with word timestamps for ad regions, ad_spans).
    """
    # Pass 1: Fast segment-level transcription
    print("  Pass 1: Segment-level transcription (fast)...")
    segments = transcribe_audio(
        audio_path,
        model_name=model_name,
        device=device,
        word_timestamps=False,  # Fast mode
        duration=duration,
        progress_callback=progress_callback,
    )

    # Run keyword detection
    print("  Finding ad candidates...")
    if progress_callback:
        progress_callback(100)  # Pass 1 complete
    candidates = find_ad_candidates(segments, duration, sponsors=sponsors, podcast_name=podcast_name)

    if not candidates:
        print("  No ad candidates found")
        return segments, []

    # Run LLM refinement on segment-level transcript
    print("  Refining with LLM...")
    ad_spans = llm_client.refine_candidates(segments, candidates, config, sponsors=sponsors, podcast_title=podcast_name)

    if not ad_spans:
        print("  LLM found no ads")
        return segments, []

    print(f"  Found {len(ad_spans)} ad region(s)")

    # Pass 2: Word-level transcription only for ad regions
    ad_regions = [(span.start, span.end) for span in ad_spans]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract audio regions for word-level transcription
        print("  Extracting ad regions for word-level transcription...")
        extracted = extract_audio_regions(
            audio_path,
            tmpdir,
            ad_regions,
            buffer=buffer_seconds,
        )

        if not extracted:
            print("  No regions to extract")
            return segments, ad_spans

        # Transcribe each extracted region with word timestamps
        print(f"  Pass 2: Word-level transcription for {len(extracted)} region(s)...")
        word_level_segments: list[TranscriptSegment] = []

        for idx, (segment_path, original_start, buffered_start) in enumerate(extracted):
            if progress_callback:
                # Report Pass 2 progress as percentage of regions processed
                pass2_progress = int(100 + (idx / len(extracted)) * 100)  # 100-200%
                progress_callback(min(pass2_progress, 200))

            region_segments = transcribe_audio(
                segment_path,
                model_name=model_name,
                device=device,
                word_timestamps=True,
            )

            # Adjust timestamps to match original audio
            for seg in region_segments:
                adjusted_words = None
                if seg.words:
                    adjusted_words = [
                        WordTimestamp(
                            word=w.word,
                            start=w.start + buffered_start,
                            end=w.end + buffered_start,
                            probability=w.probability,
                        )
                        for w in seg.words
                    ]
                word_level_segments.append(
                    TranscriptSegment(
                        index=seg.index,
                        start=seg.start + buffered_start,
                        end=seg.end + buffered_start,
                        text=seg.text,
                        words=adjusted_words,
                    )
                )

    # Merge word-level segments into the original transcript
    print("  Merging word-level timestamps...")
    merged_segments = _merge_word_timestamps(segments, word_level_segments, ad_regions)

    # Refine ad spans using word-level timestamps
    print("  Refining ad boundaries with word timestamps...")
    if progress_callback:
        progress_callback(200)  # All processing complete
    refined_spans = _refine_spans_with_words(ad_spans, merged_segments)

    return merged_segments, refined_spans


def _merge_word_timestamps(
    segment_level: list[TranscriptSegment],
    word_level: list[TranscriptSegment],
    ad_regions: list[tuple[float, float]],
) -> list[TranscriptSegment]:
    """Merge word-level timestamps into segment-level transcript.

    For segments that overlap with ad regions, replace with word-level data.
    For other segments, keep the original segment-level data.
    """
    if not word_level:
        return segment_level

    # Create lookup for which segments overlap with ad regions
    def overlaps_ad_region(seg: TranscriptSegment) -> bool:
        for region_start, region_end in ad_regions:
            # Add some tolerance for boundary matching
            if seg.end >= region_start - 5 and seg.start <= region_end + 5:
                return True
        return False

    # Build merged list
    merged: list[TranscriptSegment] = []
    word_level_by_time = sorted(word_level, key=lambda s: s.start)
    word_idx = 0

    for seg in segment_level:
        if overlaps_ad_region(seg):
            # Find word-level segments that cover this time range
            matching_word_segs = []
            for ws in word_level_by_time:
                if ws.end >= seg.start - 1 and ws.start <= seg.end + 1:
                    matching_word_segs.append(ws)

            if matching_word_segs:
                # Use word-level segments instead
                for ws in matching_word_segs:
                    # Check we haven't already added this segment
                    if not merged or ws.start > merged[-1].end - 0.5:
                        merged.append(
                            TranscriptSegment(
                                index=len(merged),
                                start=ws.start,
                                end=ws.end,
                                text=ws.text,
                                words=ws.words,
                            )
                        )
            else:
                # No word-level data, keep original
                merged.append(
                    TranscriptSegment(
                        index=len(merged),
                        start=seg.start,
                        end=seg.end,
                        text=seg.text,
                        words=None,
                    )
                )
        else:
            # Outside ad regions, keep original segment
            merged.append(
                TranscriptSegment(
                    index=len(merged),
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    words=None,
                )
            )

    return merged


def _refine_spans_with_words(
    spans: list[AdSpan],
    segments: list[TranscriptSegment],
) -> list[AdSpan]:
    """Refine ad span boundaries using word-level timestamps.

    Adjusts span start/end to align with actual word boundaries when available.
    """
    if not spans or not segments:
        return spans

    refined: list[AdSpan] = []

    for span in spans:
        # Find segments that overlap with this span
        overlapping = [
            s for s in segments
            if s.end >= span.start and s.start <= span.end
        ]

        if not overlapping:
            refined.append(span)
            continue

        # Find first and last words in the span
        first_word_time = span.start
        last_word_time = span.end

        for seg in overlapping:
            if seg.words:
                for w in seg.words:
                    # Find first word at or after span start
                    if w.start >= span.start - 0.5 and w.start < first_word_time + 10:
                        first_word_time = min(first_word_time, w.start)
                    # Find last word at or before span end
                    if w.end <= span.end + 0.5 and w.end > last_word_time - 10:
                        last_word_time = max(last_word_time, w.end)

        refined.append(
            AdSpan(
                start=first_word_time,
                end=last_word_time,
                confidence=span.confidence,
                reason=span.reason,
                candidate_indices=span.candidate_indices,
            )
        )

    return refined
