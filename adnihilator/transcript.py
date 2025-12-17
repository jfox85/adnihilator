"""Transcript formatting with ad markers."""

from .models import AdSpan, DetectionResult, TranscriptSegment


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def generate_marked_transcript(result: DetectionResult) -> str:
    """Generate a human-readable transcript with ad boundaries marked.

    Args:
        result: The detection result containing segments and ad spans.

    Returns:
        A formatted string with the transcript and ad markers.
    """
    if not result.segments:
        return "No transcript segments found."

    # Sort ad spans by start time
    ad_spans = sorted(result.ad_spans, key=lambda s: s.start)

    # Build a list of events (segment text, ad start, ad end)
    output_lines: list[str] = []

    # Header
    output_lines.append(f"# Transcript: {result.audio_path}")
    output_lines.append(f"# Duration: {format_timestamp(result.duration)}")
    output_lines.append(f"# Detected ad spans: {len(ad_spans)}")
    output_lines.append("")

    # Track which ad span we're currently inside
    current_ad_idx = 0
    inside_ad = False

    for segment in result.segments:
        # Check if we need to start an ad section
        while current_ad_idx < len(ad_spans):
            ad = ad_spans[current_ad_idx]

            # If segment starts after this ad ends, move to next ad
            if segment.start >= ad.end:
                if inside_ad:
                    output_lines.append("")
                    output_lines.append(f"### END AD [{format_timestamp(ad.start)} - {format_timestamp(ad.end)}] (confidence: {ad.confidence:.0%}) ###")
                    output_lines.append("")
                    inside_ad = False
                current_ad_idx += 1
                continue

            # If segment starts within this ad
            if segment.start >= ad.start and not inside_ad:
                output_lines.append("")
                output_lines.append(f"### START AD [{format_timestamp(ad.start)} - {format_timestamp(ad.end)}] (confidence: {ad.confidence:.0%}) ###")
                output_lines.append(f"### Reason: {ad.reason} ###")
                output_lines.append("")
                inside_ad = True

            break

        # Output the segment
        timestamp = format_timestamp(segment.start)
        output_lines.append(f"[{timestamp}] {segment.text}")

    # Close any remaining ad
    if inside_ad and current_ad_idx < len(ad_spans):
        ad = ad_spans[current_ad_idx]
        output_lines.append("")
        output_lines.append(f"### END AD [{format_timestamp(ad.start)} - {format_timestamp(ad.end)}] (confidence: {ad.confidence:.0%}) ###")
        output_lines.append("")

    return "\n".join(output_lines)


def generate_summary(result: DetectionResult) -> str:
    """Generate a summary of detected ads.

    Args:
        result: The detection result.

    Returns:
        A formatted summary string.
    """
    lines = [
        f"AdNihilator Detection Summary",
        f"=" * 40,
        f"File: {result.audio_path}",
        f"Duration: {format_timestamp(result.duration)}",
        f"Total segments: {len(result.segments)}",
        f"Ad candidates (heuristic): {len(result.candidates)}",
        f"Final ad spans: {len(result.ad_spans)}",
        "",
    ]

    if result.ad_spans:
        lines.append("Detected Advertisements:")
        lines.append("-" * 40)

        total_ad_time = 0.0
        for i, span in enumerate(sorted(result.ad_spans, key=lambda s: s.start), 1):
            duration = span.end - span.start
            total_ad_time += duration
            lines.append(
                f"{i}. {format_timestamp(span.start)} - {format_timestamp(span.end)} "
                f"({duration:.0f}s, {span.confidence:.0%} confidence)"
            )
            lines.append(f"   Reason: {span.reason}")

        lines.append("")
        lines.append(f"Total ad time: {format_timestamp(total_ad_time)} ({total_ad_time/result.duration*100:.1f}% of episode)")
    else:
        lines.append("No advertisements detected.")

    return "\n".join(lines)
