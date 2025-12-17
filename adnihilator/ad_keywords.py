"""Heuristic-based advertisement detection using keyword patterns."""

from .models import AdCandidate, TranscriptSegment, SponsorInfo
from .sponsors import generate_sponsor_keywords

# Keyword patterns with their categories
# Format: (pattern, category)
KEYWORD_PATTERNS: list[tuple[str, str]] = [
    # Strong sponsor indicators - these trigger span extension
    ("brought to you by", "intro_sponsor"),
    ("our sponsor today", "intro_sponsor"),
    ("today's sponsor", "intro_sponsor"),
    ("this episode is sponsored", "intro_sponsor"),
    ("sponsored by", "intro_sponsor"),
    ("thanks to our sponsor", "intro_sponsor"),
    ("thank our sponsor", "intro_sponsor"),
    ("word from our sponsor", "intro_sponsor"),
    ("message from our sponsor", "intro_sponsor"),
    # Promo codes and CTAs
    ("promo code", "promo_code"),
    ("use code", "promo_code"),
    ("discount code", "promo_code"),
    ("coupon code", "promo_code"),
    ("special offer", "promo_code"),
    ("exclusive offer", "promo_code"),
    ("limited time", "promo_code"),
    # Call to action
    ("go to", "cta"),
    ("head over to", "cta"),
    ("check out", "cta"),
    ("sign up at", "cta"),
    ("get started at", "cta"),
    # URL patterns
    (".com/", "url"),
    (".io/", "url"),
    (".co/", "url"),
    (".org/", "url"),
    (".net/", "url"),
    # Pricing and offers
    ("free trial", "offer"),
    ("money back", "offer"),
    ("percent off", "offer"),
    ("% off", "offer"),
    ("first month free", "offer"),
    ("risk free", "offer"),
    ("money back guarantee", "offer"),
]

# Category weights for scoring
CATEGORY_WEIGHTS: dict[str, float] = {
    "intro_sponsor": 0.5,  # Strong indicator
    "promo_code": 0.3,
    "cta": 0.15,
    "url": 0.1,
    "offer": 0.25,
}

# Strong intro patterns that definitely indicate an ad start
STRONG_AD_PATTERNS = [
    "brought to you by",
    "our sponsor today",
    "today's sponsor",
    "this episode is sponsored",
    "sponsored by",
    "word from our sponsor",
    "message from our sponsor",
]

# Threshold for considering a segment "ad-like"
AD_SCORE_THRESHOLD = 0.3

# How far to extend from a keyword hit (in seconds)
EXTENSION_BEFORE = 5.0  # Small lookback for intro
EXTENSION_AFTER = 60.0  # Extend forward to capture full ad

# Pre-roll region settings - always check first 2 minutes for house ads and network promos
PRE_ROLL_REGION_DURATION = 120.0  # First 2 minutes
PRE_ROLL_MIN_EPISODE_LENGTH = 180.0  # Only add pre-roll region for episodes > 3 minutes

# Outro region settings - always check final minutes for post-roll ads
OUTRO_REGION_DURATION = 120.0  # Last 2 minutes (matches pre-roll)
OUTRO_MIN_EPISODE_LENGTH = 180.0  # Only add outro region for episodes > 3 minutes


def score_segment(
    segment: TranscriptSegment,
    duration: float,
    context_text: str = "",
    sponsors: SponsorInfo | None = None,
    podcast_name: str | None = None,
) -> tuple[float, list[str], bool, list[str]]:
    """Score a transcript segment for ad likelihood.

    Args:
        segment: The transcript segment to score.
        duration: Total duration of the audio in seconds.
        context_text: Optional combined text from adjacent segments for cross-boundary matching.
        sponsors: Optional sponsor information for enhanced detection.
        podcast_name: Optional podcast name to filter promo codes that match the podcast name.

    Returns:
        A tuple of (score, trigger keywords, is_strong_indicator, sponsors_found).
    """
    text = segment.text.lower()
    # Use context text if provided (for cross-segment matching)
    search_text = context_text.lower() if context_text else text

    score = 0.0
    triggers: list[str] = []
    is_strong = False
    sponsors_found: list[str] = []

    # Check for keyword matches
    for pattern, category in KEYWORD_PATTERNS:
        if pattern in search_text:
            weight = CATEGORY_WEIGHTS.get(category, 0.1)
            score += weight
            triggers.append(pattern)
            if pattern in STRONG_AD_PATTERNS:
                is_strong = True

    # Check for sponsor-specific keywords
    if sponsors and sponsors.sponsors:
        for sponsor in sponsors.sponsors:
            keywords = generate_sponsor_keywords(sponsor, podcast_name)
            for keyword in keywords:
                if keyword in search_text:
                    score += 0.3  # Sponsor match weight
                    triggers.append(f"sponsor:{keyword}")
                    if sponsor.name not in sponsors_found:
                        sponsors_found.append(sponsor.name)
                    # Promo codes are strong signals
                    if sponsor.code and sponsor.code.lower() in keyword:
                        is_strong = True
                    break  # One match per sponsor is enough

    # Positional boosts
    # Intro ads (first 90 seconds)
    if segment.start < 90:
        has_sponsor_or_cta = any(
            cat in ["intro_sponsor", "cta"]
            for pattern, cat in KEYWORD_PATTERNS
            if pattern in search_text
        )
        if has_sponsor_or_cta:
            score += 0.2

    # Midroll ads (middle 20% of episode, i.e., 40% to 60%)
    midpoint_start = duration * 0.4
    midpoint_end = duration * 0.6
    if midpoint_start <= segment.start <= midpoint_end:
        if triggers:  # Only boost if we found something
            score += 0.15

    # Outro ads (last 90 seconds)
    if segment.start > (duration - 90):
        if triggers:
            score += 0.1

    # Clamp score to [0, 1]
    score = min(max(score, 0.0), 1.0)

    return score, triggers, is_strong, sponsors_found


def find_ad_candidates(
    segments: list[TranscriptSegment],
    duration: float,
    extend_before: float = EXTENSION_BEFORE,
    extend_after: float = EXTENSION_AFTER,
    sponsors: SponsorInfo | None = None,
    podcast_name: str | None = None,
) -> list[AdCandidate]:
    """Find advertisement candidates from transcript segments.

    Uses a two-phase approach:
    1. Find segments with strong ad indicators (e.g., "brought to you by")
    2. Extend those segments forward/backward to capture the full ad
    3. Merge overlapping extended spans

    Args:
        segments: List of transcript segments.
        duration: Total duration of the audio in seconds.
        extend_before: Seconds to extend before a keyword hit.
        extend_after: Seconds to extend after a keyword hit.
        sponsors: Optional sponsor information for enhanced detection.
        podcast_name: Optional podcast name to filter promo codes that match the podcast name.

    Returns:
        List of AdCandidate objects representing potential ad segments.
    """
    if not segments:
        return []

    # Sort segments by start time for sliding window
    sorted_segments = sorted(segments, key=lambda s: s.start)

    # Phase 1: Score all segments with sliding window context
    # This catches phrases split across segment boundaries like "brought to you" | "by"
    scored: list[tuple[TranscriptSegment, float, list[str], bool, list[str]]] = []

    for i, seg in enumerate(sorted_segments):
        # Build context from current segment plus 1 before and 1 after
        context_parts = []
        if i > 0:
            context_parts.append(sorted_segments[i - 1].text)
        context_parts.append(seg.text)
        if i < len(sorted_segments) - 1:
            context_parts.append(sorted_segments[i + 1].text)

        context_text = " ".join(context_parts)

        score, triggers, is_strong, sponsors_found = score_segment(
            seg, duration, context_text, sponsors, podcast_name
        )
        scored.append((seg, score, triggers, is_strong, sponsors_found))

    # Track all found sponsors across all segments
    all_sponsors_found: set[str] = set()
    for _, _, _, _, sponsors_found in scored:
        all_sponsors_found.update(sponsors_found)

    # Calculate missing sponsors
    sponsors_missing: list[str] = []
    if sponsors and sponsors.sponsors:
        all_sponsor_names = {s.name for s in sponsors.sponsors}
        sponsors_missing = sorted(all_sponsor_names - all_sponsors_found)

    # Phase 2: Create extended spans from strong indicators
    raw_spans: list[tuple[float, float, list[int], list[str], float, list[str]]] = []

    for seg, score, triggers, is_strong, sponsors_found in scored:
        if is_strong or score >= AD_SCORE_THRESHOLD:
            # Extend the span
            span_start = max(0, seg.start - extend_before)
            span_end = min(duration, seg.end + extend_after)

            # Find all segment indices within this span
            indices_in_span = [
                s.index for s in segments
                if s.start >= span_start and s.end <= span_end
            ]

            raw_spans.append((span_start, span_end, indices_in_span, triggers, score, sponsors_found))

    # Phase 3: Merge overlapping spans
    # Note: Don't return early if no raw_spans - we still want to add pre-roll/outro regions
    merged_spans: list[tuple[float, float, list[int], list[str], float, list[str]]] = []

    if raw_spans:
        # Sort by start time
        raw_spans.sort(key=lambda x: x[0])

        current = raw_spans[0]
        for span in raw_spans[1:]:
            # Check for overlap (spans overlap if one starts before the other ends)
            if span[0] <= current[1]:
                # Merge: extend end, combine indices and triggers, take max score
                combined_indices = list(set(current[2] + span[2]))
                combined_indices.sort()
                combined_triggers = list(dict.fromkeys(current[3] + span[3]))  # Unique, preserve order
                max_score = max(current[4], span[4])
                combined_sponsors = list(dict.fromkeys(current[5] + span[5]))  # Unique sponsors
                current = (
                    current[0],
                    max(current[1], span[1]),
                    combined_indices,
                    combined_triggers,
                    max_score,
                    combined_sponsors,
                )
            else:
                merged_spans.append(current)
                current = span

        merged_spans.append(current)

    # Phase 4: Create AdCandidate objects
    candidates: list[AdCandidate] = []
    for start, end, indices, triggers, score, sponsors_found in merged_spans:
        candidates.append(
            AdCandidate(
                start=start,
                end=end,
                segment_indices=indices,
                trigger_keywords=triggers,
                heuristic_score=score,
                sponsors_found=sponsors_found,
                sponsors_missing=sponsors_missing,
            )
        )

    # Phase 5: Add dedicated pre-roll candidate region for house ads and network promos
    # Pre-rolls often promote other shows or network content rather than sponsors
    if duration >= PRE_ROLL_MIN_EPISODE_LENGTH:
        pre_roll_start = 0.0
        pre_roll_end = min(PRE_ROLL_REGION_DURATION, duration)

        # Check if we already have ANY candidate in the pre-roll region (substantial overlap)
        # Only add pre-roll candidate if there's minimal coverage (<30% overlap)
        total_overlap = sum(
            max(0, min(c.end, pre_roll_end) - max(c.start, pre_roll_start))
            for c in candidates
        )
        pre_roll_covered = total_overlap > (pre_roll_end - pre_roll_start) * 0.3

        if not pre_roll_covered:
            # Find segment indices in the pre-roll region
            pre_roll_indices = [
                s.index for s in segments
                if s.start <= pre_roll_end and s.end >= pre_roll_start
            ]

            if pre_roll_indices:  # Only add if there are segments in this region
                candidates.append(
                    AdCandidate(
                        start=pre_roll_start,
                        end=pre_roll_end,
                        segment_indices=pre_roll_indices,
                        trigger_keywords=["pre_roll_region"],
                        heuristic_score=0.3,  # Moderate score - LLM determines if it's house ad
                        sponsors_found=[],
                        sponsors_missing=sponsors_missing,
                    )
                )

    # Phase 6: Add dedicated outro candidate region for post-roll ads
    # Post-rolls often promote other shows or have dynamically inserted ads that lack keyword triggers
    if duration >= OUTRO_MIN_EPISODE_LENGTH:
        outro_start = duration - OUTRO_REGION_DURATION
        outro_end = duration

        # Check if we already have ANY candidate in the outro region (substantial overlap)
        # Only add outro candidate if there's minimal coverage (<30% overlap)
        total_overlap = sum(
            max(0, min(c.end, outro_end) - max(c.start, outro_start))
            for c in candidates
        )
        outro_covered = total_overlap > (outro_end - outro_start) * 0.3

        if not outro_covered:
            # Find segment indices in the outro region
            outro_indices = [
                s.index for s in segments
                if s.start >= outro_start or s.end >= outro_start
            ]

            if outro_indices:  # Only add if there are segments in this region
                candidates.append(
                    AdCandidate(
                        start=outro_start,
                        end=outro_end,
                        segment_indices=outro_indices,
                        trigger_keywords=["outro_region"],
                        heuristic_score=0.3,  # Moderate score - LLM will decide (matches pre-roll)
                        sponsors_found=[],
                        sponsors_missing=sponsors_missing,
                    )
                )

    return candidates
