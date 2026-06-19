"""Regression tests for no-inference ad accuracy improvements."""

from adnihilator.ad_keywords import score_segment
from adnihilator.models import AdSpan, Sponsor, SponsorInfo, TranscriptSegment, WordTimestamp
from adnihilator.splice import _snap_span_to_transcript
from worker.daemon import WorkerDaemon


def test_sponsor_keyword_does_not_match_inside_ordinary_words() -> None:
    """Short sponsor names like Scribe should not match describe/subscribed."""
    segment = TranscriptSegment(
        index=0,
        start=100.0,
        end=110.0,
        text="We describe how subscribed terminals change personal computing.",
    )
    sponsors = SponsorInfo(
        sponsors=[Sponsor(name="Scribe", url="https://scribe.com")],
        extraction_method="patterns",
    )

    score, triggers, _, sponsors_found = score_segment(segment, 3600.0, sponsors=sponsors)

    assert score == 0.0
    assert triggers == []
    assert sponsors_found == []


def test_gemini_timestamp_rescue_shifts_to_nearby_brand_evidence() -> None:
    """Misaligned Gemini dynamic ads are shifted to nearby transcript evidence."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    segments = [
        TranscriptSegment(index=0, start=8240.0, end=8268.0, text="Walmart Business saves your company time. Learn more at business.walmart.com."),
        TranscriptSegment(index=1, start=8269.0, end=8300.0, text="Fred Meyer pickup has fresh groceries and free delivery. Terms apply."),
        TranscriptSegment(index=2, start=8316.0, end=8330.0, text="Vine is back. Did you see this?"),
        TranscriptSegment(index=3, start=8340.0, end=8360.0, text="This is regular show discussion."),
    ]
    candidate = {
        "start": 8320.0,
        "end": 8400.0,
        "confidence": 1.0,
        "reason": "Gemini chunked: Walmart Business/Fred Meyer Pickup (en) - dynamic_insertion",
        "source": "gemini",
        "ad_type": "dynamic_insertion",
    }

    validated, rejected = daemon._validate_gemini_candidates([candidate], segments, [], duration=9500.0)

    assert rejected == []
    assert len(validated) == 1
    assert validated[0]["start"] < 8241.0
    assert 8299.0 <= validated[0]["end"] <= 8302.0
    assert "timestamp rescued" in validated[0]["reason"]


def test_splice_boundary_snap_moves_mid_word_cuts_to_word_edges() -> None:
    """Cut boundaries inside words are expanded to word edges with padding."""
    segments = [
        TranscriptSegment(
            index=0,
            start=9.5,
            end=12.0,
            text="hello sponsor world",
            words=[
                WordTimestamp(word="hello", start=9.5, end=10.0, probability=1.0),
                WordTimestamp(word="sponsor", start=10.1, end=10.9, probability=1.0),
                WordTimestamp(word="world", start=11.0, end=11.5, probability=1.0),
            ],
        )
    ]
    span = AdSpan(start=10.4, end=11.2, confidence=1.0, reason="test")

    snapped = _snap_span_to_transcript(span, segments, duration=60.0)

    assert snapped.start < 10.1
    assert snapped.end > 11.5
