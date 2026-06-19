"""Regression tests for Tier 1 ad accuracy improvements."""

from adnihilator.models import AdSpan, Sponsor, SponsorInfo, TranscriptSegment
from worker.daemon import WorkerDaemon


def _seg(index: int, start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(index=index, start=start, end=end, text=text)


def test_verify_long_span_splits_scattered_evidence_into_subspans() -> None:
    """Two sponsor mentions minutes apart must not bridge the conversation between."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    sponsors = SponsorInfo(
        sponsors=[Sponsor(name="Acme"), Sponsor(name="Globex")],
        extraction_method="patterns",
    )
    segments = [
        _seg(0, 0.0, 20.0, "This episode is brought to you by Acme. Use code SAVE."),
        _seg(1, 20.0, 200.0, "Now back to the show, we were discussing the weather and sports."),
        _seg(2, 200.0, 400.0, "More normal conversation about movies and books and travel plans."),
        _seg(3, 400.0, 420.0, "Our sponsor Globex makes great products. Check them out."),
    ]
    # One over-expanded 420s span covering both mentions plus the chatter between.
    spans = [AdSpan(start=0.0, end=420.0, confidence=0.8, reason="Keywords: sponsor")]

    result = daemon._verify_long_ad_spans(spans, segments, sponsors)

    assert len(result) == 2, "scattered evidence should split into two clusters"
    # The long stretch of normal conversation (20-400s) must not be fully removed.
    assert result[0].end < 200.0
    assert result[1].start > 200.0


def test_verify_long_span_keeps_continuous_host_read_intact() -> None:
    """A genuinely long, continuous host-read ad should stay a single span."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    sponsors = SponsorInfo(sponsors=[Sponsor(name="Meter")], extraction_method="patterns")
    segments = [
        _seg(i, i * 30.0, i * 30.0 + 30.0, "Let me tell you about Meter, our sponsor. Brought to you by Meter.")
        for i in range(10)  # 0..300s, continuous evidence every segment
    ]
    spans = [AdSpan(start=0.0, end=300.0, confidence=0.9, reason="Gemini: Meter host_read")]

    result = daemon._verify_long_ad_spans(spans, segments, sponsors)

    assert len(result) == 1
    assert result[0].end - result[0].start > 240.0


def test_apply_gemini_boundaries_snaps_overextended_keyword_span() -> None:
    """A keyword span overlapping a tighter Gemini candidate adopts Gemini's edges."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    spans = [AdSpan(start=100.0, end=300.0, confidence=0.6, reason="Keywords", sources=["keywords"])]
    gemini = [{"start": 130.0, "end": 210.0, "confidence": 1.0, "ad_type": "host_read"}]

    result = daemon._apply_gemini_boundaries(spans, gemini)

    assert len(result) == 1
    assert result[0].start == 130.0
    assert result[0].end == 210.0
    assert "gemini" in result[0].sources


def test_apply_gemini_boundaries_ignores_distant_candidate() -> None:
    """Non-overlapping Gemini candidates leave keyword spans unchanged."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    spans = [AdSpan(start=100.0, end=200.0, confidence=0.6, reason="Keywords", sources=["keywords"])]
    gemini = [{"start": 5000.0, "end": 5100.0, "confidence": 1.0, "ad_type": "host_read"}]

    result = daemon._apply_gemini_boundaries(spans, gemini)

    assert result[0].start == 100.0
    assert result[0].end == 200.0


def test_merge_overlapping_spans_collapses_adjacent() -> None:
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    spans = [
        AdSpan(start=0.0, end=50.0, confidence=0.5, reason="a"),
        AdSpan(start=50.5, end=80.0, confidence=0.9, reason="b"),
        AdSpan(start=200.0, end=250.0, confidence=0.7, reason="c"),
    ]
    result = daemon._merge_overlapping_spans(spans)
    assert len(result) == 2
    assert result[0].start == 0.0 and result[0].end == 80.0
    assert result[0].confidence == 0.9
