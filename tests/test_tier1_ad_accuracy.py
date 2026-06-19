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


def test_apply_gemini_boundaries_is_shrink_only() -> None:
    """Gemini boundaries wider than the span must not expand it."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    spans = [AdSpan(start=100.0, end=200.0, confidence=0.6, reason="Keywords", sources=["keywords"])]
    gemini = [{"start": 50.0, "end": 300.0, "confidence": 1.0, "ad_type": "host_read"}]

    result = daemon._apply_gemini_boundaries(spans, gemini)

    # Span unchanged: never grows beyond its original bounds.
    assert result[0].start == 100.0
    assert result[0].end == 200.0


def test_apply_gemini_boundaries_never_emits_out_of_range() -> None:
    """A negative/over-range Gemini candidate cannot push a span out of range."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    spans = [AdSpan(start=10.0, end=20.0, confidence=0.6, reason="k", sources=["keywords"])]
    gemini = [{"start": -5.0, "end": 25.0, "confidence": 1.0, "ad_type": "dynamic_insertion"}]

    result = daemon._apply_gemini_boundaries(spans, gemini)

    assert result[0].start == 10.0
    assert result[0].end == 20.0


def test_apply_gemini_boundaries_does_not_collapse_two_spans() -> None:
    """One wide Gemini candidate overlapping two spans must keep them separate."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    spans = [
        AdSpan(start=100.0, end=140.0, confidence=0.6, reason="a", sources=["keywords"]),
        AdSpan(start=160.0, end=200.0, confidence=0.6, reason="b", sources=["keywords"]),
    ]
    gemini = [{"start": 120.0, "end": 180.0, "confidence": 1.0, "ad_type": "host_read"}]

    result = daemon._apply_gemini_boundaries(spans, gemini)
    result = daemon._merge_overlapping_spans(result)

    assert len(result) == 2, "shrink-only snapping must not collapse distinct spans"
    assert result[0].end <= 140.0
    assert result[1].start >= 160.0


def test_apply_gemini_boundaries_requires_min_overlap() -> None:
    """A barely-overlapping Gemini candidate is not treated as the same ad."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    spans = [AdSpan(start=100.0, end=200.0, confidence=0.6, reason="k", sources=["keywords"])]
    # Overlap is only 100-105 = 5s of a 100s span (5%), below min_overlap 25%.
    gemini = [{"start": 5.0, "end": 105.0, "confidence": 1.0, "ad_type": "host_read"}]

    result = daemon._apply_gemini_boundaries(spans, gemini)

    assert result[0].start == 100.0
    assert result[0].end == 200.0


def test_verify_long_span_splits_three_clusters() -> None:
    """Three scattered sponsor mentions produce three sub-spans."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    sponsors = SponsorInfo(sponsors=[Sponsor(name="Acme")], extraction_method="patterns")
    segments = []
    # Evidence at 0, 300, 600s with 200s+ evidence-free gaps between.
    for cluster_i, base in enumerate((0.0, 300.0, 600.0)):
        segments.append(_seg(cluster_i * 3, base, base + 20.0, "Brought to you by Acme."))
        segments.append(_seg(cluster_i * 3 + 1, base + 20.0, base + 200.0, "Normal show talk about many unrelated things here."))
    spans = [AdSpan(start=0.0, end=800.0, confidence=0.8, reason="Keywords: sponsor")]

    result = daemon._verify_long_ad_spans(spans, segments, sponsors)

    assert len(result) == 3


def test_verify_long_span_gap_boundary_stays_merged() -> None:
    """Evidence exactly at the max gap stays one cluster (<=, not >)."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    sponsors = SponsorInfo(sponsors=[Sponsor(name="Acme")], extraction_method="patterns")
    # Two evidence segments whose gap is exactly 60s (the default max).
    segments = [
        _seg(0, 0.0, 20.0, "Brought to you by Acme."),
        _seg(1, 20.0, 80.0, "some filler conversation in the middle of the read here."),
        _seg(2, 80.0, 300.0, "Acme is our sponsor, check them out now please."),
    ]
    spans = [AdSpan(start=0.0, end=300.0, confidence=0.8, reason="k")]

    result = daemon._verify_long_ad_spans(spans, segments, sponsors)

    assert len(result) == 1


def test_verify_long_span_rejects_no_evidence() -> None:
    """A long span with no ad evidence is dropped entirely."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    sponsors = SponsorInfo(sponsors=[Sponsor(name="Acme")], extraction_method="patterns")
    segments = [_seg(0, 0.0, 300.0, "Just a long stretch of normal conversation with no ads.")]
    spans = [AdSpan(start=0.0, end=300.0, confidence=0.8, reason="k")]

    result = daemon._verify_long_ad_spans(spans, segments, sponsors)

    assert result == []


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


def test_merge_overlapping_spans_unions_provenance() -> None:
    """Merging must not drop a later span's source or ad_type."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    spans = [
        AdSpan(start=0.0, end=50.0, confidence=0.5, reason="a", sources=["keywords"], ad_type=None),
        AdSpan(start=50.5, end=80.0, confidence=0.9, reason="b", sources=["gemini"], ad_type="host_read"),
    ]
    result = daemon._merge_overlapping_spans(spans)
    assert len(result) == 1
    assert set(result[0].sources) == {"keywords", "gemini"}
    assert result[0].ad_type == "host_read"


def test_merge_overlapping_spans_keeps_separated() -> None:
    """Spans just over the gap threshold are not merged."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    spans = [
        AdSpan(start=0.0, end=50.0, confidence=0.5, reason="a"),
        AdSpan(start=51.5, end=80.0, confidence=0.9, reason="b"),  # gap 1.5s > 1.0
    ]
    result = daemon._merge_overlapping_spans(spans)
    assert len(result) == 2


def test_merge_overlapping_spans_drops_degenerate() -> None:
    """Zero/negative-length spans must be filtered out before splicing."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    spans = [
        AdSpan(start=10.0, end=10.0, confidence=0.9, reason="zero"),
        AdSpan(start=30.0, end=20.0, confidence=0.9, reason="inverted"),
        AdSpan(start=40.0, end=60.0, confidence=0.9, reason="ok"),
    ]
    result = daemon._merge_overlapping_spans(spans)
    assert len(result) == 1
    assert result[0].start == 40.0 and result[0].end == 60.0


def test_final_review_skipped_on_clean_result() -> None:
    """All sponsors covered, no anomaly/hunt/long span -> skip the LLM review."""
    spans = [AdSpan(start=0.0, end=60.0, confidence=0.9, reason="ad")]
    validation_info = {"total_sponsors": 2, "covered_count": 2, "missing_count": 0}
    assert WorkerDaemon._needs_final_review(spans, [], False, validation_info) is False


def test_final_review_triggers_on_each_signal() -> None:
    short = [AdSpan(start=0.0, end=60.0, confidence=0.9, reason="ad")]
    long = [AdSpan(start=0.0, end=400.0, confidence=0.9, reason="ad")]
    clean = {"total_sponsors": 1, "covered_count": 1, "missing_count": 0}

    # Missing sponsors
    assert WorkerDaemon._needs_final_review(short, ["Acme"], False, clean) is True
    # Hunt added spans
    assert WorkerDaemon._needs_final_review(short, [], True, clean) is True
    # Anomaly flagged
    assert WorkerDaemon._needs_final_review(short, [], False, {"anomaly": "more_ads_than_expected"}) is True
    # No sponsors in description (coverage uncheckable)
    assert WorkerDaemon._needs_final_review(short, [], False, {"no_sponsors_in_description": True}) is True
    # Long span remains
    assert WorkerDaemon._needs_final_review(long, [], False, clean) is True
