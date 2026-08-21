"""Regression tests for no-inference ad accuracy improvements."""

from types import SimpleNamespace

import pytest

from adnihilator.ad_llm import OpenAIClient
from adnihilator.ad_keywords import find_ad_candidates, score_segment
from adnihilator.models import AdSpan, Sponsor, SponsorInfo, TranscriptSegment, WordTimestamp
from adnihilator.splice import _snap_span_to_transcript
from worker.daemon import DetectionSafetyError, WorkerDaemon


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


def test_gemini_host_read_is_relocated_even_when_old_buffer_would_validate() -> None:
    """A shifted host read is relocated instead of accepted by a broad buffer."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    segments = [
        TranscriptSegment(
            index=0,
            start=2877.0,
            end=2885.0,
            text="Finding other entrepreneurs to learn from is difficult.",
        ),
        TranscriptSegment(
            index=1,
            start=2885.0,
            end=2891.0,
            text="I started a company called Hampton. Check it out at joinhampton.com.",
        ),
        TranscriptSegment(
            index=2,
            start=2912.0,
            end=2922.0,
            text="If you are a founder, check out joinhampton.com.",
        ),
        TranscriptSegment(
            index=3,
            start=2932.0,
            end=3000.0,
            text="The guest explains how Uber expanded the taxi market.",
        ),
    ]
    candidate = {
        "start": 2938.0,
        "end": 3000.0,
        "confidence": 1.0,
        "reason": "Gemini chunked: Podcast Network/Hampton (en) - network_bumper",
        "source": "gemini",
        "ad_type": "network_bumper",
    }

    validated, rejected = daemon._validate_gemini_candidates(
        [candidate], segments, [], duration=4687.0
    )

    assert rejected == []
    assert len(validated) == 1
    assert validated[0]["start"] < 2890.0
    assert 2920.0 < validated[0]["end"] < 2930.0
    assert "timestamp rescued" in validated[0]["reason"]


def test_gemini_host_read_duration_mismatch_is_annotated() -> None:
    """Large reported/localized length disagreement is visible in evidence."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    segments = [
        TranscriptSegment(
            index=0,
            start=1372.0,
            end=1374.0,
            text="Let's take a quick break.",
        ),
        TranscriptSegment(
            index=1,
            start=1382.0,
            end=1400.0,
            text="HubSpot can help. Check out HubSpot.com.",
        ),
        TranscriptSegment(
            index=2,
            start=1451.0,
            end=1500.0,
            text="Normal discussion of an economic principle.",
        ),
    ]
    candidate = {
        "start": 1451.0,
        "end": 1500.0,
        "confidence": 1.0,
        "reason": "Gemini chunked: HubSpot (en) - host_read",
        "source": "gemini",
        "ad_type": "host_read",
    }

    validated, rejected = daemon._validate_gemini_candidates(
        [candidate], segments, [], duration=3173.0
    )

    assert rejected == []
    assert validated[0]["start"] < 1372.0
    assert validated[0]["end"] <= 1401.0
    assert "duration mismatch" in validated[0]["reason"]


def test_simple_merge_does_not_expand_gemini_ad_to_full_outro() -> None:
    """A generic outro hint cannot inherit confidence from a short Gemini ad."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    gemini = [{
        "start": 4663.0,
        "end": 4687.0,
        "confidence": 1.0,
        "reason": "Gemini: Success Story",
        "source": "gemini",
        "ad_type": "host_read",
    }]
    keywords = [{
        "start": 4567.0,
        "end": 4687.0,
        "confidence": 0.3,
        "reason": "Keywords: outro_region",
        "matched_keywords": ["outro_region"],
        "source": "keywords",
    }]

    spans = daemon._simple_merge_candidates(gemini, keywords)

    assert len(spans) == 1
    assert spans[0].start == 4663.0
    assert spans[0].end == 4687.0


def test_llm_merge_failure_aborts_instead_of_using_unsafe_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A required merge failure must stop processing before audio is spliced."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    daemon.config = SimpleNamespace(
        llm=SimpleNamespace(model="gpt-4o-mini"),
    )
    client = OpenAIClient(api_key="test")

    def fail_merge(*args: object, **kwargs: object) -> None:
        raise ImportError("wrong architecture")

    monkeypatch.setattr(client, "merge_and_refine", fail_merge)
    monkeypatch.setattr("worker.daemon.create_llm_client", lambda config: client)

    with pytest.raises(DetectionSafetyError, match="refusing to splice"):
        daemon._llm_merge_candidates(
            gemini_candidates=[{
                "start": 10.0,
                "end": 30.0,
                "confidence": 1.0,
                "reason": "Gemini: Acme",
                "source": "gemini",
            }],
            keyword_candidates=[],
            segments=[
                TranscriptSegment(
                    index=0,
                    start=10.0,
                    end=30.0,
                    text="Acme sponsors this episode.",
                )
            ],
            sponsors=SponsorInfo(sponsors=[], extraction_method="none"),
            duration=60.0,
        )


def test_quick_break_host_read_becomes_keyword_candidate() -> None:
    """An explicit ad-break transition is strong enough to preserve recall."""
    segments = [
        TranscriptSegment(
            index=0,
            start=1444.0,
            end=1446.0,
            text="Hey, let's take a quick break.",
        ),
        TranscriptSegment(
            index=1,
            start=1446.0,
            end=1472.0,
            text="The Breeze assistant from HubSpot can help. Check out HubSpot.com.",
        ),
        TranscriptSegment(
            index=2,
            start=1472.0,
            end=1480.0,
            text="Back to the interview.",
        ),
    ]

    candidates = find_ad_candidates(segments, duration=4687.0)

    assert any(
        candidate.start <= 1444.0
        and candidate.end >= 1472.0
        and "let's take a quick break" in candidate.trigger_keywords
        for candidate in candidates
    )


def test_candidate_context_preserves_late_episode_candidate_when_truncated() -> None:
    """A post-roll candidate remains visible when the context budget is small."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    segments = [
        TranscriptSegment(
            index=0,
            start=0.0,
            end=100.0,
            text="Early candidate " + ("intro " * 100),
        ),
        TranscriptSegment(
            index=1,
            start=1500.0,
            end=1600.0,
            text="Middle candidate " + ("middle " * 100),
        ),
        TranscriptSegment(
            index=2,
            start=3148.0,
            end=3173.0,
            text="Late Success Story post-roll wherever you get your podcasts.",
        ),
    ]
    candidates = [
        {"start": 0.0, "end": 100.0},
        {"start": 1500.0, "end": 1600.0},
        {"start": 3148.0, "end": 3173.0},
    ]

    context = daemon._get_combined_transcript_context(
        segments, candidates, max_chars=600
    )

    assert len(context) <= 600
    assert "Early candidate" in context
    assert "Middle candidate" in context
    assert "Success Story post-roll" in context


def test_explicit_transcript_lane_finds_ads_gemini_omitted() -> None:
    """Explicit intros plus CTAs recover omitted mid-roll and post-roll ads."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    segments = [
        TranscriptSegment(
            index=0,
            start=2336.0,
            end=2339.0,
            text="Today's podcast is brought to you by my friends at Mercury.",
        ),
        TranscriptSegment(
            index=1,
            start=2375.0,
            end=2387.0,
            text="Go to Mercury.com and learn more. Services provided by members FDIC.",
        ),
        TranscriptSegment(
            index=2,
            start=3148.0,
            end=3151.0,
            text="All right, let's take a quick break to talk about a podcast.",
        ),
        TranscriptSegment(
            index=3,
            start=3157.0,
            end=3173.0,
            text="Success Story is great. Check it out wherever you get your podcasts.",
        ),
    ]

    spans = daemon._find_explicit_transcript_ad_spans(segments, duration=3173.0)

    assert len(spans) == 2
    assert spans[0].start < 2336.0 and spans[0].end > 2387.0
    assert spans[1].start < 3148.0 and spans[1].end == 3173.0
    assert all("transcript_explicit" in span.sources for span in spans)


def test_explicit_transcript_lane_rejects_intro_only_url() -> None:
    """A URL in the intro alone does not prove a complete removable ad."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    segments = [
        TranscriptSegment(
            index=0,
            start=100.0,
            end=104.0,
            text="Today's podcast is brought to you by example.com.",
        ),
        TranscriptSegment(
            index=1,
            start=104.0,
            end=112.0,
            text="Now let us return to the interview.",
        ),
    ]

    spans = daemon._find_explicit_transcript_ad_spans(segments, duration=600.0)

    assert spans == []


def test_parallel_detection_failures_stop_splicing() -> None:
    """A failed required detection lane must fail closed."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)

    with pytest.raises(DetectionSafetyError, match="Gemini detection failed"):
        daemon._ensure_parallel_detection_succeeded("malformed JSON", None)

    with pytest.raises(DetectionSafetyError, match="transcript/keyword"):
        daemon._ensure_parallel_detection_succeeded(None, "transcriber crashed")

    daemon._ensure_parallel_detection_succeeded(None, None)


def test_explicit_transcript_span_replaces_broad_overlapping_guess() -> None:
    """Exact transcript evidence shrinks a destructive broad keyword span."""
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    broad = [AdSpan(
        start=3053.0,
        end=3173.0,
        confidence=0.3,
        reason="outro region",
        sources=["keywords"],
    )]
    explicit = [AdSpan(
        start=3147.5,
        end=3173.0,
        confidence=0.9,
        reason="explicit",
        sources=["transcript_explicit"],
    )]

    result = daemon._apply_explicit_transcript_spans(broad, explicit)

    assert len(result) == 1
    assert result[0].start == 3147.5
    assert result[0].end == 3173.0
    assert set(result[0].sources) == {"keywords", "transcript_explicit"}


def test_house_promo_with_check_out_and_url_becomes_candidate() -> None:
    """A promotional URL combination catches house ads without sponsor intros."""
    segments = [
        TranscriptSegment(
            index=0,
            start=2885.0,
            end=2891.0,
            text="I started a company called Hampton. Check it out at joinhampton.com.",
        ),
        TranscriptSegment(
            index=1,
            start=2912.0,
            end=2922.0,
            text="Check out joinhampton.com. Again, the URL is joinhampton.com.",
        ),
        TranscriptSegment(
            index=2,
            start=2922.0,
            end=2930.0,
            text="You said something fascinating about competing differently.",
        ),
    ]

    candidates = find_ad_candidates(segments, duration=4687.0)

    assert any(
        candidate.start <= 2885.0
        and candidate.end >= 2922.0
        and "check out" in candidate.trigger_keywords
        and ".com" in candidate.trigger_keywords
        for candidate in candidates
    )


def test_house_promo_with_free_guide_and_link_becomes_candidate() -> None:
    """A free-resource pitch with link/QR CTAs is retained for refinement."""
    segments = [
        TranscriptSegment(
            index=0,
            start=536.0,
            end=548.0,
            text="The team at Starter Story put together a free guide.",
        ),
        TranscriptSegment(
            index=1,
            start=548.0,
            end=558.0,
            text="It contains tactics for getting attention for your business.",
        ),
        TranscriptSegment(
            index=2,
            start=558.0,
            end=568.0,
            text="Click the link in the description or scan the QR code.",
        ),
    ]

    candidates = find_ad_candidates(segments, duration=4687.0)

    assert any(
        candidate.start <= 558.0
        and candidate.end >= 568.0
        and "free guide" in candidate.trigger_keywords
        and "click the link" in candidate.trigger_keywords
        for candidate in candidates
    )


def test_generic_sponsoring_discussion_is_not_a_strong_ad_intro() -> None:
    """Ordinary discussion of event sponsorship should not start an ad."""
    segments = [
        TranscriptSegment(
            index=0,
            start=600.0,
            end=608.0,
            text="The company was criticized for sponsoring the controversial event.",
        ),
    ]

    candidates = find_ad_candidates(segments, duration=3600.0)

    assert candidates == []


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
