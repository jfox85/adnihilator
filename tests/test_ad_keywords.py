"""Tests for heuristic ad detection."""

import pytest

from adnihilator.ad_keywords import (
    AD_SCORE_THRESHOLD,
    CATEGORY_WEIGHTS,
    KEYWORD_PATTERNS,
    STRONG_AD_PATTERNS,
    find_ad_candidates,
    score_segment,
)
from adnihilator.models import SponsorInfo, Sponsor, TranscriptSegment


class TestKeywordPatterns:
    def test_patterns_have_categories(self):
        """All patterns should have a valid category."""
        for pattern, category in KEYWORD_PATTERNS:
            assert isinstance(pattern, str)
            assert len(pattern) > 0
            assert category in CATEGORY_WEIGHTS

    def test_category_weights_are_valid(self):
        """Category weights should be between 0 and 1."""
        for category, weight in CATEGORY_WEIGHTS.items():
            assert 0 < weight <= 1

    def test_strong_patterns_are_in_keyword_patterns(self):
        """All strong patterns should also be in KEYWORD_PATTERNS."""
        all_patterns = [p for p, _ in KEYWORD_PATTERNS]
        for strong in STRONG_AD_PATTERNS:
            assert strong in all_patterns


class TestScoreSegment:
    def test_no_keywords_zero_score(self):
        """Segment with no keywords should have low score."""
        segment = TranscriptSegment(
            index=0,
            start=100.0,
            end=105.0,
            text="This is a regular conversation about the weather.",
        )
        score, triggers, is_strong, sponsors_found = score_segment(segment, 3600.0)
        assert score == 0.0
        assert triggers == []
        assert is_strong is False
        assert sponsors_found == []

    def test_strong_sponsor_keyword(self):
        """Strong sponsor keywords should produce high scores and is_strong=True."""
        segment = TranscriptSegment(
            index=0,
            start=100.0,
            end=105.0,
            text="This episode is brought to you by Acme Corp.",
        )
        score, triggers, is_strong, sponsors_found = score_segment(segment, 3600.0)
        assert score >= 0.4
        assert "brought to you by" in triggers
        assert is_strong is True
        assert sponsors_found == []

    def test_promo_code_keyword(self):
        """Promo code keywords should be detected but not be strong."""
        segment = TranscriptSegment(
            index=0,
            start=100.0,
            end=105.0,
            text="Use code PODCAST for 20% off.",
        )
        score, triggers, is_strong, sponsors_found = score_segment(segment, 3600.0)
        assert score > 0
        assert "use code" in triggers
        assert is_strong is False
        assert sponsors_found == []

    def test_url_pattern(self):
        """URL patterns should be detected."""
        segment = TranscriptSegment(
            index=0,
            start=100.0,
            end=105.0,
            text="Visit example.com/podcast for more info.",
        )
        score, triggers, is_strong, sponsors_found = score_segment(segment, 3600.0)
        assert ".com/" in triggers
        assert sponsors_found == []

    def test_intro_positional_boost(self):
        """Early segments with sponsor keywords get boosted."""
        # Early segment with a CTA keyword
        early_segment = TranscriptSegment(
            index=0,
            start=30.0,  # Within first 90 seconds
            end=35.0,
            text="Go to example.com for this episode.",
        )
        early_score, _, _, _ = score_segment(early_segment, 3600.0)

        # Same content but later in episode (outside intro and midroll zones)
        late_segment = TranscriptSegment(
            index=100,
            start=1000.0,  # Much later, not in midroll zone either
            end=1005.0,
            text="Go to example.com for this episode.",
        )
        late_score, _, _, _ = score_segment(late_segment, 3600.0)

        # Early segment should have higher score due to positional boost
        assert early_score > late_score

    def test_midroll_positional_boost(self):
        """Segments with keywords in the middle of episode get boosted."""
        duration = 3600.0  # 1 hour
        # Midpoint is 1800s, range is 1440-2160

        mid_segment = TranscriptSegment(
            index=50,
            start=1800.0,  # Exact middle
            end=1805.0,
            text="Use code SAVE for discount.",  # Has a keyword
        )
        mid_score, _, _, _ = score_segment(mid_segment, duration)

        edge_segment = TranscriptSegment(
            index=50,
            start=500.0,  # Not in mid range
            end=505.0,
            text="Use code SAVE for discount.",  # Same keyword
        )
        edge_score, _, _, _ = score_segment(edge_segment, duration)

        # Mid segment should have higher score
        assert mid_score > edge_score

    def test_score_clamped_to_one(self):
        """Score should never exceed 1.0."""
        # Segment with many keywords
        segment = TranscriptSegment(
            index=0,
            start=30.0,  # Early for boost
            end=35.0,
            text="This episode is brought to you by our sponsor. Use promo code SAVE at example.com/offer for a special offer and free trial.",
        )
        score, _, _, _ = score_segment(segment, 3600.0)
        assert score <= 1.0


class TestFindAdCandidates:
    def test_empty_segments(self):
        """Empty segment list should return no candidates."""
        candidates = find_ad_candidates([], 3600.0)
        assert candidates == []

    def test_no_ads_detected(self):
        """Regular conversation should produce pre-roll candidate for LLM review."""
        segments = [
            TranscriptSegment(index=i, start=i * 10.0, end=(i + 1) * 10.0, text=text)
            for i, text in enumerate(
                [
                    "Welcome to the show.",
                    "Today we're talking about science.",
                    "Our guest is a researcher.",
                    "Thanks for listening.",
                ]
            )
        ]
        candidates = find_ad_candidates(segments, 600.0)
        # Should have pre-roll region candidate (LLM will determine if it's a house ad)
        assert len(candidates) == 1
        assert "pre_roll_region" in candidates[0].trigger_keywords

    def test_single_ad_candidate_with_extension(self):
        """Single strong ad indicator should create extended candidate."""
        segments = [
            TranscriptSegment(
                index=0, start=0.0, end=10.0, text="Welcome to the show."
            ),
            TranscriptSegment(
                index=1,
                start=10.0,
                end=20.0,
                text="This episode is brought to you by Acme Corp.",
            ),
            TranscriptSegment(
                index=2, start=20.0, end=30.0, text="They make great products."
            ),
            TranscriptSegment(
                index=3, start=30.0, end=40.0, text="Use code SAVE for discount."
            ),
            TranscriptSegment(
                index=4, start=40.0, end=50.0, text="Now back to our topic."
            ),
        ]
        # With 60s extension, should capture segments after the trigger
        candidates = find_ad_candidates(segments, 600.0, extend_before=5.0, extend_after=60.0)
        assert len(candidates) == 1
        # Should include the trigger segment and segments within extension
        assert 1 in candidates[0].segment_indices

    def test_overlapping_spans_merged(self):
        """Overlapping extended spans should be merged."""
        segments = [
            TranscriptSegment(
                index=0, start=0.0, end=10.0, text="Welcome to the show."
            ),
            TranscriptSegment(
                index=1,
                start=10.0,
                end=20.0,
                text="This episode is brought to you by Acme Corp.",
            ),
            TranscriptSegment(
                index=2,
                start=50.0,
                end=60.0,
                text="Also sponsored by Beta Inc.",  # Within 60s of first
            ),
            TranscriptSegment(
                index=3, start=100.0, end=110.0, text="Back to our topic."
            ),
        ]
        candidates = find_ad_candidates(segments, 600.0, extend_before=5.0, extend_after=60.0)
        # Both triggers are within 60s, so should merge into one
        assert len(candidates) == 1

    def test_separate_candidates_when_far_apart(self):
        """Ad segments far apart should produce separate candidates."""
        # Two ad segments far apart with no intervening content
        # (no segments between them means sliding window can't bleed keywords)
        segments = [
            TranscriptSegment(
                index=0,
                start=10.0,
                end=20.0,
                text="This episode brought to you by Sponsor A.",
            ),
            TranscriptSegment(
                index=1,
                start=500.0,
                end=510.0,
                text="Also sponsored by Sponsor B.",
            ),
        ]
        # With 60s extension (10-80s and 495-570s), these should be separate
        candidates = find_ad_candidates(segments, 600.0, extend_before=5.0, extend_after=60.0)
        assert len(candidates) == 2
        # Verify they're actually separate spans
        assert candidates[0].end < candidates[1].start

    def test_trigger_keywords_collected(self):
        """Trigger keywords should be collected in candidate."""
        segments = [
            TranscriptSegment(
                index=0,
                start=10.0,
                end=20.0,
                text="Brought to you by Acme. Use promo code SAVE.",
            ),
        ]
        candidates = find_ad_candidates(segments, 600.0)

        assert len(candidates) == 1
        keywords = candidates[0].trigger_keywords
        assert "brought to you by" in keywords
        assert "promo code" in keywords

    def test_candidate_includes_extended_segments(self):
        """Candidate should include segment indices within extended range."""
        segments = [
            TranscriptSegment(index=0, start=0.0, end=10.0, text="Intro."),
            TranscriptSegment(
                index=1,
                start=10.0,
                end=20.0,
                text="This episode is brought to you by Acme.",
            ),
            TranscriptSegment(index=2, start=20.0, end=30.0, text="Product description."),
            TranscriptSegment(index=3, start=30.0, end=40.0, text="More about Acme."),
            TranscriptSegment(index=4, start=40.0, end=50.0, text="Visit acme.com today."),
            TranscriptSegment(index=5, start=50.0, end=60.0, text="Back to the show."),
        ]
        candidates = find_ad_candidates(segments, 600.0, extend_before=5.0, extend_after=40.0)

        assert len(candidates) >= 1
        # Should include segments 1-4 (within 40s extension)
        indices = candidates[0].segment_indices
        assert 1 in indices
        assert 2 in indices
        assert 3 in indices


class TestSponsorIntegration:
    def test_sponsor_name_boosts_score(self):
        """Sponsor name in transcript should boost ad score."""
        segment = TranscriptSegment(
            index=0,
            start=100.0,
            end=105.0,
            text="And now a word from ExpressVPN, the best VPN service.",
        )
        sponsors = SponsorInfo(
            sponsors=[Sponsor(name="ExpressVPN", url="expressvpn.com/podcast")],
            extraction_method="patterns"
        )

        score, triggers, is_strong, sponsors_found = score_segment(segment, 3600.0, sponsors=sponsors)

        assert score > 0
        assert any("expressvpn" in t.lower() for t in triggers)
        assert "ExpressVPN" in sponsors_found

    def test_promo_code_is_strong_signal(self):
        """Promo code mention should be a strong ad signal."""
        segment = TranscriptSegment(
            index=0,
            start=100.0,
            end=105.0,
            text="Use code TWIT at checkout for 20% off.",
        )
        sponsors = SponsorInfo(
            sponsors=[Sponsor(name="Canary", code="TWIT")],
            extraction_method="patterns"
        )

        score, triggers, is_strong, sponsors_found = score_segment(segment, 3600.0, sponsors=sponsors)

        assert score >= 0.3  # At least the sponsor match weight
        assert is_strong  # Promo codes are strong signals
        assert any("twit" in t.lower() for t in triggers)

    def test_find_candidates_tracks_found_sponsors(self):
        """find_ad_candidates should track which sponsors were found."""
        segments = [
            TranscriptSegment(index=0, start=0, end=5, text="Welcome to the show."),
            TranscriptSegment(index=1, start=60, end=70,
                text="This episode is brought to you by ExpressVPN."),
            TranscriptSegment(index=2, start=70, end=80, text="Back to our topic."),
        ]
        sponsors = SponsorInfo(
            sponsors=[
                Sponsor(name="ExpressVPN"),
                Sponsor(name="Squarespace"),  # Not mentioned
            ],
            extraction_method="patterns"
        )

        candidates = find_ad_candidates(segments, 3600.0, sponsors=sponsors)

        assert len(candidates) >= 1
        # Check sponsor tracking in candidates
        all_found = set()
        for c in candidates:
            all_found.update(c.sponsors_found)

        assert "ExpressVPN" in all_found

    def test_find_candidates_tracks_missing_sponsors(self):
        """find_ad_candidates should track which sponsors were NOT found."""
        segments = [
            TranscriptSegment(index=0, start=60, end=70,
                text="This episode is brought to you by ExpressVPN."),
        ]
        sponsors = SponsorInfo(
            sponsors=[
                Sponsor(name="ExpressVPN"),
                Sponsor(name="Squarespace"),  # Not mentioned
            ],
            extraction_method="patterns"
        )

        candidates = find_ad_candidates(segments, 3600.0, sponsors=sponsors)

        assert len(candidates) >= 1
        # Missing sponsors should be tracked
        all_missing = set()
        for c in candidates:
            all_missing.update(c.sponsors_missing)

        assert "Squarespace" in all_missing
