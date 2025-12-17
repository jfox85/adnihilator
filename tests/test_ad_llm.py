"""Tests for LLM-based advertisement refinement."""

import pytest

from adnihilator.ad_llm import merge_nearby_candidates
from adnihilator.models import AdCandidate


class TestMergeNearbyCandidates:
    """Tests for merge_nearby_candidates function."""

    def test_preserves_sponsor_fields_when_merging(self):
        """Test that sponsor tracking fields are preserved during merge."""
        # Create candidates with sponsor information
        candidate1 = AdCandidate(
            start=10.0,
            end=20.0,
            segment_indices=[0, 1],
            trigger_keywords=["sponsor", "brought to you by"],
            heuristic_score=0.8,
            sponsors_found=["Athletic Greens", "BetterHelp"],
            sponsors_missing=["HelloFresh"],
        )

        candidate2 = AdCandidate(
            start=25.0,  # Close enough to merge (within 300s default threshold)
            end=35.0,
            segment_indices=[2, 3],
            trigger_keywords=["promo code"],
            heuristic_score=0.7,
            sponsors_found=["HelloFresh"],  # This sponsor was missing in candidate1
            sponsors_missing=["Shopify"],
        )

        # Merge candidates
        merged = merge_nearby_candidates([candidate1, candidate2])

        # Should have one merged candidate
        assert len(merged) == 1
        indices, merged_candidate = merged[0]

        # Check that all sponsors are combined and deduplicated
        assert set(merged_candidate.sponsors_found) == {"Athletic Greens", "BetterHelp", "HelloFresh"}
        assert set(merged_candidate.sponsors_missing) == {"HelloFresh", "Shopify"}

        # Check that order is preserved (dict.fromkeys maintains insertion order)
        assert merged_candidate.sponsors_found == ["Athletic Greens", "BetterHelp", "HelloFresh"]
        assert merged_candidate.sponsors_missing == ["HelloFresh", "Shopify"]

        # Verify other fields are still merged correctly
        assert merged_candidate.start == 10.0
        assert merged_candidate.end == 35.0
        assert set(merged_candidate.segment_indices) == {0, 1, 2, 3}
        assert merged_candidate.heuristic_score == 0.8

    def test_preserves_empty_sponsor_lists(self):
        """Test that empty sponsor lists are handled correctly."""
        candidate1 = AdCandidate(
            start=10.0,
            end=20.0,
            segment_indices=[0],
            trigger_keywords=["sponsor"],
            heuristic_score=0.8,
            sponsors_found=[],
            sponsors_missing=[],
        )

        candidate2 = AdCandidate(
            start=25.0,
            end=35.0,
            segment_indices=[1],
            trigger_keywords=["promo"],
            heuristic_score=0.7,
            sponsors_found=[],
            sponsors_missing=[],
        )

        merged = merge_nearby_candidates([candidate1, candidate2])
        assert len(merged) == 1
        indices, merged_candidate = merged[0]

        assert merged_candidate.sponsors_found == []
        assert merged_candidate.sponsors_missing == []

    def test_deduplicates_sponsors(self):
        """Test that duplicate sponsors are removed during merge."""
        candidate1 = AdCandidate(
            start=10.0,
            end=20.0,
            segment_indices=[0],
            trigger_keywords=["sponsor"],
            heuristic_score=0.8,
            sponsors_found=["Athletic Greens", "BetterHelp"],
            sponsors_missing=["HelloFresh"],
        )

        candidate2 = AdCandidate(
            start=25.0,
            end=35.0,
            segment_indices=[1],
            trigger_keywords=["promo"],
            heuristic_score=0.7,
            sponsors_found=["Athletic Greens"],  # Duplicate
            sponsors_missing=["HelloFresh"],  # Duplicate
        )

        merged = merge_nearby_candidates([candidate1, candidate2])
        assert len(merged) == 1
        indices, merged_candidate = merged[0]

        # Should have deduplicated lists
        assert merged_candidate.sponsors_found == ["Athletic Greens", "BetterHelp"]
        assert merged_candidate.sponsors_missing == ["HelloFresh"]

    def test_candidates_too_far_apart_not_merged(self):
        """Test that distant candidates preserve their own sponsor info."""
        candidate1 = AdCandidate(
            start=10.0,
            end=20.0,
            segment_indices=[0],
            trigger_keywords=["sponsor"],
            heuristic_score=0.8,
            sponsors_found=["Athletic Greens"],
            sponsors_missing=[],
        )

        candidate2 = AdCandidate(
            start=400.0,  # Far away (> 300s threshold)
            end=410.0,
            segment_indices=[1],
            trigger_keywords=["promo"],
            heuristic_score=0.7,
            sponsors_found=["BetterHelp"],
            sponsors_missing=[],
        )

        merged = merge_nearby_candidates([candidate1, candidate2])

        # Should have two separate candidates
        assert len(merged) == 2

        # Each should preserve its own sponsors
        indices1, merged1 = merged[0]
        assert merged1.sponsors_found == ["Athletic Greens"]

        indices2, merged2 = merged[1]
        assert merged2.sponsors_found == ["BetterHelp"]
