"""Integration tests for sponsor extraction pipeline."""

import pytest

from adnihilator.sponsors import extract_sponsors
from adnihilator.ad_keywords import find_ad_candidates
from adnihilator.models import TranscriptSegment, Sponsor, SponsorInfo


class TestSponsorPipeline:
    def test_full_pipeline_with_twit_format(self):
        """Test full pipeline with TWiT-style show notes."""
        description = '''
        <p><strong>Sponsors:</strong><ul>
        <li><a href="http://expressvpn.com/twit">expressvpn.com/twit</a></li>
        <li><a href="http://canary.tools/twit">canary.tools/twit - use code: TWIT</a></li>
        </ul></p>
        '''

        # Extract sponsors
        sponsor_info = extract_sponsors(description)
        assert len(sponsor_info.sponsors) == 2
        assert sponsor_info.extraction_method == "patterns"

        # Simulate transcript with sponsor mention
        segments = [
            TranscriptSegment(index=0, start=0, end=30,
                text="Welcome to the show, let's get started."),
            TranscriptSegment(index=1, start=60, end=90,
                text="This episode is brought to you by ExpressVPN."),
            TranscriptSegment(index=2, start=90, end=120,
                text="Go to expressvpn.com/twit to get started."),
            TranscriptSegment(index=3, start=500, end=530,
                text="Thanks to Canary for sponsoring. Use code TWIT."),
            TranscriptSegment(index=4, start=530, end=560,
                text="Visit canary.tools/twit for more info."),
            TranscriptSegment(index=5, start=600, end=630,
                text="Back to our main topic for today."),
        ]

        # Find candidates with sponsor info
        candidates = find_ad_candidates(segments, 3600.0, sponsors=sponsor_info)

        # Should find ad segments
        assert len(candidates) >= 1

        # At least one sponsor should be found
        all_found = set()
        for c in candidates:
            all_found.update(c.sponsors_found)

        assert len(all_found) >= 1

    def test_pipeline_tracks_missing_sponsors(self):
        """Pipeline should track sponsors not found in transcript."""
        description = '''
        <p><strong>Sponsors:</strong><ul>
        <li><a href="http://sponsor1.com">sponsor1.com</a></li>
        <li><a href="http://sponsor2.com">sponsor2.com</a></li>
        <li><a href="http://sponsor3.com">sponsor3.com</a></li>
        </ul></p>
        '''

        sponsor_info = extract_sponsors(description)
        assert len(sponsor_info.sponsors) == 3

        # Only mention one sponsor
        segments = [
            TranscriptSegment(index=0, start=60, end=90,
                text="This episode is brought to you by Sponsor1."),
        ]

        candidates = find_ad_candidates(segments, 3600.0, sponsors=sponsor_info)

        # Should have missing sponsors tracked
        assert len(candidates) >= 1

        # Check that missing sponsors are tracked
        all_missing = set()
        for c in candidates:
            all_missing.update(c.sponsors_missing)

        assert len(all_missing) >= 2  # sponsor2 and sponsor3

    def test_pipeline_with_lex_fridman_format(self):
        """Test extraction from Lex Fridman-style format."""
        description = '''
        <p><b>SPONSORS:</b><br />
        <b>Shopify:</b> Sell stuff online.<br />
        Go to <a href="https://shopify.com/lex">https://shopify.com/lex</a><br />
        <b>LMNT:</b> Zero-sugar electrolyte drink mix.<br />
        Go to <a href="https://drinkLMNT.com/lex">https://drinkLMNT.com/lex</a><br />
        </p>
        '''

        sponsor_info = extract_sponsors(description)
        assert len(sponsor_info.sponsors) >= 2

        names_lower = [s.name.lower() for s in sponsor_info.sponsors]
        assert "shopify" in names_lower
        assert "lmnt" in names_lower

    def test_pipeline_with_promo_codes(self):
        """Test that promo codes boost detection."""
        sponsor_info = SponsorInfo(
            sponsors=[Sponsor(name="TestSponsor", code="PODCAST20")],
            extraction_method="patterns"
        )

        # Segment mentions promo code
        segments = [
            TranscriptSegment(index=0, start=100, end=130,
                text="Use code PODCAST20 at checkout for twenty percent off."),
        ]

        candidates = find_ad_candidates(segments, 3600.0, sponsors=sponsor_info)

        # Should find this as a candidate due to promo code match
        assert len(candidates) >= 1

        # At least one trigger should mention the promo code
        all_triggers = []
        for c in candidates:
            all_triggers.extend(c.trigger_keywords)

        assert any("podcast20" in t.lower() for t in all_triggers)

    def test_no_sponsors_returns_empty(self):
        """Description with no sponsor info returns empty list."""
        description = "This is a tech podcast about artificial intelligence and machine learning."

        sponsor_info = extract_sponsors(description)

        assert sponsor_info.sponsors == []
        assert sponsor_info.extraction_method == "none"

    def test_empty_description(self):
        """Empty description handled gracefully."""
        sponsor_info = extract_sponsors("")

        assert sponsor_info.sponsors == []
        assert sponsor_info.extraction_method == "none"

    def test_none_description(self):
        """None description handled gracefully."""
        sponsor_info = extract_sponsors(None)

        assert sponsor_info.sponsors == []
        assert sponsor_info.extraction_method == "none"
