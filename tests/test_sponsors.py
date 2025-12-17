"""Tests for sponsor extraction from show notes."""

import pytest
from unittest.mock import Mock, patch

from adnihilator.sponsors import extract_sponsors, extract_sponsors_with_patterns, generate_sponsor_keywords
from adnihilator.models import Sponsor, SponsorInfo


class TestPatternExtraction:
    def test_twit_html_format(self):
        """Extract sponsors from TWiT-style HTML."""
        description = '''
        <p><strong>Sponsors:</strong><ul>
        <li><a href="http://miro.com" rel="sponsored">miro.com</a></li>
        <li><a href="http://expressvpn.com/twit" rel="sponsored">expressvpn.com/twit</a></li>
        <li><a href="http://canary.tools/twit" rel="sponsored">canary.tools/twit - use code: TWIT</a></li>
        </ul></p>
        '''
        sponsors = extract_sponsors_with_patterns(description)

        assert len(sponsors) == 3
        names = [s.name for s in sponsors]
        assert "miro" in [n.lower() for n in names]
        assert "expressvpn" in [n.lower() for n in names]
        assert "canary" in [n.lower() for n in names]

        # Check code extraction
        canary = next(s for s in sponsors if "canary" in s.name.lower())
        assert canary.code == "TWIT"

    def test_lex_fridman_format(self):
        """Extract sponsors from Lex Fridman-style format."""
        description = '''
        <p><b>SPONSORS:</b><br />
        <b>Shopify:</b> Sell stuff online.<br />
        Go to <a href="https://shopify.com/lex">https://shopify.com/lex</a><br />
        <b>LMNT:</b> Zero-sugar electrolyte drink mix.<br />
        Go to <a href="https://drinkLMNT.com/lex">https://drinkLMNT.com/lex</a><br />
        '''
        sponsors = extract_sponsors_with_patterns(description)

        assert len(sponsors) >= 2
        names_lower = [s.name.lower() for s in sponsors]
        assert "shopify" in names_lower
        assert "lmnt" in names_lower

    def test_megaphone_plain_text_format(self):
        """Extract sponsors from Megaphone plain text format."""
        description = '''
        Partner Deals
        Copilot: Free 2 months access with code HACKS2
        NetSuite: Free KPI checklist
        DeleteMe: 20% off with code DELETE20
        '''
        sponsors = extract_sponsors_with_patterns(description)

        assert len(sponsors) >= 2

        copilot = next((s for s in sponsors if "copilot" in s.name.lower()), None)
        assert copilot is not None
        assert copilot.code == "HACKS2"

    def test_no_sponsors_returns_empty(self):
        """Description without sponsors returns empty list."""
        description = "This is a podcast about technology and science."
        sponsors = extract_sponsors_with_patterns(description)
        assert sponsors == []

    def test_simple_sponsors_list(self):
        """Extract from simple comma-separated list."""
        description = "Sponsors: Squarespace, HelloFresh, BetterHelp"
        sponsors = extract_sponsors_with_patterns(description)

        assert len(sponsors) == 3
        names_lower = [s.name.lower() for s in sponsors]
        assert "squarespace" in names_lower
        assert "hellofresh" in names_lower
        assert "betterhelp" in names_lower


class TestLLMExtraction:
    def test_llm_fallback_triggered_when_patterns_fail(self):
        """LLM fallback runs when patterns find nothing but keywords present."""
        description = "Thanks to our sponsors for making this episode possible."

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content='{"sponsors": [{"name": "Acme", "url": "acme.com"}]}'))]
        )

        with patch('adnihilator.sponsors._extract_sponsors_with_llm') as mock_llm:
            mock_llm.return_value = [Sponsor(name="Acme", url="acme.com")]

            result = extract_sponsors(description, llm_client=mock_client)

            mock_llm.assert_called_once()
            assert result.extraction_method == "llm"
            assert len(result.sponsors) == 1

    def test_no_llm_when_no_trigger_keywords(self):
        """LLM not called when no sponsor keywords in description."""
        description = "This episode discusses artificial intelligence."

        mock_client = Mock()
        result = extract_sponsors(description, llm_client=mock_client)

        assert result.extraction_method == "none"
        assert result.sponsors == []

    def test_llm_extraction_with_json_response(self):
        """Test LLM extraction with a valid JSON response."""
        from adnihilator.sponsors import _extract_sponsors_with_llm

        description = "Special thanks to our partners who made this possible."

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='{"sponsors": [{"name": "TechCorp", "url": "techcorp.com/podcast", "code": "TECH50"}]}'))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        sponsors = _extract_sponsors_with_llm(description, mock_client)

        assert len(sponsors) == 1
        assert sponsors[0].name == "TechCorp"
        assert sponsors[0].url == "techcorp.com/podcast"
        assert sponsors[0].code == "TECH50"

    def test_llm_extraction_with_markdown_wrapped_json(self):
        """Test LLM extraction when response is wrapped in markdown code blocks."""
        from adnihilator.sponsors import _extract_sponsors_with_llm

        description = "Our sponsors this week include some amazing companies."

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='```json\n{"sponsors": [{"name": "DataCo", "url": "dataco.io"}]}\n```'))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        sponsors = _extract_sponsors_with_llm(description, mock_client)

        assert len(sponsors) == 1
        assert sponsors[0].name == "DataCo"
        assert sponsors[0].url == "dataco.io"

    def test_llm_extraction_with_multiple_sponsors(self):
        """Test LLM extraction with multiple sponsors."""
        from adnihilator.sponsors import _extract_sponsors_with_llm

        description = "Thanks to all our partners."

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='{"sponsors": [{"name": "VPN Pro", "code": "SAVE20"}, {"name": "CloudHost", "url": "cloudhost.com"}]}'))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        sponsors = _extract_sponsors_with_llm(description, mock_client)

        assert len(sponsors) == 2
        assert sponsors[0].name == "VPN Pro"
        assert sponsors[0].code == "SAVE20"
        assert sponsors[1].name == "CloudHost"
        assert sponsors[1].url == "cloudhost.com"

    def test_llm_extraction_handles_empty_response(self):
        """Test LLM extraction handles empty sponsor list."""
        from adnihilator.sponsors import _extract_sponsors_with_llm

        description = "No actual sponsors here."

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"sponsors": []}'))]
        mock_client.chat.completions.create.return_value = mock_response

        sponsors = _extract_sponsors_with_llm(description, mock_client)

        assert len(sponsors) == 0

    def test_llm_extraction_handles_api_error(self):
        """Test LLM extraction returns empty list on API error."""
        from adnihilator.sponsors import _extract_sponsors_with_llm

        description = "Some description"

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        sponsors = _extract_sponsors_with_llm(description, mock_client)

        assert len(sponsors) == 0

    def test_llm_extraction_handles_invalid_json(self):
        """Test LLM extraction returns empty list on invalid JSON."""
        from adnihilator.sponsors import _extract_sponsors_with_llm

        description = "Some description"

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='This is not valid JSON'))]
        mock_client.chat.completions.create.return_value = mock_response

        sponsors = _extract_sponsors_with_llm(description, mock_client)

        assert len(sponsors) == 0


class TestKeywordGeneration:
    def test_name_variations(self):
        """Generate name variations for matching."""
        sponsor = Sponsor(name="ExpressVPN")
        keywords = generate_sponsor_keywords(sponsor)

        assert "expressvpn" in keywords
        assert "express vpn" in keywords  # CamelCase split

    def test_url_slug_extraction(self):
        """Extract URL components as keywords."""
        sponsor = Sponsor(name="Shopify", url="shopify.com/lex")
        keywords = generate_sponsor_keywords(sponsor)

        assert "shopify" in keywords
        assert "shopify.com/lex" in keywords
        assert "shopify dot com slash lex" in keywords

    def test_promo_code_keyword(self):
        """Include promo code as keyword."""
        sponsor = Sponsor(name="Canary", code="TWIT")
        keywords = generate_sponsor_keywords(sponsor)

        assert "twit" in keywords  # Function returns lowercase
        assert "code twit" in keywords

    def test_lmnt_special_case(self):
        """Handle acronym-style names."""
        sponsor = Sponsor(name="LMNT")
        keywords = generate_sponsor_keywords(sponsor)

        assert "lmnt" in keywords
        # May also include "l m n t" or "element" variations

    def test_get_all_sponsor_keywords(self):
        """Test batch keyword generation."""
        from adnihilator.sponsors import get_all_sponsor_keywords

        sponsors = [
            Sponsor(name="ExpressVPN", code="LEX"),
            Sponsor(name="Shopify", url="shopify.com/lex"),
        ]

        result = get_all_sponsor_keywords(sponsors)

        assert len(result) == 2
        assert "ExpressVPN" in result
        assert "Shopify" in result
        assert "expressvpn" in result["ExpressVPN"]
        assert "shopify" in result["Shopify"]

    def test_empty_name_returns_empty_list(self):
        """Empty sponsor name returns empty keyword list."""
        sponsor = Sponsor(name="")
        keywords = generate_sponsor_keywords(sponsor)
        assert keywords == []
