# Sponsor Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract sponsor info from episode show notes to improve ad detection accuracy.

**Architecture:** Hybrid extraction (regex patterns + LLM fallback), passed to both keyword detector and LLM refiner. Hunt mode searches gaps when sponsors are missing.

**Tech Stack:** Python, Pydantic, OpenAI API, pytest

---

## Task 1: Add Sponsor Models

**Files:**
- Modify: `adnihilator/models.py`
- Test: `tests/test_models.py`

**Step 1: Write failing test for Sponsor model**

Add to `tests/test_models.py`:

```python
class TestSponsorModels:
    def test_sponsor_with_all_fields(self):
        """Sponsor with name, url, and code."""
        from adnihilator.models import Sponsor

        sponsor = Sponsor(name="ExpressVPN", url="expressvpn.com/twit", code="TWIT")
        assert sponsor.name == "ExpressVPN"
        assert sponsor.url == "expressvpn.com/twit"
        assert sponsor.code == "TWIT"

    def test_sponsor_name_only(self):
        """Sponsor with just a name."""
        from adnihilator.models import Sponsor

        sponsor = Sponsor(name="Miro")
        assert sponsor.name == "Miro"
        assert sponsor.url is None
        assert sponsor.code is None

    def test_sponsor_info(self):
        """SponsorInfo container."""
        from adnihilator.models import Sponsor, SponsorInfo

        info = SponsorInfo(
            sponsors=[Sponsor(name="Shopify", url="shopify.com/lex")],
            extraction_method="patterns"
        )
        assert len(info.sponsors) == 1
        assert info.extraction_method == "patterns"

    def test_sponsor_info_empty(self):
        """SponsorInfo with no sponsors."""
        from adnihilator.models import SponsorInfo

        info = SponsorInfo(sponsors=[], extraction_method="none")
        assert info.sponsors == []
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_models.py::TestSponsorModels -v
```

Expected: FAIL with `ImportError: cannot import name 'Sponsor'`

**Step 3: Implement Sponsor and SponsorInfo models**

Add to `adnihilator/models.py` after `AdSpan`:

```python
class Sponsor(BaseModel):
    """A sponsor extracted from episode show notes."""

    name: str
    url: Optional[str] = None
    code: Optional[str] = None


class SponsorInfo(BaseModel):
    """Sponsor information extracted from episode description."""

    sponsors: list[Sponsor]
    extraction_method: str  # "patterns", "llm", or "none"
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_models.py::TestSponsorModels -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add adnihilator/models.py tests/test_models.py
git commit -m "feat(models): add Sponsor and SponsorInfo models"
```

---

## Task 2: Create Sponsor Extraction Module - Pattern Matching

**Files:**
- Create: `adnihilator/sponsors.py`
- Create: `tests/test_sponsors.py`

**Step 1: Write failing tests for pattern extraction**

Create `tests/test_sponsors.py`:

```python
"""Tests for sponsor extraction from show notes."""

import pytest

from adnihilator.sponsors import extract_sponsors, extract_sponsors_with_patterns
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_sponsors.py::TestPatternExtraction -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'adnihilator.sponsors'`

**Step 3: Implement pattern extraction**

Create `adnihilator/sponsors.py`:

```python
"""Sponsor extraction from episode show notes."""

import re
from html import unescape
from typing import Optional

from .models import Sponsor, SponsorInfo


# Trigger keywords for LLM fallback
SPONSOR_TRIGGER_KEYWORDS = [
    "sponsor",
    "partner",
    "code",
    "deal",
    "discount",
    "promo",
]


def extract_sponsors_with_patterns(description: str) -> list[Sponsor]:
    """Extract sponsors using regex patterns.

    Handles common formats:
    - HTML lists with <strong>Sponsors:</strong> or <b>SPONSORS:</b>
    - Plain text "Partner Deals" or "Sponsor Deals" sections
    - Simple "Sponsors: Name1, Name2" lists

    Args:
        description: Episode description/show notes HTML or text.

    Returns:
        List of Sponsor objects found.
    """
    if not description:
        return []

    sponsors: list[Sponsor] = []
    seen_names: set[str] = set()

    # Normalize HTML entities
    text = unescape(description)

    # Pattern 1: HTML list items with links after Sponsors header
    # Matches: <li><a href="url">text</a></li>
    sponsor_section = re.search(
        r'<(?:strong|b)>(?:SPONSORS?|Sponsors?):?</(?:strong|b)>(.*?)(?:<(?:strong|b)>|$)',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if sponsor_section:
        section_text = sponsor_section.group(1)
        # Extract links from list items
        links = re.findall(
            r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>',
            section_text,
            re.IGNORECASE
        )
        for url, link_text in links:
            sponsor = _parse_sponsor_from_link(url, link_text)
            if sponsor and sponsor.name.lower() not in seen_names:
                sponsors.append(sponsor)
                seen_names.add(sponsor.name.lower())

    # Pattern 2: Lex Fridman style - <b>Name:</b> description Go to <a href="url">
    lex_pattern = re.findall(
        r'<b>([^:<]+):</b>[^<]*(?:Go to\s*)?<a[^>]+href=["\']([^"\']+)["\']',
        text,
        re.IGNORECASE
    )
    for name, url in lex_pattern:
        name = name.strip()
        if name.upper() not in ('SPONSORS', 'SPONSOR', 'OUTLINE', 'CONTACT LEX',
                                 'TRANSCRIPT', 'EPISODE LINKS', 'CONTACT', 'OTHER'):
            if name.lower() not in seen_names:
                sponsors.append(Sponsor(name=name, url=_clean_url(url)))
                seen_names.add(name.lower())

    # Pattern 3: Plain text "Partner Deals" or "Sponsor Deals" section
    deals_section = re.search(
        r'(?:Partner|Sponsor)\s+Deals?\s*\n(.*?)(?:\n\n|\n[A-Z]|\Z)',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if deals_section:
        section_text = deals_section.group(1)
        # Lines like "Name: description with code CODE"
        lines = section_text.strip().split('\n')
        for line in lines:
            match = re.match(r'^([A-Za-z0-9\s]+):\s*(.*)$', line.strip())
            if match:
                name = match.group(1).strip()
                rest = match.group(2)
                code = _extract_promo_code(rest)
                if name.lower() not in seen_names:
                    sponsors.append(Sponsor(name=name, code=code))
                    seen_names.add(name.lower())

    # Pattern 4: Simple comma-separated list
    simple_list = re.search(
        r'Sponsors?:\s*([A-Za-z0-9,\s]+?)(?:\.|<|\n|$)',
        text,
        re.IGNORECASE
    )
    if simple_list and not sponsors:  # Only if nothing else found
        names = [n.strip() for n in simple_list.group(1).split(',')]
        for name in names:
            if name and len(name) > 1 and name.lower() not in seen_names:
                sponsors.append(Sponsor(name=name))
                seen_names.add(name.lower())

    return sponsors


def _parse_sponsor_from_link(url: str, link_text: str) -> Optional[Sponsor]:
    """Parse sponsor info from a URL and link text."""
    # Extract domain/name from URL
    url_clean = _clean_url(url)

    # Try to get name from URL domain
    domain_match = re.search(r'(?:https?://)?(?:www\.)?([^./]+)', url_clean)
    name = domain_match.group(1) if domain_match else link_text

    # Check for promo code in link text
    code = _extract_promo_code(link_text)

    # Skip if name looks like a generic domain
    if name.lower() in ('http', 'https', 'www', 'com', 'org', 'net'):
        return None

    return Sponsor(name=name.title(), url=url_clean, code=code)


def _clean_url(url: str) -> str:
    """Clean up URL for storage."""
    # Remove tracking redirects like lexfridman.com/s/...
    if '/s/' in url and '-ep' in url:
        # Extract the actual destination from tracking URL text if available
        pass
    return url.strip()


def _extract_promo_code(text: str) -> Optional[str]:
    """Extract promo code from text like 'use code TWIT' or 'code: HACKS2'."""
    patterns = [
        r'(?:use\s+)?code[:\s]+([A-Z0-9]+)',
        r'(?:promo|coupon)[:\s]+([A-Z0-9]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def _should_use_llm_fallback(description: str) -> bool:
    """Check if LLM fallback should be triggered."""
    if not description:
        return False

    text_lower = description.lower()
    return any(keyword in text_lower for keyword in SPONSOR_TRIGGER_KEYWORDS)


def extract_sponsors(
    description: str,
    llm_client=None,
) -> SponsorInfo:
    """Extract sponsors from episode description.

    Uses hybrid approach:
    1. Try regex patterns first
    2. If nothing found but sponsor keywords present, use LLM fallback

    Args:
        description: Episode description/show notes.
        llm_client: Optional LLM client for fallback extraction.

    Returns:
        SponsorInfo with extracted sponsors and method used.
    """
    # Try patterns first
    sponsors = extract_sponsors_with_patterns(description)

    if sponsors:
        return SponsorInfo(sponsors=sponsors, extraction_method="patterns")

    # Check if LLM fallback should be used
    if llm_client and _should_use_llm_fallback(description):
        sponsors = _extract_sponsors_with_llm(description, llm_client)
        if sponsors:
            return SponsorInfo(sponsors=sponsors, extraction_method="llm")

    return SponsorInfo(sponsors=[], extraction_method="none")


def _extract_sponsors_with_llm(description: str, llm_client) -> list[Sponsor]:
    """Extract sponsors using LLM when patterns fail.

    Args:
        description: Episode description text.
        llm_client: OpenAI client instance.

    Returns:
        List of extracted sponsors.
    """
    # LLM extraction implementation - Task 3
    return []
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_sponsors.py::TestPatternExtraction -v
```

Expected: Most tests PASS (some edge cases may need tuning)

**Step 5: Commit**

```bash
git add adnihilator/sponsors.py tests/test_sponsors.py
git commit -m "feat(sponsors): add pattern-based sponsor extraction"
```

---

## Task 3: Add LLM Fallback Extraction

**Files:**
- Modify: `adnihilator/sponsors.py`
- Modify: `tests/test_sponsors.py`

**Step 1: Write failing test for LLM extraction**

Add to `tests/test_sponsors.py`:

```python
from unittest.mock import Mock, patch


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
```

**Step 2: Run test to verify behavior**

```bash
pytest tests/test_sponsors.py::TestLLMExtraction -v
```

**Step 3: Implement LLM extraction**

Update `_extract_sponsors_with_llm` in `adnihilator/sponsors.py`:

```python
import json


LLM_EXTRACTION_PROMPT = '''Extract sponsor/advertiser information from this podcast episode description.

Return a JSON object with this structure:
{
  "sponsors": [
    {"name": "Sponsor Name", "url": "sponsor.com/podcast", "code": "PROMO"}
  ]
}

Rules:
- Only include paid sponsors/advertisers, not general links
- Extract promo codes if mentioned (e.g., "use code SAVE20")
- URL should be the sponsor's URL mentioned, not tracking links
- If no sponsors found, return {"sponsors": []}

Episode description:
"""
{description}
"""'''


def _extract_sponsors_with_llm(description: str, llm_client) -> list[Sponsor]:
    """Extract sponsors using LLM when patterns fail."""
    try:
        response = llm_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": LLM_EXTRACTION_PROMPT.format(description=description[:4000])}
            ],
            temperature=0.1,
            max_tokens=500,
        )

        content = response.choices[0].message.content or ""

        # Parse JSON response
        content = content.strip()
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    content = part
                    break

        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            content = content[start:end]

        data = json.loads(content)
        sponsors = []
        for s in data.get("sponsors", []):
            sponsors.append(Sponsor(
                name=s.get("name", ""),
                url=s.get("url"),
                code=s.get("code"),
            ))
        return sponsors

    except Exception:
        return []
```

**Step 4: Run tests**

```bash
pytest tests/test_sponsors.py -v
```

**Step 5: Commit**

```bash
git add adnihilator/sponsors.py tests/test_sponsors.py
git commit -m "feat(sponsors): add LLM fallback extraction"
```

---

## Task 4: Generate Dynamic Keywords from Sponsors

**Files:**
- Create: function in `adnihilator/sponsors.py`
- Add tests to `tests/test_sponsors.py`

**Step 1: Write failing tests for keyword generation**

Add to `tests/test_sponsors.py`:

```python
from adnihilator.sponsors import generate_sponsor_keywords


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
        assert "shopify.com/lex" in keywords or "shopify dot com slash lex" in keywords

    def test_promo_code_keyword(self):
        """Include promo code as keyword."""
        sponsor = Sponsor(name="Canary", code="TWIT")
        keywords = generate_sponsor_keywords(sponsor)

        assert "twit" in keywords or "TWIT" in keywords
        assert "code twit" in keywords

    def test_lmnt_special_case(self):
        """Handle acronym-style names."""
        sponsor = Sponsor(name="LMNT")
        keywords = generate_sponsor_keywords(sponsor)

        assert "lmnt" in keywords
        # May also include "l m n t" or "element" variations
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_sponsors.py::TestKeywordGeneration -v
```

**Step 3: Implement keyword generation**

Add to `adnihilator/sponsors.py`:

```python
def generate_sponsor_keywords(sponsor: Sponsor) -> list[str]:
    """Generate keyword variations for matching sponsor in transcript.

    Handles:
    - CamelCase splitting (ExpressVPN -> express vpn)
    - URL slug extraction (shopify.com/lex -> shopify, shopify dot com slash lex)
    - Promo codes (code TWIT)

    Args:
        sponsor: Sponsor object to generate keywords for.

    Returns:
        List of lowercase keyword strings.
    """
    keywords: set[str] = set()

    name = sponsor.name

    # Original name (lowercase)
    keywords.add(name.lower())

    # CamelCase splitting: ExpressVPN -> express vpn
    # Insert space before uppercase letters that follow lowercase
    split_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    if split_name != name:
        keywords.add(split_name.lower())

    # Handle all-caps acronyms: LMNT -> l m n t
    if name.isupper() and len(name) <= 5:
        keywords.add(' '.join(name.lower()))

    # URL-based keywords
    if sponsor.url:
        url = sponsor.url.lower()

        # Remove protocol
        url_clean = re.sub(r'^https?://', '', url)

        # Extract domain
        domain_match = re.match(r'([^/]+)', url_clean)
        if domain_match:
            domain = domain_match.group(1)
            # Remove www
            domain = re.sub(r'^www\.', '', domain)
            # Add domain without TLD
            domain_name = domain.split('.')[0]
            keywords.add(domain_name)

        # Full URL path for matching "shopify.com/lex"
        keywords.add(url_clean)

        # Spoken URL: "shopify dot com slash lex"
        spoken = url_clean.replace('.', ' dot ').replace('/', ' slash ')
        keywords.add(spoken.strip())

    # Promo code keywords
    if sponsor.code:
        code = sponsor.code.lower()
        keywords.add(code)
        keywords.add(f"code {code}")
        keywords.add(f"promo code {code}")

    return list(keywords)


def get_all_sponsor_keywords(sponsors: list[Sponsor]) -> dict[str, list[str]]:
    """Generate keywords for all sponsors.

    Args:
        sponsors: List of sponsors.

    Returns:
        Dict mapping sponsor name to list of keywords.
    """
    return {
        sponsor.name: generate_sponsor_keywords(sponsor)
        for sponsor in sponsors
    }
```

**Step 4: Run tests**

```bash
pytest tests/test_sponsors.py::TestKeywordGeneration -v
```

**Step 5: Commit**

```bash
git add adnihilator/sponsors.py tests/test_sponsors.py
git commit -m "feat(sponsors): add keyword generation from sponsor info"
```

---

## Task 5: Integrate Sponsors into Keyword Detection

**Files:**
- Modify: `adnihilator/ad_keywords.py`
- Modify: `tests/test_ad_keywords.py`

**Step 1: Write failing test for sponsor integration**

Add to `tests/test_ad_keywords.py`:

```python
from adnihilator.models import SponsorInfo, Sponsor


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

        score, triggers, is_strong = score_segment(segment, 3600.0, sponsors=sponsors)

        assert score > 0
        assert any("expressvpn" in t.lower() for t in triggers)

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

        score, triggers, is_strong = score_segment(segment, 3600.0, sponsors=sponsors)

        assert score >= AD_SCORE_THRESHOLD
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
        # Check sponsor tracking
        all_found = set()
        all_missing = set()
        for c in candidates:
            all_found.update(c.sponsors_found)
            all_missing.update(c.sponsors_missing)

        assert "ExpressVPN" in all_found or "expressvpn" in [s.lower() for s in all_found]
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_ad_keywords.py::TestSponsorIntegration -v
```

Expected: FAIL - `score_segment` doesn't accept `sponsors` param

**Step 3: Update AdCandidate model**

Modify `adnihilator/models.py` - update `AdCandidate`:

```python
class AdCandidate(BaseModel):
    """A candidate advertisement segment identified by heuristics."""

    start: float
    end: float
    segment_indices: list[int]
    trigger_keywords: list[str]
    heuristic_score: float
    sponsors_found: list[str] = []
    sponsors_missing: list[str] = []
```

**Step 4: Update ad_keywords.py**

Modify `adnihilator/ad_keywords.py`:

Add import at top:
```python
from .models import AdCandidate, TranscriptSegment, SponsorInfo, Sponsor
from .sponsors import generate_sponsor_keywords
```

Update `score_segment` signature and implementation:

```python
def score_segment(
    segment: TranscriptSegment,
    duration: float,
    context_text: str = "",
    sponsors: SponsorInfo | None = None,
) -> tuple[float, list[str], bool, list[str]]:
    """Score a transcript segment for ad likelihood.

    Args:
        segment: The transcript segment to score.
        duration: Total duration of the audio in seconds.
        context_text: Optional combined text for cross-boundary matching.
        sponsors: Optional sponsor info for dynamic keywords.

    Returns:
        A tuple of (score, trigger keywords, is_strong_indicator, sponsors_found).
    """
    text = segment.text.lower()
    search_text = context_text.lower() if context_text else text

    score = 0.0
    triggers: list[str] = []
    is_strong = False
    sponsors_found: list[str] = []

    # Check for standard keyword matches
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
            keywords = generate_sponsor_keywords(sponsor)
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

    # Positional boosts (existing code)
    if segment.start < 90:
        has_sponsor_or_cta = any(
            cat in ["intro_sponsor", "cta"]
            for pattern, cat in KEYWORD_PATTERNS
            if pattern in search_text
        )
        if has_sponsor_or_cta:
            score += 0.2

    midpoint_start = duration * 0.4
    midpoint_end = duration * 0.6
    if midpoint_start <= segment.start <= midpoint_end:
        if triggers:
            score += 0.15

    if segment.start > (duration - 90):
        if triggers:
            score += 0.1

    score = min(max(score, 0.0), 1.0)

    return score, triggers, is_strong, sponsors_found
```

Update `find_ad_candidates`:

```python
def find_ad_candidates(
    segments: list[TranscriptSegment],
    duration: float,
    sponsors: SponsorInfo | None = None,
    extend_before: float = EXTENSION_BEFORE,
    extend_after: float = EXTENSION_AFTER,
) -> list[AdCandidate]:
    """Find advertisement candidates from transcript segments.

    ... existing docstring ...
    """
    if not segments:
        return []

    sorted_segments = sorted(segments, key=lambda s: s.start)

    # Track which sponsors we find
    all_sponsors_found: set[str] = set()
    all_sponsor_names: set[str] = set()
    if sponsors:
        all_sponsor_names = {s.name for s in sponsors.sponsors}

    # Phase 1: Score all segments
    scored: list[tuple[TranscriptSegment, float, list[str], bool, list[str]]] = []

    for i, seg in enumerate(sorted_segments):
        context_parts = []
        if i > 0:
            context_parts.append(sorted_segments[i - 1].text)
        context_parts.append(seg.text)
        if i < len(sorted_segments) - 1:
            context_parts.append(sorted_segments[i + 1].text)

        context_text = " ".join(context_parts)

        score, triggers, is_strong, seg_sponsors = score_segment(
            seg, duration, context_text, sponsors
        )
        scored.append((seg, score, triggers, is_strong, seg_sponsors))
        all_sponsors_found.update(seg_sponsors)

    # Phase 2: Create extended spans (existing logic)
    raw_spans: list[tuple[float, float, list[int], list[str], float, list[str]]] = []

    for seg, score, triggers, is_strong, seg_sponsors in scored:
        if is_strong or score >= AD_SCORE_THRESHOLD:
            span_start = max(0, seg.start - extend_before)
            span_end = min(duration, seg.end + extend_after)

            indices_in_span = [
                s.index for s in segments
                if s.start >= span_start and s.end <= span_end
            ]

            raw_spans.append((span_start, span_end, indices_in_span, triggers, score, seg_sponsors))

    # Phase 3: Merge overlapping spans (existing logic, updated for sponsors)
    if not raw_spans:
        # No candidates found - return empty with all sponsors missing
        return []

    raw_spans.sort(key=lambda x: x[0])
    merged_spans: list[tuple[float, float, list[int], list[str], float, list[str]]] = []

    current = raw_spans[0]
    for span in raw_spans[1:]:
        if span[0] <= current[1]:
            combined_indices = list(set(current[2] + span[2]))
            combined_indices.sort()
            combined_triggers = list(dict.fromkeys(current[3] + span[3]))
            max_score = max(current[4], span[4])
            combined_sponsors = list(set(current[5] + span[5]))
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
    sponsors_missing = list(all_sponsor_names - all_sponsors_found)

    candidates: list[AdCandidate] = []
    for start, end, indices, triggers, score, span_sponsors in merged_spans:
        candidates.append(
            AdCandidate(
                start=start,
                end=end,
                segment_indices=indices,
                trigger_keywords=triggers,
                heuristic_score=score,
                sponsors_found=span_sponsors,
                sponsors_missing=sponsors_missing,
            )
        )

    return candidates
```

**Step 5: Run tests**

```bash
pytest tests/test_ad_keywords.py -v
```

**Step 6: Commit**

```bash
git add adnihilator/ad_keywords.py adnihilator/models.py tests/test_ad_keywords.py
git commit -m "feat(ad_keywords): integrate sponsor info into keyword detection"
```

---

## Task 6: Pass Sponsors to LLM Refinement

**Files:**
- Modify: `adnihilator/ad_llm.py`

**Step 1: Update refine_candidates signature**

Update `AdLLMClient` base class and implementations to accept sponsors:

```python
class AdLLMClient(ABC):
    @abstractmethod
    def refine_candidates(
        self,
        segments: list[TranscriptSegment],
        candidates: list[AdCandidate],
        config: Config,
        sponsors: SponsorInfo | None = None,
    ) -> list[AdSpan]:
        pass
```

**Step 2: Update prompt to include sponsor info**

In `OpenAIClient._build_prompt`, add sponsor context:

```python
def _build_prompt(
    self,
    context_segments: list[TranscriptSegment],
    candidate: AdCandidate,
    context_start: float,
    context_end: float,
    sponsors: SponsorInfo | None = None,
) -> str:
    # ... existing transcript formatting ...

    # Add sponsor context
    sponsor_context = ""
    if sponsors and sponsors.sponsors:
        sponsor_lines = []
        for s in sponsors.sponsors:
            parts = [s.name]
            if s.url:
                parts.append(f"({s.url})")
            if s.code:
                parts.append(f"code: {s.code}")
            found = "FOUND" if s.name in candidate.sponsors_found else "NOT YET FOUND"
            sponsor_lines.append(f"- {' '.join(parts)} [{found}]")

        sponsor_context = f"""
Known sponsors from show notes:
{chr(10).join(sponsor_lines)}

Pay special attention to sponsors marked NOT YET FOUND - they may appear in this segment.
"""

    return f"""Analyze this transcript segment for advertisements.
{sponsor_context}
... rest of existing prompt ...
"""
```

**Step 3: Commit**

```bash
git add adnihilator/ad_llm.py
git commit -m "feat(ad_llm): pass sponsor context to LLM prompt"
```

---

## Task 7: Implement Hunt Mode for Missing Sponsors

**Files:**
- Modify: `adnihilator/ad_llm.py`

**Step 1: Add hunt mode logic**

Add after normal refinement in `OpenAIClient.refine_candidates`:

```python
def refine_candidates(
    self,
    segments: list[TranscriptSegment],
    candidates: list[AdCandidate],
    config: Config,
    sponsors: SponsorInfo | None = None,
) -> list[AdSpan]:
    # ... existing refinement code ...

    # Hunt mode: search for missing sponsors in gaps
    if sponsors and candidates:
        missing = set()
        for c in candidates:
            missing.update(c.sponsors_missing)

        if missing:
            hunt_spans = self._hunt_missing_sponsors(
                segments, spans, sponsors, missing, config
            )
            spans.extend(hunt_spans)

    return self._merge_overlapping_spans(spans)


def _hunt_missing_sponsors(
    self,
    segments: list[TranscriptSegment],
    existing_spans: list[AdSpan],
    sponsors: SponsorInfo,
    missing: set[str],
    config: Config,
) -> list[AdSpan]:
    """Search gaps in coverage for missing sponsors."""
    if not segments:
        return []

    duration = segments[-1].end

    # Find gaps > 10 minutes between ads
    gaps: list[tuple[float, float]] = []
    sorted_spans = sorted(existing_spans, key=lambda s: s.start)

    prev_end = 0.0
    for span in sorted_spans:
        if span.start - prev_end > 600:  # 10 minute gap
            gaps.append((prev_end, span.start))
        prev_end = max(prev_end, span.end)

    # Check final gap
    if duration - prev_end > 600:
        gaps.append((prev_end, duration))

    # Search up to 3 gaps
    hunt_spans: list[AdSpan] = []
    for gap_start, gap_end in gaps[:3]:
        midpoint = (gap_start + gap_end) / 2
        window_start = max(0, midpoint - 120)  # 2 min before
        window_end = min(duration, midpoint + 120)  # 2 min after

        window_segments = [
            s for s in segments
            if s.end >= window_start and s.start <= window_end
        ]

        if not window_segments:
            continue

        # Build hunt prompt
        prompt = self._build_hunt_prompt(
            window_segments, list(missing), window_start, window_end
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            result = self._parse_response(response.choices[0].message.content or "")

            for ad in result.get("ads", []):
                if ad.get("is_ad") and ad.get("ad_start") and ad.get("ad_end"):
                    hunt_spans.append(AdSpan(
                        start=ad["ad_start"],
                        end=ad["ad_end"],
                        confidence=ad.get("confidence", 0.6),
                        reason=f"hunt_mode: {ad.get('reason', 'found missing sponsor')}",
                        candidate_indices=[],
                    ))
        except Exception:
            continue

    return hunt_spans


def _build_hunt_prompt(
    self,
    segments: list[TranscriptSegment],
    missing_sponsors: list[str],
    window_start: float,
    window_end: float,
) -> str:
    """Build prompt for hunting missing sponsors."""
    transcript_lines = []
    for s in segments:
        mins = int(s.start // 60)
        secs = s.start % 60
        transcript_lines.append(f"[{mins:02d}:{secs:05.2f}] {s.text}")

    return f"""We're looking for ads that our initial scan may have missed.

Missing sponsors (from show notes): {', '.join(missing_sponsors)}

Search window: {window_start:.0f}s to {window_end:.0f}s

Transcript:
{chr(10).join(transcript_lines)}

Look carefully for any mention of these sponsors or their products. They may be subtle.

Respond with JSON:
{{
  "ads": [
    {{"is_ad": true, "ad_start": <seconds>, "ad_end": <seconds>, "confidence": 0.0-1.0, "reason": "found [sponsor name]"}}
  ]
}}

If no ads found: {{"ads": [], "reason": "no sponsor mentions found"}}"""
```

**Step 2: Commit**

```bash
git add adnihilator/ad_llm.py
git commit -m "feat(ad_llm): add hunt mode for missing sponsors"
```

---

## Task 8: Update Worker Pipeline

**Files:**
- Modify: `worker/client.py`
- Modify: `worker/daemon.py`
- Modify: `web/routes/api.py`

**Step 1: Add description to EpisodeJob**

Update `worker/client.py`:

```python
@dataclass
class EpisodeJob:
    """An episode job to process."""

    id: str
    podcast_id: str
    guid: str
    title: Optional[str]
    original_audio_url: str
    description: Optional[str] = None  # NEW
```

Update `WorkerClient.claim` to handle description:

```python
def claim(self) -> Optional[EpisodeJob]:
    response = httpx.post(
        f"{self.api_url}/api/queue/claim",
        headers=self._headers(),
        timeout=self.timeout,
    )
    response.raise_for_status()
    data = response.json()
    if data is None:
        return None
    return EpisodeJob(
        id=data["id"],
        podcast_id=data["podcast_id"],
        guid=data["guid"],
        title=data.get("title"),
        original_audio_url=data["original_audio_url"],
        description=data.get("description"),  # NEW
    )
```

**Step 2: Update API to return description**

Update `web/routes/api.py` claim endpoint:

```python
@router.post("/queue/claim")
async def claim_episode(...) -> Optional[dict]:
    # ... existing code ...

    return {
        "id": episode.id,
        "podcast_id": episode.podcast_id,
        "guid": episode.guid,
        "title": episode.title,
        "original_audio_url": episode.original_audio_url,
        "description": episode.description,  # NEW
        "status": episode.status,
    }
```

**Step 3: Update worker daemon**

Update `worker/daemon.py`:

Add import:
```python
from adnihilator.sponsors import extract_sponsors
```

Update `process_job`:

```python
def process_job(self, job: EpisodeJob) -> None:
    print(f"Processing: {job.title or job.guid}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Extract sponsors from description (NEW)
        print("  Extracting sponsors...")
        sponsor_info = extract_sponsors(
            job.description or "",
            llm_client=self._get_openai_client() if self.config.llm.provider == "openai" else None,
        )
        if sponsor_info.sponsors:
            print(f"    Found {len(sponsor_info.sponsors)} sponsors via {sponsor_info.extraction_method}")

        # Download audio
        # ... existing code ...

        # Detect ads - pass sponsors (CHANGED)
        print("  Detecting ads...")
        self.api_client.update_progress(job.id, "detecting")
        candidates = find_ad_candidates(segments, duration, sponsors=sponsor_info)

        # Refine with LLM - pass sponsors (CHANGED)
        print("  Refining with LLM...")
        self.api_client.update_progress(job.id, "refining")
        llm_client = create_llm_client(self.config)
        ad_spans = llm_client.refine_candidates(
            segments, candidates, self.config, sponsors=sponsor_info
        )

        # Save sponsor artifact (NEW)
        if self.artifacts_dir and sponsor_info.sponsors:
            sponsor_path = artifact_dir / f"{job.id}_sponsors.json"
            sponsor_path.write_text(sponsor_info.model_dump_json(indent=2))
            print(f"    Saved sponsors to {sponsor_path}")

        # ... rest of existing code ...
```

Add helper method:
```python
def _get_openai_client(self):
    """Get OpenAI client for sponsor extraction LLM fallback."""
    try:
        from openai import OpenAI
        api_key = self.config.llm.api_key
        if api_key:
            return OpenAI(api_key=api_key)
    except ImportError:
        pass
    return None
```

**Step 4: Commit**

```bash
git add worker/client.py worker/daemon.py web/routes/api.py
git commit -m "feat(worker): integrate sponsor extraction into pipeline"
```

---

## Task 9: Add Integration Tests

**Files:**
- Create: `tests/test_sponsor_integration.py`

**Step 1: Write integration test**

```python
"""Integration tests for sponsor extraction pipeline."""

import pytest

from adnihilator.sponsors import extract_sponsors
from adnihilator.ad_keywords import find_ad_candidates
from adnihilator.models import TranscriptSegment


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

        # Should find both ad segments
        assert len(candidates) >= 2

        # Both sponsors should be found
        all_found = set()
        for c in candidates:
            all_found.update(c.sponsors_found)

        assert len(all_found) == 2

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

        # Only mention one sponsor
        segments = [
            TranscriptSegment(index=0, start=60, end=90,
                text="This episode is brought to you by Sponsor1."),
        ]

        candidates = find_ad_candidates(segments, 3600.0, sponsors=sponsor_info)

        # Should have missing sponsors tracked
        assert len(candidates) >= 1
        assert len(candidates[0].sponsors_missing) >= 2
```

**Step 2: Run tests**

```bash
pytest tests/test_sponsor_integration.py -v
```

**Step 3: Commit**

```bash
git add tests/test_sponsor_integration.py
git commit -m "test: add sponsor extraction integration tests"
```

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Add Sponsor and SponsorInfo models |
| 2 | Create sponsor extraction with pattern matching |
| 3 | Add LLM fallback extraction |
| 4 | Generate dynamic keywords from sponsors |
| 5 | Integrate sponsors into keyword detection |
| 6 | Pass sponsors to LLM refinement |
| 7 | Implement hunt mode for missing sponsors |
| 8 | Update worker pipeline |
| 9 | Add integration tests |
