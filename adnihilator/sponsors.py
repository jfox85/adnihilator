"""Sponsor extraction from episode show notes."""

import json
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
    # Matches: <li><a href="url">text</a></li> or <li><a href="url">text</a> - Use code XYZ</li>
    sponsor_section = re.search(
        r'<(?:strong|b)>(?:SPONSORS?|Sponsors?):?</(?:strong|b)>(.*?)(?:<(?:strong|b)>|$)',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if sponsor_section:
        section_text = sponsor_section.group(1)
        # Extract links from list items, including text after the link for promo codes
        links = re.findall(
            r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>([^<]*)',
            section_text,
            re.IGNORECASE
        )
        for url, link_text, after_text in links:
            sponsor = _parse_sponsor_from_link(url, link_text, after_text)
            if sponsor and sponsor.name.lower() not in seen_names:
                sponsors.append(sponsor)
                seen_names.add(sponsor.name.lower())

    # Pattern 2: Lex Fridman style - <b>Name:</b> description Go to <a href="url">
    # Use negative lookahead to not cross other <b> tags
    lex_pattern = re.findall(
        r'<b>([^:<]+?):</b>(?:(?!<b>).)*?<a[^>]+href=["\']([^"\']+)["\']',
        text,
        re.IGNORECASE | re.DOTALL
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


def _parse_sponsor_from_link(
    url: str,
    link_text: str,
    after_text: str = "",
) -> Optional[Sponsor]:
    """Parse sponsor info from a URL, link text, and surrounding context.

    Args:
        url: The sponsor URL.
        link_text: The text inside the <a> tag.
        after_text: Optional text following the </a> tag (may contain promo codes).

    Returns:
        Sponsor object or None if invalid.
    """
    # Extract domain/name from URL
    url_clean = _clean_url(url)

    # Try to get name from URL domain
    domain_match = re.search(r'(?:https?://)?(?:www\.)?([^./]+)', url_clean)
    name = domain_match.group(1) if domain_match else link_text

    # Check for promo code in link text first, then in after_text
    code = _extract_promo_code(link_text) or _extract_promo_code(after_text)

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


LLM_EXTRACTION_PROMPT = '''Extract sponsor/advertiser information from this podcast episode description.

Return a JSON object with this structure:
{{
  "sponsors": [
    {{"name": "Sponsor Name", "url": "sponsor.com/podcast", "code": "PROMO"}}
  ]
}}

Rules:
- Only include paid sponsors/advertisers, not general links
- Extract promo codes if mentioned (e.g., "use code SAVE20")
- URL should be the sponsor's URL mentioned, not tracking links
- If no sponsors found, return {{"sponsors": []}}

Episode description:
"""
{description}
"""'''


def generate_sponsor_keywords(
    sponsor: Sponsor,
    podcast_name: str | None = None,
) -> list[str]:
    """Generate keyword variations for matching sponsor in transcript.

    Handles:
    - CamelCase splitting (ExpressVPN -> express vpn)
    - Common word boundary splitting (Bitwarden -> bit warden)
    - URL slug extraction (shopify.com/lex -> shopify, shopify dot com slash lex)
    - Promo codes (code TWIT) - excludes codes matching podcast name

    Args:
        sponsor: Sponsor object to generate keywords for.
        podcast_name: Optional podcast name to filter out false positive promo codes.

    Returns:
        List of lowercase keyword strings.
    """
    keywords: set[str] = set()

    name = sponsor.name

    # Guard against empty name
    if not name or not name.strip():
        return []

    # Original name (lowercase)
    keywords.add(name.lower())

    # CamelCase splitting: ExpressVPN -> express vpn
    # Insert space before uppercase letters that follow lowercase
    split_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    if split_name != name:
        keywords.add(split_name.lower())

    # Common word boundary splitting for compound words
    # Handles cases where Whisper transcribes as separate words
    # e.g., "Bitwarden" -> "bit warden", "CacheFly" -> "cache fly"
    word_boundary_variants = _generate_word_boundary_splits(name)
    keywords.update(word_boundary_variants)

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

            # Also try word boundary splitting on domain name
            domain_variants = _generate_word_boundary_splits(domain_name)
            keywords.update(domain_variants)

        # Full URL path for matching "shopify.com/lex"
        keywords.add(url_clean)

        # Spoken URL: "shopify dot com slash lex"
        spoken = url_clean.replace('.', ' dot ').replace('/', ' slash ')
        keywords.add(spoken.strip())

    # Promo code keywords - but filter out codes that match podcast name
    if sponsor.code:
        code = sponsor.code.lower()
        # Check if code matches podcast name (to avoid false positives)
        # e.g., "TWIT" code should not match "TWiT" podcast name
        should_include_code = True
        if podcast_name:
            podcast_lower = podcast_name.lower()
            # Check various forms of the podcast name
            words = podcast_lower.split()
            # Check direct match, word match, or concatenated match
            if (code == podcast_lower or
                code in words or
                code == ''.join(words)):
                should_include_code = False
            # Check if code is an acronym of the podcast name
            # e.g., "twit" matches "This Week in Tech"
            elif len(code) == len(words):
                acronym = ''.join(w[0] for w in words if w)
                if code == acronym:
                    should_include_code = False

        if should_include_code:
            keywords.add(code)
            keywords.add(f"code {code}")
            keywords.add(f"promo code {code}")

    return list(keywords)


# Common word parts that Whisper often transcribes separately
COMMON_WORD_PARTS = {
    'bit': True, 'warden': True,  # Bitwarden -> bit warden
    'cache': True, 'fly': True,   # CacheFly -> cache fly
    'out': True, 'systems': True, # OutSystems -> out systems
    'space': True, 'ship': True,  # Spaceship -> space ship
    'mail': True,                 # SpaceMail -> space mail
    'express': True, 'vpn': True, # ExpressVPN -> express vpn
    'zip': True, 'recruiter': True,  # ZipRecruiter -> zip recruiter
    'hello': True, 'fresh': True, # HelloFresh -> hello fresh
    'square': True,               # Squarespace -> square space
    'better': True, 'help': True, # BetterHelp -> better help
    'nord': True, 'pass': True,   # NordPass -> nord pass, NordVPN -> nord vpn
    'surf': True, 'shark': True,  # Surfshark -> surf shark
    'rocket': True, 'money': True,  # Rocket Money
    'aura': True, 'frames': True,  # Aura Frames
}


def _generate_word_boundary_splits(name: str) -> set[str]:
    """Generate word boundary split variations for a name.

    Uses a dictionary of common word parts to split compound names
    that Whisper might transcribe as separate words.

    Args:
        name: The sponsor name to split.

    Returns:
        Set of split variations (lowercase).
    """
    variants: set[str] = set()
    name_lower = name.lower()

    # Try splitting at each position
    for i in range(2, len(name_lower) - 1):
        left = name_lower[:i]
        right = name_lower[i:]

        # Check if both parts are known word parts
        if left in COMMON_WORD_PARTS and right in COMMON_WORD_PARTS:
            variants.add(f"{left} {right}")

        # Also check if just the split makes sense without dictionary
        # (for longer names that might have internal word boundaries)

    # Also try matching common prefixes/suffixes
    for word in COMMON_WORD_PARTS:
        if name_lower.startswith(word) and len(name_lower) > len(word):
            rest = name_lower[len(word):]
            if rest and rest[0].isalpha():
                variants.add(f"{word} {rest}")
        if name_lower.endswith(word) and len(name_lower) > len(word):
            prefix = name_lower[:-len(word)]
            if prefix and prefix[-1].isalpha():
                variants.add(f"{prefix} {word}")

    return variants


def get_all_sponsor_keywords(
    sponsors: list[Sponsor],
    podcast_name: str | None = None,
) -> dict[str, list[str]]:
    """Generate keywords for all sponsors.

    Args:
        sponsors: List of sponsors.
        podcast_name: Optional podcast name to filter out false positive promo codes.

    Returns:
        Dict mapping sponsor name to list of keywords.
    """
    return {
        sponsor.name: generate_sponsor_keywords(sponsor, podcast_name)
        for sponsor in sponsors
    }


def _extract_sponsors_with_llm(description: str, llm_client) -> list[Sponsor]:
    """Extract sponsors using LLM when patterns fail.

    Args:
        description: Episode description text.
        llm_client: OpenAI client instance.

    Returns:
        List of extracted sponsors.
    """
    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
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

    except Exception as e:
        # Silent failure - return empty list
        # During development, you can uncomment this to debug:
        # import traceback
        # traceback.print_exc()
        return []
