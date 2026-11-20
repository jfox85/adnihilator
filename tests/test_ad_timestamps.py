"""Tests for extracting ad spans from episode description timestamps."""

import pytest

from adnihilator.ad_timestamps import extract_ad_timestamps

# Trimmed from the real Lex Fridman #505 show notes (HTML as served in the RSS feed).
LEX_505_OUTLINE = """<p><b>SPONSORS:</b><br />
To support this podcast, check out our sponsors &#038; get discounts:<br />
<b>Fin:</b> AI agent for customer service.<br />
Go to <a href="https://lexfridman.com/s/fin-ep505-sc">https://fin.ai/lex</a></p>
<p><b>OUTLINE:</b><br />
(00:00) &#8211; Introduction<br />
(01:32) &#8211; Sponsors, Comments, and Reflections<br />
(07:23) &#8211; GTA 6<br />
(28:31) &#8211; 1980s Miami crime era<br />
(1:05:18) &#8211; Surviving solitary confinement</p>"""


def _high_confidence(description: str) -> list[tuple[float, float]]:
    """Spans the worker would accept for the timestamp fast path."""
    return [(t.start, t.end) for t in extract_ad_timestamps(description) if t.confidence >= 0.85]


class TestChapterOutlines:
    def test_lex_fridman_html_outline_cuts_whole_sponsor_chapter(self):
        """The sponsor chapter runs until the next chapter starts."""
        assert _high_confidence(LEX_505_OUTLINE) == [(92, 443)]

    def test_sponsor_chapter_label_is_kept(self):
        """The chapter label is preserved for diagnostics."""
        [stamp] = extract_ad_timestamps(LEX_505_OUTLINE)
        assert stamp.label == "Sponsors, Comments, and Reflections"
        assert stamp.extraction_method == "chapter_span"

    def test_plain_text_outline_with_hours(self):
        """Plain-text outlines and H:MM:SS times work the same way."""
        description = "0:00 - Intro\n1:02:03 - Sponsor break\n1:04:33 - Back to the show\n"
        assert _high_confidence(description) == [(3723, 3873)]

    @pytest.mark.parametrize(
        "label",
        [
            "Sponsors: Eight Sleep & AG1",
            "Sponsor: LMNT",
            "2) Sponsors: LMNT & Waking Up",
            "Disclaimer & Sponsors: AG1, LMNT & Waking Up",
            "Protocols Book; Sponsor: Carbon App",
            "Thanking the Sponsors",
            "Acknowledgement of Sponsors",
            "Sponsors",
            "Ad break",
        ],
    )
    def test_explicit_sponsor_chapter_labels_are_cut(self, label: str):
        """Real sponsor-chapter labels seen across subscribed feeds are cut."""
        description = f"(00:00) – Introduction\n(10:00) – {label}\n(12:00) – Next topic\n"
        assert _high_confidence(description) == [(600, 720)]

    @pytest.mark.parametrize(
        "label",
        [
            "Ad blockers and privacy",
            "Advertising industry",
            "Breaking news",
            # Real chapter titles from subscribed feeds that discuss ads/sponsors.
            "Paid ads tips",
            "The power of creative in ads",
            "Investing in brand before paid ads",
            "Making sponsorship deals win/win.",
            "Mentors vs. sponsors, and advice for finding people who will",
            "Industry-Sponsored Studies",
            "The Reason We Have Ads in the Show",
            "Transition to ads and joining Patrick’s network.",
            "Are Bill’s podcast ads reads — or roasts?",
            "GLP-1s, Weight Loss, Alcohol Addiction; Pharmaceutical Advertising",
        ],
    )
    def test_topic_chapters_that_mention_ads_are_not_cut(self, label: str):
        """A whole chapter is only cut for an explicit sponsor/ad-read label."""
        description = f"(00:00) – Introduction\n(10:00) – {label}\n(40:00) – Next topic\n"
        assert _high_confidence(description) == []

    def test_implausibly_long_sponsor_chapter_is_not_trusted(self):
        """A sponsor chapter longer than 20 minutes is not trusted as an exact span."""
        description = "(00:00) – Introduction\n(01:00) – Sponsors\n(45:00) – Main topic\n"
        assert _high_confidence(description) == []

    def test_final_sponsor_chapter_without_end_is_low_confidence(self):
        """Without a following chapter the end is a guess, so it is not trusted."""
        description = "(00:00) – Introduction\n(1:30:00) – Sponsors\n"
        assert _high_confidence(description) == []


class TestExistingFormats:
    def test_explicit_time_range_with_ad_label(self):
        """Explicit ranges remain high confidence."""
        assert _high_confidence("(00:30 - 01:10) Sponsor: Acme") == [(30, 70)]

    def test_no_ads_disclaimer_returns_empty(self):
        """Ad-free disclaimers suppress extraction."""
        assert extract_ad_timestamps("This episode is ad-free.\n(01:00) – Sponsors\n(05:00) – Topic") == []

    def test_empty_description(self):
        """Empty input returns no spans."""
        assert extract_ad_timestamps("") == []
