"""Tests for remapping chapter timestamps after ad removal."""

import pytest

from adnihilator.chapters import remap_description_timestamps, remap_time

# Lex #505: sponsor chapter 92-443 removed.
KEEP = [(0.0, 92.0), (443.0, 12417.69)]


class TestRemapTime:
    def test_before_cut_is_unchanged(self):
        assert remap_time(30, KEEP) == 30

    def test_after_cut_shifts_by_removed_duration(self):
        assert remap_time(1711, KEEP) == 1711 - 351

    def test_inside_cut_snaps_to_cut_point(self):
        """A chapter that started inside an ad now starts where the ad was."""
        assert remap_time(200, KEEP) == 92

    def test_multiple_cuts_accumulate(self):
        keep = [(0.0, 100.0), (160.0, 500.0), (600.0, 1000.0)]
        assert remap_time(700, keep) == 700 - 60 - 100


LEX_OUTLINE = """<p><b>OUTLINE:</b><br />
(00:00) &#8211; Introduction<br />
(01:32) &#8211; Sponsors, Comments, and Reflections<br />
(07:23) &#8211; GTA 6<br />
(28:31) &#8211; 1980s Miami crime era<br />
(1:05:18) &#8211; Surviving solitary confinement</p>
<p>See <a href="https://example.com/x?t=12:30">link 12:30</a></p>"""


class TestRemapDescription:
    def test_lex_outline_drops_sponsor_chapter_and_shifts_rest(self):
        out = remap_description_timestamps(LEX_OUTLINE, KEEP)
        assert "(00:00) &#8211; Introduction<br />" in out
        assert "Sponsors, Comments" not in out
        assert "(01:32) &#8211; GTA 6<br />" in out
        assert "(22:40) &#8211; 1980s Miami crime era<br />" in out
        assert "(59:27) &#8211; Surviving solitary confinement</p>" in out

    def test_non_outline_text_is_untouched(self):
        """Times inside links or prose are not chapter markers."""
        out = remap_description_timestamps(LEX_OUTLINE, KEEP)
        assert '<a href="https://example.com/x?t=12:30">link 12:30</a>' in out

    def test_plain_text_huberman_outline_keeps_hours_format(self):
        description = (
            "Timestamps\n\n(00:00:00) Twyla Tharp\n\n(00:15:59) Sponsors: Our Place & Eight Sleep\n\n"
            "(00:19:09) Cubby-Holing\n\n(01:00:00) Later topic\n"
        )
        out = remap_description_timestamps(description, [(0.0, 959.0), (1149.0, 9000.0)])
        assert "Sponsors:" not in out
        assert "(00:15:59) Cubby-Holing" in out
        assert "(00:56:50) Later topic" in out
        assert "(00:00:00) Twyla Tharp" in out

    def test_list_outline_stays_balanced(self):
        """Dropping a list item keeps the surrounding list markup intact."""
        description = "<ul><li>(00:00) Sponsors</li><li>(05:00) Topic</li><li>(09:00) More</li></ul>"
        out = remap_description_timestamps(description, [(300.0, 1000.0)])
        assert out == "<ul><li></li><li>(00:00) Topic</li><li>(04:00) More</li></ul>"

    def test_paragraph_outline_drops_text_not_markup(self):
        description = "<p>(00:00) Intro</p><p>(01:00) Sponsors</p><p>(05:00) Topic</p>"
        out = remap_description_timestamps(description, [(0.0, 60.0), (300.0, 1000.0)])
        assert out == "<p>(00:00) Intro</p><p></p><p>(01:00) Topic</p>"

    def test_no_cuts_returns_description_unchanged(self):
        assert remap_description_timestamps(LEX_OUTLINE, [(0.0, 12417.69)]) == LEX_OUTLINE

    def test_incoherent_outline_is_left_alone(self):
        """Out-of-order times are not an outline; don't rewrite them."""
        description = "(10:00) – B\n(05:00) – A\n"
        assert remap_description_timestamps(description, KEEP) == description

    @pytest.mark.parametrize("description", ["", None])
    def test_empty(self, description):
        assert remap_description_timestamps(description, KEEP) == description
