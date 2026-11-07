"""Real HTTP and durable route regressions for the bounded D20 exception."""

import pytest

from web.models import Episode, Podcast
from .test_feed_policy import AUTH, web
from .test_feed_policy_acceptance import responses, subscribe
from .test_feed_d19_alignment import atom


class TestD20Provenance:
    @pytest.mark.parametrize('recovered', [False, True])
    @pytest.mark.parametrize('reference,expected', [
        ('<link rel="enclosure" href="actual.pdf" x:type="audio/mpeg"/>', None),
        ('<link rel="enclosure" x:href="foreign.mp3" type="audio/mpeg"/>', None),
        ('<link rel="enclosure" x:url="foreign.mp3" type="audio/mpeg"/>', None),
        ('<link rel="enclosure" x:uri="foreign.mp3" type="audio/mpeg"/>', None),
        ('<link rel="enclosure" href="actual.mp3" x:type="text/html"/>', 'actual.mp3'),
        ('<link x:rel="enclosure" href="actual.mp3" type="audio/mpeg"/>', None),
        ('<link rel="alternate" x:rel="enclosure" href="actual.mp3" type="audio/mpeg"/>', None),
        ('<link rel="enclosure" href="actual.mp3" type="text/html" x:type="audio/mpeg"/>', None),
        ('<link rel="enclosure" href="actual.mp3" x:type="audio/mpeg" type="text/html"/>', None),
        ('<link rel="enclosure" href="actual.mp3" x:href="foreign.mp3" type="audio/mpeg"/>', 'actual.mp3'),
        ('<link rel="enclosure" x:href="foreign.mp3" href="actual.mp3" type="audio/mpeg"/>', 'actual.mp3'),
        ('<link rel="enclosure" x:rel="alternate" href="actual.mp3" type="audio/mpeg"/>', 'actual.mp3'),
        ('<link x:rel="alternate" rel="ENCLOSURE" href="actual.mp3" type="audio/mpeg"/>', 'actual.mp3'),
        ('<link rel="enclosure" href="actual.mp3" type="application/octet-stream" x:extra="safe"/>', 'actual.mp3'),
    ])
    def test_only_unqualified_source_attributes_can_admit_audio(self, web: tuple, recovered: bool, reference: str, expected: str | None) -> None:
        """Foreign attributes cannot fabricate or override genuine enclosure data."""
        client, factory = web
        with responses() as (base, routes):
            title = 'Research & Design' if recovered else 'Research &amp; Design'
            routes['/feed'] = (200, {'Content-Type': 'application/atom+xml'}, atom(f'<entry><id>urn:source</id><title>{title}</title>{reference}</entry>'))
            pid = subscribe(client, factory, base + '/feed')
            with factory() as db:
                podcast = db.get(Podcast, pid)
                assert podcast.last_sync_error is None
                row = db.query(Episode).one()
                assert row.original_audio_url == (base + '/' + expected if expected else None)
                eid = row.id
            assert client.post(f'/podcasts/{pid}/episodes/{eid}/queue', auth=AUTH).status_code == (200 if expected else 409)
            assert routes['_requests'] == ['/feed']

    @pytest.mark.parametrize('recovered', [False, True])
    @pytest.mark.parametrize('reference', [
        '<link rel="enclosure" href="clip.mp3" title="Ep &#34;one&#34;" type="audio/mpeg"/>',
        "<link rel='enclosure' href='it&#39;s.mp3' type='audio/mpeg'/>",
        '<link rel="enclosure" href="clip.mp3" title="Ep &quot;one&quot;" type="audio/mpeg"/>',
        '<link rel="enclosure" href="clip.mp3" title="Ep &#x22;one&#x22;" type="audio/mpeg"/>',
        '<enclosure url="clip.mp3" title="Ep &#34;one&#34;" type="audio/mpeg"/>',
    ])
    def test_complete_quote_reference_feed_recovers_without_losing_other_entries(self, web: tuple, recovered: bool, reference: str) -> None:
        """A valid quoted attribute must not reject an otherwise complete feed."""
        client, factory = web
        with responses() as (base, routes):
            title = 'Research & Design' if recovered else 'Research &amp; Design'
            entries = f'<entry><id>urn:quoted</id><title>{title}</title>{reference}</entry><entry><id>urn:neighbor</id><title>Neighbor</title><link rel="enclosure" href="neighbor.mp3" type="audio/mpeg"/></entry>'
            routes['/feed'] = (200, {'Content-Type': 'application/atom+xml'}, atom(entries))
            pid = subscribe(client, factory, base + '/feed')
            with factory() as db:
                assert db.get(Podcast, pid).last_sync_error is None
                rows = {row.guid: row for row in db.query(Episode)}
                assert set(rows) == {'urn:quoted', 'urn:neighbor'}
                # Decimal quotes can already truncate an upstream loose-parser
                # attribute. This regression concerns whole-feed rejection;
                # do not turn inherited safe exclusions into URL repair.
                numeric_quote_recovery = recovered and ('&#34;' in reference or 'it&#39;s.mp3' in reference)
                if not numeric_quote_recovery:
                    expected_name = "it's.mp3" if 'it&#39;s.mp3' in reference else 'clip.mp3'
                    assert rows['urn:quoted'].original_audio_url == base + '/' + expected_name
                assert rows['urn:neighbor'].original_audio_url == base + '/neighbor.mp3'
            assert client.post(f'/podcasts/{pid}/sync', auth=AUTH).json()['episodes_added'] == 0
