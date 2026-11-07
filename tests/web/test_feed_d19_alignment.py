"""Namespace and identity regressions covered by the explicit D19 exception."""

import feedparser
import pytest

from web.models import Episode, Podcast
from .test_feed_policy import AUTH, web
from .test_feed_policy_acceptance import responses, subscribe

ATOM = 'http://www.w3.org/2005/Atom'


def atom(entries: str, base: str = '') -> bytes:
    """Wrap independent source entries in an actual Atom vocabulary."""
    return f'<feed xmlns="{ATOM}" xmlns:x="urn:outside" {base}><id>urn:feed</id><title>Actual feed</title>{entries}</feed>'.encode()


class TestD19Alignment:
    @pytest.mark.parametrize('foreign', ['link', 'entry', 'xhtml'])
    def test_foreign_vocabulary_does_not_abort_actual_atom_entries(self, web: tuple, foreign: str) -> None:
        """Ignore extensions without counting them as Atom entries or links."""
        client, factory = web
        with responses() as (base, routes):
            extra = '<x:link rel="enclosure" href="foreign.mp3"/>' if foreign == 'link' else '<h:link xmlns:h="http://www.w3.org/1999/xhtml" rel="enclosure" href="foreign.mp3"/>' if foreign == 'xhtml' else ''
            preceding = '<x:entry><x:id>urn:foreign</x:id><x:link rel="enclosure" href="foreign.mp3"/></x:entry>' if foreign == 'entry' else ''
            routes['/feed'] = (200, {'Content-Type': 'application/atom+xml'}, atom(preceding + f'<entry><id>urn:actual</id><title>Actual</title>{extra}<link rel="enclosure" href="{base}/actual.mp3"/></entry>'))
            pid = subscribe(client, factory, base + '/feed')
            with factory() as db:
                assert db.get(Podcast, pid).last_sync_error is None
                row = db.query(Episode).one()
                assert row.guid == 'urn:actual' and row.original_audio_url == base + '/actual.mp3'
            assert client.post(f'/podcasts/{pid}/sync', auth=AUTH).json()['episodes_added'] == 0

    @pytest.mark.parametrize('mime,foreign_href,eligible', [('text/html', 'foreign.mp3', False), ('audio/mpeg', '', True)])
    def test_rss_style_enclosure_never_borrows_foreign_link_provenance(self, web: tuple, mime: str, foreign_href: str, eligible: bool) -> None:
        """Equal counts cannot strip real HTML MIME or erase actual audio."""
        client, factory = web
        with responses() as (base, routes):
            source = f'<entry><id>urn:actual</id><title>Actual</title><enclosure url="{base}/actual.mp3" type="{mime}"/><x:link rel="enclosure" href="{foreign_href}"/></entry>'
            routes['/feed'] = (200, {'Content-Type': 'application/atom+xml'}, atom(source))
            pid = subscribe(client, factory, base + '/feed')
            with factory() as db:
                assert db.get(Podcast, pid).last_sync_error is None
                row = db.query(Episode).one()
                assert row.original_audio_url == (base + '/actual.mp3' if eligible else None)
                eid = row.id
            assert client.post(f'/podcasts/{pid}/episodes/{eid}/queue', auth=AUTH).status_code == (200 if eligible else 409)

    def test_author_enclosure_is_distinct_from_content_and_source_links(self, web: tuple) -> None:
        """Preserve actual parser enclosure context without importing note markup."""
        client, factory = web
        with responses() as (base, routes):
            source = '<entry><id>urn:actual</id><title>Actual</title><author><name>Author</name><link rel="enclosure" href="author.mp3" type="audio/mpeg"/></author><source><title>Source</title><link rel="enclosure" href="source.mp3"/></source><summary type="xhtml"><div xmlns="http://www.w3.org/1999/xhtml"><link xmlns="http://www.w3.org/2005/Atom" rel="enclosure" href="notes.mp3"/></div></summary><link rel="enclosure" href="html.mp3" type="text/html"/></entry>'
            routes['/feed'] = (200, {'Content-Type': 'application/atom+xml'}, atom(source))
            pid = subscribe(client, factory, base + '/feed')
            with factory() as db:
                assert db.get(Podcast, pid).last_sync_error is None
                assert db.query(Episode).one().original_audio_url == base + '/author.mp3'

    def test_namespace_shadowing_restores_real_prefix_and_xml_base(self, web: tuple) -> None:
        """Scope namespace declarations, preserving genuine prefixed Atom links."""
        client, factory = web
        with responses() as (base, routes):
            body = f'<a:feed xmlns:a="{ATOM}" xmlns:p="{ATOM}" xml:base="{base}/media/"><a:title>Prefixed</a:title><a:entry><a:id>urn:one</a:id><a:title>One</a:title><p:link xmlns:p="urn:outside" rel="enclosure" href="foreign.mp3"/><a:link rel="ENCLOSURE" href="one.mp3"/></a:entry><a:entry><a:id>urn:two</a:id><a:title>Two</a:title><p:link rel="enclosure" href="two.mp3" type="audio/mpeg"/></a:entry></a:feed>'
            routes['/start'] = (302, {'Location': '/feed'}, b'')
            routes['/feed'] = (200, {'Content-Type': 'application/atom+xml'}, body.encode())
            pid = subscribe(client, factory, base + '/start')
            with factory() as db:
                assert db.get(Podcast, pid).last_sync_error is None
                assert {e.original_audio_url for e in db.query(Episode)} == {base + '/media/one.mp3', base + '/media/two.mp3'}

    @pytest.mark.parametrize('has_valid_omission', [False, True])
    def test_identical_normalized_links_keep_raw_mime_and_url_admission(self, web: tuple, has_valid_omission: bool) -> None:
        """Matching duplicate identities must not rehabilitate an invalid raw URL."""
        client, factory = web
        with responses() as (base, routes):
            valid = '<link rel="enclosure" href="clip.mp3"/>' if has_valid_omission else ''
            entry = f'<entry><id>urn:duplicates</id><title>Duplicates</title><link rel="enclosure" href="{base}/clip.mp3" type="text/html"/><link rel="enclosure" href="http:///127.0.0.1:{base.rsplit(":", 1)[1]}/clip.mp3"/>{valid}</entry>'
            routes['/feed'] = (200, {'Content-Type': 'application/atom+xml'}, atom(entry))
            pid = subscribe(client, factory, base + '/feed')
            with factory() as db:
                assert db.get(Podcast, pid).last_sync_error is None
                assert db.query(Episode).one().original_audio_url == (base + '/clip.mp3' if has_valid_omission else None)

    @pytest.mark.parametrize('recovered', [False, True])
    def test_content_location_and_entity_decoding_keep_actual_parser_url(self, web: tuple, recovered: bool) -> None:
        """Provenance matching must preserve the existing parser's URL resolution."""
        client, factory = web
        with responses() as (base, routes):
            title = 'Research & Design' if recovered else 'Research &amp; Design'
            entry = f'<entry><id>urn:encoded</id><title>{title}</title><link rel="enclosure" href="clip.mp3?a=1&amp;amp;b=2"/></entry>'
            routes['/feed'] = (200, {'Content-Type': 'application/atom+xml', 'Content-Location': '/advertised/feed.xml'}, atom(entry))
            expected = feedparser.parse(base + '/feed').entries[0].enclosures[0].href
            pid = subscribe(client, factory, base + '/feed')
            with factory() as db:
                assert db.get(Podcast, pid).last_sync_error is None
                assert db.query(Episode).one().original_audio_url == expected
            assert routes['_requests'] == ['/feed', '/feed']

    @pytest.mark.parametrize('recovered', [False, True])
    def test_foreign_entry_reusing_known_prefix_is_not_imported(self, web: tuple, recovered: bool) -> None:
        """Qualified source identity wins over a parser's retained prefix alias."""
        client, factory = web
        with responses() as (base, routes):
            title = 'Actual & Recovered' if recovered else 'Actual'
            body = f'<a:feed xmlns:a="{ATOM}" xmlns:p="{ATOM}"><a:title>Feed</a:title><p:entry xmlns:p="urn:foreign"><p:id>urn:foreign</p:id><p:title>Foreign</p:title><p:link rel="enclosure" href="foreign.mp3" type="audio/mpeg"/></p:entry><a:entry><a:id>urn:actual</a:id><a:title>{title}</a:title><a:link rel="enclosure" href="actual.mp3" type="audio/mpeg"/></a:entry></a:feed>'
            routes['/feed'] = (200, {'Content-Type': 'application/atom+xml'}, body.encode())
            pid = subscribe(client, factory, base + '/feed')
            with factory() as db:
                assert db.get(Podcast, pid).last_sync_error is None
                row = db.query(Episode).one()
                assert row.guid == 'urn:actual'
                assert row.original_audio_url == base + '/actual.mp3'

    @pytest.mark.parametrize('href,eligible', [('clip&#32;.mp3', False), ('clip&#x09;.mp3', False), ('clip&nbsp;.mp3', False), ('clip&amp;nbsp;.mp3', True)])
    def test_recovered_attribute_entities_keep_existing_raw_url_guards(self, web: tuple, href: str, eligible: bool) -> None:
        """Validation decoding rejects spaces without rewriting an imported URL."""
        client, factory = web
        with responses() as (base, routes):
            entry = f'<entry><id>urn:guard</id><title>Research & Design</title><link rel="enclosure" href="{href}" type="audio/mpeg"/></entry>'
            routes['/feed'] = (200, {'Content-Type': 'application/atom+xml'}, atom(entry))
            pid = subscribe(client, factory, base + '/feed')
            with factory() as db:
                assert db.get(Podcast, pid).last_sync_error is None
                row = db.query(Episode).one()
                assert row.original_audio_url == (base + '/clip&nbsp;.mp3' if eligible else None)
                eid = row.id
            assert client.post(f'/podcasts/{pid}/episodes/{eid}/queue', auth=AUTH).status_code == (200 if eligible else 409)

    @pytest.mark.parametrize('mime', ['audio&sol;mpeg', 'text&sol;html'])
    def test_validation_entity_rewrite_cannot_upgrade_original_mime(self, web: tuple, mime: str) -> None:
        """Only the actual parsed source type controls explicit MIME admission."""
        client, factory = web
        with responses() as (base, routes):
            entry = f'<entry><id>urn:mime</id><title>Research & Design</title><link rel="enclosure" href="clip.mp3" type="{mime}"/></entry>'
            routes['/feed'] = (200, {'Content-Type': 'application/atom+xml'}, atom(entry))
            pid = subscribe(client, factory, base + '/feed')
            with factory() as db:
                assert db.get(Podcast, pid).last_sync_error is None
                row = db.query(Episode).one()
                assert row.original_audio_url is None
                eid = row.id
            assert client.post(f'/podcasts/{pid}/episodes/{eid}/queue', auth=AUTH).status_code == 409
