"""Real transport regressions covered by the explicit D17 repair exception."""

import gzip
from html import escape
import zlib

import pytest

from web.models import Episode, Podcast
from .test_feed_policy import AUTH, web
from .test_feed_policy_acceptance import responses, rss, subscribe


class TestD17Compatibility:
    @pytest.mark.parametrize('variant', ['bare_amp', 'nbsp', 'prefix', 'query', 'atom_nbsp'])
    def test_complete_recovered_feed_imports_new_episode_and_retry_dedupes(self, web: tuple, variant: str) -> None:
        """Recovered complete documents remain usable without admitting truncation."""
        client, factory = web
        with responses() as (base, routes):
            routes['/feed'] = (200, {'Content-Type': 'application/rss+xml'}, rss('<item><guid isPermaLink="false">old</guid><title>Old</title></item>'))
            pid = subscribe(client, factory, base + '/feed')
            titles = {'bare_amp': 'Research & Design', 'nbsp': 'Hello&nbsp;world', 'prefix': 'Prefix', 'query': 'Query'}
            if variant == 'atom_nbsp':
                body = f'<feed xmlns="http://www.w3.org/2005/Atom"><title>Atom&nbsp;publisher</title><entry><id>urn:new</id><title>New&nbsp;entry</title><link rel="enclosure" href="{base}/audio.mp3"/></entry></feed>'.encode()
                content_type = 'application/atom+xml'
            else:
                extra = '<itunes:duration>60</itunes:duration>' if variant == 'prefix' else ''
                url = base + ('/audio.mp3?id=1&src=rss' if variant == 'query' else '/audio.mp3')
                body = rss(f'<item><guid isPermaLink="false">new</guid><title>{titles[variant]}</title>{extra}<enclosure url="{url}" type="audio/mpeg"/></item>', 'Publisher')
                content_type = 'application/rss+xml'
            routes['/feed'] = (200, {'Content-Type': content_type}, body)
            for added in [1, 0]:
                result = client.post(f'/podcasts/{pid}/sync', auth=AUTH)
                assert result.status_code == 200, result.text
                assert result.json()['episodes_added'] == added
                with factory() as db:
                    podcast = db.get(Podcast, pid)
                    assert podcast.last_sync_error is None
                    assert db.query(Episode).count() == 2
                    new = db.query(Episode).filter_by(guid='urn:new' if variant == 'atom_nbsp' else 'new').one()
                    assert new.original_audio_url == base + ('/audio.mp3?id=1&src=rss' if variant == 'query' else '/audio.mp3')
            assert routes['_requests'].count('/feed') == 3

    @pytest.mark.parametrize('tail', ['', '</channel></rss>', '</item></rss>'])
    def test_recoverable_token_before_broken_structure_never_commits_partial_data(self, web: tuple, tail: str) -> None:
        """A recoverable token cannot hide later truncation or mismatched tags."""
        client, factory = web
        with responses() as (base, routes):
            routes['/feed'] = (200, {}, rss('<item><guid isPermaLink="false">old</guid></item>'))
            pid = subscribe(client, factory, base + '/feed')
            with factory() as db:
                p = db.get(Podcast, pid)
                before = (p.title, p.last_synced_at)
            routes['/feed'] = (200, {}, ('<rss version="2.0"><channel><title>Changed & Broken</title><item><guid isPermaLink="false">partial</guid><title>Half</title>' + tail).encode())
            assert client.post(f'/podcasts/{pid}/sync', auth=AUTH).status_code == 502
            with factory() as db:
                p = db.get(Podcast, pid)
                assert (p.title, p.last_synced_at) == before
                assert p.last_sync_error
                assert [e.guid for e in db.query(Episode)] == ['old']

    def test_raw_atom_url_validation_precedes_parser_repairs_and_keeps_relative_base(self, web: tuple) -> None:
        """Queue only original valid references, preserving MIME and XML-base policy."""
        client, factory = web
        with responses() as (base, routes):
            cases = [('malformed', f'http:///127.0.0.1:{base.rsplit(":", 1)[1]}/bad.mp3', ''), ('empty', '', 'type="audio/mpeg"'), ('space', '  ', 'type="audio/mpeg"'), ('valid', 'clip.mp3?episode=2&src=rss', ''), ('html', 'page.mp3', 'type="text/html"'), ('uppercase_html', 'page.mp3', 'TYPE="TEXT/HTML"')]
            entries = ''.join(f'<entry><id>{name}</id><title>{name}</title><link rel="enclosure" href="{escape(href, quote=True)}" {typ}/></entry>' for name, href, typ in cases)
            routes['/feed'] = (200, {'Content-Type': 'application/atom+xml'}, f'<feed xmlns="http://www.w3.org/2005/Atom" xml:base="{base}/media/"><title>Atom</title>{entries}</feed>'.encode())
            pid = subscribe(client, factory, base + '/feed')
            with factory() as db:
                rows = {e.title: (e.id, e.original_audio_url) for e in db.query(Episode)}
                assert rows['valid'][1] == base + '/media/clip.mp3?episode=2&src=rss'
                assert all(rows[name][1] is None for name in ['malformed', 'empty', 'space', 'html', 'uppercase_html'])
            for name, (eid, _) in rows.items():
                assert client.post(f'/podcasts/{pid}/episodes/{eid}/queue', auth=AUTH).status_code == (200 if name == 'valid' else 409)

    def test_uppercase_atom_enclosure_rel_does_not_reject_whole_feed(self, web: tuple) -> None:
        """Source metadata aligns with feedparser's normalized relation tokens."""
        client, factory = web
        with responses() as (base, routes):
            entries = ''.join(f'<entry><id>{rel}</id><title>{rel}</title><link rel="{rel}" href="{base}/{rel}.mp3"/></entry>' for rel in ['ENCLOSURE', 'enclosure'])
            routes['/feed'] = (200, {'Content-Type': 'application/atom+xml'}, f'<feed xmlns="http://www.w3.org/2005/Atom"><title>Atom</title>{entries}</feed>'.encode())
            pid = subscribe(client, factory, base + '/feed')
            with factory() as db:
                assert db.get(Podcast, pid).last_sync_error is None
                assert db.query(Episode).count() == 2
                assert all(e.original_audio_url for e in db.query(Episode))

    @pytest.mark.parametrize('compression', ['none', 'gzip', 'deflate'])
    @pytest.mark.parametrize('encoding', ['utf-8', 'iso-8859-1'])
    def test_recovered_atom_provenance_uses_same_encoded_compressed_response(self, web: tuple, compression: str, encoding: str) -> None:
        """Recover entities without losing MIME provenance or making another fetch."""
        client, factory = web
        with responses() as (base, routes):
            body = (f'<?xml version="1.0" encoding="{encoding}"?><feed xmlns="http://www.w3.org/2005/Atom"><title>Café&nbsp;feed</title><entry><id>urn:recovered</id><title>Recovered</title><link rel="ENCLOSURE" href="{base}/clip.mp3"/></entry></feed>').encode(encoding)
            headers = {'Content-Type': f'application/atom+xml; charset={encoding}'}
            if compression == 'gzip':
                body = gzip.compress(body)
                headers['Content-Encoding'] = 'gzip'
            elif compression == 'deflate':
                body = zlib.compress(body)
                headers['Content-Encoding'] = 'deflate'
            routes['/feed'] = (200, headers, body)
            pid = subscribe(client, factory, base + '/feed')
            with factory() as db:
                podcast = db.get(Podcast, pid)
                assert podcast.last_sync_error is None
                assert podcast.title == 'Café\u00a0feed'
                episode = db.query(Episode).one()
                assert episode.original_audio_url == base + '/clip.mp3'
            assert routes['_requests'] == ['/feed']

    def test_structural_validation_does_not_open_external_xml_entities(self, web: tuple) -> None:
        """The validation parser must not fetch publisher-supplied DTDs."""
        client, factory = web
        with responses() as (base, routes):
            body = (f'<!DOCTYPE rss SYSTEM "{base}/external.dtd">' + rss('<item><guid isPermaLink="false">safe</guid><title>Safe & Recovered</title></item>').decode()).encode()
            routes['/feed'] = (200, {'Content-Type': 'application/rss+xml'}, body)
            routes['/external.dtd'] = (200, {}, b'<!ENTITY outside "unexpected">')
            pid = subscribe(client, factory, base + '/feed')
            with factory() as db:
                assert db.get(Podcast, pid).last_sync_error is None
                assert db.query(Episode).count() == 1
            assert routes['_requests'] == ['/feed']
