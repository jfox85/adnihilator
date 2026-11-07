"""HTTP/route/SQLite contracts for source authority and bounded quote recovery."""

import feedparser
import pytest

from web.models import Episode, Podcast
from .test_feed_policy import AUTH, web
from .test_feed_policy_acceptance import responses, subscribe


def document(kind: str, mode: str, content: str, base: str = '', extra: str = '') -> bytes:
    """Create complete source XML with an intentional, asserted parser mode."""
    title = {'strict': 'Source &amp; title', 'bare': 'Source & title', 'nbsp': 'Source&nbsp;title'}[mode]
    if kind == 'atom':
        return f'<feed xmlns="http://www.w3.org/2005/Atom" xmlns:x="urn:outside" {extra} {base}><title>{title}</title>{content}</feed>'.encode()
    return f'<rss version="2.0" xmlns:x="urn:outside" {extra} {base}><channel><title>{title}</title>{content}</channel></rss>'.encode()


def entry(kind: str, name: str, content: str) -> str:
    """Bind expected rows to explicit stable source IDs."""
    return f'<entry><id>urn:{name}</id><title>{name}</title>{content}</entry>' if kind == 'atom' else f'<item><guid isPermaLink="false">urn:{name}</guid><title>{name}</title>{content}</item>'


def enclosure(kind: str, attrs: str) -> str:
    """Keep publisher attribute order intact in both source formats."""
    return f'<link rel="enclosure" {attrs}/>' if kind == 'atom' else f'<enclosure {attrs}/>'


class TestD21Adapter:
    @pytest.mark.parametrize('kind', ['atom', 'rss'])
    @pytest.mark.parametrize('mode', ['strict', 'bare', 'nbsp'])
    @pytest.mark.parametrize('alias', ['url', 'uri'])
    @pytest.mark.parametrize('first', [False, True])
    def test_true_href_beats_foreign_uri_with_actual_base(self, web: tuple, kind: str, mode: str, alias: str, first: bool) -> None:
        """Only the genuine href, resolved at its true base, may reach Queue."""
        client, factory = web
        with responses() as (base, routes):
            true = 'href="actual.mp3?episode=1&amp;sig=a%2Fb#chapter"'
            foreign = f'x:{alias}="http:///wrong.pdf"'
            attrs = f'{foreign} {true}' if first else f'{true} {foreign}'
            body = document(kind, mode, entry(kind, 'target', enclosure(kind, attrs + ' type="audio/mpeg"')), base=f'xml:base="{base}/media/"')
            assert bool(feedparser.parse(body).bozo) == (mode != 'strict')
            routes['/start'] = (302, {'Location': '/feed'}, b'')
            routes['/feed'] = (200, {'Content-Type': 'application/xml'}, body)
            pid = subscribe(client, factory, base + '/start')
            with factory() as db:
                assert db.get(Podcast, pid).last_sync_error is None
                row = db.query(Episode).one()
                assert row.original_audio_url == base + '/media/actual.mp3?episode=1&sig=a%2Fb#chapter'
                eid = row.id
            assert client.post(f'/podcasts/{pid}/episodes/{eid}/queue', auth=AUTH).status_code == 200
            assert routes['_requests'] == ['/start', '/feed']

    @pytest.mark.parametrize('kind', ['atom', 'rss'])
    @pytest.mark.parametrize('mode', ['strict', 'bare', 'nbsp'])
    @pytest.mark.parametrize('attrs,expected', [
        ('href="actual.mp3" x:type="text/html"', 'actual.mp3'),
        ('href="document.pdf" x:type="audio/mpeg"', None),
        ('href="actual.mp3" type="text/html" x:type="audio/mpeg"', None),
        ('x:url="actual.mp3" type="audio/mpeg"', None),
        ('href="actual.mp3" type="&#97;udio/mpeg"', 'actual.mp3'),
        ('href="document.pdf" type="&aacute;udio/mpeg"', None),
    ])
    def test_source_mime_absence_and_xml_value_are_authoritative(self, web: tuple, kind: str, mode: str, attrs: str, expected: str | None) -> None:
        """A qualified or validation-only value cannot replace true source MIME."""
        if '&aacute;' in attrs and mode == 'strict':
            pytest.skip('An undeclared entity intentionally requires recovery')
        client, factory = web
        with responses() as (base, routes):
            routes['/feed'] = (200, {'Content-Type': 'application/xml'}, document(kind, mode, entry(kind, 'target', enclosure(kind, attrs))))
            pid = subscribe(client, factory, base + '/feed')
            with factory() as db:
                assert db.get(Podcast, pid).last_sync_error is None
                row = db.query(Episode).one()
                assert row.original_audio_url == (base + '/' + expected if expected else None)
                eid = row.id
            assert client.post(f'/podcasts/{pid}/episodes/{eid}/queue', auth=AUTH).status_code == (200 if expected else 409)

    @pytest.mark.parametrize('kind', ['atom', 'rss'])
    @pytest.mark.parametrize('mode', ['bare', 'nbsp'])
    @pytest.mark.parametrize('true_type', ['', 'type="text/html"'])
    def test_quoted_title_is_a_value_not_a_second_mime_attribute(self, web: tuple, kind: str, mode: str, true_type: str) -> None:
        """Reject a PDF while retaining every healthy later episode."""
        client, factory = web
        with responses() as (base, routes):
            attrs = f'{true_type} title="x&#34; type=&#34;audio/mpeg" href="{base}/document.pdf"'
            content = entry(kind, 'target', enclosure(kind, attrs)) + entry(kind, 'after', enclosure(kind, f'href="{base}/after.mp3" type="audio/mpeg"'))
            routes['/feed'] = (200, {'Content-Type': 'application/xml'}, document(kind, mode, content))
            pid = subscribe(client, factory, base + '/feed')
            with factory() as db:
                assert db.get(Podcast, pid).last_sync_error is None
                rows = {row.guid: row for row in db.query(Episode)}
                assert set(rows) == {'urn:target', 'urn:after'}
                assert rows['urn:target'].original_audio_url is None
                assert rows['urn:after'].original_audio_url == base + '/after.mp3'
                eid = rows['urn:target'].id
            assert client.post(f'/podcasts/{pid}/episodes/{eid}/queue', auth=AUTH).status_code == 409

    @pytest.mark.parametrize('kind', ['atom', 'rss'])
    @pytest.mark.parametrize('mode', ['bare', 'nbsp'])
    @pytest.mark.parametrize('quote,entity,name,decoded', [('"', '&#34;', 'a', 'a"b.mp3'), ("'", '&#39;', 'it', "it'b.mp3"), ('"', '&quot;', 'a', 'a"b.mp3'), ("'", '&#x27;', 'it', "it'b.mp3")])
    def test_type_before_quoted_href_never_queues_a_truncated_or_serialized_target(self, web: tuple, kind: str, mode: str, quote: str, entity: str, name: str, decoded: str) -> None:
        """Safe NULL is allowed; any admitted URI must be the full true XML value."""
        client, factory = web
        with responses() as (base, routes):
            attrs = f'type="audio/mpeg" href={quote}{base}/{name}{entity}b.mp3{quote}'
            content = entry(kind, 'target', enclosure(kind, attrs)) + entry(kind, 'after', enclosure(kind, f'href="{base}/after.mp3" type="audio/mpeg"'))
            body = document(kind, mode, content)
            assert bool(feedparser.parse(body).bozo)
            routes['/feed'] = (200, {'Content-Type': 'application/xml'}, body)
            pid = subscribe(client, factory, base + '/feed')
            with factory() as db:
                assert db.get(Podcast, pid).last_sync_error is None
                rows = {row.guid: row for row in db.query(Episode)}
                assert set(rows) == {'urn:target', 'urn:after'}
                assert rows['urn:target'].original_audio_url in {None, base + '/' + decoded}
                assert rows['urn:after'].original_audio_url == base + '/after.mp3'
                url, eid = rows['urn:target'].original_audio_url, rows['urn:target'].id
            assert client.post(f'/podcasts/{pid}/episodes/{eid}/queue', auth=AUTH).status_code == (200 if url else 409)

    @pytest.mark.parametrize('kind', ['atom', 'rss'])
    @pytest.mark.parametrize('mode', ['bare', 'nbsp'])
    @pytest.mark.parametrize('cross', [False, True])
    def test_mixed_single_and_multiline_quotes_import_all_legitimate_rows(self, web: tuple, kind: str, mode: str, cross: bool) -> None:
        """Complete recovery cannot silently omit healthy later entries or just fail."""
        client, factory = web
        with responses() as (base, routes):
            plain = lambda name: enclosure(kind, f'href="{base}/{name}.mp3" type="audio/mpeg"')
            single = enclosure(kind, f'title="x&#34;one&#34;" href="{base}/single.mp3" type="audio/mpeg"')
            multi = enclosure(kind, f'href="{base}/target.mp3" type="audio/mpeg"\n title="x&#34;two&#34;"')
            content = entry(kind, 'before', plain('before'))
            content += entry(kind, 'single', single) + entry(kind, 'target', multi) if cross else entry(kind, 'target', single + multi)
            content += entry(kind, 'after1', plain('after1')) + entry(kind, 'after2', plain('after2'))
            routes['/feed'] = (200, {'Content-Type': 'application/xml'}, document(kind, mode, content))
            pid = subscribe(client, factory, base + '/feed')
            with factory() as db:
                assert db.get(Podcast, pid).last_sync_error is None
                rows = {row.guid: row for row in db.query(Episode)}
                assert set(rows) == {'urn:before', 'urn:target', 'urn:after1', 'urn:after2'} | ({'urn:single'} if cross else set())
                for name in ['before', 'after1', 'after2']:
                    assert rows['urn:' + name].original_audio_url == base + '/' + name + '.mp3'
            assert client.post(f'/podcasts/{pid}/sync', auth=AUTH).json()['episodes_added'] == 0
