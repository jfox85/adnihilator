"""Original feed-policy acceptance gaps confirmed independently before fixes."""
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from threading import Thread
import gzip
import zlib
from html import escape

import pytest
from sqlalchemy import event

from web.models import Episode,Podcast
from .test_feed_policy import AUTH,web


@contextmanager
def responses() -> Iterator[tuple[str,dict]]:
    """Serve mutable synthetic responses including redirected and gzip feeds."""
    routes={}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            """Return only a response explicitly installed by this test."""
            routes.setdefault('_requests', []).append(self.path)
            status,headers,body=routes.get(self.path,(404,{},b'not found'))
            self.send_response(status)
            for name,value in headers.items():self.send_header(name,value)
            self.end_headers();self.wfile.write(body)
        def log_message(self,*args: object) -> None:
            pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield f'http://127.0.0.1:{server.server_port}',routes
    finally:server.shutdown();server.server_close();thread.join(timeout=2)


def rss(items: str,title: str='Healthy') -> bytes:
    """Build well-formed RSS around caller-supplied synthetic item XML."""
    return f'<rss version="2.0"><channel><title>{title}</title>{items}</channel></rss>'.encode()


def subscribe(client: object,factory: object,url: str) -> str:
    """Return the one subscription created through the real authenticated form."""
    response=client.post('/podcasts',auth=AUTH,data={'source_rss_url':url},follow_redirects=False)
    assert response.status_code==303
    with factory() as db:return db.query(Podcast).filter_by(source_rss_url=url).one().id


class TestOriginalFeedPolicyAcceptance:
    @pytest.mark.parametrize('after_error',[False,True])
    def test_fatal_xml_never_commits_partial_episodes_or_clears_failure(self,web: tuple,after_error: bool) -> None:
        """Fatal truncation preserves prior good data and a durable retry state."""
        client,factory=web
        with responses() as (base,routes):
            good=rss('<item><guid isPermaLink="false">good</guid><title>Good item</title></item>')
            routes['/feed']=(503,{},b'outage') if after_error else (200,{'Content-Type':'application/rss+xml'},good)
            pid=subscribe(client,factory,base+'/feed')
            with factory() as db:
                p=db.get(Podcast,pid);old=(p.title,p.last_synced_at,db.query(Episode).count())
            routes['/feed']=(200,{'Content-Type':'application/rss+xml'},b'<rss version="2.0"><channel><item><guid isPermaLink="false">partial</guid><title>half')
            result=client.post(f'/podcasts/{pid}/sync',auth=AUTH)
            assert result.status_code==502
            with factory() as db:
                p=db.get(Podcast,pid)
                assert (p.title,p.last_synced_at,db.query(Episode).count())==old
                assert p.last_sync_error
                assert not db.query(Episode).filter_by(guid='partial').first()
            assert 'Retry' in client.get(f'/podcasts/{pid}/episodes',auth=AUTH).text

    def test_foreign_xml_entries_do_not_make_a_feed(self,web: tuple) -> None:
        """Entries in an unrelated XML root cannot fabricate podcast episodes."""
        client,factory=web
        with responses() as (base,routes):
            routes['/feed']=(200,{'Content-Type':'application/xml'},b'<directory><entry><title>Not a podcast</title><id>fake</id></entry></directory>')
            pid=subscribe(client,factory,base+'/feed')
        with factory() as db:
            p=db.get(Podcast,pid)
            assert p.last_sync_error and p.last_synced_at is None
            assert db.query(Episode).count()==0

    def test_non_xml_content_type_warning_keeps_well_formed_rss_usable(self,web: tuple) -> None:
        """A recoverable feedparser MIME warning is not a fatal XML error."""
        client,factory=web
        with responses() as (base,routes):
            routes['/feed']=(200,{'Content-Type':'text/plain'},rss('<item><guid>usable</guid><title>Usable</title></item>'))
            pid=subscribe(client,factory,base+'/feed')
        with factory() as db:
            assert db.get(Podcast,pid).last_sync_error is None
            assert db.query(Episode).count()==1

    @pytest.mark.parametrize('compression',['none','gzip','deflate'])
    @pytest.mark.parametrize('encoding',['utf-8','iso-8859-1'])
    def test_atom_omitted_mime_is_distinct_from_explicit_html(self,web: tuple,compression: str,encoding: str) -> None:
        """Actual Atom link attributes survive parser defaulting, including gzip."""
        client,factory=web
        with responses() as (base,routes):
            entries=''.join(f'<entry><id>urn:{name}</id><title>{name}</title><link rel="enclosure" href="clip.mp3" {typ}/></entry>' for name,typ in [('omitted',''),('binary','type="application/octet-stream"'),('html','type="text/html"')])
            body=f'<feed xmlns="http://www.w3.org/2005/Atom" xml:base="{base}/media/"><title>Atom Café</title><id>urn:feed</id>{entries}</feed>'.encode(encoding)
            headers={'Content-Type':f'application/atom+xml; charset={encoding}'}
            if compression=='gzip':body=gzip.compress(body);headers['Content-Encoding']='gzip'
            if compression=='deflate':body=zlib.compress(body);headers['Content-Encoding']='deflate'
            routes['/feed']=(200,headers,body)
            pid=subscribe(client,factory,base+'/feed')
            assert routes['_requests'].count('/feed')==1  # source metadata needs no second fetch
            with factory() as db:
                rows={e.title:e for e in db.query(Episode).all()}
                assert rows['omitted'].original_audio_url==base+'/media/clip.mp3'
                assert rows['binary'].original_audio_url==base+'/media/clip.mp3'
                assert rows['html'].original_audio_url is None
                omitted_id,html_id=rows['omitted'].id,rows['html'].id
            assert client.post(f'/podcasts/{pid}/episodes/{omitted_id}/queue',auth=AUTH).status_code==200
            assert client.post(f'/podcasts/{pid}/episodes/{html_id}/queue',auth=AUTH).status_code==409

    def test_relative_enclosures_use_final_redirect_url_and_stay_http_only(self,web: tuple) -> None:
        """Resolve sibling/root references before selecting audio and identity."""
        client,factory=web
        with responses() as (base,routes):
            routes['/start']=(302,{'Location':'/destination/rss'},b'')
            items=''.join(f'<item><title>{name}</title><enclosure url="{escape(url,quote=True)}" type="audio/mpeg"/></item>' for name,url in [('sibling','audio.mp3'),('root','/media/root.mp3'),('bad','file:///tmp/forbidden.mp3')])
            routes['/destination/rss']=(200,{'Content-Type':'application/rss+xml'},rss(items))
            pid=subscribe(client,factory,base+'/start')
            with factory() as db:
                rows={e.title:e for e in db.query(Episode).all()}
                assert rows['sibling'].original_audio_url==base+'/destination/audio.mp3'
                assert rows['root'].original_audio_url==base+'/media/root.mp3'
                assert 'bad' not in rows  # no GUID/link/usable fallback identity
                assert all(e.guid.startswith('enclosure-sha256:') for e in rows.values())
            assert client.post(f'/podcasts/{pid}/sync',auth=AUTH).json()['episodes_added']==0

    def test_opml_rejects_unsupported_sources_without_blocking_private_http(self,web: tuple) -> None:
        """All source entrypoints share the HTTP policy and truthful import counts."""
        client,factory=web
        urls=['file:///tmp/forbidden.xml','ftp://127.0.0.1/feed','data:text/xml,secret','javascript:secret','/tmp/local.xml','http://127.0.0.1/private.xml']
        xml='<opml><body>'+''.join(f'<outline xmlUrl="{escape(url,quote=True)}" text="Synthetic"/>' for url in urls)+'</body></opml>'
        result=client.post('/opml/import',auth=AUTH,files={'file':('settings.opml',xml,'application/xml')})
        assert result.status_code==200,result.text
        assert result.json()['imported']==1
        assert result.json()['skipped_invalid']==5
        assert result.json()['skipped']==5
        assert all(x['reason']=='unsupported source URL' for x in result.json()['feeds']['skipped'])
        with factory() as db:assert [p.source_rss_url for p in db.query(Podcast).all()]==[urls[-1]]

    def test_stale_skip_cannot_turn_completion_into_an_implicit_reprocess(self,web: tuple) -> None:
        """Completed output and diagnostics survive Queue/Skip/Queue requests."""
        client,factory=web
        with factory() as db:
            p=Podcast(source_rss_url='http://127.0.0.1/feed');db.add(p);db.flush()
            e=Episode(podcast_id=p.id,guid='done',status='complete',original_audio_url='http://127.0.0.1/a.mp3',processed_audio_key='keep.mp3',retry_count=4,error_message='keep diagnostic')
            db.add(e);db.commit();pid,eid=p.id,e.id
        for action in ['queue','skip','queue']:
            assert client.post(f'/podcasts/{pid}/episodes/{eid}/{action}',auth=AUTH).status_code==409
        with factory() as db:
            e=db.get(Episode,eid)
            assert (e.status,e.processed_audio_key,e.retry_count,e.error_message)==('complete','keep.mp3',4,'keep diagnostic')

    def test_completion_committed_after_skip_read_wins(self,web: tuple) -> None:
        """A stale Skip UPDATE cannot overwrite a real intervening completion."""
        client,factory=web
        with factory() as db:
            p=Podcast(source_rss_url='http://127.0.0.1/feed');db.add(p);db.flush()
            e=Episode(podcast_id=p.id,guid='skip-race',status='pending',original_audio_url='http://127.0.0.1/a.mp3');db.add(e);db.commit();pid,eid=p.id,e.id
        engine=factory.kw['bind'];fired=False
        def complete(conn: object,cursor: object,statement: str,parameters: object,context: object,executemany: bool) -> None:
            nonlocal fired
            if fired or not statement.startswith('UPDATE episodes SET status'):return
            fired=True
            with factory() as other:
                other.query(Episode).filter_by(id=eid).update({Episode.status:'complete',Episode.processed_audio_key:'won.mp3'});other.commit()
        event.listen(engine,'before_cursor_execute',complete)
        try:result=client.post(f'/podcasts/{pid}/episodes/{eid}/skip',auth=AUTH)
        finally:event.remove(engine,'before_cursor_execute',complete)
        assert fired and result.status_code==409
        with factory() as db:
            e=db.get(Episode,eid);assert (e.status,e.processed_audio_key)==('complete','won.mp3')

    @pytest.mark.parametrize('query',['episode','id'])
    def test_distinct_query_identified_audio_remains_distinct_and_stable(self,web: tuple,query: str) -> None:
        """Opaque identity-bearing query values cannot silently omit an episode."""
        client,factory=web
        with responses() as (base,routes):
            items=''.join(f'<item><title>Episode {n}</title><enclosure url="{base}/download.php?{query}={n}" type="audio/mpeg"/></item>' for n in [1,2])
            routes['/feed']=(200,{'Content-Type':'application/rss+xml'},rss(items))
            pid=subscribe(client,factory,base+'/feed')
            with factory() as db:
                before=[(e.id,e.guid,e.original_audio_url) for e in db.query(Episode).order_by(Episode.title)]
                assert len(before)==2 and len({x[1] for x in before})==2
            assert client.post(f'/podcasts/{pid}/sync',auth=AUTH).json()['episodes_added']==0
            with factory() as db:assert [(e.id,e.guid,e.original_audio_url) for e in db.query(Episode).order_by(Episode.title)]==before


    @pytest.mark.parametrize('path,status',[('/choice',300),('/loop',302)])
    def test_http_error_response_is_not_reported_as_connection_failure(self,web: tuple,path: str,status: int) -> None:
        """HTTPError is a URLError subtype but still represents a real response."""
        client,factory=web
        with responses() as (base,routes):
            routes[path]=(status,{'Location':path},b'')
            pid=subscribe(client,factory,base+path)
            with factory() as db:
                diagnostic=db.get(Podcast,pid).last_sync_error
                assert f'HTTP {status}' in diagnostic and 'connect' not in diagnostic.lower()

    def test_recoverable_encoding_warning_is_not_a_fatal_parse_error(self,web: tuple) -> None:
        """Honor a complete feed when its XML encoding overrides a bad header."""
        client,factory=web
        with responses() as (base,routes):
            body=('<?xml version="1.0" encoding="iso-8859-1"?>'+rss('<item><guid>encoding</guid></item>',title='Café').decode()).encode('iso-8859-1')
            routes['/feed']=(200,{'Content-Type':'application/rss+xml; charset=utf-8'},body)
            pid=subscribe(client,factory,base+'/feed')
        with factory() as db:
            p=db.get(Podcast,pid)
            assert p.title=='Café' and p.last_sync_error is None
            assert db.query(Episode).count()==1
