"""HTTP regressions for independently confirmed subscription error states."""
import socket

from bs4 import BeautifulSoup

from web.models import Podcast
from .test_feed_policy import AUTH, feed_server, web


def error_text(html: str) -> str:
    """Read the actual visible error paragraph from the rendered document."""
    return BeautifulSoup(html, "html.parser").select_one('.feed-sync-error').get_text(' ', strip=True)


class TestFeedErrorPresentation:
    def test_http_failure_explains_status_and_retry_without_internal_class(self, web: tuple) -> None:
        """Keep useful HTTP details while removing implementation vocabulary."""
        client, factory=web
        with feed_server() as (url,state):
            state.update(status=503,body=b'fixture unavailable')
            client.post('/podcasts',auth=AUTH,data={'source_rss_url':url+'/feed'})
        message=error_text(client.get('/',auth=AUTH).text)
        assert '503' in message and 'Retry' in message
        assert 'FeedSyncError' not in message and 'RuntimeError' not in message
        assert 'Check' in message or 'try' in message.lower()

    def test_connection_refusal_and_invalid_content_have_distinct_durable_reasons(self, web: tuple) -> None:
        """An actual refused connection must not be diagnosed as malformed RSS."""
        client,factory=web
        sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
        refused=f'http://127.0.0.1:{port}/refused'
        client.post('/podcasts',auth=AUTH,data={'source_rss_url':refused})
        with feed_server() as (url,state):
            state['body']=b'<html><title>Not a feed</title></html>'
            client.post('/podcasts',auth=AUTH,data={'source_rss_url':url+'/feed'})
        with factory() as db:
            rows=db.query(Podcast).all()
            network=next(p for p in rows if p.source_rss_url==refused)
            content=next(p for p in rows if p.source_rss_url!=refused)
            assert 'connect' in network.last_sync_error.lower()
            assert network.last_sync_error!=content.last_sync_error
            assert 'RSS' in content.last_sync_error or 'feed' in content.last_sync_error
        page=client.get('/',auth=AUTH).text
        assert 'connect' in page.lower()

    def test_browser_rejection_renders_inline_error_and_retains_value(self, web: tuple) -> None:
        """A normal invalid form POST stays editable inside the application."""
        client,factory=web
        value='file:///tmp/unsupported-feed.xml'
        response=client.post('/podcasts',auth=AUTH,headers={'Accept':'text/html'},data={'source_rss_url':value})
        assert response.status_code==422 and 'text/html' in response.headers['content-type']
        doc=BeautifulSoup(response.text,'html.parser')
        field=doc.select_one('input[name=source_rss_url]')
        assert field and field.get('value')==value
        assert field.get('aria-invalid')=='true'
        assert doc.select_one('#source-rss-error[role=alert]')
        assert doc.select_one('form[action="/podcasts"]')
        with factory() as db:
            assert db.query(Podcast).count()==0
        api=client.post('/podcasts',auth=AUTH,headers={'Accept':'application/json'},data={'source_rss_url':value})
        assert api.status_code==422 and 'detail' in api.json()

    def test_failed_episode_page_keeps_attention_error_and_retry(self, web: tuple) -> None:
        """Following a failed subscription must not hide its recorded failure."""
        client,factory=web
        with feed_server() as (url,state):
            state.update(status=503,body=b'fixture unavailable')
            client.post('/podcasts',auth=AUTH,data={'source_rss_url':url+'/feed'})
            with factory() as db:
                pid=db.query(Podcast).one().id
            page=client.get(f'/podcasts/{pid}/episodes',auth=AUTH)
            doc=BeautifulSoup(page.text,'html.parser')
            assert 'Subscription needs attention' in doc.select_one('h2').get_text()
            assert '503' in error_text(page.text)
            assert doc.select_one('#refreshBtn').get_text(strip=True)=='Retry'
            assert 'will be synced automatically' not in page.text
            assert client.post(f'/podcasts/{pid}/sync',auth=AUTH).status_code==502
            assert '503' in error_text(client.get(f'/podcasts/{pid}/episodes',auth=AUTH).text)
            state.update(status=200,body=b'<rss version="2.0"><channel><title>Recovered</title></channel></rss>')
            assert client.post(f'/podcasts/{pid}/sync',auth=AUTH).status_code==200
            recovered=BeautifulSoup(client.get(f'/podcasts/{pid}/episodes',auth=AUTH).text,'html.parser')
            assert not recovered.select_one('.feed-sync-error')
            assert recovered.select_one('#refreshBtn').get_text(strip=True)=='Refresh'

    def test_previously_saved_diagnostics_are_presented_without_rewriting_them(self, web: tuple) -> None:
        """Both template environments format old safe errors without a backfill."""
        client,factory=web
        diagnostic='Feed sync failed (FeedSyncError): Feed server returned HTTP 503'
        with factory() as db:
            p=Podcast(source_rss_url='http://127.0.0.1/feed',last_sync_error=diagnostic)
            db.add(p);db.commit();pid=p.id
        for path in ['/',f'/podcasts/{pid}/episodes']:
            message=error_text(client.get(path,auth=AUTH).text)
            assert '503' in message and 'Retry' in message and 'FeedSyncError' not in message
        with factory() as db:
            assert db.get(Podcast,pid).last_sync_error==diagnostic
