"""Source-origin credentials must not travel to a redirected origin."""
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from base64 import b64encode
from email.message import Message
import secrets
from urllib.request import Request

import pytest
from sqlalchemy.orm import Session

from web.database import get_engine,init_db
from web.models import Podcast
from web.services.feed_sync import _HTTPOnlyRedirect,sync_podcast


@pytest.fixture(autouse=True)
def disable_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep synthetic loopback credentials away from ambient proxy settings."""
    for name in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setenv("NO_PROXY", "*")


@contextmanager
def origins(code: int,mode: str,userinfo: bool) -> Iterator[tuple[str,dict]]:
    """Capture only credential presence/equality booleans on owned listeners."""
    username='synthetic-user'
    password=secrets.token_hex(16)
    expected='Basic '+b64encode(f'{username}:{password}'.encode()).decode()
    observed={'source_matched':[],'destination_present':[],'destination_matched':[]}
    rss=b'<rss version="2.0"><channel><title>Redirect destination</title></channel></rss>'
    def destination(handler: BaseHTTPRequestHandler) -> None:
        """Compare sensitive values in memory and never serialize headers."""
        value=handler.headers.get('Authorization')
        observed['destination_present'].append(value is not None)
        observed['destination_matched'].append(value==expected)
        handler.send_response(200);handler.send_header('Content-Type','application/rss+xml');handler.end_headers();handler.wfile.write(rss)
    class Second(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            """Serve the cross-port destination."""
            destination(self)
        def log_message(self,*args: object) -> None:
            pass
    second=ThreadingHTTPServer(('127.0.0.1',0),Second)
    class First(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            """Redirect a credentialed source or serve its same-origin target."""
            if self.path=='/start':
                observed['source_matched'].append(self.headers.get('Authorization')==expected)
                host='localhost' if mode=='cross-host' else '127.0.0.1'
                port=second.server_port if mode=='cross-port' else self.server.server_port
                self.send_response(code);self.send_header('Location',f'http://{host}:{port}/destination');self.end_headers()
            else:destination(self)
        def log_message(self,*args: object) -> None:
            pass
    first=ThreadingHTTPServer(('127.0.0.1',0),First)
    threads=[Thread(target=s.serve_forever,kwargs={'poll_interval':0.05},daemon=True) for s in [first,second]]
    for thread in threads:thread.start()
    # This URL is never printed, asserted, or saved to public artifacts.
    auth=f'{username}:{password}@' if userinfo else ''
    try:yield f'http://{auth}127.0.0.1:{first.server_port}/start',observed
    finally:
        for server in [first,second]:server.shutdown();server.server_close()
        for thread in threads:thread.join(timeout=1)


class TestRedirectCredentials:
    @pytest.mark.parametrize('code',[301,302,303,307,308])
    @pytest.mark.parametrize('mode,userinfo,expected_present',[
        ('same',True,True),('cross-port',True,False),('cross-host',True,False),('cross-port',False,False),
    ])
    def test_real_feed_sync_scopes_basic_credentials_to_source_origin(self,tmp_path: Path,code: int,mode: str,userinfo: bool,expected_present: bool) -> None:
        """Real feedparser redirects preserve same-origin auth and strip others."""
        engine=init_db(get_engine('sqlite:///'+str(tmp_path/'redirect.db')))
        try:
            with origins(code,mode,userinfo) as (url,observed):
                with Session(engine) as db:
                    p=Podcast(source_rss_url=url);db.add(p);db.commit()
                    assert sync_podcast(db,p)==0
                    assert p.title=='Redirect destination' and p.last_sync_error is None
                assert observed['source_matched']==[userinfo]
                assert observed['destination_present']==[expected_present]
                assert observed['destination_matched']==[expected_present]
        finally:engine.dispose()

    @pytest.mark.parametrize('credential_kind',['Basic','Bearer','Digest','Custom'])
    def test_cross_origin_request_drops_the_whole_authorization_header(self,credential_kind: str) -> None:
        """Header removal is independent of the credential scheme/value format."""
        request=Request('http://127.0.0.1:8100/start',headers={'Authorization':credential_kind+' synthetic-opaque-value','User-Agent':'Synthetic-control'})
        result=_HTTPOnlyRedirect().redirect_request(request,None,302,'Found',Message(),'http://127.0.0.1:8101/destination')
        assert result is not None and not result.has_header('Authorization')
        assert result.get_header('User-agent')=='Synthetic-control'

    @pytest.mark.parametrize('source,target,retained',[
        ('http://EXAMPLE.invalid/a','http://example.invalid:80/b',True),
        ('https://example.invalid/a','https://EXAMPLE.invalid:443/b',True),
        ('https://example.invalid/a','http://example.invalid/b',False),
        ('http://example.invalid:8000/a','http://example.invalid:8001/b',False),
    ])
    def test_origin_comparison_uses_scheme_host_and_effective_port(self,source: str,target: str,retained: bool) -> None:
        """Pure Request controls never make DNS or external network requests."""
        request=Request(source,headers={'Authorization':'Basic synthetic-opaque-value'})
        result=_HTTPOnlyRedirect().redirect_request(request,None,302,'Found',Message(),target)
        assert result is not None and result.has_header('Authorization') is retained
