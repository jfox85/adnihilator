"""Real overlapping feed responses must reconcile durable subscription errors."""
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Lock, Thread

import pytest

from web.models import Podcast
from web.services.feed_sync import FeedSyncError, sync_podcast
from .test_feed_policy import web


@pytest.mark.parametrize('run',[1,2])
def test_last_healthy_commit_clears_error_from_concurrent_failed_sync(web: tuple, run: int) -> None:
    """A session's stale NULL snapshot must not hide an intervening saved error."""
    client,factory=web
    entered,release=Event(),Event()
    lock=Lock()
    count=0

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            nonlocal count
            with lock:
                count+=1
                ordinal=count
            if ordinal==1:
                entered.set()
                assert release.wait(timeout=5)
                self.send_response(200);self.end_headers()
                self.wfile.write(b'<rss version="2.0"><channel><title>Success won</title></channel></rss>')
            else:
                self.send_response(503);self.end_headers();self.wfile.write(b'failed overlap')

        def log_message(self,*args: object) -> None:
            pass

    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    with factory() as db:
        p=Podcast(source_rss_url=f'http://127.0.0.1:{server.server_port}/race-{run}',title='Before')
        db.add(p);db.commit();pid=p.id

    def healthy() -> int:
        """Hold a real feed response after loading the old error view."""
        with factory() as db:
            p=db.get(Podcast,pid)
            assert p.last_sync_error is None
            return sync_podcast(db,p)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(healthy)
            try:
                assert entered.wait(timeout=3)
                with factory() as db:
                    with pytest.raises(FeedSyncError,match='503'):
                        sync_podcast(db,db.get(Podcast,pid))
                with factory() as db:
                    assert '503' in db.get(Podcast,pid).last_sync_error
            finally:
                release.set()
            assert future.result(timeout=5)==0
        with factory() as db:
            p=db.get(Podcast,pid)
            assert p.title=='Success won' and p.last_synced_at is not None
            assert p.last_sync_error is None
    finally:
        release.set();server.shutdown();server.server_close();thread.join(timeout=2)
