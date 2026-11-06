"""Owner-selected feed policies exercised through real HTTP, SQLite and routes."""
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from web.database import init_db
from web.models import Episode, Podcast
from web.services.feed_sync import FeedSyncError, sync_podcast

AUTH = ("policy-admin", "policy-password")
RSS = b'<rss version="2.0"><channel><title>Private fixture</title><item><title>Episode</title><enclosure url="http://127.0.0.1/episode.mp3?sig=first" type="application/octet-stream"/></item></channel></rss>'


@contextmanager
def feed_server() -> Iterator[tuple[str, dict[str, object]]]:
    """Serve only synthetic local responses, including unsafe redirect attempts."""
    state: dict[str, object] = {"body": RSS, "status": 200, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            state["requests"].append(self.path)
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", state.get("redirect", "file:///tmp/forbidden-policy-feed.xml"))
            elif self.path == "/private-redirect":
                self.send_response(302)
                self.send_header("Location", "/feed")
            else:
                self.send_response(state["status"])
                self.send_header("Content-Type", "application/rss+xml")
            self.end_headers()
            if self.path not in {"/redirect", "/private-redirect"}:
                self.wfile.write(state["body"])

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[tuple]:
    """Use an authenticated app with a disposable database, never a worker."""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "policy.db"))
    monkeypatch.setenv("ADMIN_USERNAME", AUTH[0])
    monkeypatch.setenv("ADMIN_PASSWORD", AUTH[1])
    monkeypatch.setenv("WORKER_API_KEY", "policy-synthetic-worker")
    from web.app import app
    from web import dependencies
    with TestClient(app) as client:
        yield client, dependencies.SessionFactory


class TestFeedPolicy:
    def test_failed_subscription_persists_visible_error_and_retry_recovers(self, web: tuple) -> None:
        """A failed initial subscription survives a new request and a successful retry."""
        client, factory = web
        with feed_server() as (url, state):
            state.update(status=503, body=b"temporary fixture outage")
            assert client.post("/podcasts", auth=AUTH, data={"source_rss_url": url + "/feed"}, follow_redirects=False).status_code == 303
            with factory() as db:
                p = db.query(Podcast).one()
                pid = p.id
                assert p.last_sync_error and p.last_synced_at is None
                assert db.query(Episode).count() == 0
            # A separate request/session must expose the durable error and retry.
            page = client.get("/", auth=AUTH)
            assert "Untitled" not in page.text
            assert "Subscription needs attention" in page.text and "Retry" in page.text
            assert 'role="alert"' in page.text
            state.update(status=200, body=RSS)
            response = client.post(f"/podcasts/{pid}/sync", auth=AUTH)
            assert response.status_code == 200
            with factory() as db:
                p = db.get(Podcast, pid)
                assert p.last_sync_error is None and p.title == "Private fixture"
                assert db.query(Episode).one().original_audio_url.endswith("sig=first")
            assert "Subscription needs attention" not in client.get("/", auth=AUTH).text

    @pytest.mark.parametrize("source", ["/tmp/private.xml", "file:///tmp/private.xml", "ftp://127.0.0.1/feed", "data:text/xml,secret"])
    def test_non_http_subscription_is_rejected_without_reading(self, web: tuple, source: str) -> None:
        """Unsupported sources must never reach the parser or subscription storage."""
        client, factory = web
        with patch("web.services.feed_sync.feedparser.parse") as parse:
            response = client.post("/podcasts", auth=AUTH, data={"source_rss_url": source})
        assert response.status_code == 422
        parse.assert_not_called()
        with factory() as db:
            assert db.query(Podcast).count() == 0

    @pytest.mark.parametrize("path", ["/feed", "/private-redirect"])
    def test_private_http_and_private_redirect_work_without_guid(self, web: tuple, path: str) -> None:
        """Private feeds and redirects deduplicate stable fallback identities on refresh."""
        client, factory = web
        with feed_server() as (url, state):
            client.post("/podcasts", auth=AUTH, data={"source_rss_url": url + path})
            with factory() as db:
                episode = db.query(Episode).one()
                pid, guid, eid = episode.podcast_id, episode.guid, episode.id
                assert guid.startswith("enclosure-sha256:")
            state["body"] = RSS.replace(b"sig=first", b"sig=rotated")
            assert client.post(f"/podcasts/{pid}/sync", auth=AUTH).json()["episodes_added"] == 0
            with factory() as db:
                episode = db.query(Episode).one()
                assert (episode.id, episode.guid) == (eid, guid)
                # Identity dedupe must not silently rewrite an existing URL.
                assert episode.original_audio_url.endswith("sig=first")

    @pytest.mark.parametrize("target", ["file:///tmp/forbidden-policy-feed.xml", "ftp://127.0.0.1/forbidden.xml", "data:text/xml,private"])
    def test_redirect_to_file_is_failed_and_persisted(self, web: tuple, target: str) -> None:
        """Unsupported redirect schemes cannot open local files or FTP handlers."""
        client, factory = web
        with feed_server() as (url, state):
            state["redirect"] = target
            with patch("urllib.request.FTPHandler.ftp_open", side_effect=AssertionError("FTP must not open")) as ftp:
                client.post("/podcasts", auth=AUTH, data={"source_rss_url": url + "/redirect"})
                ftp.assert_not_called()
        with factory() as db:
            p = db.query(Podcast).one()
            assert p.last_sync_error and p.last_synced_at is None
            assert db.query(Episode).count() == 0

    @pytest.mark.parametrize("body", [b"<html><title>not a feed</title></html>", b"not XML", b""])
    def test_non_feed_200_is_error_not_success(self, web: tuple, body: bytes) -> None:
        """An HTML error page with HTTP 200 cannot be recorded as a successful feed."""
        client, factory = web
        with feed_server() as (url, state):
            state["body"] = body
            client.post("/podcasts", auth=AUTH, data={"source_rss_url": url + "/feed"})
        with factory() as db:
            p = db.query(Podcast).one()
            assert p.last_sync_error and p.last_synced_at is None

    @pytest.mark.parametrize("mime,expected", [("application/octet-stream", True), ("binary/octet-stream", True), ("application/octet-stream; x=1", True), ("image/jpeg", False), ("video/mp4", False), ("application/pdf", False)])
    def test_generic_binary_audio_extension_only(self, mime: str, expected: bool) -> None:
        """Only absent or generic-binary MIME may use recognized audio extensions."""
        from web.services.audio import select_audio_enclosure
        assert (select_audio_enclosure([{"href": "http://127.0.0.1/show.MP3?sig=x", "type": mime}]) is not None) is expected
        assert select_audio_enclosure([{"href": "http://127.0.0.1/no-extension", "type": mime}]) is None

    def test_completed_queue_is_rejected_without_mutating_output(self, web: tuple) -> None:
        """Ordinary Queue preserves completed output and retry metadata."""
        client, factory = web
        with factory() as db:
            p = Podcast(source_rss_url="http://127.0.0.1/feed")
            db.add(p);db.flush()
            e = Episode(podcast_id=p.id, guid="finished", status="complete", original_audio_url="http://127.0.0.1/a.mp3", processed_audio_key="retained.mp3", retry_count=2, error_message="retain diagnostic")
            db.add(e);db.commit();pid,eid=p.id,e.id
        assert client.post(f"/podcasts/{pid}/episodes/{eid}/queue", auth=AUTH).status_code == 409
        with factory() as db:
            e = db.get(Episode, eid)
            assert (e.status,e.processed_audio_key,e.retry_count,e.error_message) == ("complete","retained.mp3",2,"retain diagnostic")

    def test_additive_existing_database_upgrade_is_idempotent(self, tmp_path: Path) -> None:
        """Startup adds the nullable field once while preserving a legacy row."""
        engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
        # Produce the actual pre-change schema, including an existing row.
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE podcasts (id VARCHAR(36) PRIMARY KEY, feed_token VARCHAR(64), source_rss_url TEXT NOT NULL, title TEXT, description TEXT, artwork_url TEXT, auto_process BOOLEAN, last_synced_at DATETIME, created_at DATETIME, updated_at DATETIME)"))
            connection.execute(text("INSERT INTO podcasts (id,source_rss_url,title) VALUES ('legacy','http://127.0.0.1/feed','Keep me')"))
        init_db(engine);init_db(engine)
        with Session(engine) as db:
            p = db.get(Podcast, "legacy")
            assert p.title == "Keep me" and p.last_sync_error is None
        engine.dispose()

    def test_existing_guid_is_not_silently_backfilled(self, web: tuple) -> None:
        """Refreshing a legacy GUID never changes its missing URL or completed output."""
        client, factory = web
        rss = RSS.replace(b'<title>Episode</title>', b'<guid isPermaLink="false">legacy-guid</guid><title>Episode</title>')
        with feed_server() as (url, state):
            state["body"] = rss
            with factory() as db:
                p = Podcast(source_rss_url=url + "/feed")
                db.add(p);db.flush()
                e = Episode(podcast_id=p.id, guid="legacy-guid", original_audio_url=None, status="complete", processed_audio_key="keep.mp3")
                db.add(e);db.commit();pid,eid=p.id,e.id
            assert client.post(f"/podcasts/{pid}/sync", auth=AUTH).json()["episodes_added"] == 0
            with factory() as db:
                e = db.get(Episode, eid)
                assert (e.original_audio_url,e.status,e.processed_audio_key) == (None,"complete","keep.mp3")

    def test_completion_between_queue_read_and_write_wins(self, web: tuple) -> None:
        """An intervening durable completion wins against a stale Queue request."""
        from sqlalchemy import event
        client, factory = web
        with factory() as db:
            p = Podcast(source_rss_url="http://127.0.0.1/feed")
            db.add(p);db.flush()
            e = Episode(podcast_id=p.id,guid="race",status="failed",original_audio_url="http://127.0.0.1/a.mp3")
            db.add(e);db.commit();pid,eid=p.id,e.id
        engine=factory.kw["bind"]
        fired=False

        def complete_before_queue(conn: object, cursor: object, statement: str, parameters: object, context: object, executemany: bool) -> None:
            nonlocal fired
            if fired or not statement.startswith("UPDATE episodes SET status"):
                return
            fired=True
            with factory() as other:
                other.query(Episode).filter_by(id=eid).update({Episode.status:"complete",Episode.processed_audio_key:"won.mp3"})
                other.commit()
        event.listen(engine,"before_cursor_execute",complete_before_queue)
        try:
            assert client.post(f"/podcasts/{pid}/episodes/{eid}/queue",auth=AUTH).status_code==409
        finally:
            event.remove(engine,"before_cursor_execute",complete_before_queue)
        assert fired
        with factory() as db:
            e=db.get(Episode,eid)
            assert (e.status,e.processed_audio_key)==("complete","won.mp3")

    def test_identity_precedence_path_distinction_and_token_rotation(self) -> None:
        """Fallback identity ignores expiring query credentials while distinguishing paths."""
        from web.services.feed_sync import episode_identity
        audio="HTTPS://name:password@Example.INVALID:443/path.mp3?sig=first#part"
        assert episode_identity({"id":"explicit","link":"https://example.invalid/page"},audio)=="explicit"
        assert episode_identity({"link":"https://example.invalid/page"},audio)=="https://example.invalid/page"
        identity=episode_identity({},audio)
        assert identity==episode_identity({},"https://example.invalid/path.mp3?sig=rotated")
        assert identity!=episode_identity({},"https://example.invalid/other.mp3")
        assert identity!=episode_identity({},"http://example.invalid/path.mp3")
        assert episode_identity({},None) is None

    def test_failed_refresh_preserves_title_success_time_and_exposes_error(self, web: tuple) -> None:
        """An existing healthy subscription retains metadata through a failed retry."""
        client, factory = web
        with feed_server() as (url, state):
            client.post("/podcasts", auth=AUTH, data={"source_rss_url": url + "/feed"})
            with factory() as db:
                p = db.query(Podcast).one()
                pid, succeeded = p.id, p.last_synced_at
            state.update(status=503, body=b"outage")
            response=client.post(f"/podcasts/{pid}/sync",auth=AUTH)
            assert response.status_code==502 and response.json()["status"]=="failed"
            with factory() as db:
                p=db.get(Podcast,pid)
                assert p.title=="Private fixture" and p.last_synced_at==succeeded
                assert p.last_sync_error==response.json()["error"]
                assert db.query(Episode).count()==1
            listed=client.get('/podcasts',auth=AUTH).json()[0]
            assert listed['last_sync_error']==response.json()['error']
            assert 'Retry' in client.get('/',auth=AUTH).text
