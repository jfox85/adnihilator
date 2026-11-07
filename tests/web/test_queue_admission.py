"""Feed parsing and HTTP admission rules for safe queue transitions."""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import feedparser
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event

from web.models import Episode, Podcast

AUTH = ("synthetic-admin", "synthetic-password")


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[tuple]:
    """Keep feed parsing and HTTP requests inside a temporary database."""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "admission.db"))
    monkeypatch.setenv("ADMIN_USERNAME", AUTH[0])
    monkeypatch.setenv("ADMIN_PASSWORD", AUTH[1])
    monkeypatch.setenv("WORKER_API_KEY", "synthetic-worker-key")
    from web.app import app
    from web import dependencies
    with TestClient(app) as client:
        with dependencies.SessionFactory() as db:
            podcast = Podcast(source_rss_url="http://127.0.0.1/feed.xml", title="Synthetic", auto_process=True)
            db.add(podcast)
            db.commit()
            podcast_id = podcast.id
        yield client, dependencies.SessionFactory, podcast_id


def add_episode(factory: Any, podcast_id: str, status: str, url: str | None) -> str:
    """Create a synthetic episode with retry metadata worth preserving."""
    import uuid
    with factory() as db:
        episode = Episode(podcast_id=podcast_id, guid=str(uuid.uuid4()), status=status,
                          original_audio_url=url, retry_count=1, error_message="preserved error")
        db.add(episode)
        db.commit()
        return episode.id


class TestAudioAdmission:
    def test_rendered_episode_and_queue_actions_match_admission(self, harness: tuple) -> None:
        """Unavailable audio and active work cannot expose misleading actions."""
        client, factory, podcast_id = harness
        invalid = add_episode(factory, podcast_id, "failed", None)
        valid = add_episode(factory, podcast_id, "skipped", "http://127.0.0.1/audio.mp3")
        active = add_episode(factory, podcast_id, "processing", "http://127.0.0.1/audio.mp3")
        episodes = client.get(f"/podcasts/{podcast_id}/episodes", auth=AUTH)
        assert episodes.status_code == 200
        def actions(episode_id: str) -> str:
            return re.search(rf'id="actions-{episode_id}">(.*?)</div>', episodes.text, re.S).group(1)
        assert "queueEpisode(" not in actions(invalid)
        assert "queueEpisode(" in actions(valid)
        assert "skipEpisode(" not in actions(active)
        assert f'id="audio-{invalid}" >No usable audio enclosure' in episodes.text
        queue = client.get("/queue", auth=AUTH)
        assert queue.status_code == 200
        assert f"skipEpisode('{podcast_id}', '{active}')" not in queue.text
        assert f"retryEpisode('{podcast_id}', '{invalid}')" not in queue.text
        assert "No usable audio enclosure" in queue.text

    @pytest.mark.skipif(not shutil.which("node"), reason="Node is unavailable")
    def test_ajax_consumes_exact_bulk_ids_and_keeps_409_state(self, harness: tuple) -> None:
        """Execute rendered JavaScript against actual HTTP endpoint results."""
        client, factory, podcast_id = harness
        valid = add_episode(factory, podcast_id, "skipped", "http://127.0.0.1/audio.mp3")
        add_episode(factory, podcast_id, "skipped", None)
        active = add_episode(factory, podcast_id, "processing", "http://127.0.0.1/audio.mp3")
        html = client.get(f"/podcasts/{podcast_id}/episodes", auth=AUTH).text
        script = next(script for script in re.findall(r"<script>(.*?)</script>", html, re.S) if "const podcastId =" in script)
        bulk = client.post(f"/podcasts/{podcast_id}/episodes/queue-all", auth=AUTH).json()
        conflicts = [client.post(f"/podcasts/{podcast_id}/episodes/{active}/{action}", auth=AUTH).status_code for action in ("queue", "skip")]
        program = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const updates = [], alerts = [];
const sandbox = {window:{location:{origin:'http://fixture.invalid'}},
 document:{addEventListener(){},querySelectorAll(){throw Error('Bulk changed unreturned rows')}},
 setInterval(){return 1},setTimeout(){},confirm(){return true},alert(text){alerts.push(text)},console,
 fetch:async()=>({ok:true,json:async()=>input.bulk})};
vm.createContext(sandbox); vm.runInContext(input.script,sandbox);
sandbox.updateEpisodeUI=(id,data)=>updates.push([id,data.status]);
(async()=>{
 await sandbox.queueAll(); assert.deepStrictEqual(updates,[[input.valid,'pending']]);
 updates.length=0;
 for(const [index,action] of ['queueEpisode','skipEpisode'].entries()){
  sandbox.fetch=async()=>({ok:input.conflicts[index]<400,status:input.conflicts[index]});
  await sandbox[action]('podcast','active');
 }
 assert.deepStrictEqual(updates,[]); assert.strictEqual(alerts.length,2);
})().catch(error=>{console.error(error);process.exitCode=1});
"""
        result = subprocess.run(["node", "-e", program], input=json.dumps({"script": script, "bulk": bulk, "valid": valid, "conflicts": conflicts}), capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

    @pytest.mark.parametrize("enclosures,expected", [
        ('<enclosure url="http://127.0.0.1/image.jpg" type="image/jpeg"/><enclosure url="http://127.0.0.1/show.mp3" type="audio/mpeg"/>', "http://127.0.0.1/show.mp3"),
        ('<enclosure url="http://127.0.0.1/first.mp3"/><enclosure url="http://127.0.0.1/preferred" type="audio/mpeg"/>', "http://127.0.0.1/preferred"),
        ('<enclosure url="http://127.0.0.1/show.MP3?signature=synthetic"/>', "http://127.0.0.1/show.MP3?signature=synthetic"),
        ('<enclosure url="http://127.0.0.1/fake.mp3" type="image/jpeg"/>', None),
        ('<enclosure url="file:///tmp/show.mp3" type="audio/mpeg"/>', None),
        ('<enclosure url="ftp://127.0.0.1/show.mp3" type="audio/mpeg"/>', None),
        ('<enclosure url="http:///show.mp3" type="audio/mpeg"/>', None),
        ('<enclosure type="audio/mpeg"/>', None),
        ('<enclosure url="" type="audio/mpeg"/>', None),
        ('<enclosure url=" audio.mp3" type="audio/mpeg"/>', None),
        ('<enclosure url="http://127.0.0.1/image.jpg"/>', None),
        ('', None),
    ])
    def test_actual_rss_parser_selects_usable_audio(
        self, harness: tuple, enclosures: str, expected: str | None
    ) -> None:
        """Use real feedparser output, including multiple enclosure choices."""
        _, factory, podcast_id = harness
        from web.services.feed_sync import sync_podcast
        rss = f'<rss version="2.0"><channel><title>Synthetic</title><item><guid>entry-1</guid><title>Episode</title>{enclosures}</item></channel></rss>'
        parsed = feedparser.parse(rss.encode())
        assert len(parsed.entries) == 1
        with factory() as db:
            with patch("web.services.feed_sync.feedparser.parse", return_value=parsed):
                assert sync_podcast(db, db.get(Podcast, podcast_id)) == 1
            episode = db.query(Episode).one()
            assert episode.original_audio_url == expected
            assert episode.status == ("pending" if expected else "skipped")

    @pytest.mark.parametrize("url", [None, "", "file:///tmp/show.mp3", "ftp://example.invalid/show.mp3", "http:///show.mp3", "https://example.invalid:invalid/show.mp3"])
    def test_individual_queue_rejects_unusable_existing_audio(self, harness: tuple, url: str | None) -> None:
        """Legacy missing and unsupported URLs cannot enter the queue manually."""
        client, factory, podcast_id = harness
        episode_id = add_episode(factory, podcast_id, "failed", url)
        response = client.post(f"/podcasts/{podcast_id}/episodes/{episode_id}/queue", auth=AUTH)
        assert response.status_code == 409, response.text
        with factory() as db:
            episode = db.get(Episode, episode_id)
            assert (episode.status, episode.retry_count, episode.error_message) == ("failed", 1, "preserved error")

    def test_bulk_queue_counts_only_eligible_skipped_or_failed(self, harness: tuple) -> None:
        """Bulk admission retains existing scopes and leaves active rows alone."""
        client, factory, podcast_id = harness
        rows = [(status, url, add_episode(factory, podcast_id, status, url)) for status, url in [
            ("skipped", "http://127.0.0.1/audio.mp3"),
            ("failed", "https://fixture.invalid/audio"),
            ("skipped", None), ("failed", "file:///tmp/audio.mp3"),
            ("processing", "https://fixture.invalid/audio"),
            ("pending", "https://fixture.invalid/audio"),
            ("complete", "https://fixture.invalid/audio"),
        ]]
        response = client.post(f"/podcasts/{podcast_id}/episodes/queue-all", auth=AUTH)
        assert response.status_code == 200
        assert response.json()["count"] == 2
        assert set(response.json()["episode_ids"]) == {rows[0][2], rows[1][2]}
        with factory() as db:
            for index, (status, url, episode_id) in enumerate(rows):
                episode = db.get(Episode, episode_id)
                assert episode.status == ("pending" if index < 2 else status)
                assert episode.retry_count == (0 if index < 2 else 1)

    @pytest.mark.parametrize("action", ["queue", "skip"])
    def test_processing_job_cannot_be_reset(self, harness: tuple, action: str) -> None:
        """Admin actions cannot change an active worker's processing status."""
        client, factory, podcast_id = harness
        episode_id = add_episode(factory, podcast_id, "processing", "https://fixture.invalid/audio")
        response = client.post(f"/podcasts/{podcast_id}/episodes/{episode_id}/{action}", auth=AUTH)
        assert response.status_code == 409
        assert response.json()["detail"] == "Episode is processing"
        with factory() as db:
            episode = db.get(Episode, episode_id)
            assert (episode.status, episode.retry_count, episode.error_message) == ("processing", 1, "preserved error")

    def test_pending_queue_is_idempotent(self, harness: tuple) -> None:
        """Double Queue preserves the attempt's retry and diagnostic metadata."""
        client, factory, podcast_id = harness
        episode_id = add_episode(factory, podcast_id, "pending", "https://fixture.invalid/audio")
        response = client.post(f"/podcasts/{podcast_id}/episodes/{episode_id}/queue", auth=AUTH)
        assert response.json() == {"status": "already_queued"}
        with factory() as db:
            episode = db.get(Episode, episode_id)
            assert (episode.status, episode.retry_count, episode.error_message) == ("pending", 1, "preserved error")

    @pytest.mark.parametrize("action", ["queue", "skip"])
    def test_admin_transition_loses_to_intervening_worker_claim(self, harness: tuple, action: str) -> None:
        """A real claim between admin read and write remains processing."""
        client, factory, podcast_id = harness
        episode_id = add_episode(factory, podcast_id, "failed" if action == "queue" else "pending", "https://fixture.invalid/audio")
        engine = factory.kw["bind"]
        fired = False

        def claim_before_admin_write(
            conn: Any, cursor: Any, statement: str, parameters: Any,
            context: Any, executemany: bool,
        ) -> None:
            nonlocal fired
            if fired or not statement.startswith("UPDATE episodes SET status"):
                return
            fired = True
            if action == "queue":
                # Another administrator queues the failed episode just before
                # a worker claims it, while the first action still has old data.
                with factory() as db:
                    db.query(Episode).filter_by(id=episode_id).update({Episode.status: "pending"})
                    db.commit()
            other = TestClient(client.app)
            try:
                claimed = other.post("/api/queue/claim", headers={"X-Worker-API-Key": "synthetic-worker-key"})
                assert claimed.status_code == 200
                assert claimed.json()["id"] == episode_id
            finally:
                other.close()

        event.listen(engine, "before_cursor_execute", claim_before_admin_write)
        try:
            response = client.post(f"/podcasts/{podcast_id}/episodes/{episode_id}/{action}", auth=AUTH)
        finally:
            event.remove(engine, "before_cursor_execute", claim_before_admin_write)
        assert fired
        assert response.status_code == 409
        with factory() as db:
            episode = db.get(Episode, episode_id)
            assert (episode.status, episode.retry_count, episode.error_message) == ("processing", 1, "preserved error")

    def test_status_poll_exposes_audio_eligibility(self, harness: tuple) -> None:
        """The UI can consistently hide unavailable queue actions after polling."""
        client, factory, podcast_id = harness
        valid = add_episode(factory, podcast_id, "skipped", "http://127.0.0.1/audio.mp3")
        invalid = add_episode(factory, podcast_id, "skipped", None)
        episodes = client.get(f"/podcasts/{podcast_id}/episodes/status", auth=AUTH).json()["episodes"]
        assert episodes[valid]["audio_eligible"] is True
        assert episodes[invalid]["audio_eligible"] is False
