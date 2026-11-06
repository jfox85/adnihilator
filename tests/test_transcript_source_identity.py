"""Confirmed identity and importer-shape failures with neighboring controls."""

import json
from pathlib import Path

import pytest

from adnihilator.transcript_document import audio_identity
from scripts.import_transcripts import import_local
from tests.web.test_transcript_ui_regressions import transcript_ui_client


class TestEnclosureIdentity:
    @pytest.mark.parametrize("url", [
        "https://publisher-one.example/TCP12345.mp3?x=1",
        "https://unrelated-provider.example/TCP12345.mp3?different=1",
        "https://unrelated-provider.example/traffic.megaphone.fm/TCP12345.mp3",
        "https://traffic.megaphone.fm.evil.example/TCP12345.mp3",
    ])
    def test_unknown_providers_remain_exact(self, url: str) -> None:
        assert audio_identity(url) == url

    @pytest.mark.parametrize("url", [
        "https://traffic.megaphone.fm/TCP12345.mp3?updated=2",
        "https://chrt.fm/track/synthetic/traffic.megaphone.fm/TCP12345.mp3?ad=2",
        "https://dts.podtrac.com/redirect.mp3/traffic.megaphone.fm/TCP12345.mp3",
    ])
    def test_explicit_megaphone_and_known_tracking_paths_share_identity(self, url: str) -> None:
        assert audio_identity(url) == "TCP12345.mp3"

    def test_worker_http_rejects_same_basename_from_another_provider(self, transcript_ui_client: tuple) -> None:
        client, factory = transcript_ui_client
        from web.models import Episode
        with factory() as db:
            db.get(Episode, "ui-episode").original_audio_url = "https://publisher-one.example/TCP12345.mp3"
            db.commit()
        document = dict(episode_id="ui-episode", podcast_id="ui-podcast", guid="ui-guid", source="synthetic",
                        original_audio_url="https://different-provider.example/TCP12345.mp3",
                        segments=[dict(index=0, start=0, end=1, text="wrong source")])
        assert client.put("/api/worker/transcripts/ui-episode", headers={"X-Worker-API-Key": "synthetic-worker"}, json={"document": document}).status_code == 409
        assert client.get("/api/transcripts/episodes/ui-episode").json()["segments"][0]["text"] == "a remembered wombat conversation"


class TestImportArtifactShapes:
    @pytest.mark.parametrize("invalid", [[], {"audio_path": None}, {"audio_path": 123}, "not a document"])
    def test_bad_shape_does_not_abort_a_valid_neighbor(self, tmp_path: Path, invalid: object) -> None:
        artifacts = tmp_path / "artifacts"
        (artifacts / "p").mkdir(parents=True)
        feed = tmp_path / "feed.xml"
        feed.write_text('<rss><channel><title>Synthetic</title><item><guid>g</guid><title>One</title><pubDate>Wed, 30 Sep 2026 10:00:00 GMT</pubDate><enclosure url="https://example.invalid/one.mp3"/></item></channel></rss>')
        (artifacts / "p/good.json").write_text(json.dumps(dict(audio_path="https://example.invalid/one.mp3", segments=[dict(index=0, start=.123, end=1.456, text="healthy neighbor")], ad_spans=[])))
        (artifacts / "p/bad.json").write_text(json.dumps(invalid))
        report = import_local(tmp_path / "fixture.sqlite", feed, artifacts, "p", "https://example.invalid/feed")
        assert report["available"] == 1
        assert report["invalid_artifacts"] == 1


class TestNestedArtifactIsolation:
    @pytest.mark.parametrize("changes", [
        {"segments": 1, "duration": 3},
        {"segments": [None], "duration": 3},
        {"model_info": ["not metadata"]},
    ])
    def test_malformed_latest_artifact_cannot_abort_healthy_older_episode(self, tmp_path: Path, changes: dict) -> None:
        """Use actual importer iteration, not just standalone parser rejection."""
        artifacts = tmp_path / "artifacts" / "p"
        artifacts.mkdir(parents=True)
        feed = tmp_path / "feed.xml"
        feed.write_text('<rss><channel><title>Synthetic</title>' + ''.join(
            f'<item><guid>{name}</guid><title>{name}</title><pubDate>{date}</pubDate><enclosure url="https://example.invalid/{name}.mp3"/></item>'
            for name, date in [("bad", "Wed, 30 Sep 2026 10:00:00 GMT"), ("good", "Tue, 29 Sep 2026 10:00:00 GMT")]
        ) + '</channel></rss>')
        for name in ("bad", "good"):
            data = dict(audio_path=f"https://example.invalid/{name}.mp3", segments=[dict(index=0, start=0, end=2, text="healthy independent content")])
            if name == "bad":
                data.update(changes)
            (artifacts / f"{name}.json").write_text(json.dumps(data))
        report = import_local(tmp_path / "fixture.sqlite", feed, artifacts.parent, "p", "https://example.invalid/feed")
        assert report["failed"] == 1
        assert report["available"] == 1
        import sqlite3
        with sqlite3.connect(tmp_path / "fixture.sqlite") as db:
            assert db.execute("SELECT count(*) FROM transcript_versions").fetchone()[0] == 1
            assert "healthy independent content" in db.execute("SELECT canonical_json FROM transcript_versions").fetchone()[0]


@pytest.mark.parametrize("artifact_url,available", [
    ("https://publisher-two.invalid/XYZ123.mp3", False),
    ("https://publisher-one.invalid/XYZ123.mp3", True),
    ("https://publisher-one.invalid/ABC999.mp3", False),
])
def test_prototype_inventory_and_fts_do_not_attach_another_publishers_audio(tmp_path: Path, artifact_url: str, available: bool) -> None:
    """Exercise inventory through SQLite retrieval with exact and absent controls."""
    from tests.test_transcript_search_poc import search as prototype
    artifacts = tmp_path / "artifacts" / "p"
    artifacts.mkdir(parents=True)
    feed = tmp_path / "feed.xml"
    feed.write_text('<rss><channel><title>Synthetic</title><item><guid>g</guid><title>One</title><pubDate>Wed, 30 Sep 2026 10:00:00 GMT</pubDate><enclosure url="https://publisher-one.invalid/XYZ123.mp3"/></item></channel></rss>')
    (artifacts / "worker.json").write_text(json.dumps(dict(audio_path=artifact_url, segments=[dict(index=0, start=0, end=2, text="uniquely remembered quokkaword")])) )
    report = prototype.inventory(feed, artifacts.parent, "p", 20)
    assert report["episodes"][0]["coverage"] == ("available" if available else "missing")
    connection = prototype.connect(tmp_path / "prototype.sqlite")
    try:
        prototype.ingest(connection, report)
        hits = prototype.search(connection, "quokkaword", "p")["hits"]
        assert bool(hits) is available
    finally:
        connection.close()
