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
