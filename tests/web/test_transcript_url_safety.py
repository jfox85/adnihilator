"""Credential-bearing provenance is filtered at each transcript read boundary."""

import hashlib
import json

import pytest

from tests.web.test_transcript_ui_regressions import transcript_ui_client


@pytest.mark.parametrize("userinfo", [":SYNTHETIC_PROVENANCE_SECRET@", "@", ":@", "reader:SYNTHETIC_PROVENANCE_SECRET@"])
def test_current_and_retained_userinfo_is_filtered_without_rewriting_content(
    transcript_ui_client: tuple, monkeypatch: pytest.MonkeyPatch, userinfo: str
) -> None:
    """Actual bearer REST/MCP and admin HTML must also protect old snapshots."""
    client, factory = transcript_ui_client
    from web.models import Episode, TranscriptVersion
    from sqlalchemy import select

    url = f"https://{userinfo}example.invalid/audio.mp3"
    with factory() as db:
        episode = db.get(Episode, "ui-episode")
        episode.source_url = episode.original_audio_url = url
        version = db.scalar(select(TranscriptVersion))
        stored = json.loads(version.episode_metadata_json)
        stored.update(source_url=url, original_audio_url=url)
        version.episode_metadata_json = json.dumps(stored)
        canonical, snapshot, checksum = version.canonical_json, version.episode_metadata_json, version.checksum
        db.commit()

    token = "synthetic-url-safety-read-token"
    monkeypatch.setenv("TRANSCRIPT_READ_TOKEN_SHA256", hashlib.sha256(token.encode()).hexdigest())
    headers = {"Authorization": "Bearer " + token}
    response = client.get("/api/transcripts/episodes/ui-episode", headers=headers, auth=None)
    assert response.status_code == 200
    passage = response.json()
    for identity in (passage["episode"], passage["version_episode"]):
        assert identity["source_url"] is None
        assert identity["original_audio_url"] is None
    assert passage["checksum"] == checksum
    assert passage["segments"][0]["text"] == "a remembered wombat conversation"
    mcp = client.post("/mcp", headers={**headers, "Accept": "application/json, text/event-stream"}, auth=None,
                      json=dict(jsonrpc="2.0", id=1, method="tools/call", params=dict(name="read_passage", arguments=dict(episode_id="ui-episode"))))
    assert mcp.status_code == 200 and mcp.json()["result"]["isError"] is False
    assert url not in mcp.text and "SYNTHETIC_PROVENANCE_SECRET" not in mcp.text
    html = client.get("/transcripts/episodes/ui-episode")
    assert html.status_code == 200
    assert url not in html.text and "SYNTHETIC_PROVENANCE_SECRET" not in html.text
    with factory() as db:
        version = db.scalar(select(TranscriptVersion))
        assert (version.canonical_json, version.episode_metadata_json, version.checksum) == (canonical, snapshot, checksum)


def test_plain_provenance_urls_and_transcript_content_remain_usable(transcript_ui_client: tuple) -> None:
    """No-userinfo URLs remain visible and available as reader links."""
    client, _ = transcript_ui_client
    response = client.get("/api/transcripts/episodes/ui-episode")
    assert response.status_code == 200
    assert response.json()["episode"]["original_audio_url"] == "https://example.invalid/audio.mp3"
    assert response.json()["version_episode"]["original_audio_url"] == "https://example.invalid/audio.mp3"
    assert 'href="https://example.invalid/audio.mp3"' in client.get("/transcripts/episodes/ui-episode").text
