"""Real HTML/SQLite regressions for the independently reported transcript flows."""

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from urllib.parse import urldefrag

from bs4 import BeautifulSoup
from fastapi.testclient import TestClient
import pytest

from adnihilator.transcript_document import TranscriptDocument


@pytest.fixture
def transcript_ui_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[tuple]:
    """Use an explicitly migrated disposable database and synthetic auth only."""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "transcript-ui.sqlite"))
    monkeypatch.setenv("ADMIN_USERNAME", "synthetic-admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "synthetic-password")
    monkeypatch.setenv("WORKER_API_KEY", "synthetic-worker")
    monkeypatch.setenv("FEED_SYNC_ENABLED", "false")
    monkeypatch.delenv("TRANSCRIPT_SCHEMA_OPT_IN", raising=False)
    from web.app import app
    from web import dependencies
    from web.models import Episode, Podcast
    from web.services.transcripts import store
    from web.transcript_migration import upgrade

    with TestClient(app, raise_server_exceptions=False) as client:
        client.auth = ("synthetic-admin", "synthetic-password")
        upgrade(dependencies.SessionFactory.kw["bind"])
        with dependencies.SessionFactory() as db:
            db.add(Podcast(id="ui-podcast", title="Synthetic podcast", source_rss_url="https://example.invalid/feed"))
            db.flush()
            db.add(Episode(id="ui-episode", podcast_id="ui-podcast", guid="ui-guid", title="Synthetic conversation",
                           published_at=datetime(2026, 9, 30), original_audio_url="https://example.invalid/audio.mp3", status="complete"))
            db.commit()
            store(db, ui_document(["a remembered wombat conversation"], start=3444.32))
        yield client, dependencies.SessionFactory


def ui_document(texts: list[str], start: float = 0) -> TranscriptDocument:
    """Build accepted original-audio source content with stable identities."""
    return TranscriptDocument(episode_id="ui-episode", podcast_id="ui-podcast", guid="ui-guid",
                              original_audio_url="https://example.invalid/audio.mp3", source="synthetic-fixture",
                              segments=[dict(index=n, start=start + n * 3, end=start + n * 3 + 2.16, text=text)
                                        for n, text in enumerate(texts)])


class TestDefaultSearchForm:
    @pytest.mark.parametrize("dates", [{}, {"after": "2020-01-01"}, {"before": "2030-01-01"}])
    def test_native_get_form_defaults_and_partially_blank_dates_work(self, transcript_ui_client: tuple, dates: dict) -> None:
        """Serialize the actual emitted fields, including browser-submitted blanks."""
        client, _ = transcript_ui_client
        form = BeautifulSoup(client.get("/transcripts", headers={"Accept": "text/html"}).text, "html.parser").select_one("form.transcript-form")
        values = {field["name"]: field.get("value", "") for field in form.select("input[name]") if field.get("type") != "checkbox"}
        values.update(podcast_id="ui-podcast", query="wombat", **dates)
        result = client.get(form["action"], params=values, headers={"Accept": "text/html"})
        assert result.status_code == 200
        assert BeautifulSoup(result.text, "html.parser").select("article.card a")
        assert "wombat" in result.text


class TestTranscriptJSONErrors:
    def test_json_error_contract_is_unchanged(self, transcript_ui_client: tuple) -> None:
        client, _ = transcript_ui_client
        response = client.get("/transcripts/episodes/no-such-episode", headers={"Accept": "application/json"})
        assert response.status_code == 404
        assert response.json() == {"detail": "Episode not found"}


class TestHumanTranscriptMetadata:
    def test_dates_and_original_times_are_readable_without_changing_source(self, transcript_ui_client: tuple) -> None:
        client, factory = transcript_ui_client
        search = BeautifulSoup(client.get("/transcripts?podcast_id=ui-podcast&query=wombat").text, "html.parser").get_text(" ", strip=True)
        reader = BeautifulSoup(client.get("/transcripts/episodes/ui-episode").text, "html.parser").get_text(" ", strip=True)
        assert "Sep 30, 2026" in search and "Sep 30, 2026" in reader
        assert "57:24" in reader and "57:26" in reader
        assert "2026-09-30T00:00:00" not in search + reader
        assert "3444.32" not in reader
        from web.services.transcripts import TranscriptReader
        with factory() as db:
            passage = TranscriptReader(db).passage("ui-episode")
            assert passage["segments"][0]["start"] == 3444.32
            assert passage["segments"][0]["end"] == 3446.48


class TestSearchCitationNavigation:
    def test_long_preceding_segments_cannot_exclude_the_cited_match(self, transcript_ui_client: tuple) -> None:
        """Follow the real emitted link instead of guessing a reader offset."""
        client, factory = transcript_ui_client
        from web.services.transcripts import store
        with factory() as db:
            store(db, ui_document(["contextword " * 833] * 12 + ["rareword independent match"]))
        search = BeautifulSoup(client.get("/transcripts?podcast_id=ui-podcast&query=rareword").text, "html.parser")
        link = search.select_one("article.card h3 a")["href"]
        target, fragment = urldefrag(link)
        reader = BeautifulSoup(client.get(target).text, "html.parser")
        anchor = reader.find(id=fragment)
        assert anchor is not None
        assert "rareword" in anchor.get_text()


class TestSearchWindowFocus:
    @pytest.mark.parametrize("query,phrase,texts,focus", [
        ("wombat", False, ["plain context"] * 30 + ["wombat target"] + ["plain context"] * 10, 30),
        ("cafe", False, ["plain context"] * 30 + ["café target"] + ["plain context"] * 10, 30),
        ("silver otter", True, ["plain context"] * 30 + ["silver", "otter"] + ["plain context"] * 10, 30),
    ])
    def test_derived_focus_keeps_window_reference_and_points_to_actual_match(
        self, transcript_ui_client: tuple, query: str, phrase: bool, texts: list[str], focus: int
    ) -> None:
        """FTS tokenization, including accents/phrase seams, determines focus."""
        client, factory = transcript_ui_client
        from web.services.transcripts import store
        with factory() as db:
            store(db, ui_document(texts))
        hit = client.get("/api/transcripts/search", params={"podcast_id": "ui-podcast", "query": query, "phrase": str(phrase).lower()}).json()["hits"][0]
        assert hit["focus_ordinal"] == focus
        assert hit["match_ordinal"] <= focus <= hit["match_ordinal"] + 2
        for segment in hit["segments"]:
            assert segment["reference"].endswith("/" + str(segment["ordinal"]))
        page = BeautifulSoup(client.get("/transcripts", params={"podcast_id": "ui-podcast", "query": query, "phrase": str(phrase).lower()}).text, "html.parser")
        target, fragment = urldefrag(page.select_one("article.card h3 a")["href"])
        reader = BeautifulSoup(client.get(target).text, "html.parser")
        assert fragment == "segment-" + str(focus)
        assert texts[focus] in reader.find(id=fragment).get_text()
