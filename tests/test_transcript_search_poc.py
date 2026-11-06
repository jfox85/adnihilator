"""Standard-library prototype tests; never use worker data or network."""

import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location("transcript_search", Path(__file__).parents[1] / "prototypes/transcript_search.py")
search = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(search)


class TestTranscriptSearch(unittest.TestCase):
    """Prove identity, coverage, idempotence, and source-context behavior."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifacts = self.root / "artifacts"
        (self.artifacts / "p1").mkdir(parents=True)
        self.feed = self.root / "feed.xml"
        self.feed.write_text('''<rss><channel><title>Example Podcast</title>
          <item><guid>old</guid><title>Older</title><pubDate>Mon, 28 Sep 2026 10:00:00 GMT</pubDate><enclosure url="https://traffic.megaphone.fm/TCP100.mp3"/></item>
          <item><guid>new</guid><title>Newest</title><pubDate>Wed, 30 Sep 2026 12:00:00 +0200</pubDate><enclosure url="https://track/traffic.megaphone.fm/TCP200.mp3?updated=1"/></item>
          </channel></rss>''')
        self.segments = [dict(index=9, start=0.15, end=4.31, text="before context", words=[dict(word="before", start=0.15, end=1.0)]),
                         dict(index=12, start=5.125, end=9.999, text="Wonderful memoir with a telescope"),
                         dict(index=13, start=10.2, end=12.6, text="after context")]
        (self.artifacts / "p1" / "worker1.json").write_text(json.dumps(dict(audio_path="https://traffic.megaphone.fm/TCP100.mp3?updated=2", segments=self.segments)))
        self.connection = search.connect(self.root / "index.sqlite")
        self.addCleanup(self.connection.close)

    def report(self) -> dict:
        """Inventory fixture episodes."""
        return search.inventory(self.feed, self.artifacts, "p1", 20)

    def test_order_identity_and_missing_latest(self) -> None:
        """Newest by UTC publication, stable GUID identity, no fallback."""
        report = self.report()
        self.assertEqual([e["guid"] for e in report["episodes"]], ["new", "old"])
        self.assertEqual(report["episodes"][0]["published_at"], "2026-09-30T10:00:00+00:00")
        self.assertEqual(report["episodes"][1]["worker_episode_id"], "worker1")
        search.ingest(self.connection, report)
        result = search.search(self.connection, "memoir", "p1", latest=True)
        self.assertEqual(result["status"], "latest_transcript_unavailable")
        self.assertEqual(result["hits"], [])
        self.assertEqual(report["episodes"][0]["id"], self.report()["episodes"][0]["id"])

    def test_repeat_ingest_and_canonical_timestamps(self) -> None:
        """Repeat ingestion preserves every segment, word timing, and decimal."""
        first = search.ingest(self.connection, self.report())
        self.assertEqual(first, search.ingest(self.connection, self.report()))
        self.assertEqual(first, dict(episodes=2, segments=3, passages=3))
        canonical = json.loads(self.connection.execute("SELECT canonical_json FROM episodes WHERE guid='old'").fetchone()[0])
        self.assertEqual(canonical["segments"], self.segments)
        self.assertEqual(self.connection.execute("SELECT start FROM segments WHERE ordinal=1").fetchone()[0], 5.125)

    def test_search_context_filters_and_negative(self) -> None:
        """Lexical hit has surrounding conversation; filters and negatives work."""
        search.ingest(self.connection, self.report())
        result = search.search(self.connection, "What memoir was he talking about?", "Example Podcast", before="2026-09-29", context_seconds=10)
        self.assertEqual(result["status"], "hits")
        self.assertEqual([s["text"] for s in result["hits"][0]["context"]], [s["text"] for s in self.segments])
        self.assertEqual(result["timestamp_basis"], "original_audio")
        for kwargs in [dict(query="xylophonemars"), dict(query="memoir", podcast="other"), dict(query="memoir", after="2026-09-29")]:
            self.assertEqual(search.search(self.connection, **kwargs)["status"], "no_hits")

    def test_changed_transcript_replaces_old_fts(self) -> None:
        """Updates remove stale search tokens instead of retaining duplicate hits."""
        report = self.report()
        search.ingest(self.connection, report)
        report["episodes"][1]["segments"] = [dict(start=1, end=2, text="new mango")]
        search.ingest(self.connection, report)
        self.assertEqual(search.search(self.connection, "telescope")["status"], "no_hits")
        self.assertEqual(search.search(self.connection, "mango")["status"], "hits")

    def test_duplicate_media_identity_is_ambiguous(self) -> None:
        """Conflicting worker IDs cannot silently identify an episode."""
        p = self.artifacts / "p1" / "worker1.json"
        p.with_name("worker2.json").write_text(p.read_text())
        self.assertEqual(self.report()["episodes"][1]["coverage"], "ambiguous_artifacts")

    def test_text_fallback_and_empty_detection(self) -> None:
        """Missing JSON segments can use a full timestamped text artifact."""
        p = self.artifacts / "p1" / "worker1.json"
        p.write_text(json.dumps(dict(audio_path="https://traffic.megaphone.fm/TCP100.mp3", segments=[])))
        self.assertEqual(self.report()["episodes"][1]["coverage"], "no_segments")
        p.with_name("worker1_transcript.txt").write_text("[0.1s - 1.9s] hello memoir\n")
        self.assertEqual(self.report()["episodes"][1]["segments"][0]["start"], 0.1)
        self.assertEqual(self.report()["episodes"][1]["canonical_text"], "[0.1s - 1.9s] hello memoir\n")

    def test_fts_syntax_is_treated_as_plain_text(self) -> None:
        """Quoted user tokens cannot execute FTS syntax or SQL."""
        search.ingest(self.connection, self.report())
        self.assertEqual(search.search(self.connection, '"telescope" OR ) --')["status"], "hits")

    def test_identity_is_namespaced_by_podcast(self) -> None:
        """The same GUID in two podcasts is never merged."""
        self.assertNotEqual(search.feed_episodes(self.feed, "p1")[0]["id"], search.feed_episodes(self.feed, "p2")[0]["id"])

    def test_refuses_existing_unrelated_database(self) -> None:
        """A mistaken database argument cannot alter production tables."""
        path = self.root / "unrelated.sqlite"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE podcasts(id TEXT)")
        with self.assertRaises(ValueError):
            search.connect(path)
        with sqlite3.connect(path) as connection:
            self.assertEqual(connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), [("podcasts",)])

    def test_latest_available_and_date_bound_never_fall_back(self) -> None:
        """Available newest is searched; an excluded newest never selects older."""
        report = self.report()
        report["episodes"][0].update(coverage="available", segments=[dict(start=2, end=3, text="latest memoir")])
        search.ingest(self.connection, report)
        result = search.search(self.connection, "memoir", "p1", latest=True)
        self.assertEqual(result["hits"][0]["episode"]["guid"], "new")
        self.assertEqual(search.search(self.connection, "memoir", "p1", before="2026-09-29", latest=True)["hits"], [])


if __name__ == "__main__":
    unittest.main()
