"""Offline artifact import to an explicitly created local fixture database.

Feed metadata and existing artifacts are read only. No API, auth, or transcription.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import select

from adnihilator.artifact_import import read_artifact
from adnihilator.transcript_document import audio_identity
from web.database import get_engine, get_session_factory, init_db
from web.transcript_migration import upgrade
from web.models import Episode, EpisodeTranscript, Podcast
from web.services.transcripts import store


def import_local(db_path: Path, feed: Path, artifacts: Path, podcast_id: str,
                 feed_url: str, limit: int = 20) -> dict:
    """Commit each episode independently; rerunning resumes after interruption."""
    if db_path.resolve().is_relative_to(artifacts.resolve()) or not 1 <= limit <= 100:
        raise ValueError("Invalid output path or selection limit")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Check the marker read-only before the engine can change journal settings.
    import sqlite3
    if db_path.exists() and db_path.stat().st_size:
        with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if tables and "local_transcript_fixture" not in tables:
            raise ValueError("Refusing a database without the local fixture marker")
    engine = get_engine("sqlite:///" + str(db_path.resolve()))
    from sqlalchemy import text
    init_db(engine)
    upgrade(engine)  # Explicit fixture import owns this disposable schema.
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE IF NOT EXISTS local_transcript_fixture(version INTEGER)"))
    channel = ET.parse(feed).getroot().find("channel")
    entries = []
    seen = set()
    for item in channel.findall("item"):
        guid = item.findtext("guid")
        date = parsedate_to_datetime(item.findtext("pubDate"))
        date = date.replace(tzinfo=timezone.utc) if date.tzinfo is None else date.astimezone(timezone.utc)
        if guid in seen:
            raise ValueError("Duplicate RSS GUID")
        seen.add(guid)
        entries.append((date.replace(tzinfo=None), guid, item))
    artifact_map = {}
    invalid = 0
    for path in sorted((artifacts / podcast_id).glob("*.json")):
        if path.name.endswith("_sponsors.json"):
            continue
        try:
            data = json.loads(path.read_bytes())
            artifact_map.setdefault(audio_identity(data.get("audio_path", "")), []).append(path)
        except (ValueError, OSError):
            invalid += 1
    counts = dict(available=0, missing=0, failed=0, ambiguous=0, invalid_artifacts=invalid)
    with get_session_factory(engine)() as db:
        podcast = db.get(Podcast, podcast_id)
        if not podcast:
            podcast = Podcast(id=podcast_id, title=channel.findtext("title"), source_rss_url=feed_url,
                              feed_token="local-fixture-only-" + podcast_id, auto_process=False)
            db.add(podcast); db.commit()
        for date, guid, item in sorted(entries, key=lambda x: (x[0], x[1]), reverse=True)[:limit]:
            audio = item.find("enclosure").get("url")
            matches = artifact_map.get(audio_identity(audio), [])
            episode = db.scalar(select(Episode).where(Episode.podcast_id == podcast_id, Episode.guid == guid))
            if not episode:
                episode = Episode(id=str(uuid5(NAMESPACE_URL, podcast_id + "\0" + guid)), podcast_id=podcast_id,
                                  guid=guid, title=item.findtext("title"), published_at=date,
                                  original_audio_url=audio, source_url=item.findtext("link"), status="skipped")
                db.add(episode); db.commit()
            if len(matches) != 1:
                state, error = ("missing", None) if not matches else ("failed", "ambiguous_artifacts")
                counts["missing" if not matches else "ambiguous"] += 1
            else:
                try:
                    doc = read_artifact(matches[0], episode.id, podcast_id, guid, audio)
                    if doc:
                        store(db, doc)
                        counts["available"] += 1
                        continue
                    state, error = "missing", "no_segments"
                    counts["missing"] += 1
                except (ValueError, OSError):
                    db.rollback()
                    state, error = "failed", "artifact_invalid"
                    counts["failed"] += 1
            row = db.get(EpisodeTranscript, episode.id)
            if row is None:
                row = EpisodeTranscript(episode_id=episode.id)
                db.add(row)
            if not row.active_checksum:
                row.state = state
            row.last_error = error
            db.commit()
    engine.dispose()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--feed", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--podcast-id", required=True)
    parser.add_argument("--feed-url", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(import_local(args.db, args.feed, args.artifacts, args.podcast_id, args.feed_url, args.limit)))


if __name__ == "__main__":
    main()
