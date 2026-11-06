"""Offline transcript search; only the explicitly supplied output database is written.

RSS is a saved public metadata snapshot, not a replacement for subscription state.
Run directly with Python; no AdNihilator imports, credentials, or network calls.
"""

import argparse
import datetime as dt
import email.utils
import hashlib
import json
import re
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


STOP_WORDS = set("what were they talking about in my latest episode of the one guy really it a an and is was he she i me to for with that this podcast".split())


def utc_date(value: str) -> str:
    """Normalize RSS and ISO dates to sortable UTC, rejecting malformed dates."""
    try:
        date = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        date = email.utils.parsedate_to_datetime(value)
    if date.tzinfo is None:
        date = date.replace(tzinfo=dt.timezone.utc)
    return date.astimezone(dt.timezone.utc).isoformat()


def audio_identity(url: str) -> str:
    """Match the core's conservative provenance rule without runtime imports."""
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or parsed.username is not None or parsed.password is not None or parsed.port not in (None, 80, 443):
            return url
        if parsed.hostname == "traffic.megaphone.fm":
            match = re.fullmatch(r"/([A-Z]+\d+\.mp3)", parsed.path)
        elif parsed.hostname in {"chrt.fm", "chtbl.com", "dts.podtrac.com", "pdst.fm"}:
            match = re.search(r"(?:^|/)traffic\.megaphone\.fm/([A-Z]+\d+\.mp3)$", parsed.path)
        else:
            return url
    except ValueError:
        return url
    return match.group(1) if match else url


def feed_episodes(feed: Path, podcast_id: str) -> list[dict[str, Any]]:
    """Read all feed identities, ordering by publication time rather than XML order."""
    channel = ET.parse(feed).getroot().find("channel")
    if channel is None:
        raise ValueError("Expected an RSS channel")
    episodes = {}
    for item in channel.findall("item"):
        enclosure = item.find("enclosure")
        audio = enclosure.get("url", "") if enclosure is not None else ""
        guid = item.findtext("guid") or audio
        if not guid:
            raise ValueError("Episode has no GUID or enclosure identity")
        key = hashlib.sha256((podcast_id + "\0" + guid).encode()).hexdigest()
        episode = dict(id=key, podcast_id=podcast_id, podcast=channel.findtext("title"),
                       guid=guid, title=item.findtext("title"),
                       published_at=utc_date(item.findtext("pubDate") or ""),
                       original_audio_url=audio, source_url=item.findtext("link"),
                       description=item.findtext("description"), metadata_source=str(feed.resolve()))
        if key in episodes and episodes[key] != episode:
            raise ValueError("Conflicting feed entries for the same GUID")
        episodes[key] = episode
    return sorted(episodes.values(), key=lambda e: (e["published_at"], e["guid"]), reverse=True)


def segments_from_text(path: Path) -> list[dict[str, Any]]:
    """Read the worker's timestamped transcript format without discarding any line."""
    segments = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"\[([\d.]+)s - ([\d.]+)s\] (.*)", line)
        if not match:
            raise ValueError(f"Unrecognized transcript line in {path}")
        segments.append(dict(index=len(segments), start=float(match[1]), end=float(match[2]), text=match[3]))
    return segments


def inventory(feed: Path, artifacts: Path, podcast_id: str, limit: int) -> dict[str, Any]:
    """Match artifact media identities to RSS; retain missing coverage explicitly."""
    episodes = feed_episodes(feed, podcast_id)
    artifacts_by_audio: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path in sorted((artifacts / podcast_id).glob("*.json")):
        if path.name.endswith("_sponsors.json"):
            continue
        data = json.loads(path.read_text())
        artifacts_by_audio.setdefault(audio_identity(data.get("audio_path", "")), []).append((path, data))
    all_audio = {audio_identity(e["original_audio_url"]) for e in episodes}
    selected = episodes[:limit]
    for episode in selected:
        matches = artifacts_by_audio.get(audio_identity(episode["original_audio_url"]), [])
        episode.update(coverage="missing", segments=[], canonical_text=None, artifact_path=None,
                       worker_episode_id=None, timestamp_basis="original_audio", model_info={})
        if len(matches) > 1:
            episode["coverage"] = "ambiguous_artifacts"
            episode["artifact_candidates"] = [str(p.resolve()) for p, _ in matches]
            continue
        if matches:
            path, data = matches[0]
            text_path = path.with_name(path.stem + "_transcript.txt")
            segments = data.get("segments") or (segments_from_text(text_path) if text_path.exists() else [])
            for segment in segments:
                if not isinstance(segment.get("text"), str) or not (0 <= segment["start"] <= segment["end"]):
                    raise ValueError(f"Invalid segment in {path}")
            episode.update(coverage="available" if segments else "no_segments", segments=segments,
                           canonical_text=text_path.read_text() if text_path.exists() else None,
                           artifact_path=str(path.resolve()), worker_episode_id=path.stem,
                           model_info=data.get("model_info", {}), duration=data.get("duration"))
    return dict(feed_episode_count=len(episodes), artifact_count=sum(map(len, artifacts_by_audio.values())),
                unmatched_artifact_count=sum(len(v) for k, v in artifacts_by_audio.items() if k not in all_audio),
                selection="latest feed episodes, including missing transcripts", episodes=selected)


def connect(db: Path) -> sqlite3.Connection:
    """Create a separate prototype database with canonical transcripts and FTS5."""
    db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db)
    connection.row_factory = sqlite3.Row
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if tables and "transcript_search_prototype" not in tables:
        connection.close()
        raise ValueError("Refusing to write an existing database without the prototype marker")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS transcript_search_prototype (version INTEGER);
        CREATE TABLE IF NOT EXISTS episodes (
            id TEXT PRIMARY KEY, podcast_id TEXT, podcast TEXT, guid TEXT,
            title TEXT, published_at TEXT, coverage TEXT, canonical_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY, episode_id TEXT, ordinal INTEGER,
            start REAL, end REAL, text TEXT, UNIQUE(episode_id, ordinal));
        CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(text, segment_id UNINDEXED);
    """)
    return connection


def ingest(connection: sqlite3.Connection, report: dict[str, Any]) -> dict[str, int]:
    """Atomically replace each episode; repeat ingestion never duplicates passages."""
    with connection:
        for episode in report["episodes"]:
            connection.execute("DELETE FROM passages WHERE segment_id IN (SELECT id FROM segments WHERE episode_id=?)", (episode["id"],))
            connection.execute("DELETE FROM segments WHERE episode_id=?", (episode["id"],))
            connection.execute("INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?)", (
                episode["id"], episode["podcast_id"], episode["podcast"], episode["guid"],
                episode["title"], episode["published_at"], episode["coverage"], json.dumps(episode)))
            for ordinal, segment in enumerate(episode["segments"]):
                cursor = connection.execute("INSERT INTO segments(episode_id,ordinal,start,end,text) VALUES (?,?,?,?,?)",
                                            (episode["id"], ordinal, segment["start"], segment["end"], segment["text"]))
                connection.execute("INSERT INTO passages(text,segment_id) VALUES (?,?)", (segment["text"], cursor.lastrowid))
    return {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in ("episodes", "segments", "passages")}


def search(connection: sqlite3.Connection, query: str, podcast: str | None = None,
           after: str | None = None, before: str | None = None, latest: bool = False,
           limit: int = 8, context_seconds: float = 30) -> dict[str, Any]:
    """Return ranked lexical hits plus nearby canonical conversation and provenance.

    Latest is resolved before date/coverage filtering so a gap cannot select an
    older transcript. User query tokens are quoted; FTS operators are not accepted.
    """
    filters, parameters = [], []
    if podcast:
        filters.append("(podcast_id=? OR lower(podcast)=lower(?))")
        parameters.extend([podcast, podcast])
    where = " WHERE " + " AND ".join(filters) if filters else ""
    episodes = list(connection.execute("SELECT * FROM episodes" + where + " ORDER BY published_at DESC, guid DESC", parameters))
    if latest and not podcast and len({e["podcast_id"] for e in episodes}) > 1:
        raise ValueError("--latest requires --podcast when multiple podcasts are indexed")
    if latest:
        episodes = episodes[:1]
    if after:
        episodes = [e for e in episodes if e["published_at"] >= utc_date(after)]
    if before:
        episodes = [e for e in episodes if e["published_at"] < utc_date(before)]
    metadata = [{k: e[k] for k in ("id", "podcast", "title", "published_at", "coverage")} for e in episodes]
    result: dict[str, Any] = dict(query=query, episodes=metadata, hits=[], timestamp_basis="original_audio",
                                  warning="Timestamps precede ad splicing; no speaker diarization or semantic answer generation.")
    if latest and episodes and episodes[0]["coverage"] != "available":
        result["status"] = "latest_transcript_unavailable"
        return result
    if not episodes:
        result["status"] = "no_hits"
        return result
    # A scoped show's name is metadata, not a useful transcript search term.
    ignored = STOP_WORDS | (set(re.findall(r"\w+", episodes[0]["podcast"].lower())) if podcast else set())
    tokens = list(dict.fromkeys(t.lower() for t in re.findall(r"\w+", query) if t.lower() not in ignored))
    if not tokens or not episodes:
        result["status"] = "no_hits"
        return result
    expression = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
    ids = [e["id"] for e in episodes]
    rows = connection.execute(f"""SELECT s.*, bm25(passages) AS score FROM passages
        JOIN segments s ON s.id=passages.segment_id
        WHERE passages MATCH ? AND s.episode_id IN ({','.join('?' for _ in ids)})
        ORDER BY score, s.start, s.ordinal""", [expression, *ids])
    episode_map = {e["id"]: json.loads(e["canonical_json"]) for e in episodes}
    seen: set[tuple[str, int]] = set()
    for row in rows:
        if (row["episode_id"], row["ordinal"]) in seen:
            continue
        context = list(connection.execute("""SELECT ordinal,start,end,text FROM segments WHERE episode_id=?
            AND end>=? AND start<=? ORDER BY ordinal""", (row["episode_id"], row["start"] - context_seconds, row["end"] + context_seconds)))
        seen.update((row["episode_id"], s["ordinal"]) for s in context)
        episode = episode_map[row["episode_id"]]
        result["hits"].append(dict(episode={k: episode.get(k) for k in (
            "id", "podcast", "guid", "worker_episode_id", "title", "published_at",
            "original_audio_url", "source_url", "artifact_path", "metadata_source", "model_info")},
            match=dict(row), context=[dict(s) for s in context],
            start=min(s["start"] for s in context), end=max(s["end"] for s in context)))
        if len(result["hits"]) >= limit:
            break
    result["status"] = "hits" if result["hits"] else "no_hits"
    return result


def main() -> None:
    """Run offline ingestion, coverage reporting, or lexical retrieval."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("ingest")
    build.add_argument("--feed", type=Path, required=True)
    build.add_argument("--artifacts", type=Path, required=True)
    build.add_argument("--podcast-id", required=True)
    build.add_argument("--limit", type=int, default=20)
    build.add_argument("--inventory", type=Path, required=True)
    build.add_argument("--db", type=Path, required=True)
    find = sub.add_parser("search")
    find.add_argument("query")
    find.add_argument("--db", type=Path, required=True)
    find.add_argument("--podcast")
    find.add_argument("--after")
    find.add_argument("--before", help="Exclusive upper publication bound")
    find.add_argument("--latest", action="store_true")
    find.add_argument("--limit", type=int, default=8)
    find.add_argument("--context-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.command == "ingest":
        if args.limit < 1:
            parser.error("--limit must be positive")
        for output in (args.db, args.inventory):
            if output.resolve().is_relative_to(args.artifacts.resolve()) or output.resolve() == args.feed.resolve():
                parser.error("Outputs must be separate from the source feed and worker artifacts")
        report = inventory(args.feed, args.artifacts, args.podcast_id, args.limit)
        with connect(args.db) as connection:
            counts = ingest(connection, report)
        # Keep full transcripts in SQLite, and compact availability in this report.
        compact = {**report, "episodes": [{k: v for k, v in e.items() if k not in ("segments", "canonical_text", "description")} | {"segment_count": len(e["segments"])} for e in report["episodes"]]}
        args.inventory.parent.mkdir(parents=True, exist_ok=True)
        args.inventory.write_text(json.dumps(compact, indent=2))
        print(json.dumps(counts))
    else:
        if args.limit < 1 or args.context_seconds < 0:
            parser.error("limit must be positive and context seconds nonnegative")
        # Search cannot create or alter a database.
        connection = sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        with connection:
            print(json.dumps(search(connection, args.query, args.podcast, args.after, args.before,
                                    args.latest, args.limit, args.context_seconds), indent=2))


if __name__ == "__main__":
    main()
