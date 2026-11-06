"""Private durable storage and shared bounded read operations for UI/REST/MCP."""

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import case, func, select, text, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from adnihilator.transcript_document import TranscriptDocument, audio_identity
from ..models import Episode, EpisodeTranscript, Podcast, TranscriptJob, TranscriptVersion
from ..transcript_dates import utc_naive


MAX_RESULTS = 20
MAX_CONTEXT_SEGMENTS = 40
MAX_TEXT_CHARS = 16000
STOP = set("which where when why how can tell remember said what were they talking about in my latest episode of the one guy really it a an and is was he she i me to for with that this podcast".split())


class TranscriptError(ValueError):
    """A safe domain error; messages never contain transcript or credential values."""


def coverage(db: Session, episode: Episode) -> dict[str, Any]:
    """Canonical availability and independent request/index state."""
    row = db.get(EpisodeTranscript, episode.id)
    job = db.get(TranscriptJob, episode.id)
    return dict(state=row.state if row else "missing", checksum=row.active_checksum if row else None,
                index_ready=bool(row and row.active_checksum and row.indexed_checksum == row.active_checksum),
                request_state=job.state if job else None, error=row.last_error if row else None,
                downloaded_at=row.downloaded_at.isoformat() if row and row.downloaded_at else None)


def identity(episode: Episode) -> dict[str, Any]:
    """Bounded public-to-the-admin identity, without feed or storage tokens."""
    return dict(episode_id=episode.id, podcast_id=episode.podcast_id, guid=episode.guid[:2000],
                title=(episode.title or episode.guid)[:1000], podcast=(episode.podcast.title or "")[:300],
                published_at=episode.published_at.isoformat() + "Z" if episode.published_at else None,
                source_url=safe_url(episode.source_url), original_audio_url=safe_url(episode.original_audio_url))


def safe_url(url: str | None) -> str | None:
    """Prevent script URLs and userinfo from becoming reader links."""
    from urllib.parse import urlsplit
    if not url:
        return None
    try:
        parsed = urlsplit(url)
        return url[:4000] if parsed.scheme in ("http", "https") and parsed.hostname and not parsed.username else None
    except ValueError:
        return None


def store(db: Session, document: TranscriptDocument, claim_id: str | None = None,
          downloaded_at: datetime | None = None) -> str:
    """Commit complete canonical content/pointer atomically; index is independent.

    Job attempt IDs reject late workers. Equal checksums are idempotent; a new
    checksum creates an immutable version. No incomplete document is published.
    """
    episode = db.get(Episode, document.episode_id)
    if not episode or (episode.podcast_id, episode.guid) != (document.podcast_id, document.guid):
        raise TranscriptError("Episode identity mismatch")
    # Only tracking/query wrappers may differ; opaque cross-provider URLs stay exact.
    if audio_identity(episode.original_audio_url or "") != audio_identity(document.original_audio_url):
        raise TranscriptError("Episode audio identity mismatch")
    checksum = document.checksum()
    if claim_id:
        # A successful conditional write holds the lease through canonical/job
        # commit. A read snapshot alone cannot fence an in-flight old request.
        fenced = db.execute(update(TranscriptJob).where(
            TranscriptJob.episode_id == episode.id, TranscriptJob.state == "processing",
            TranscriptJob.claim_id == claim_id).values(claim_id=TranscriptJob.claim_id),
            execution_options={"synchronize_session": False})
        if not fenced.rowcount:
            db.rollback()
            job = db.get(TranscriptJob, episode.id, populate_existing=True)
            if job and job.state == "complete" and db.scalar(select(TranscriptVersion).where(
                    TranscriptVersion.episode_id == episode.id, TranscriptVersion.checksum == checksum)):
                return checksum  # Lost successful response remains idempotent.
            raise TranscriptError("Stale transcript claim")
    job = db.get(TranscriptJob, episode.id, populate_existing=True)
    row = db.get(EpisodeTranscript, episode.id, populate_existing=True)
    if not row:
        row = EpisodeTranscript(episode_id=episode.id)
        db.add(row)
    existing = db.scalar(select(TranscriptVersion).where(TranscriptVersion.episode_id == episode.id, TranscriptVersion.checksum == checksum))
    if existing and row.active_checksum and row.active_checksum != checksum:
        return checksum  # Old outbox replay must never replace newer content.
    if not existing:
        db.add(TranscriptVersion(episode_id=episode.id, checksum=checksum,
                                 canonical_json=document.canonical_bytes().decode(),
                                 episode_metadata_json=json.dumps(identity(episode))))
    row.active_checksum = checksum
    row.state = "available"
    row.last_error = None
    row.updated_at = datetime.utcnow()
    if downloaded_at:
        row.downloaded_at = utc_naive(downloaded_at)
    if claim_id or (job and job.state == "queued"):
        job.state = "complete"
        job.claim_id = None
    db.commit()
    try:
        rebuild_episode(db, episode.id)
    except Exception:
        db.rollback()
        # Availability belongs to canonical content; reader remains usable.
        row = db.get(EpisodeTranscript, episode.id)
        row.last_error = "index_pending"
        db.commit()
    return checksum


def get_version(db: Session, episode_id: str, checksum: str | None = None) -> TranscriptVersion:
    """Read the active or a cited immutable revision without index dependence."""
    row = db.get(EpisodeTranscript, episode_id)
    checksum = checksum or (row.active_checksum if row else None)
    version = db.scalar(select(TranscriptVersion).where(TranscriptVersion.episode_id == episode_id, TranscriptVersion.checksum == checksum)) if checksum else None
    if not version:
        raise TranscriptError("Transcript unavailable")
    return version


def rebuild_episode(db: Session, episode_id: str) -> None:
    """Replace one episode's FTS rows in a single transaction; canonical is untouched."""
    # Serialize selection and replacement, including against another publish.
    # The no-op write takes SQLite's writer reservation (and a row lock on
    # engines with row locking), before refreshing any cached active pointer.
    db.execute(text("UPDATE episode_transcripts SET active_checksum=active_checksum WHERE episode_id=:id"),
               dict(id=episode_id))
    row = db.get(EpisodeTranscript, episode_id, populate_existing=True)
    version = get_version(db, episode_id)
    document = json.loads(version.canonical_json)
    segments = document["segments"]
    db.execute(text("DELETE FROM transcript_fts WHERE episode_id=:id"), dict(id=episode_id))
    db.execute(text("INSERT INTO transcript_fts(text,episode_id,checksum,ordinal) VALUES (:text,:id,:checksum,:ordinal)"),
               [dict(text=" ".join(x["text"] for x in segments[n:n + 3]), id=episode_id, checksum=version.checksum, ordinal=n) for n, s in enumerate(document["segments"])])
    row.indexed_checksum = version.checksum
    row.last_error = None
    db.commit()


def request_transcript(db: Session, episode_id: str) -> dict[str, Any]:
    """Explicit per-episode queue; never changes audio status/keys or bulk queues."""
    episode = db.get(Episode, episode_id)
    if not episode or not safe_url(episode.original_audio_url):
        raise TranscriptError("Episode unavailable for transcription")
    row = db.get(EpisodeTranscript, episode_id)
    if row and row.active_checksum:
        return coverage(db, episode)
    # Concurrent first requests must converge on one row/job. The first insert
    # also acquires the writer reservation before the state is refreshed.
    db.execute(sqlite_insert(EpisodeTranscript).values(episode_id=episode_id, state="queued")
               .on_conflict_do_nothing(index_elements=["episode_id"]))
    row = db.get(EpisodeTranscript, episode_id, populate_existing=True)
    if row.active_checksum:
        db.commit()
        return coverage(db, episode)
    db.execute(sqlite_insert(TranscriptJob).values(episode_id=episode_id)
               .on_conflict_do_nothing(index_elements=["episode_id"]))
    job = db.get(TranscriptJob, episode_id, populate_existing=True)
    if job.state not in ("processing", "queued"):
        job.state, job.attempt, job.claim_id = "queued", 0, None
    row.state, row.last_error = job.state, None
    db.commit()
    return coverage(db, episode)


def claim_transcript(db: Session) -> dict[str, Any] | None:
    """Recover expired attempts and claim with a compare-and-swap update."""
    now = datetime.utcnow()
    for job in db.scalars(select(TranscriptJob).where(TranscriptJob.state == "processing", TranscriptJob.claimed_at < now - timedelta(hours=2))):
        next_state = "queued" if job.attempt < 3 else "failed"
        changed = db.execute(update(TranscriptJob).where(
            TranscriptJob.episode_id == job.episode_id, TranscriptJob.state == "processing",
            TranscriptJob.claim_id == job.claim_id, TranscriptJob.claimed_at == job.claimed_at,
            TranscriptJob.claimed_at < now - timedelta(hours=2)).values(state=next_state, claim_id=None),
            execution_options={"synchronize_session": False})
        if changed.rowcount:
            db.execute(update(EpisodeTranscript).where(EpisodeTranscript.episode_id == job.episode_id,
                       EpisodeTranscript.active_checksum.is_(None)).values(state=next_state, last_error="worker_timeout"),
                       execution_options={"synchronize_session": False})
            db.expire(job)
    db.commit()
    for job in db.scalars(select(TranscriptJob).where(TranscriptJob.state == "queued").order_by(TranscriptJob.updated_at).limit(10)):
        available = db.get(EpisodeTranscript, job.episode_id)
        if available and available.active_checksum:
            job.state = "complete"
            db.commit()
            continue
        claim_id = str(uuid4())
        claimed = db.execute(update(TranscriptJob).where(TranscriptJob.episode_id == job.episode_id, TranscriptJob.state == "queued").values(
            state="processing", claim_id=claim_id, claimed_at=now, attempt=TranscriptJob.attempt + 1))
        if claimed.rowcount:
            row = db.get(EpisodeTranscript, job.episode_id)
            if row and not row.active_checksum:
                row.state = "processing"
            db.commit()
            episode = db.get(Episode, job.episode_id)
            return {**identity(episode), "id": episode.id, "claim_id": claim_id, "description": None}
    db.rollback()
    return None


def fail_transcript(db: Session, episode_id: str, claim_id: str) -> None:
    """Bounded retries; expired attempts cannot alter a newer claim or audio status."""
    changed = db.execute(update(TranscriptJob).where(
        TranscriptJob.episode_id == episode_id, TranscriptJob.state == "processing",
        TranscriptJob.claim_id == claim_id).values(
            state=case((TranscriptJob.attempt < 3, "queued"), else_="failed"), claim_id=None),
        execution_options={"synchronize_session": False})
    if not changed.rowcount:
        db.rollback()
        raise TranscriptError("Stale transcript claim")
    job = db.get(TranscriptJob, episode_id, populate_existing=True)
    db.execute(update(EpisodeTranscript).where(EpisodeTranscript.episode_id == episode_id,
               EpisodeTranscript.active_checksum.is_(None)).values(state=job.state, last_error="transcription_failed"),
               execution_options={"synchronize_session": False})
    db.commit()


class TranscriptReader:
    """Read-only facade: bounded data, stable citations, and explicit coverage."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def episodes(self, podcast_id: str, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        if not 1 <= limit <= 100 or not 0 <= offset <= 100000:
            raise TranscriptError("Invalid pagination")
        query = select(Episode).where(Episode.podcast_id == podcast_id).order_by(Episode.published_at.desc(), Episode.id)
        episodes = self.db.scalars(query.offset(offset).limit(limit)).all()
        latest = self.db.scalar(query.limit(1))
        available = self.db.scalar(query.join(EpisodeTranscript).where(EpisodeTranscript.active_checksum.isnot(None)).limit(1))
        downloaded = self.db.scalar(query.join(EpisodeTranscript).where(EpisodeTranscript.downloaded_at.isnot(None)).limit(1))
        return dict(episodes=[dict(**identity(e), coverage=coverage(self.db, e)) for e in episodes],
                    latest_published=identity(latest) if latest else None,
                    latest_downloaded=identity(downloaded) if downloaded else None,
                    latest_transcribed=identity(available) if available else None,
                    latest_meaning="Newest publication in each coverage class; unknown download times stay unknown")

    def passage(self, episode_id: str, checksum: str | None = None, ordinal: int = 0,
                count: int = 20) -> dict[str, Any]:
        if not 0 <= ordinal <= 30000 or not 1 <= count <= MAX_CONTEXT_SEGMENTS:
            raise TranscriptError("Invalid passage bounds")
        episode = self.db.get(Episode, episode_id)
        if not episode:
            raise TranscriptError("Episode not found")
        version = get_version(self.db, episode_id, checksum)
        doc = json.loads(version.canonical_json)
        selected = doc["segments"][ordinal:ordinal + count]
        segments, remaining = [], MAX_TEXT_CHARS
        for n, s in enumerate(selected, ordinal):
            if remaining <= 0:
                break
            if len(s["text"]) > remaining:
                break  # Page at segment boundaries so the reader never loses a tail.
            value = s["text"]
            remaining -= len(value)
            segments.append(dict(ordinal=n, start=s["start"], end=s["end"], text=value,
                                 text_truncated=len(value) < len(s["text"]),
                                 ad=any(s["start"] < a["end"] and s["end"] > a["start"] for a in doc["ad_spans"]),
                                 reference=f"{episode_id}/{version.checksum}/{n}"))
        return dict(episode=identity(episode), coverage=coverage(self.db, episode), checksum=version.checksum,
                    segments=segments, total_segments=len(doc["segments"]),
                    next_ordinal=ordinal + len(segments) if ordinal + len(segments) < len(doc["segments"]) else None,
                    timestamp_basis=doc["timestamp_basis"], processed_seek_url=None,
                    source=doc["source"], model=doc["model"], schema_version=doc["schema_version"],
                    version_episode=json.loads(version.episode_metadata_json), producer_version=doc["producer_version"],
                    artifact_id=doc.get("source_artifact_id"), artifact_checksum=doc.get("source_artifact_checksum"),
                    content_is_untrusted_data=True)

    def search(self, query: str, podcast_id: str, after: datetime | None = None,
               before: datetime | None = None, phrase: bool = False,
               latest: bool = False, limit: int = 8) -> dict[str, Any]:
        if not 1 <= len(query) <= 500 or not 1 <= limit <= MAX_RESULTS:
            raise TranscriptError("Invalid search bounds")
        base = select(Episode).where(Episode.podcast_id == podcast_id)
        newest = self.db.scalar(base.order_by(Episode.published_at.desc(), Episode.id).limit(1))
        diagnostics = self.episodes(podcast_id, 1)
        result = dict(hits=[], query=query, latest_published=diagnostics["latest_published"],
                      latest_transcribed=diagnostics["latest_transcribed"], latest_downloaded=diagnostics["latest_downloaded"],
                      coverage_warning=None, content_is_untrusted_data=True,
                      answer_status="retrieval_only", ambiguity="Passages may discuss multiple candidates; no speaker or unique answer is inferred")
        if latest and newest:
            cov = coverage(self.db, newest)
            if cov["state"] != "available" or not cov["index_ready"]:
                result["coverage_warning"] = dict(episode=identity(newest), coverage=cov)
                return result
        podcast = self.db.get(Podcast, podcast_id)
        ignored = STOP | set(re.findall(r"\w+", (podcast.title or "").lower())) if podcast else STOP
        tokens = re.findall(r"\w+", query.lower())[:30]
        if not phrase:
            tokens = list(dict.fromkeys(t for t in tokens if t not in ignored))
        state = func.coalesce(EpisodeTranscript.state, "missing")
        selected = select(state, func.count(Episode.id)).select_from(Episode).outerjoin(EpisodeTranscript).where(Episode.podcast_id == podcast_id)
        if latest:
            selected = selected.where(Episode.id == (newest.id if newest else ""))
        if after:
            selected = selected.where(Episode.published_at >= utc_naive(after))
        if before:
            selected = selected.where(Episode.published_at < utc_naive(before))
        result["coverage_counts"] = dict(self.db.execute(selected.group_by(state)).all())
        if not tokens:
            return result
        expression = '"' + " ".join(tokens) + '"' if phrase else " OR ".join('"' + t + '"' for t in tokens)
        # Podcast/date constraints remain inside the bounded SQL query.
        conditions = "e.podcast_id=:podcast AND c.active_checksum=f.checksum AND c.indexed_checksum=f.checksum"
        values: dict[str, Any] = dict(podcast=podcast_id, expression=expression, anchor="%" + tokens[0] + "%", limit=limit * 40)
        if latest:
            conditions += " AND e.id=:latest"
            values["latest"] = newest.id if newest else ""
        for label, date, operator in (("after", after, ">="), ("before", before, "<")):
            if date:
                conditions += f" AND e.published_at {operator} :{label}"
                values[label] = utc_naive(date)
        rows = self.db.execute(text(f"""SELECT f.episode_id,f.checksum,f.ordinal,bm25(transcript_fts) score,f.rowid,f.text
            FROM transcript_fts f JOIN episodes e ON e.id=f.episode_id
            JOIN episode_transcripts c ON c.episode_id=e.id
            WHERE transcript_fts MATCH :expression AND {conditions}
            ORDER BY CASE WHEN f.text LIKE :anchor THEN 0 ELSE 1 END, score LIMIT :limit"""), values)
        seen = set()
        for episode_id, checksum, ordinal, score, row_id, indexed_text in rows:
            ordinal = int(ordinal)
            if (episode_id, checksum, ordinal) in seen:
                continue
            # Search previews have a per-segment budget; the reader pages full text.
            begin = max(0, ordinal - 20)
            passage = self.passage(episode_id, checksum, begin, 40)
            document = json.loads(get_version(self.db, episode_id, checksum).canonical_json)
            # Use the same FTS tokenizer for focus, including diacritics and
            # phrases crossing segments. Choose a marker absent from this text.
            prefix = "\x1e" + uuid4().hex + ":"
            occupied = set(re.findall(re.escape(prefix) + r"([0-9]+)\x1f", indexed_text))
            number = 0
            while str(number) in occupied:
                number += 1
            marker = prefix + str(number) + "\x1f"
            marked = self.db.scalar(text("""SELECT highlight(transcript_fts,0,:marker,'')
                FROM transcript_fts WHERE rowid=:row_id AND episode_id=:episode_id
                AND checksum=:checksum AND ordinal=:ordinal
                AND transcript_fts MATCH :expression"""),
                dict(marker=marker, row_id=row_id, episode_id=episode_id, checksum=checksum,
                     ordinal=ordinal, expression=expression))
            if marked is None or marker not in marked:
                continue  # A concurrently replaced index is not a broken citation.
            position = marked.index(marker)
            focus = ordinal
            for target in range(ordinal, min(ordinal + 3, len(document["segments"]))):
                length = len(document["segments"][target]["text"])
                if position < length:
                    focus = target
                    break
                position -= length + 1  # FTS windows join source text with a space.
            preview = []
            for n, segment in enumerate(document["segments"][begin:begin + 40], begin):
                value = segment["text"]
                start_char = 0
                if len(value) > 400:
                    matches = [m.start() for token in tokens for m in re.finditer(re.escape(token), value, re.I)]
                    start_char = max(0, min(matches) - 150) if matches else max(0, len(value) - 400) if n < ordinal else 0
                preview.append(dict(ordinal=n, start=segment["start"], end=segment["end"],
                                    text=value[start_char:start_char + 400], text_start_char=start_char,
                                    text_truncated=len(value) > 400, reference=f"{episode_id}/{checksum}/{n}",
                                    ad=any(segment["start"] < a["end"] and segment["end"] > a["start"] for a in document["ad_spans"])))
            passage["segments"] = preview
            passage["next_ordinal"] = begin + len(preview) if begin + len(preview) < len(document["segments"]) else None
            passage["match_ordinal"] = ordinal  # Existing indexed window start.
            passage["focus_ordinal"] = focus  # Segment where the first FTS match starts.
            result["hits"].append(passage)
            seen.update((episode_id, checksum, s["ordinal"]) for s in passage["segments"])
            if len(result["hits"]) >= limit:
                break
        return result


def heartbeat_transcript(db: Session, episode_id: str, claim_id: str) -> None:
    """Renew a live transcript-only lease without modifying audio job state."""
    changed = db.execute(update(TranscriptJob).where(TranscriptJob.episode_id == episode_id,
                        TranscriptJob.state == "processing", TranscriptJob.claim_id == claim_id)
                        .values(claimed_at=datetime.utcnow()))
    if not changed.rowcount:
        db.rollback()
        raise TranscriptError("Stale transcript claim")
    db.commit()
