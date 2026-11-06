"""Authenticated transcript API. Feed tokens never authorize transcript content."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from adnihilator.transcript_document import TranscriptDocument
from ..auth import SESSION_COOKIE, verify_admin
from ..transcript_auth import verify_transcript_read
from ..dependencies import get_db, require_transcripts
from ..services.transcripts import (TranscriptError, TranscriptReader, claim_transcript,
                                    fail_transcript, request_transcript, store)
from .api import get_worker_auth

router = APIRouter(tags=["transcripts"], dependencies=[Depends(require_transcripts)])


def check_origin(request: Request, require_cookie_origin: bool = False) -> None:
    """Reject cross-origin operations; session-authenticated writes require Origin."""
    expected = str(request.base_url).rstrip("/")
    origin = request.headers.get("origin")
    if origin and origin != expected:
        raise HTTPException(403, "Origin denied")
    if require_cookie_origin and request.cookies.get(SESSION_COOKIE) and origin != expected:
        raise HTTPException(403, "Origin required")


class PublishRequest(BaseModel):
    document: TranscriptDocument
    claim_id: str | None = Field(default=None, max_length=36)
    downloaded_at: datetime | None = None


class FailRequest(BaseModel):
    claim_id: str = Field(min_length=1, max_length=36)


@router.get("/api/transcripts/episodes", dependencies=[Depends(verify_transcript_read)])
def list_episodes(podcast_id: str = Query(max_length=100), limit: int = Query(20, ge=1, le=100),
                  offset: int = Query(0, ge=0, le=100000), db: Session = Depends(get_db)) -> dict:
    return TranscriptReader(db).episodes(podcast_id, limit, offset)


@router.get("/api/transcripts/search", dependencies=[Depends(verify_transcript_read)])
def search(query: str = Query(min_length=1, max_length=500), podcast_id: str = Query(max_length=100),
           after: datetime | None = None, before: datetime | None = None, phrase: bool = False,
           latest: bool = False, limit: int = Query(8, ge=1, le=20), db: Session = Depends(get_db)) -> dict:
    return TranscriptReader(db).search(query, podcast_id, after, before, phrase, latest, limit)


@router.get("/api/transcripts/episodes/{episode_id}", dependencies=[Depends(verify_transcript_read)])
def read_episode(episode_id: str, checksum: str | None = Query(None, pattern="^[a-f0-9]{64}$"),
                 ordinal: int = Query(0, ge=0, le=30000), count: int = Query(20, ge=1, le=40),
                 db: Session = Depends(get_db)) -> dict:
    try:
        return TranscriptReader(db).passage(episode_id, checksum, ordinal, count)
    except TranscriptError as exc:
        raise HTTPException(404, str(exc)) from None


@router.post("/api/transcripts/episodes/{episode_id}/request", dependencies=[Depends(verify_admin)])
def queue_transcript(episode_id: str, request: Request, db: Session = Depends(get_db)) -> dict:
    check_origin(request, True)
    try:
        return request_transcript(db, episode_id)
    except TranscriptError as exc:
        raise HTTPException(404, str(exc)) from None


@router.put("/api/worker/transcripts/{episode_id}", dependencies=[Depends(get_worker_auth)])
def publish(episode_id: str, body: PublishRequest, db: Session = Depends(get_db)) -> dict:
    if episode_id != body.document.episode_id:
        raise HTTPException(409, "Episode identity mismatch")
    try:
        return dict(checksum=store(db, body.document, body.claim_id, body.downloaded_at))
    except TranscriptError as exc:
        raise HTTPException(409, str(exc)) from None


@router.post("/api/worker/transcript-jobs/claim", dependencies=[Depends(get_worker_auth)])
def claim(db: Session = Depends(get_db)) -> dict | None:
    return claim_transcript(db)


@router.post("/api/worker/transcript-jobs/{episode_id}/fail", dependencies=[Depends(get_worker_auth)])
def fail(episode_id: str, body: FailRequest, db: Session = Depends(get_db)) -> dict:
    try:
        fail_transcript(db, episode_id, body.claim_id)
        return dict(status="recorded")
    except TranscriptError as exc:
        raise HTTPException(409, str(exc)) from None


@router.get("/api/transcripts/episodes/{episode_id}/coverage", dependencies=[Depends(verify_transcript_read)])
def read_coverage(episode_id: str, db: Session = Depends(get_db)) -> dict:
    from ..models import Episode
    from ..services.transcripts import coverage, identity
    episode = db.get(Episode, episode_id)
    if not episode:
        raise HTTPException(404, "Episode not found")
    return dict(episode=identity(episode), coverage=coverage(db, episode))


@router.post("/api/worker/transcript-jobs/{episode_id}/heartbeat", dependencies=[Depends(get_worker_auth)])
def heartbeat(episode_id: str, body: FailRequest, db: Session = Depends(get_db)) -> dict:
    from ..services.transcripts import heartbeat_transcript
    try:
        heartbeat_transcript(db, episode_id, body.claim_id)
        return dict(status="renewed")
    except TranscriptError as exc:
        raise HTTPException(409, str(exc)) from None
