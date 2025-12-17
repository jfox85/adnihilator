"""Worker API routes."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..app import get_db
from ..auth import verify_worker_api_key
from ..models import Episode, EpisodeStatus

router = APIRouter(prefix="/api", tags=["worker"])

MAX_RETRIES = 2


def get_worker_auth(x_worker_api_key: str = Header(...)):
    """Dependency to verify worker API key."""
    verify_worker_api_key(x_worker_api_key)


class CompleteRequest(BaseModel):
    """Request body for completing an episode."""

    processed_audio_key: str
    processed_duration: float
    ads_removed_seconds: float
    detection_result_path: Optional[str] = None


class FailRequest(BaseModel):
    """Request body for failing an episode."""

    error: str


class ProgressRequest(BaseModel):
    """Request body for updating progress."""

    step: str
    percent: Optional[int] = None


@router.post("/queue/claim")
async def claim_episode(
    db: Session = Depends(get_db),
    _: None = Depends(get_worker_auth),
) -> Optional[dict]:
    """Atomically claim the next pending episode.

    Returns the episode data if one was claimed, or None if queue is empty.
    """
    # Atomic update: find and claim in one query
    episode = (
        db.query(Episode)
        .filter(Episode.status == EpisodeStatus.PENDING.value)
        .order_by(Episode.created_at.asc())
        .with_for_update(skip_locked=True)
        .first()
    )

    if not episode:
        return None

    episode.status = EpisodeStatus.PROCESSING.value
    episode.claimed_at = datetime.utcnow()
    db.commit()

    return {
        "id": episode.id,
        "podcast_id": episode.podcast_id,
        "podcast_title": episode.podcast.title if episode.podcast else None,
        "guid": episode.guid,
        "title": episode.title,
        "original_audio_url": episode.original_audio_url,
        "description": episode.description,
        "source_url": episode.source_url,
        "status": episode.status,
    }


@router.post("/queue/{episode_id}/complete")
async def complete_episode(
    episode_id: str,
    request: CompleteRequest,
    db: Session = Depends(get_db),
    _: None = Depends(get_worker_auth),
):
    """Mark an episode as successfully processed."""
    episode = db.query(Episode).filter_by(id=episode_id).first()
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")

    # Idempotent: ignore if already complete
    if episode.status == EpisodeStatus.COMPLETE.value:
        return {"status": "already_complete"}

    episode.status = EpisodeStatus.COMPLETE.value
    episode.processed_audio_key = request.processed_audio_key
    episode.processed_duration = request.processed_duration
    episode.ads_removed_seconds = request.ads_removed_seconds
    episode.detection_result_path = request.detection_result_path
    episode.claimed_at = None
    episode.progress_step = None
    episode.progress_percent = None
    episode.error_message = None  # Clear any previous error on success
    db.commit()

    return {"status": "complete"}


@router.post("/queue/{episode_id}/fail")
async def fail_episode(
    episode_id: str,
    request: FailRequest,
    db: Session = Depends(get_db),
    _: None = Depends(get_worker_auth),
):
    """Mark an episode as failed. Will retry if under max retries."""
    episode = db.query(Episode).filter_by(id=episode_id).first()
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")

    # Idempotent: ignore if already failed
    if episode.status == EpisodeStatus.FAILED.value:
        return {"status": "already_failed"}

    episode.error_message = request.error
    episode.claimed_at = None
    episode.progress_step = None
    episode.progress_percent = None

    if episode.retry_count < MAX_RETRIES:
        # Retry: back to pending
        episode.retry_count += 1
        episode.status = EpisodeStatus.PENDING.value
        db.commit()
        return {"status": "retrying", "retry_count": episode.retry_count}
    else:
        # Max retries reached: permanent failure
        episode.status = EpisodeStatus.FAILED.value
        db.commit()
        return {"status": "failed"}


@router.post("/queue/{episode_id}/progress")
async def update_progress(
    episode_id: str,
    request: ProgressRequest,
    db: Session = Depends(get_db),
    _: None = Depends(get_worker_auth),
):
    """Update the progress of a processing episode."""
    episode = db.query(Episode).filter_by(id=episode_id).first()
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")

    # Only update progress for processing episodes
    if episode.status != EpisodeStatus.PROCESSING.value:
        return {"status": "ignored", "reason": "not_processing"}

    episode.progress_step = request.step
    episode.progress_percent = request.percent
    db.commit()

    return {"status": "updated"}
