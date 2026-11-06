"""Worker API routes."""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import case
from sqlalchemy.orm import Session

from ..app import get_db
from ..auth import verify_worker_api_key
from ..models import Episode, EpisodeStatus, EpisodeProviderHold
from ..services import provider_outages as outages

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["worker"])

MAX_RETRIES = 2

# Episodes stuck in 'processing' longer than this are assumed to be from a
# worker that died mid-job (e.g. watchdog restart) and are recovered on claim.
DEFAULT_STUCK_TIMEOUT_SECONDS = 7200  # 2 hours
# Upper bound so a misconfigured env value cannot overflow timedelta or
# effectively disable recovery (30 days).
MAX_STUCK_TIMEOUT_SECONDS = 30 * 24 * 3600


def _stuck_timeout_seconds() -> int:
    """Read the stuck-job timeout from the environment, falling back to default.

    Invalid, non-positive, or out-of-range values are rejected (with a warning)
    in favour of the default so a misconfiguration cannot break claim handling.
    """
    raw = os.getenv("WORKER_STUCK_TIMEOUT_SECONDS")
    if not raw:
        return DEFAULT_STUCK_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid WORKER_STUCK_TIMEOUT_SECONDS=%r, using default %ds",
            raw,
            DEFAULT_STUCK_TIMEOUT_SECONDS,
        )
        return DEFAULT_STUCK_TIMEOUT_SECONDS
    if value <= 0 or value > MAX_STUCK_TIMEOUT_SECONDS:
        logger.warning(
            "WORKER_STUCK_TIMEOUT_SECONDS=%d out of range (1..%d), using default %ds",
            value,
            MAX_STUCK_TIMEOUT_SECONDS,
            DEFAULT_STUCK_TIMEOUT_SECONDS,
        )
        return DEFAULT_STUCK_TIMEOUT_SECONDS
    return value


def requeue_stuck_episodes(db: Session, *, commit: bool = True) -> int:
    """Recover episodes stuck in 'processing' so the queue is not blocked.

    An episode is considered stuck when it has been in 'processing' with a
    ``claimed_at`` older than the configured timeout. ``claimed_at`` is renewed
    on every progress update, so a stale value means the worker that claimed it
    died before completing or failing the job.

    Recovery consumes the retry budget like an explicit failure: each stuck
    episode has its ``retry_count`` incremented and is returned to 'pending'
    while under ``MAX_RETRIES``, or marked permanently 'failed' once exhausted.
    This prevents a poison episode that repeatedly crashes the worker from
    being requeued forever.

    Returns the number of episodes that were recovered.
    """
    cutoff = datetime.utcnow() - timedelta(seconds=_stuck_timeout_seconds())
    stuck = (
        db.query(Episode)
        .filter(
            Episode.status == EpisodeStatus.PROCESSING.value,
            Episode.claimed_at.isnot(None),
            Episode.claimed_at < cutoff,
            ~Episode.id.in_(db.query(EpisodeProviderHold.episode_id)),
        )
        .all()
    )

    recovered = 0
    for episode in stuck:
        # Another request can renew or recover this lease after our SELECT.
        # Only the unchanged expired lease may spend a recovery attempt.
        retry_available = Episode.retry_count < MAX_RETRIES
        recovered += db.query(Episode).filter(
            Episode.id == episode.id,
            Episode.status == EpisodeStatus.PROCESSING.value,
            Episode.claimed_at == episode.claimed_at,
            Episode.claimed_at < cutoff,
        ).update({
            Episode.claimed_at: None,
            Episode.progress_step: None,
            Episode.progress_percent: None,
            Episode.error_message: "Reclaimed: worker stopped responding while processing",
            Episode.retry_count: case((retry_available, Episode.retry_count + 1), else_=Episode.retry_count),
            Episode.status: case((retry_available, EpisodeStatus.PENDING.value), else_=EpisodeStatus.FAILED.value),
        }, synchronize_session=False)

    if stuck and commit:
        db.commit()

    return recovered


def get_worker_auth(x_worker_api_key: str = Header(...)):
    """Dependency to verify worker API key."""
    verify_worker_api_key(x_worker_api_key)


class RecoveryProof(BaseModel):
    """Only the worker holding this outage-probe lease can finish it."""
    provider: Literal["gemini", "openai"]
    token: str = Field(min_length=32, max_length=32, pattern=r"^[a-f0-9]{32}$")


class ProviderHoldRequest(BaseModel):
    """An authenticated global failure holds rather than spends retries."""
    provider: Literal["gemini", "openai"]
    category: Literal["billing", "authentication", "model"]
    error: str = Field(min_length=1, max_length=500)
    recovery_tokens: list[RecoveryProof] = Field(default_factory=list, max_length=2)


class CompleteRequest(BaseModel):
    """Request body for completing an episode."""

    recovery_tokens: list[RecoveryProof] = Field(default_factory=list, max_length=2)
    provider_successes: list[Literal["gemini", "openai"]] = Field(default_factory=list, max_length=2)
    processed_audio_key: str
    processed_duration: float = Field(ge=0, allow_inf_nan=False)
    ads_removed_seconds: float = Field(ge=0, allow_inf_nan=False)
    detection_result_path: Optional[str] = None

    # LLM tracking fields
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_input_tokens: Optional[int] = Field(default=None, ge=0, le=2**63 - 1)
    llm_output_tokens: Optional[int] = Field(default=None, ge=0, le=2**63 - 1)
    llm_total_tokens: Optional[int] = Field(default=None, ge=0, le=2**63 - 1)
    llm_cost_usd: Optional[float] = Field(default=None, ge=0, le=9999.999999, allow_inf_nan=False)
    detection_source: Optional[str] = None

    @field_validator("processed_audio_key")
    @classmethod
    def require_audio_key(cls, value: str) -> str:
        """Reject empty keys without normalizing valid object names."""
        if not value.strip():
            raise ValueError("processed_audio_key must not be blank")
        return value


class FailRequest(BaseModel):
    """Request body for failing an episode."""

    error: str
    recovery_tokens: list[RecoveryProof] = Field(default_factory=list, max_length=2)


class ProgressRequest(BaseModel):
    """Request body for updating progress."""

    recovery_tokens: list[RecoveryProof] = Field(default_factory=list, max_length=2)

    step: str = Field(max_length=50)
    percent: Optional[int] = Field(default=None, ge=0, le=100)


@router.get("/provider-state")
async def provider_state(db: Session = Depends(get_db), _: None = Depends(get_worker_auth)) -> dict:
    """Negotiate the durable hold protocol before a new worker claims work."""
    return {"protocol": outages.PROTOCOL, "paused": bool(outages.active_outages(db))}


@router.post("/queue/{episode_id}/provider-hold")
async def provider_hold(episode_id: str, request: ProviderHoldRequest,
                        db: Session = Depends(get_db), _: None = Depends(get_worker_auth)) -> dict:
    """Hold a global outage without consuming the episode's retry budget."""
    outages.begin_write(db)
    episode = db.get(Episode, episode_id)
    if episode is None:
        raise HTTPException(404, "Episode not found")
    if episode.status == EpisodeStatus.PENDING.value and db.query(EpisodeProviderHold).filter_by(episode_id=episode_id).first():
        db.commit()
        return {"status": "already_held"}
    outages.report_outage(db, episode, request.provider, request.category, request.error,
                          [proof.model_dump() for proof in request.recovery_tokens])
    db.commit()
    return {"status": "held"}


@router.post("/queue/claim-with-provider-recovery")
async def claim_with_provider_recovery(
    db: Session = Depends(get_db), _: None = Depends(get_worker_auth),
    x_worker_protocol: Optional[str] = Header(default=None),
) -> Optional[dict]:
    """A versioned path cannot accidentally claim work on an older server."""
    if x_worker_protocol != outages.PROTOCOL:
        raise HTTPException(409, "The provider recovery worker protocol is required")
    return await claim_episode(db=db, _=None, x_worker_protocol=x_worker_protocol)


@router.post("/queue/claim")
async def claim_episode(
    db: Session = Depends(get_db), _: None = Depends(get_worker_auth),
    x_worker_protocol: Optional[str] = Header(default=None),
) -> Optional[dict]:
    """Serialize claiming with durable pauses and one leased recovery job."""
    # Existing expired-snapshot recovery remains optimistic/CAS so a fresh
    # heartbeat can win. Held/probe episodes are excluded from that budget path.
    requeue_stuck_episodes(db)
    outages.begin_write(db)
    allowed, recovery = outages.prepare_claim(db, x_worker_protocol, _stuck_timeout_seconds())
    if not allowed:
        db.commit()
        return None
    candidates = db.query(Episode).filter(Episode.status == EpisodeStatus.PENDING.value)
    if recovery:
        held = candidates.filter(Episode.id.in_(db.query(EpisodeProviderHold.episode_id))).order_by(Episode.created_at, Episode.id).first()
        episode = held or candidates.order_by(Episode.created_at, Episode.id).first()
    else:
        episode = candidates.filter(~Episode.id.in_(db.query(EpisodeProviderHold.episode_id))).order_by(Episode.created_at, Episode.id).first()
    if episode is None:
        db.commit()
        return None
    episode.status = EpisodeStatus.PROCESSING.value
    episode.claimed_at = datetime.utcnow()
    proofs = outages.reserve_probe(db, episode, recovery)
    db.commit()
    return {
        "id": episode.id, "podcast_id": episode.podcast_id,
        "podcast_title": episode.podcast.title if episode.podcast else None,
        "guid": episode.guid, "title": episode.title,
        "original_audio_url": episode.original_audio_url,
        "description": episode.description, "source_url": episode.source_url,
        "status": episode.status, "provider_recovery": proofs,
    }


@router.post("/queue/{episode_id}/complete")
async def complete_episode(
    episode_id: str,
    request: CompleteRequest,
    db: Session = Depends(get_db),
    _: None = Depends(get_worker_auth),
):
    """Mark an episode as successfully processed."""
    outages.begin_write(db)
    episode = db.query(Episode).filter_by(id=episode_id).first()
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")

    # Idempotent: ignore if already complete
    if episode.status == EpisodeStatus.COMPLETE.value:
        return {"status": "already_complete"}

    outages.complete_probe(db, episode,
                           [proof.model_dump() for proof in request.recovery_tokens],
                           request.provider_successes)

    episode.status = EpisodeStatus.COMPLETE.value
    episode.processed_audio_key = request.processed_audio_key
    episode.processed_duration = request.processed_duration
    episode.ads_removed_seconds = request.ads_removed_seconds
    episode.detection_result_path = request.detection_result_path
    episode.claimed_at = None
    episode.progress_step = None
    episode.progress_percent = None
    episode.error_message = None  # Clear any previous error on success

    # LLM tracking fields
    episode.llm_provider = request.llm_provider
    episode.llm_model = request.llm_model
    episode.llm_input_tokens = request.llm_input_tokens
    episode.llm_output_tokens = request.llm_output_tokens
    episode.llm_total_tokens = request.llm_total_tokens
    episode.llm_cost_usd = request.llm_cost_usd
    episode.detection_source = request.detection_source

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

    # Normal failures retain the existing conditional status update. Only
    # recovery results need a serialized lease snapshot; do not block ordinary
    # concurrent completion/heartbeat delivery while loading a normal episode.
    has_hold = db.query(EpisodeProviderHold).filter_by(episode_id=episode_id).first() is not None
    if has_hold or request.recovery_tokens:
        db.rollback()
        outages.begin_write(db)
        episode = db.get(Episode, episode_id)
        if episode is None:
            raise HTTPException(404, "Episode not found")
    if episode.status == EpisodeStatus.PENDING.value and db.query(EpisodeProviderHold).filter_by(episode_id=episode_id).first():
        db.commit()
        return {"status": "already_held"}
    proofs = [proof.model_dump() for proof in request.recovery_tokens]
    leases = outages.validate_proofs(db, episode, proofs)
    if leases:
        outages.report_probe_failure(db, episode, request.error, proofs)
        db.commit()
        return {"status": "held"}

    # The status predicate also protects concurrent duplicate deliveries and a
    # completion committed after the read above. Attempt fencing is separate.
    retry_available = Episode.retry_count < MAX_RETRIES
    changed = db.query(Episode).filter(
        Episode.id == episode_id,
        Episode.status == EpisodeStatus.PROCESSING.value,
    ).update({
        Episode.error_message: request.error,
        Episode.claimed_at: None,
        Episode.progress_step: None,
        Episode.progress_percent: None,
        Episode.retry_count: case((retry_available, Episode.retry_count + 1), else_=Episode.retry_count),
        Episode.status: case((retry_available, EpisodeStatus.PENDING.value), else_=EpisodeStatus.FAILED.value),
    }, synchronize_session=False)
    db.commit()
    db.refresh(episode)
    if not changed:
        if episode.status == EpisodeStatus.FAILED.value:
            return {"status": "already_failed"}
        return {"status": "ignored", "reason": "not_processing"}
    if episode.status == EpisodeStatus.PENDING.value:
        return {"status": "retrying", "retry_count": episode.retry_count}
    return {"status": "failed"}


@router.post("/queue/{episode_id}/progress")
async def update_progress(
    episode_id: str,
    request: ProgressRequest,
    db: Session = Depends(get_db),
    _: None = Depends(get_worker_auth),
):
    """Update the progress of a processing episode."""
    outages.begin_write(db)
    episode = db.query(Episode).filter_by(id=episode_id).first()
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")

    # Only update progress for processing episodes
    if episode.status != EpisodeStatus.PROCESSING.value:
        return {"status": "ignored", "reason": "not_processing"}

    outages.validate_proofs(db, episode, [proof.model_dump() for proof in request.recovery_tokens])
    episode.progress_step = request.step
    episode.progress_percent = request.percent
    # Renew the claim lease: a job that is still reporting progress is alive,
    # so refresh claimed_at to keep stuck-job recovery from requeueing it.
    episode.claimed_at = datetime.utcnow()
    db.commit()

    return {"status": "updated"}
