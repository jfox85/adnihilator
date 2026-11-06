"""Durable holds, serialized claim gating and bounded provider recovery."""

import os
import secrets
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from adnihilator.redaction import redact_secrets
from ..models import Episode, EpisodeProviderHold, EpisodeStatus, ProviderOutage

PROTOCOL = "provider-outage-v1"


def begin_write(db: Session) -> None:
    """Serialize SQLite claim/hold/probe decisions before taking a snapshot."""
    db.execute(text("BEGIN IMMEDIATE"))


def cooldown(failures: int) -> int:
    """Default recovery waits 300,600,1200,1800 seconds and remains capped."""
    try:
        base = int(os.environ.get("PROVIDER_RECOVERY_BASE_SECONDS", "300"))
        maximum = int(os.environ.get("PROVIDER_RECOVERY_MAX_SECONDS", "1800"))
    except ValueError:
        base, maximum = 300, 1800
    if not 1 <= base <= maximum <= 1800:
        base, maximum = 300, 1800
    return min(base * (2 ** min(max(failures - 1, 0), 12)), maximum)


def safe_message(message: str) -> str:
    """Never persist the authenticated worker key in an outage or banner."""
    return redact_secrets(message, (os.environ.get("WORKER_API_KEY"),))[:500]


def active_outages(db: Session) -> list[ProviderOutage]:
    """Return current pauses in a stable order for gating and dashboards."""
    return db.query(ProviderOutage).filter(ProviderOutage.active.is_(True)).order_by(ProviderOutage.provider).all()


def hold_episode(db: Session, episode: Episode, provider: str, message: str) -> None:
    """Park an episode without incrementing or resetting its retry count."""
    episode.status = EpisodeStatus.PENDING.value
    episode.claimed_at = None
    episode.progress_step = None
    episode.progress_percent = None
    episode.error_message = safe_message(message)
    if db.get(EpisodeProviderHold, (episode.id, provider)) is None:
        db.add(EpisodeProviderHold(episode_id=episode.id, provider=provider))


def validate_proofs(db: Session, episode: Episode, proofs: list[dict]) -> list[ProviderOutage]:
    """Fence only outage-probe results; broad callback fencing stays separate."""
    expected = db.query(ProviderOutage).filter(
        ProviderOutage.active.is_(True), ProviderOutage.probe_episode_id == episode.id,
    ).order_by(ProviderOutage.provider).all()
    supplied = {item["provider"]: item["token"] for item in proofs}
    if len(supplied) != len(proofs):
        raise HTTPException(409, "Duplicate recovery proof")
    if expected:
        if episode.status != EpisodeStatus.PROCESSING.value or set(supplied) != {row.provider for row in expected}:
            raise HTTPException(409, "Recovery lease is no longer current")
        if any(not secrets.compare_digest(row.probe_token or "", supplied[row.provider]) for row in expected):
            raise HTTPException(409, "Recovery lease is no longer current")
    elif proofs or db.query(EpisodeProviderHold).filter_by(episode_id=episode.id).first() is not None:
        raise HTTPException(409, "Held episode has no current recovery lease")
    return expected


def reconcile_probes(db: Session, rows: list[ProviderOutage], timeout: int) -> None:
    """A dead/skipped probe returns to a bounded hold without losing attempts."""
    now = datetime.utcnow()
    cutoff = now - timedelta(seconds=timeout)
    for episode_id in {row.probe_episode_id for row in rows if row.probe_episode_id}:
        episode = db.get(Episode, episode_id)
        if episode and episode.status == EpisodeStatus.PROCESSING.value and episode.claimed_at and episode.claimed_at >= cutoff:
            continue
        for row in rows:
            if row.probe_episode_id != episode_id:
                continue
            if episode and episode.status == EpisodeStatus.PROCESSING.value:
                hold_episode(db, episode, row.provider, "Recovery worker stopped responding; episode remains held")
            row.failure_count += 1
            row.retry_after = now + timedelta(seconds=cooldown(row.failure_count))
            row.probe_episode_id = None
            row.probe_token = None
            row.last_probe_error = "Recovery did not finish; waiting before trying another episode"
            row.updated_at = now
    db.flush()


def prepare_claim(db: Session, protocol: str | None, timeout: int) -> tuple[bool, list[ProviderOutage]]:
    """Pause normal claiming; admit one recovery lease when all pauses are due."""
    rows = active_outages(db)
    if not rows:
        return True, []
    reconcile_probes(db, rows, timeout)
    now = datetime.utcnow()
    if any(row.manual_pause or row.probe_token or row.retry_after is None or row.retry_after > now for row in rows):
        return False, []
    if protocol != PROTOCOL:
        return False, []
    return True, rows


def select_recovery_episode(db: Session) -> Episode | None:
    """Rotate pending probes fairly; prefer held work among untried episodes."""
    recent = db.query(
        EpisodeProviderHold.episode_id.label("episode_id"),
        func.max(EpisodeProviderHold.last_probe_at).label("last_probe_at"),
    ).group_by(EpisodeProviderHold.episode_id).subquery()
    return db.query(Episode).outerjoin(recent, recent.c.episode_id == Episode.id).filter(
        Episode.status == EpisodeStatus.PENDING.value,
    ).order_by(
        recent.c.last_probe_at.is_not(None), recent.c.last_probe_at,
        recent.c.episode_id.is_(None), Episode.created_at, Episode.id,
    ).first()


def reserve_probe(db: Session, episode: Episode, rows: list[ProviderOutage]) -> list[dict[str, str]]:
    """All active provider pauses share the same single recovery episode."""
    proofs = []
    now = datetime.utcnow()
    for row in rows:
        row.probe_episode_id = episode.id
        row.probe_token = uuid4().hex
        row.last_probe_error = None
        row.updated_at = datetime.utcnow()
        hold = db.get(EpisodeProviderHold, (episode.id, row.provider))
        if hold is None:
            hold = EpisodeProviderHold(episode_id=episode.id, provider=row.provider)
            db.add(hold)
        hold.last_probe_at = now
        proofs.append({"provider": row.provider, "token": row.probe_token})
    return proofs


def report_outage(db: Session, episode: Episode, provider: str, category: str,
                  message: str, proofs: list[dict]) -> None:
    """Record a global failure, preserving budget and existing cooldowns."""
    leases = validate_proofs(db, episode, proofs)
    if episode.status != EpisodeStatus.PROCESSING.value:
        raise HTTPException(409, "Only a processing episode can be held")
    now = datetime.utcnow()
    row = db.get(ProviderOutage, provider)
    leased_failure = any(lease.provider == provider for lease in leases)
    if row is None:
        row = ProviderOutage(provider=provider, category=category, message=safe_message(message),
                             failure_count=1, active=True, manual_pause=False,
                             retry_after=now + timedelta(seconds=cooldown(1)))
        db.add(row)
    elif not row.active:
        row.active = True
        row.manual_pause = False
        row.failure_count = 1
        row.retry_after = now + timedelta(seconds=cooldown(1))
        row.category = category
        row.message = safe_message(message)
    elif leased_failure:
        row.failure_count += 1
        row.retry_after = now + timedelta(seconds=cooldown(row.failure_count))
        row.message = safe_message(message)
        row.category = category
    # Concurrent already-in-flight failures join the pause; they never slide
    # the recovery deadline or multiply the backoff.
    for lease in leases:
        lease.probe_token = None
        lease.probe_episode_id = None
        lease.updated_at = now
    row.updated_at = now
    hold_episode(db, episode, provider, message)


def report_probe_failure(db: Session, episode: Episode, message: str, proofs: list[dict]) -> None:
    """An inconclusive recovery stays held; ordinary job failures are separate."""
    leases = validate_proofs(db, episode, proofs)
    if not leases:
        raise HTTPException(409, "No recovery probe is active")
    now = datetime.utcnow()
    for row in leases:
        row.failure_count += 1
        row.retry_after = now + timedelta(seconds=cooldown(row.failure_count))
        row.probe_token = None
        row.probe_episode_id = None
        row.last_probe_error = safe_message(message)
        row.updated_at = now
        hold_episode(db, episode, row.provider, message)


def complete_probe(db: Session, episode: Episode, proofs: list[dict], successes: list[str]) -> None:
    """Release only providers actually validated by the successful episode."""
    leases = validate_proofs(db, episode, proofs)
    # A recovery episode may also use healthy providers. Only leased rows below
    # can be released; successful use elsewhere does not invalidate this lease.
    now = datetime.utcnow()
    for row in leases:
        row.probe_token = None
        row.probe_episode_id = None
        row.updated_at = now
        if row.provider in successes:
            row.active = False
            row.manual_pause = False
            row.retry_after = None
            row.last_probe_error = None
            db.query(EpisodeProviderHold).filter_by(provider=row.provider).delete(synchronize_session=False)
        else:
            row.retry_after = now + timedelta(seconds=cooldown(row.failure_count))
            row.last_probe_error = "Episode finished without using this provider; recovery is not yet verified"
            db.query(EpisodeProviderHold).filter_by(provider=row.provider, episode_id=episode.id).delete(synchronize_session=False)


def retry_now(db: Session, provider: str, timeout: int) -> bool:
    """Advance recovery eligibility without duplicating a healthy probe lease."""
    rows = active_outages(db)
    reconcile_probes(db, rows, timeout)
    row = next((row for row in rows if row.provider == provider), None)
    if row is None:
        return False
    if row.probe_token:
        return False
    row.manual_pause = False
    row.retry_after = datetime.utcnow()
    row.updated_at = datetime.utcnow()
    return True


def pause_for_rollout(db: Session, provider: str = "gemini") -> None:
    """Owner-only rollout helper: call in a fresh session before worker loading.

    It performs no provider calls. The durable manual pause survives restart;
    after confirming the A1 billing/resumption window the owner uses Retry now.
    Normal SDK outages never enable this special deployment gate.
    """
    begin_write(db)
    row = db.get(ProviderOutage, provider)
    if row and row.probe_token:
        raise RuntimeError("Wait for the active recovery episode before rollout pause")
    if row is None:
        row = ProviderOutage(provider=provider, category="maintenance", message="Owner rollout pause pending provider readiness", failure_count=0)
        db.add(row)
    row.active = True
    row.manual_pause = True
    row.retry_after = None
    row.updated_at = datetime.utcnow()
    db.commit()


def banner_context(db: Session) -> dict:
    """Expose safe human-readable pause data, never recovery lease tokens."""
    result = []
    for row in active_outages(db):
        count = db.query(EpisodeProviderHold).join(Episode, Episode.id == EpisodeProviderHold.episode_id).filter(
            EpisodeProviderHold.provider == row.provider,
            Episode.status.in_([EpisodeStatus.PENDING.value, EpisodeStatus.PROCESSING.value]),
        ).count()
        result.append({"provider": row.provider, "label": "Gemini" if row.provider == "gemini" else "OpenAI",
                       "category": row.category, "manual_pause": row.manual_pause,
                       "retry_after": row.retry_after, "probing": bool(row.probe_token),
                       "held_count": count, "ready": bool(row.retry_after and row.retry_after <= datetime.utcnow()),
                       "recovery_incomplete": bool(row.last_probe_error)})
    held = {value for (value,) in db.query(EpisodeProviderHold.episode_id).all()}
    return {"provider_outages": result, "provider_held_ids": held, "provider_status_enabled": True}
