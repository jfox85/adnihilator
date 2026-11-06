"""Podcast management routes."""

import re
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..dependencies import get_db
from ..auth import verify_admin
from ..models import Podcast
from ..services.audio import is_usable_audio_url
from ..values import finite_nonnegative_number

router = APIRouter(prefix="/podcasts", tags=["podcasts"])

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def regex_replace(value: str, pattern: str, replacement: str) -> str:
    """Custom Jinja2 filter for regex replacement."""
    if value is None:
        return ""
    return re.sub(pattern, replacement, value)


def relative_time(dt: datetime | None) -> str:
    """Convert datetime to relative time string like '5m ago', '2h ago'."""
    if dt is None:
        return ""
    now = datetime.utcnow()
    diff = now - dt
    seconds = int(diff.total_seconds())

    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    if seconds < 604800:
        return f"{seconds // 86400}d ago"
    return dt.strftime("%b %d, %Y")


# Register custom filters
templates.env.filters["regex_replace"] = regex_replace
templates.env.filters["relative_time"] = relative_time


@router.get("")
async def list_podcasts(
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
) -> list[dict]:
    """List all podcasts."""
    podcasts = db.query(Podcast).order_by(Podcast.created_at.desc()).all()
    return [
        {
            "id": p.id,
            "title": p.title,
            "source_rss_url": p.source_rss_url,
            "feed_token": p.feed_token,
            "auto_process": p.auto_process,
            "last_sync_error": p.last_sync_error,
            "created_at": p.created_at.isoformat(),
        }
        for p in podcasts
    ]


@router.post("")
async def add_podcast(
    source_rss_url: str = Form(...),
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
):
    """Add a new podcast subscription and sync its feed."""
    from ..services.feed_sync import sync_podcast

    if not is_usable_audio_url(source_rss_url):
        raise HTTPException(status_code=422, detail="Feed URL must use HTTP(S) with a host")
    podcast = Podcast(source_rss_url=source_rss_url)
    db.add(podcast)
    db.commit()

    # Sync the feed to get title and episodes
    try:
        sync_podcast(db, podcast)
    except Exception:
        # sync_podcast durably records the safe error for the retained subscription.
        pass

    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/{podcast_id}/sync")
async def sync_podcast_feed(
    podcast_id: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
):
    """Sync a podcast's feed to get new episodes."""
    from ..services.feed_sync import sync_podcast

    podcast = db.query(Podcast).filter_by(id=podcast_id).first()
    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    from ..services.feed_sync import FeedSyncError
    try:
        episodes_added = sync_podcast(db, podcast)
    except FeedSyncError as error:
        return JSONResponse(status_code=502, content={"status": "failed", "error": str(error)})
    return {
        "status": "synced",
        "episodes_added": episodes_added,
        "title": podcast.title,
        "last_synced_at": podcast.last_synced_at.isoformat() if podcast.last_synced_at else None,
    }


@router.post("/sync-all")
async def sync_all_podcasts(
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
):
    """Sync all podcasts to get new episodes."""
    from ..services.feed_sync import sync_all_podcasts

    results = sync_all_podcasts(db)
    total_added = sum(r["added"] for r in results.values())
    return {"status": "synced", "total_episodes_added": total_added, "results": results}


@router.delete("/{podcast_id}")
async def delete_podcast(
    podcast_id: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
):
    """Delete a podcast subscription."""
    podcast = db.query(Podcast).filter_by(id=podcast_id).first()
    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    db.delete(podcast)
    db.commit()

    return {"status": "deleted"}


@router.get("/{podcast_id}/episodes", response_class=HTMLResponse)
async def list_episodes_html(
    podcast_id: str,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
):
    """List episodes for a podcast (HTML page)."""
    from ..models import Episode

    podcast = db.query(Podcast).filter_by(id=podcast_id).first()
    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    episodes = (
        db.query(Episode)
        .filter_by(podcast_id=podcast_id)
        .order_by(Episode.published_at.desc())
        .all()
    )

    return templates.TemplateResponse(
        request,
        "episodes.html",
        {
            "title": podcast.title or "Episodes",
            "podcast": podcast,
            "episodes": episodes,
            "audio_eligible": {episode.id: is_usable_audio_url(episode.original_audio_url) for episode in episodes},
        },
    )


@router.post("/{podcast_id}/episodes/{episode_id}/queue")
async def queue_episode(
    podcast_id: str,
    episode_id: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
):
    """Queue an episode for processing."""
    from ..models import Episode, EpisodeStatus

    episode = db.query(Episode).filter_by(id=episode_id, podcast_id=podcast_id).first()
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")

    if episode.status == EpisodeStatus.COMPLETE.value:
        raise HTTPException(status_code=409, detail="Completed episodes cannot be queued again")
    if episode.status == EpisodeStatus.PROCESSING.value:
        raise HTTPException(status_code=409, detail="Episode is processing")
    if not is_usable_audio_url(episode.original_audio_url):
        raise HTTPException(status_code=409, detail="Episode has no usable audio")
    if episode.status == EpisodeStatus.PENDING.value:
        return {"status": "already_queued"}

    updated = db.query(Episode).filter(
        Episode.id == episode_id,
        Episode.podcast_id == podcast_id,
        Episode.status.notin_([EpisodeStatus.PROCESSING.value, EpisodeStatus.PENDING.value, EpisodeStatus.COMPLETE.value]),
        Episode.original_audio_url == episode.original_audio_url,
    ).update({Episode.status: EpisodeStatus.PENDING.value, Episode.retry_count: 0,
              Episode.error_message: None}, synchronize_session=False)
    db.commit()
    if not updated:
        db.refresh(episode)
        if episode.status == EpisodeStatus.PENDING.value:
            return {"status": "already_queued"}
        raise HTTPException(status_code=409, detail="Episode changed while queueing; refresh and try again")
    return {"status": "queued"}


@router.post("/{podcast_id}/episodes/{episode_id}/skip")
async def skip_episode(
    podcast_id: str,
    episode_id: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
):
    """Mark an episode as skipped."""
    from ..models import Episode, EpisodeStatus

    episode = db.query(Episode).filter_by(id=episode_id, podcast_id=podcast_id).first()
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")

    updated = db.query(Episode).filter(
        Episode.id == episode_id,
        Episode.podcast_id == podcast_id,
        Episode.status != EpisodeStatus.PROCESSING.value,
    ).update({Episode.status: EpisodeStatus.SKIPPED.value}, synchronize_session=False)
    db.commit()
    if not updated:
        raise HTTPException(status_code=409, detail="Episode is processing")
    return {"status": "skipped"}


@router.post("/{podcast_id}/episodes/skip-all")
async def skip_all_episodes(
    podcast_id: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
):
    """Skip all unprocessed episodes for a podcast."""
    from ..models import Episode, EpisodeStatus

    podcast = db.query(Podcast).filter_by(id=podcast_id).first()
    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    # Skip all pending/failed episodes
    updated = (
        db.query(Episode)
        .filter_by(podcast_id=podcast_id)
        .filter(Episode.status.in_([EpisodeStatus.PENDING.value, EpisodeStatus.FAILED.value]))
        .update({Episode.status: EpisodeStatus.SKIPPED.value}, synchronize_session=False)
    )
    db.commit()

    return {"status": "skipped", "count": updated}


@router.post("/{podcast_id}/episodes/queue-all")
async def queue_all_episodes(
    podcast_id: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
):
    """Queue all unprocessed episodes for processing."""
    from ..models import Episode, EpisodeStatus

    podcast = db.query(Podcast).filter_by(id=podcast_id).first()
    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    # Queue all skipped/failed episodes
    candidates = (
        db.query(Episode.id, Episode.original_audio_url)
        .filter_by(podcast_id=podcast_id)
        .filter(Episode.status.in_([EpisodeStatus.SKIPPED.value, EpisodeStatus.FAILED.value]))
        .all()
    )
    queued_ids = []
    for episode_id, audio_url in candidates:
        if not is_usable_audio_url(audio_url):
            continue
        changed = db.query(Episode).filter(
            Episode.id == episode_id,
            Episode.podcast_id == podcast_id,
            Episode.status.in_([EpisodeStatus.SKIPPED.value, EpisodeStatus.FAILED.value]),
            Episode.original_audio_url == audio_url,
        ).update({Episode.status: EpisodeStatus.PENDING.value, Episode.retry_count: 0,
                  Episode.error_message: None}, synchronize_session=False)
        if changed:
            queued_ids.append(episode_id)
    db.commit()

    return {"status": "queued", "count": len(queued_ids), "episode_ids": queued_ids}


@router.post("/{podcast_id}/auto-process")
async def toggle_auto_process(
    podcast_id: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
):
    """Toggle auto-process setting for a podcast."""
    podcast = db.query(Podcast).filter_by(id=podcast_id).first()
    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    podcast.auto_process = not podcast.auto_process
    db.commit()

    return {"auto_process": podcast.auto_process}


@router.get("/{podcast_id}/episodes/status")
async def get_episode_statuses(
    podcast_id: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
):
    """Get status of all episodes for polling updates."""
    from ..models import Episode

    podcast = db.query(Podcast).filter_by(id=podcast_id).first()
    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    episodes = (
        db.query(Episode)
        .filter_by(podcast_id=podcast_id)
        .all()
    )

    return {
        "auto_process": podcast.auto_process,
        "episodes": {
            ep.id: {
                "status": ep.status,
                "audio_eligible": is_usable_audio_url(ep.original_audio_url),
                "error_message": ep.error_message,
                "ads_removed_seconds": finite_nonnegative_number(ep.ads_removed_seconds),
                "processed_duration": finite_nonnegative_number(ep.processed_duration),
                "progress_step": ep.progress_step,
                "progress_percent": finite_nonnegative_number(ep.progress_percent, maximum=100),
                "llm_cost_usd": finite_nonnegative_number(ep.llm_cost_usd),
                "llm_model": ep.llm_model,
                "detection_source": ep.detection_source,
            }
            for ep in episodes
        },
    }
