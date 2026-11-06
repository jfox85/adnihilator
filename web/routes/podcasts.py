"""Podcast management routes."""

import re
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Body, Depends, Form, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from ..dependencies import get_db
from ..auth import verify_admin
from ..models import Episode, EpisodeStatus, Podcast
from ..formatting import format_cost, format_duration
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
templates.env.filters["duration"] = format_duration
templates.env.filters["cost"] = format_cost
templates.env.filters["finite_number"] = finite_nonnegative_number


class BulkSelection(BaseModel):
    """Episode identities the owner actually confirmed for a bulk action."""

    episode_ids: list[str]


EPISODES_PER_PAGE = 100


def _episode_page(
    db: Session, podcast_id: str, page: int, search: str, state: str, sort: str,
) -> tuple[list[Episode], int, int]:
    """Filter and sort the whole catalogue before selecting a bounded page."""
    query = db.query(Episode).filter_by(podcast_id=podcast_id)
    display_title = func.coalesce(func.nullif(Episode.title, ""), Episode.guid)
    if search:
        query = query.filter(
            func.catalogue_casefold(display_title).contains(
                search.casefold(), autoescape=True,
            )
        )
    if state:
        query = query.filter(Episode.status == state)
    total = query.count()
    last_page = max(1, (total + EPISODES_PER_PAGE - 1) // EPISODES_PER_PAGE)
    page = min(page, last_page)
    title = func.catalogue_casefold(display_title)
    ordering = {
        "date-desc": Episode.published_at.desc(),
        "date-asc": Episode.published_at.asc(),
        "title-asc": title.asc(),
        "title-desc": title.desc(),
        "status": case(
            {state.value: index for index, state in enumerate([
                EpisodeStatus.PROCESSING, EpisodeStatus.PENDING, EpisodeStatus.FAILED,
                EpisodeStatus.COMPLETE, EpisodeStatus.SKIPPED,
            ])}, value=Episode.status, else_=5,
        ),
    }
    episodes = query.order_by(ordering[sort], Episode.id).offset(
        (page - 1) * EPISODES_PER_PAGE
    ).limit(EPISODES_PER_PAGE).all()
    return episodes, total, page


@router.get("/{podcast_id}/episodes/bulk-summary")
async def bulk_summary(
    podcast_id: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
) -> dict:
    """Preview the exact eligible scope, including episodes hidden by filters."""
    from ..models import Episode, EpisodeStatus

    if db.get(Podcast, podcast_id) is None:
        raise HTTPException(status_code=404, detail="Podcast not found")
    episodes = db.query(Episode).filter_by(podcast_id=podcast_id).all()
    queue_ids = [ep.id for ep in episodes if ep.status in {
        EpisodeStatus.SKIPPED.value, EpisodeStatus.FAILED.value
    } and is_usable_audio_url(ep.original_audio_url)]
    skip_ids = [ep.id for ep in episodes if ep.status in {
        EpisodeStatus.PENDING.value, EpisodeStatus.FAILED.value
    }]
    return {"queue": {"count": len(queue_ids), "episode_ids": queue_ids},
            "skip": {"count": len(skip_ids), "episode_ids": skip_ids}}


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

    podcast = Podcast(source_rss_url=source_rss_url)
    db.add(podcast)
    db.commit()

    # Sync the feed to get title and episodes
    try:
        sync_podcast(db, podcast)
    except Exception:
        # If sync fails, the podcast is still added but without episodes
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

    episodes_added = sync_podcast(db, podcast)
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
    page: int = Query(default=1, ge=1, le=1_000_000),
    q: str = Query(default="", max_length=200),
    episode_status: str = Query(default="", alias="status", pattern="^(|pending|processing|complete|failed|skipped|expired)$"),
    sort: str = Query(default="date-desc", pattern="^(date-desc|date-asc|title-asc|title-desc|status)$"),
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
):
    """List episodes for a podcast (HTML page)."""
    from ..models import Episode

    podcast = db.query(Podcast).filter_by(id=podcast_id).first()
    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    q = q.strip()
    episodes, total, page = _episode_page(db, podcast_id, page, q, episode_status, sort)
    last_page = max(1, (total + EPISODES_PER_PAGE - 1) // EPISODES_PER_PAGE)

    return templates.TemplateResponse(
        request,
        "episodes.html",
        {
            "title": podcast.title or "Episodes",
            "podcast": podcast,
            "episodes": episodes,
            "catalogue": {"page": page, "pages": last_page, "total": total,
                          "start": (page - 1) * EPISODES_PER_PAGE + 1 if total else 0,
                          "end": min(page * EPISODES_PER_PAGE, total),
                          "search": q, "status": episode_status, "sort": sort,
                          "previous": request.url.path + "?" + request.url.include_query_params(page=page - 1).query if page > 1 else None,
                          "next": request.url.path + "?" + request.url.include_query_params(page=page + 1).query if page < last_page else None},
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

    if episode.status == EpisodeStatus.PROCESSING.value:
        raise HTTPException(status_code=409, detail="Episode is processing")
    if not is_usable_audio_url(episode.original_audio_url):
        raise HTTPException(status_code=409, detail="Episode has no usable audio")
    if episode.status == EpisodeStatus.PENDING.value:
        return {"status": "already_queued"}

    updated = db.query(Episode).filter(
        Episode.id == episode_id,
        Episode.podcast_id == podcast_id,
        Episode.status.notin_([EpisodeStatus.PROCESSING.value, EpisodeStatus.PENDING.value]),
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
    selection: BulkSelection | None = Body(default=None),
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
):
    """Skip all unprocessed episodes for a podcast."""
    from ..models import Episode, EpisodeStatus

    podcast = db.query(Podcast).filter_by(id=podcast_id).first()
    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    # Skip only confirmed identities when the client supplied a preview.
    selected_ids = set(selection.episode_ids) if selection is not None else None
    candidates = db.query(Episode.id).filter_by(podcast_id=podcast_id).filter(
        Episode.status.in_([EpisodeStatus.PENDING.value, EpisodeStatus.FAILED.value])
    ).all()
    skipped_ids = []
    for (episode_id,) in candidates:
        if selected_ids is not None and episode_id not in selected_ids:
            continue
        changed = db.query(Episode).filter(
            Episode.id == episode_id, Episode.podcast_id == podcast_id,
            Episode.status.in_([EpisodeStatus.PENDING.value, EpisodeStatus.FAILED.value]),
        ).update({Episode.status: EpisodeStatus.SKIPPED.value}, synchronize_session=False)
        if changed:
            skipped_ids.append(episode_id)
    db.commit()
    return {"status": "skipped", "count": len(skipped_ids), "episode_ids": skipped_ids}


@router.post("/{podcast_id}/episodes/queue-all")
async def queue_all_episodes(
    podcast_id: str,
    selection: BulkSelection | None = Body(default=None),
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
):
    """Queue eligible episodes, optionally bounded to a confirmed selection."""
    from ..models import Episode, EpisodeStatus

    podcast = db.query(Podcast).filter_by(id=podcast_id).first()
    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    selected_ids = set(selection.episode_ids) if selection is not None else None
    candidates = (
        db.query(Episode.id, Episode.original_audio_url)
        .filter_by(podcast_id=podcast_id)
        .filter(Episode.status.in_([EpisodeStatus.SKIPPED.value, EpisodeStatus.FAILED.value]))
        .all()
    )
    queued_ids = []
    for episode_id, audio_url in candidates:
        if selected_ids is not None and episode_id not in selected_ids:
            continue
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
    episode_id: list[str] | None = Query(default=None, max_length=100),
    catalogue_page: int | None = Query(default=None, ge=1, le=1_000_000),
    q: str = Query(default="", max_length=200),
    episode_status: str = Query(default="", alias="status", pattern="^(|pending|processing|complete|failed|skipped|expired)$"),
    sort: str = Query(default="date-desc", pattern="^(date-desc|date-asc|title-asc|title-desc|status)$"),
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
):
    """Poll a bounded catalogue window, selected IDs, or legacy queue clients."""
    from ..models import Episode

    podcast = db.query(Podcast).filter_by(id=podcast_id).first()
    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    catalogue = None
    if catalogue_page is not None:
        episodes, total, page = _episode_page(db, podcast_id, catalogue_page, q.strip(), episode_status, sort)
        catalogue = {"page": page, "total": total, "ids": [ep.id for ep in episodes]}
    else:
        query = db.query(Episode).filter_by(podcast_id=podcast_id)
        if episode_id is not None:
            query = query.filter(Episode.id.in_(episode_id))
        episodes = query.all()

    return {
        "auto_process": podcast.auto_process,
        "catalogue": catalogue,
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
