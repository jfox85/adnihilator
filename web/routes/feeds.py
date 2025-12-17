"""RSS feed routes."""

from datetime import timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from feedgen.feed import FeedGenerator
from sqlalchemy.orm import Session

from ..dependencies import get_db
from ..models import Episode, EpisodeStatus, Podcast
from ..services.r2 import get_audio_url


def format_duration(seconds: int | float | None) -> str | None:
    """Format duration in seconds to HH:MM:SS format for iTunes."""
    if seconds is None:
        return None
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


router = APIRouter(tags=["feeds"])


@router.get("/feed/{feed_token}.xml")
async def get_feed(
    feed_token: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """Generate RSS feed for a podcast."""
    podcast = db.query(Podcast).filter_by(feed_token=feed_token).first()
    if not podcast:
        raise HTTPException(status_code=404, detail="Feed not found")

    # Get completed episodes only
    episodes = (
        db.query(Episode)
        .filter(
            Episode.podcast_id == podcast.id,
            Episode.status == EpisodeStatus.COMPLETE.value,
        )
        .order_by(Episode.published_at.desc())
        .all()
    )

    # Build feed
    fg = FeedGenerator()
    fg.load_extension("podcast")

    # Channel metadata
    fg.title(podcast.title or "Untitled Podcast")
    fg.link(href=podcast.source_rss_url, rel="alternate")

    # Use original description if available, otherwise create one
    if podcast.description:
        fg.description(podcast.description)
    else:
        fg.description(f"Ad-free version of {podcast.title or 'podcast'}")

    # Self-referencing link for the feed
    feed_url = str(request.url)
    fg.link(href=feed_url, rel="self", type="application/rss+xml")

    # Podcast artwork
    if podcast.artwork_url:
        fg.image(podcast.artwork_url)
        fg.podcast.itunes_image(podcast.artwork_url)

    # iTunes-specific channel tags
    fg.podcast.itunes_author(podcast.title or "Unknown")
    if podcast.description:
        fg.podcast.itunes_summary(podcast.description)

    for episode in episodes:
        fe = fg.add_entry()
        fe.id(episode.guid)
        fe.title(episode.title or "Untitled Episode")

        # Episode description/summary
        if episode.description:
            fe.description(episode.description)
            fe.podcast.itunes_summary(episode.description)

        if episode.published_at:
            # feedgen requires timezone-aware datetimes
            pub_date = episode.published_at
            if pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=timezone.utc)
            fe.pubDate(pub_date)

        # Episode duration - use processed duration if available, otherwise original
        duration = episode.processed_duration or episode.duration
        if duration:
            fe.podcast.itunes_duration(format_duration(duration))

        # Audio enclosure pointing to R2
        if episode.processed_audio_key:
            audio_url = get_audio_url(episode.processed_audio_key)
            fe.enclosure(audio_url, 0, "audio/mpeg")

    rss_xml = fg.rss_str(pretty=True)
    return Response(content=rss_xml, media_type="application/xml")
