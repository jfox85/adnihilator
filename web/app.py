"""FastAPI application for AdNihilator web service."""

import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .auth import security, verify_admin
from .database import get_engine, init_db, get_session_factory
from .dependencies import get_db
from .routes import podcasts, api, feeds, opml
from .services.scheduler import create_scheduler
from . import dependencies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    engine = init_db()
    dependencies.SessionFactory = get_session_factory(engine)

    # Start the feed sync scheduler
    scheduler = create_scheduler(dependencies.SessionFactory)
    if scheduler:
        scheduler.start()

    yield

    # Stop the scheduler on shutdown
    if scheduler:
        await scheduler.stop()


app = FastAPI(
    title="AdNihilator",
    description="Podcast ad removal service",
    version="0.1.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(podcasts.router)
app.include_router(api.router)
app.include_router(feeds.router)
app.include_router(opml.router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    db = Depends(get_db),
    _: None = Depends(verify_admin),
):
    """Dashboard home page."""
    from .models import Podcast

    podcasts = db.query(Podcast).order_by(Podcast.created_at.desc()).all()

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "title": "Dashboard",
            "podcasts": podcasts,
        },
    )


@app.get("/queue", response_class=HTMLResponse)
async def queue_page(
    request: Request,
    db = Depends(get_db),
    _: None = Depends(verify_admin),
):
    """Queue page showing all pending and processing episodes."""
    from .models import Episode, EpisodeStatus, Podcast

    # Get all pending and processing episodes with their podcasts
    episodes = (
        db.query(Episode, Podcast)
        .join(Podcast, Episode.podcast_id == Podcast.id)
        .filter(Episode.status.in_([EpisodeStatus.PENDING.value, EpisodeStatus.PROCESSING.value]))
        .order_by(
            # Processing first, then pending
            Episode.status.desc(),
            # Then by created/published date
            Episode.published_at.desc()
        )
        .all()
    )

    return templates.TemplateResponse(
        request,
        "queue.html",
        {
            "title": "Queue",
            "episodes": episodes,
        },
    )
