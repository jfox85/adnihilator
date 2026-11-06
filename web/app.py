"""FastAPI application for AdNihilator web service."""

import logging
import math
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Form, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .auth import (
    LoginRequiredError,
    SESSION_COOKIE,
    SESSION_MAX_AGE,
    check_credentials,
    create_session_token,
    verify_admin,
)
from .database import get_engine, init_db, get_session_factory
from .dependencies import get_db
from .routes import podcasts, api, feeds, opml
from .services.scheduler import create_scheduler
from .services import provider_outages as outages
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


def _json_safe_validation_value(value: Any) -> Any:
    """Keep nonfinite rejected input from breaking the JSON error response."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe_validation_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_validation_value(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def request_validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return the standard 422 detail shape even for invalid numeric input."""
    detail = _json_safe_validation_value(jsonable_encoder(exc.errors()))
    return JSONResponse(status_code=422, content={"detail": detail})


@app.exception_handler(LoginRequiredError)
async def login_redirect_handler(request: Request, exc: LoginRequiredError):
    return RedirectResponse(url=f"/login?next={exc.next_url}", status_code=303)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, next: str = "/"):
    return templates.TemplateResponse(
        request,
        "login.html",
        {"title": "Login", "next": next},
    )


@app.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next: str = Form("/"),
):
    if not check_credentials(username, password):
        return templates.TemplateResponse(
            request,
            "login.html",
            {"title": "Login", "next": next, "error": "Invalid credentials"},
            status_code=401,
        )
    response = RedirectResponse(url=next, status_code=303)
    response.set_cookie(
        SESSION_COOKIE,
        create_session_token(),
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    return response


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    return response

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
            **outages.banner_context(db),
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
    from .services.audio import is_usable_audio_url

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

    # Get recently completed episodes (last 10)
    recent_completed = (
        db.query(Episode, Podcast)
        .join(Podcast, Episode.podcast_id == Podcast.id)
        .filter(Episode.status == EpisodeStatus.COMPLETE.value)
        .order_by(Episode.updated_at.desc())
        .limit(10)
        .all()
    )

    # Get recently failed episodes (last 10)
    recent_failed = (
        db.query(Episode, Podcast)
        .join(Podcast, Episode.podcast_id == Podcast.id)
        .filter(Episode.status == EpisodeStatus.FAILED.value)
        .order_by(Episode.updated_at.desc())
        .limit(10)
        .all()
    )

    return templates.TemplateResponse(
        request,
        "queue.html",
        {
            "title": "Queue",
            "episodes": episodes,
            "recent_completed": recent_completed,
            **outages.banner_context(db),
            "recent_failed": recent_failed,
            "audio_eligible": {episode.id: is_usable_audio_url(episode.original_audio_url) for episode, _ in recent_failed},
        },
    )


@app.post("/provider-outages/{provider}/retry")
async def retry_provider(
    provider: Literal["gemini", "openai"], request: Request, db=Depends(get_db),
    _: None = Depends(verify_admin),
):
    """Let the owner advance recovery after checking provider readiness."""
    # This action can authorize billable recovery. Browser forms always send
    # Origin; require the same origin rather than ambient Basic/cookie auth alone.
    def origin_key(value: str) -> tuple:
        parsed = urlsplit(value)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
            raise ValueError("Invalid origin")
        return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        same_origin = origin_key(request.headers.get("origin", "")) == origin_key(str(request.base_url))
    except ValueError:
        same_origin = False
    if not same_origin or request.headers.get("sec-fetch-site") == "cross-site":
        from fastapi import HTTPException
        raise HTTPException(403, "A same-origin recovery form is required")
    outages.begin_write(db)
    outages.retry_now(db, provider, api._stuck_timeout_seconds())
    db.commit()
    return RedirectResponse("/queue", status_code=303)


@app.get("/provider-outages/banner", response_class=HTMLResponse)
async def provider_outage_banner(request: Request, db=Depends(get_db), _: None = Depends(verify_admin)):
    """Refresh safe pause status without provider or worker health calls."""
    return templates.TemplateResponse(request, "provider_outage_banner.html", outages.banner_context(db))
