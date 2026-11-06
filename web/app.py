"""FastAPI application for AdNihilator web service."""

import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from fastapi import Depends, FastAPI, Form, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from adnihilator.redaction import redact_secrets

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
from . import dependencies
from .presentation import register_error_filters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
register_error_filters(templates)


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


def _accept_preference(accept: str, media_type: str) -> tuple[float, int, int]:
    """Find a representation's quality using its most specific media range."""
    target_type, target_subtype = media_type.split('/')
    matches = []
    for order, value in enumerate(accept.lower().split(',')):
        media_range, *parameters = (part.strip() for part in value.split(';'))
        if '/' not in media_range:
            continue
        major, minor = media_range.split('/', 1)
        if major not in (target_type, '*') or minor not in (target_subtype, '*'):
            continue
        if major == '*' and minor != '*':
            continue
        quality = 1.0
        for parameter in parameters:
            name, separator, raw = parameter.partition('=')
            if separator and name.strip() == 'q':
                try:
                    quality = float(raw.strip())
                except ValueError:
                    quality = 0.0
                if not 0 <= quality <= 1:
                    quality = 0.0
        specificity = 0 if major == '*' else 1 if minor == '*' else 2
        matches.append((specificity, -order, quality))
    if not matches:
        return (0.0, -1, 0)
    specificity, order, quality = max(matches, key=lambda item: item[:2])
    return (quality, specificity, order)


def _browser_error_page(request: Request) -> bool:
    """Select HTML only when acceptable and preferred to the JSON default."""
    if request.url.path.startswith(('/api/', '/feed/')):
        return False
    accept = request.headers.get('accept', '')
    html = _accept_preference(accept, 'text/html')
    return html[0] > 0 and html > _accept_preference(accept, 'application/json')


@app.exception_handler(StarletteHTTPException)
async def browser_http_error(request: Request, exc: StarletteHTTPException) -> Response:
    """Give browser navigation a safe message and a way back."""
    if not _browser_error_page(request):
        return await http_exception_handler(request, exc)
    messages = {
        401: ('Sign in required', 'Sign in again to continue.'),
        403: ('Access unavailable', 'You do not have access to this page.'),
        404: ('Page not found', 'This page may have moved or is no longer available.'),
    }
    heading, message = messages.get(exc.status_code, ('Request could not be completed', 'Try again, or return to the dashboard.'))
    return templates.TemplateResponse(request, 'error.html', {
        'title': heading, 'heading': heading, 'message': message,
        'sign_in': '/login?next=' + quote(request.url.path, safe='') if exc.status_code == 401 else None,
    }, status_code=exc.status_code, headers=exc.headers)


@app.exception_handler(Exception)
async def browser_server_error(request: Request, exc: Exception) -> Response:
    """Keep exception details private even when a browser page fails."""
    logging.getLogger(__name__).error('Web request failed: %s', redact_secrets(str(exc)))
    if not _browser_error_page(request):
        return PlainTextResponse('Internal Server Error', status_code=500)
    return templates.TemplateResponse(request, 'error.html', {
        'title': 'Something went wrong', 'heading': 'Something went wrong',
        'message': 'The page could not be loaded. Try again shortly, or return to the dashboard.',
    }, status_code=500)


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
            "recent_failed": recent_failed,
        },
    )
