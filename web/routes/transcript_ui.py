"""Admin-only bounded search and transcript reader, with escaped highlighting."""

import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from pydantic import BeforeValidator
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import verify_admin
from ..dependencies import get_db, require_transcripts
from ..models import Episode, Podcast
from ..formatting import format_duration
from ..transcript_dates import BoundedDate
from ..services.transcripts import STOP, TranscriptError, TranscriptReader, coverage, identity, request_transcript
from .transcripts import check_origin

router = APIRouter(prefix="/transcripts", dependencies=[Depends(verify_admin), Depends(require_transcripts)])
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


def _blank_date(value: Any) -> Any:
    """Native date controls submit an empty string when no bound is selected."""
    return None if value == "" else value


OptionalDate = Annotated[BoundedDate | None, BeforeValidator(_blank_date)]


def publication_date(value: str | datetime | None) -> str:
    """Use the episode list's readable UTC publication date convention."""
    if value is None:
        return "Unknown"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
        return parsed.strftime("%b %d, %Y")
    except ValueError:
        return "Unknown"


def highlight(value: str, query: str) -> Markup:
    """Only the server's mark tags are HTML; every source text fragment is escaped."""
    tokens = list(dict.fromkeys(t for t in re.findall(r"\w+", query)[:30] if t.lower() not in STOP))
    if not tokens:
        return escape(value)
    expression = re.compile(r"\b(" + "|".join(re.escape(t) for t in tokens) + r")\b", re.I)
    parts = expression.split(value)
    return Markup("").join(Markup("<mark>") + escape(p) + Markup("</mark>") if n % 2 else escape(p) for n, p in enumerate(parts))


templates.env.filters["highlight"] = highlight
templates.env.filters["publication_date"] = publication_date
templates.env.filters["duration"] = format_duration


@router.get("")
def search_page(request: Request, podcast_id: str | None = Query(None, max_length=100),
                query: str = Query("", max_length=500), after: OptionalDate = None,
                before: OptionalDate = None, phrase: bool = False, latest: bool = False,
                db: Session = Depends(get_db)) -> Response:
    podcasts = db.scalars(select(Podcast).order_by(Podcast.title).limit(100)).all()
    podcast_id = podcast_id or (podcasts[0].id if podcasts else "")
    reader = TranscriptReader(db)
    data = reader.search(query, podcast_id, after, before, phrase, latest) if query else None
    return templates.TemplateResponse(request, "transcript_search.html", dict(title="Transcript search", podcasts=podcasts,
        podcast_id=podcast_id, query=query, after=after.date().isoformat() if after else "",
        before=before.date().isoformat() if before else "", phrase=phrase, latest=latest,
        result=data, listing=reader.episodes(podcast_id)))


@router.get("/episodes/{episode_id}")
def reader_page(episode_id: str, request: Request, checksum: str | None = Query(None, pattern="^[a-f0-9]{64}$"),
                ordinal: int = Query(0, ge=0, le=30000), query: str = Query("", max_length=500),
                db: Session = Depends(get_db)) -> Response:
    episode = db.get(Episode, episode_id)
    if not episode:
        raise HTTPException(404, "Episode not found")
    try:
        passage = TranscriptReader(db).passage(episode_id, checksum, ordinal, 40)
    except TranscriptError:
        if checksum:
            raise HTTPException(404, "Transcript revision not found") from None
        passage = None
    return templates.TemplateResponse(request, "transcript_reader.html", dict(title="Transcript", episode=identity(episode),
        coverage=coverage(db, episode), passage=passage, query=query, ordinal=ordinal))


@router.post("/episodes/{episode_id}/request")
def request_page(episode_id: str, request: Request, db: Session = Depends(get_db)) -> Response:
    check_origin(request, True)
    try:
        request_transcript(db, episode_id)
    except TranscriptError as exc:
        raise HTTPException(404, str(exc)) from None
    return RedirectResponse(f"/transcripts/episodes/{episode_id}", status_code=303)
