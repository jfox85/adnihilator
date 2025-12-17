# AdNihilator Web Service Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a web service that manages podcast subscriptions, serves ad-free RSS feeds, and coordinates with a local worker for audio processing.

**Architecture:** FastAPI web app on VPS with SQLite database, local worker daemon on Mac that processes episodes and uploads to Cloudflare R2. The web app serves management UI (with Basic Auth), worker API, and RSS feeds.

**Tech Stack:** FastAPI, SQLAlchemy, SQLite (WAL mode), Jinja2, feedgen, feedparser, httpx, boto3

---

## Phase 1: Web Service Foundation

### Task 1: Add Web Dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add web dependencies**

Add these to the `dependencies` list in `pyproject.toml`:

```toml
dependencies = [
    "faster-whisper>=1.0.0",
    "typer>=0.9.0",
    "rich>=10.11.0",
    "pydantic>=2.0.0",
    "tomli>=2.0.0",
    "openai>=1.0.0",
    "numpy<2.0.0",
    # Web service
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "sqlalchemy>=2.0.0",
    "jinja2>=3.1.0",
    "feedgen>=1.0.0",
    "feedparser>=6.0.0",
    "httpx>=0.25.0",
    "boto3>=1.34.0",
    "python-multipart>=0.0.6",
]
```

**Step 2: Install dependencies**

Run: `pip install -e .`
Expected: All packages install successfully

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add web service dependencies"
```

---

### Task 2: Create Database Models

**Files:**
- Create: `web/__init__.py`
- Create: `web/database.py`
- Create: `web/models.py`
- Test: `tests/web/__init__.py`
- Test: `tests/web/test_models.py`

**Step 1: Create web package**

Create `web/__init__.py`:

```python
"""AdNihilator Web Service."""
```

Create `tests/web/__init__.py`:

```python
"""Web service tests."""
```

**Step 2: Write the failing test for database models**

Create `tests/web/test_models.py`:

```python
"""Tests for database models."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from web.database import Base
from web.models import Podcast, Episode, EpisodeStatus


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_create_podcast(db_session):
    """Test creating a podcast."""
    podcast = Podcast(
        source_rss_url="https://example.com/feed.xml",
        title="Test Podcast",
    )
    db_session.add(podcast)
    db_session.commit()

    assert podcast.id is not None
    assert podcast.feed_token is not None
    assert len(podcast.feed_token) >= 32
    assert podcast.auto_process is True


def test_create_episode(db_session):
    """Test creating an episode linked to a podcast."""
    podcast = Podcast(
        source_rss_url="https://example.com/feed.xml",
        title="Test Podcast",
    )
    db_session.add(podcast)
    db_session.commit()

    episode = Episode(
        podcast_id=podcast.id,
        guid="episode-123",
        title="Test Episode",
        original_audio_url="https://example.com/episode.mp3",
    )
    db_session.add(episode)
    db_session.commit()

    assert episode.id is not None
    assert episode.status == EpisodeStatus.PENDING
    assert episode.retry_count == 0


def test_episode_status_enum(db_session):
    """Test that episode status uses the enum correctly."""
    podcast = Podcast(
        source_rss_url="https://example.com/feed.xml",
        title="Test Podcast",
    )
    db_session.add(podcast)
    db_session.commit()

    episode = Episode(
        podcast_id=podcast.id,
        guid="episode-456",
        title="Another Episode",
        original_audio_url="https://example.com/ep2.mp3",
        status=EpisodeStatus.PROCESSING,
    )
    db_session.add(episode)
    db_session.commit()

    assert episode.status == EpisodeStatus.PROCESSING
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/web/test_models.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'web.database'"

**Step 4: Create database.py**

Create `web/database.py`:

```python
"""Database configuration and session management."""

import os
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_PATH = os.environ.get(
    "DATABASE_PATH",
    str(Path.home() / ".adnihilator" / "adnihilator.db")
)

Base = declarative_base()


def get_engine(database_url: str | None = None):
    """Create database engine with WAL mode for SQLite."""
    if database_url is None:
        # Ensure directory exists
        db_path = Path(DATABASE_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        database_url = f"sqlite:///{DATABASE_PATH}"

    engine = create_engine(database_url, echo=False)

    # Enable WAL mode for SQLite
    if database_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


def get_session_factory(engine=None):
    """Create a session factory."""
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine)


def init_db(engine=None):
    """Initialize the database, creating all tables."""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    return engine
```

**Step 5: Create models.py**

Create `web/models.py`:

```python
"""SQLAlchemy models for the web service."""

import secrets
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class EpisodeStatus(str, Enum):
    """Status of an episode in the processing pipeline."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"
    EXPIRED = "expired"


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid4())


def generate_feed_token() -> str:
    """Generate a secure random feed token (32+ chars)."""
    return secrets.token_urlsafe(32)


class Podcast(Base):
    """A podcast subscription."""

    __tablename__ = "podcasts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    feed_token: Mapped[str] = mapped_column(
        String(64), unique=True, default=generate_feed_token
    )
    source_rss_url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    auto_process: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    episodes: Mapped[list["Episode"]] = relationship(
        "Episode", back_populates="podcast", cascade="all, delete-orphan"
    )


class Episode(Base):
    """An episode from a podcast."""

    __tablename__ = "episodes"
    __table_args__ = (
        UniqueConstraint("podcast_id", "guid", name="uq_podcast_episode_guid"),
        CheckConstraint(
            "status IN ('pending', 'processing', 'complete', 'failed', 'skipped', 'expired')",
            name="ck_episode_status",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    podcast_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("podcasts.id"), nullable=False
    )
    guid: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    original_audio_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), default=EpisodeStatus.PENDING.value
    )
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    claimed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    processed_audio_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    processed_duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ads_removed_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    podcast: Mapped["Podcast"] = relationship("Podcast", back_populates="episodes")

    @property
    def status_enum(self) -> EpisodeStatus:
        """Get status as enum."""
        return EpisodeStatus(self.status)

    @status_enum.setter
    def status_enum(self, value: EpisodeStatus):
        """Set status from enum."""
        self.status = value.value
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/web/test_models.py -v`
Expected: All 3 tests PASS

**Step 7: Commit**

```bash
git add web/ tests/web/
git commit -m "feat: add database models for podcasts and episodes"
```

---

### Task 3: Create Authentication Helpers

**Files:**
- Create: `web/auth.py`
- Test: `tests/web/test_auth.py`

**Step 1: Write the failing test**

Create `tests/web/test_auth.py`:

```python
"""Tests for authentication."""

import os
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials

from web.auth import verify_admin, verify_worker_api_key


def test_verify_admin_valid(monkeypatch):
    """Test valid admin credentials."""
    monkeypatch.setenv("ADMIN_USERNAME", "testuser")
    monkeypatch.setenv("ADMIN_PASSWORD", "testpass")

    credentials = HTTPBasicCredentials(username="testuser", password="testpass")
    # Should not raise
    verify_admin(credentials)


def test_verify_admin_invalid_password(monkeypatch):
    """Test invalid admin password."""
    monkeypatch.setenv("ADMIN_USERNAME", "testuser")
    monkeypatch.setenv("ADMIN_PASSWORD", "testpass")

    credentials = HTTPBasicCredentials(username="testuser", password="wrongpass")
    with pytest.raises(HTTPException) as exc_info:
        verify_admin(credentials)
    assert exc_info.value.status_code == 401


def test_verify_admin_invalid_username(monkeypatch):
    """Test invalid admin username."""
    monkeypatch.setenv("ADMIN_USERNAME", "testuser")
    monkeypatch.setenv("ADMIN_PASSWORD", "testpass")

    credentials = HTTPBasicCredentials(username="wronguser", password="testpass")
    with pytest.raises(HTTPException) as exc_info:
        verify_admin(credentials)
    assert exc_info.value.status_code == 401


def test_verify_worker_api_key_valid(monkeypatch):
    """Test valid worker API key."""
    monkeypatch.setenv("WORKER_API_KEY", "secret-key-123")

    # Should not raise
    verify_worker_api_key("secret-key-123")


def test_verify_worker_api_key_invalid(monkeypatch):
    """Test invalid worker API key."""
    monkeypatch.setenv("WORKER_API_KEY", "secret-key-123")

    with pytest.raises(HTTPException) as exc_info:
        verify_worker_api_key("wrong-key")
    assert exc_info.value.status_code == 401
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/web/test_auth.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'web.auth'"

**Step 3: Create auth.py**

Create `web/auth.py`:

```python
"""Authentication helpers for the web service."""

import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()


def get_admin_username() -> str:
    """Get admin username from environment."""
    return os.environ.get("ADMIN_USERNAME", "admin")


def get_admin_password() -> str:
    """Get admin password from environment."""
    password = os.environ.get("ADMIN_PASSWORD")
    if not password:
        raise ValueError("ADMIN_PASSWORD environment variable not set")
    return password


def get_worker_api_key() -> str:
    """Get worker API key from environment."""
    key = os.environ.get("WORKER_API_KEY")
    if not key:
        raise ValueError("WORKER_API_KEY environment variable not set")
    return key


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)) -> None:
    """Verify admin credentials via HTTP Basic Auth.

    Raises HTTPException with 401 if credentials are invalid.
    """
    correct_username = secrets.compare_digest(
        credentials.username, get_admin_username()
    )
    correct_password = secrets.compare_digest(
        credentials.password, get_admin_password()
    )

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


def verify_worker_api_key(api_key: str) -> None:
    """Verify worker API key.

    Raises HTTPException with 401 if key is invalid.
    """
    if not secrets.compare_digest(api_key, get_worker_api_key()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/web/test_auth.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add web/auth.py tests/web/test_auth.py
git commit -m "feat: add authentication helpers"
```

---

### Task 4: Create FastAPI Application Shell

**Files:**
- Create: `web/app.py`
- Create: `web/routes/__init__.py`
- Test: `tests/web/test_app.py`

**Step 1: Write the failing test**

Create `tests/web/test_app.py`:

```python
"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """Create a test client with mocked auth."""
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "testpass")
    monkeypatch.setenv("WORKER_API_KEY", "test-worker-key")
    monkeypatch.setenv("DATABASE_PATH", ":memory:")

    from web.app import app
    return TestClient(app)


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_requires_auth(client):
    """Test that root requires authentication."""
    response = client.get("/")
    assert response.status_code == 401


def test_root_with_auth(client):
    """Test root with valid auth returns HTML."""
    response = client.get("/", auth=("admin", "testpass"))
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/web/test_app.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'web.app'"

**Step 3: Create routes package**

Create `web/routes/__init__.py`:

```python
"""Route handlers for the web service."""
```

**Step 4: Create app.py**

Create `web/app.py`:

```python
"""FastAPI application for AdNihilator web service."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .auth import security, verify_admin
from .database import get_engine, init_db, get_session_factory

# Templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Database session factory (initialized at startup)
SessionFactory = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global SessionFactory
    engine = init_db()
    SessionFactory = get_session_factory(engine)
    yield


app = FastAPI(
    title="AdNihilator",
    description="Podcast ad removal service",
    version="0.1.0",
    lifespan=lifespan,
)


def get_db():
    """Dependency to get a database session."""
    if SessionFactory is None:
        raise RuntimeError("Database not initialized")
    session = SessionFactory()
    try:
        yield session
    finally:
        session.close()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    _: None = Depends(verify_admin),
):
    """Dashboard home page."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "AdNihilator"},
    )
```

**Step 5: Create basic template**

Create `web/templates/base.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} - AdNihilator</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #333; }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        button, .btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
        }
        button:hover, .btn:hover { background: #0056b3; }
        input[type="text"], input[type="url"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 5px 0;
        }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        .status-pending { color: #ffc107; }
        .status-processing { color: #17a2b8; }
        .status-complete { color: #28a745; }
        .status-failed { color: #dc3545; }
    </style>
</head>
<body>
    <h1>🎙️ AdNihilator</h1>
    {% block content %}{% endblock %}
</body>
</html>
```

Create `web/templates/index.html`:

```html
{% extends "base.html" %}

{% block content %}
<div class="card">
    <h2>Add Podcast</h2>
    <form action="/podcasts" method="post">
        <input type="url" name="rss_url" placeholder="https://example.com/feed.xml" required>
        <button type="submit">Add Podcast</button>
    </form>
</div>

<div class="card">
    <h2>Your Podcasts</h2>
    <p>No podcasts yet. Add one above!</p>
</div>
{% endblock %}
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/web/test_app.py -v`
Expected: All 3 tests PASS

**Step 7: Commit**

```bash
git add web/app.py web/routes/ web/templates/
git commit -m "feat: add FastAPI application shell with templates"
```

---

## Phase 2: Podcast Management

### Task 5: Create Podcast CRUD Routes

**Files:**
- Create: `web/routes/podcasts.py`
- Modify: `web/app.py`
- Test: `tests/web/test_podcasts.py`

**Step 1: Write the failing test**

Create `tests/web/test_podcasts.py`:

```python
"""Tests for podcast management routes."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from web.database import Base
from web.models import Podcast


@pytest.fixture
def db_session():
    """Create test database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def client(monkeypatch, db_session):
    """Create test client with mocked dependencies."""
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "testpass")
    monkeypatch.setenv("WORKER_API_KEY", "test-key")

    from web.app import app, get_db

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_list_podcasts_empty(client):
    """Test listing podcasts when empty."""
    response = client.get("/podcasts", auth=("admin", "testpass"))
    assert response.status_code == 200
    assert response.json() == []


def test_add_podcast(client, db_session):
    """Test adding a podcast."""
    response = client.post(
        "/podcasts",
        data={"rss_url": "https://example.com/feed.xml"},
        auth=("admin", "testpass"),
        follow_redirects=False,
    )
    # Should redirect after successful add
    assert response.status_code in (302, 303)

    # Verify in database
    podcast = db_session.query(Podcast).first()
    assert podcast is not None
    assert podcast.source_rss_url == "https://example.com/feed.xml"


def test_delete_podcast(client, db_session):
    """Test deleting a podcast."""
    # Create a podcast first
    podcast = Podcast(
        source_rss_url="https://example.com/feed.xml",
        title="Test Podcast",
    )
    db_session.add(podcast)
    db_session.commit()
    podcast_id = podcast.id

    response = client.delete(
        f"/podcasts/{podcast_id}",
        auth=("admin", "testpass"),
    )
    assert response.status_code == 200

    # Verify deleted
    assert db_session.query(Podcast).filter_by(id=podcast_id).first() is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/web/test_podcasts.py -v`
Expected: FAIL with route not found errors

**Step 3: Create podcasts.py routes**

Create `web/routes/podcasts.py`:

```python
"""Podcast management routes."""

from fastapi import APIRouter, Depends, Form, HTTPException, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from ..app import get_db
from ..auth import verify_admin
from ..models import Podcast

router = APIRouter(prefix="/podcasts", tags=["podcasts"])


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
    rss_url: str = Form(...),
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
):
    """Add a new podcast subscription."""
    podcast = Podcast(source_rss_url=rss_url)
    db.add(podcast)
    db.commit()

    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


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
```

**Step 4: Register the router in app.py**

Add to `web/app.py` after the imports:

```python
from .routes import podcasts
```

Add before the route definitions:

```python
app.include_router(podcasts.router)
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/web/test_podcasts.py -v`
Expected: All 3 tests PASS

**Step 6: Commit**

```bash
git add web/routes/podcasts.py web/app.py tests/web/test_podcasts.py
git commit -m "feat: add podcast CRUD routes"
```

---

### Task 6: Create Worker API Routes

**Files:**
- Create: `web/routes/api.py`
- Modify: `web/app.py`
- Test: `tests/web/test_worker_api.py`

**Step 1: Write the failing test**

Create `tests/web/test_worker_api.py`:

```python
"""Tests for worker API routes."""

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from web.database import Base
from web.models import Episode, EpisodeStatus, Podcast


@pytest.fixture
def db_session():
    """Create test database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def client(monkeypatch, db_session):
    """Create test client with mocked dependencies."""
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "testpass")
    monkeypatch.setenv("WORKER_API_KEY", "test-worker-key")

    from web.app import app, get_db

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def podcast_with_episodes(db_session):
    """Create a podcast with some episodes."""
    podcast = Podcast(
        source_rss_url="https://example.com/feed.xml",
        title="Test Podcast",
    )
    db_session.add(podcast)
    db_session.commit()

    episodes = [
        Episode(
            podcast_id=podcast.id,
            guid=f"ep-{i}",
            title=f"Episode {i}",
            original_audio_url=f"https://example.com/ep{i}.mp3",
            status=EpisodeStatus.PENDING.value,
        )
        for i in range(3)
    ]
    db_session.add_all(episodes)
    db_session.commit()

    return podcast, episodes


def test_claim_requires_auth(client):
    """Test that claim endpoint requires API key."""
    response = client.post("/api/queue/claim")
    assert response.status_code == 401


def test_claim_no_pending_jobs(client):
    """Test claiming when no jobs are pending."""
    response = client.post(
        "/api/queue/claim",
        headers={"X-Worker-API-Key": "test-worker-key"},
    )
    assert response.status_code == 200
    assert response.json() is None


def test_claim_returns_episode(client, db_session, podcast_with_episodes):
    """Test claiming returns a pending episode."""
    response = client.post(
        "/api/queue/claim",
        headers={"X-Worker-API-Key": "test-worker-key"},
    )
    assert response.status_code == 200

    data = response.json()
    assert data is not None
    assert "id" in data
    assert "original_audio_url" in data
    assert data["status"] == "processing"


def test_claim_is_atomic(client, db_session, podcast_with_episodes):
    """Test that claiming marks episode as processing."""
    podcast, episodes = podcast_with_episodes

    response = client.post(
        "/api/queue/claim",
        headers={"X-Worker-API-Key": "test-worker-key"},
    )
    claimed_id = response.json()["id"]

    # Verify status changed in DB
    db_session.refresh(episodes[0])
    episode = db_session.query(Episode).filter_by(id=claimed_id).first()
    assert episode.status == EpisodeStatus.PROCESSING.value
    assert episode.claimed_at is not None


def test_complete_episode(client, db_session, podcast_with_episodes):
    """Test marking an episode as complete."""
    podcast, episodes = podcast_with_episodes
    episode = episodes[0]
    episode.status = EpisodeStatus.PROCESSING.value
    db_session.commit()

    response = client.post(
        f"/api/queue/{episode.id}/complete",
        headers={"X-Worker-API-Key": "test-worker-key"},
        json={
            "processed_audio_key": f"{podcast.id}/{episode.id}.mp3",
            "processed_duration": 3600.0,
            "ads_removed_seconds": 120.5,
        },
    )
    assert response.status_code == 200

    db_session.refresh(episode)
    assert episode.status == EpisodeStatus.COMPLETE.value
    assert episode.processed_audio_key == f"{podcast.id}/{episode.id}.mp3"


def test_fail_episode(client, db_session, podcast_with_episodes):
    """Test marking an episode as failed."""
    podcast, episodes = podcast_with_episodes
    episode = episodes[0]
    episode.status = EpisodeStatus.PROCESSING.value
    db_session.commit()

    response = client.post(
        f"/api/queue/{episode.id}/fail",
        headers={"X-Worker-API-Key": "test-worker-key"},
        json={"error": "Transcription failed"},
    )
    assert response.status_code == 200

    db_session.refresh(episode)
    # First failure should retry (back to pending)
    assert episode.status == EpisodeStatus.PENDING.value
    assert episode.retry_count == 1


def test_fail_episode_max_retries(client, db_session, podcast_with_episodes):
    """Test that max retries marks episode as permanently failed."""
    podcast, episodes = podcast_with_episodes
    episode = episodes[0]
    episode.status = EpisodeStatus.PROCESSING.value
    episode.retry_count = 2  # Already at max
    db_session.commit()

    response = client.post(
        f"/api/queue/{episode.id}/fail",
        headers={"X-Worker-API-Key": "test-worker-key"},
        json={"error": "Final failure"},
    )
    assert response.status_code == 200

    db_session.refresh(episode)
    assert episode.status == EpisodeStatus.FAILED.value
    assert episode.error_message == "Final failure"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/web/test_worker_api.py -v`
Expected: FAIL with route not found errors

**Step 3: Create api.py routes**

Create `web/routes/api.py`:

```python
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


class FailRequest(BaseModel):
    """Request body for failing an episode."""

    error: str


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
        "guid": episode.guid,
        "title": episode.title,
        "original_audio_url": episode.original_audio_url,
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
    episode.claimed_at = None
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
```

**Step 4: Register the router in app.py**

Add to `web/app.py` imports:

```python
from .routes import podcasts, api
```

Add to router registration:

```python
app.include_router(api.router)
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/web/test_worker_api.py -v`
Expected: All 7 tests PASS

**Step 6: Commit**

```bash
git add web/routes/api.py web/app.py tests/web/test_worker_api.py
git commit -m "feat: add worker API routes with atomic claiming"
```

---

### Task 7: Create RSS Feed Routes

**Files:**
- Create: `web/routes/feeds.py`
- Create: `web/services/__init__.py`
- Create: `web/services/r2.py`
- Modify: `web/app.py`
- Test: `tests/web/test_feeds.py`

**Step 1: Write the failing test**

Create `tests/web/test_feeds.py`:

```python
"""Tests for RSS feed routes."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from web.database import Base
from web.models import Episode, EpisodeStatus, Podcast


@pytest.fixture
def db_session():
    """Create test database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def client(monkeypatch, db_session):
    """Create test client."""
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "testpass")
    monkeypatch.setenv("WORKER_API_KEY", "test-key")
    monkeypatch.setenv("R2_PUBLIC_URL", "https://test-bucket.r2.dev")

    from web.app import app, get_db

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def podcast_with_complete_episodes(db_session):
    """Create podcast with completed episodes."""
    podcast = Podcast(
        source_rss_url="https://example.com/feed.xml",
        title="Test Podcast",
    )
    db_session.add(podcast)
    db_session.commit()

    episode = Episode(
        podcast_id=podcast.id,
        guid="ep-1",
        title="Episode 1",
        original_audio_url="https://example.com/ep1.mp3",
        status=EpisodeStatus.COMPLETE.value,
        processed_audio_key=f"{podcast.id}/ep1.mp3",
        processed_duration=3600.0,
    )
    db_session.add(episode)
    db_session.commit()

    return podcast


def test_feed_not_found(client):
    """Test requesting non-existent feed."""
    response = client.get("/feed/nonexistent-token.xml")
    assert response.status_code == 404


def test_feed_returns_xml(client, podcast_with_complete_episodes):
    """Test that feed returns valid XML."""
    podcast = podcast_with_complete_episodes
    response = client.get(f"/feed/{podcast.feed_token}.xml")

    assert response.status_code == 200
    assert "application/xml" in response.headers["content-type"]
    assert "<rss" in response.text
    assert podcast.title in response.text


def test_feed_contains_complete_episodes(client, podcast_with_complete_episodes, db_session):
    """Test that feed includes only complete episodes."""
    podcast = podcast_with_complete_episodes

    # Add a pending episode (should not appear)
    pending = Episode(
        podcast_id=podcast.id,
        guid="ep-pending",
        title="Pending Episode",
        status=EpisodeStatus.PENDING.value,
    )
    db_session.add(pending)
    db_session.commit()

    response = client.get(f"/feed/{podcast.feed_token}.xml")
    assert "Episode 1" in response.text
    assert "Pending Episode" not in response.text


def test_feed_audio_urls_point_to_r2(client, podcast_with_complete_episodes):
    """Test that audio URLs point to R2."""
    podcast = podcast_with_complete_episodes
    response = client.get(f"/feed/{podcast.feed_token}.xml")

    assert "https://test-bucket.r2.dev" in response.text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/web/test_feeds.py -v`
Expected: FAIL with route not found errors

**Step 3: Create services package and r2.py**

Create `web/services/__init__.py`:

```python
"""Services for the web application."""
```

Create `web/services/r2.py`:

```python
"""Cloudflare R2 integration."""

import os


def get_r2_public_url() -> str:
    """Get the public URL for R2 bucket."""
    url = os.environ.get("R2_PUBLIC_URL", "")
    return url.rstrip("/")


def get_audio_url(audio_key: str) -> str:
    """Get the full public URL for an audio file."""
    base_url = get_r2_public_url()
    return f"{base_url}/{audio_key}"
```

**Step 4: Create feeds.py routes**

Create `web/routes/feeds.py`:

```python
"""RSS feed routes."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from feedgen.feed import FeedGenerator
from sqlalchemy.orm import Session

from ..app import get_db
from ..models import Episode, EpisodeStatus, Podcast
from ..services.r2 import get_audio_url

router = APIRouter(tags=["feeds"])


@router.get("/feed/{feed_token}.xml")
async def get_feed(
    feed_token: str,
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
    fg.title(podcast.title or "Untitled Podcast")
    fg.link(href=podcast.source_rss_url, rel="alternate")
    fg.description(f"Ad-free version of {podcast.title or 'podcast'}")
    fg.load_extension("podcast")

    for episode in episodes:
        fe = fg.add_entry()
        fe.id(episode.guid)
        fe.title(episode.title or "Untitled Episode")

        if episode.published_at:
            fe.pubDate(episode.published_at)

        # Audio enclosure pointing to R2
        if episode.processed_audio_key:
            audio_url = get_audio_url(episode.processed_audio_key)
            fe.enclosure(audio_url, 0, "audio/mpeg")

    rss_xml = fg.rss_str(pretty=True)
    return Response(content=rss_xml, media_type="application/xml")
```

**Step 5: Register the router in app.py**

Add to `web/app.py` imports:

```python
from .routes import podcasts, api, feeds
```

Add to router registration:

```python
app.include_router(feeds.router)
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/web/test_feeds.py -v`
Expected: All 4 tests PASS

**Step 7: Commit**

```bash
git add web/routes/feeds.py web/services/ web/app.py tests/web/test_feeds.py
git commit -m "feat: add RSS feed generation"
```

---

## Phase 3: Local Worker

### Task 8: Create Worker Client

**Files:**
- Create: `worker/__init__.py`
- Create: `worker/client.py`
- Test: `tests/worker/__init__.py`
- Test: `tests/worker/test_client.py`

**Step 1: Create worker package**

Create `worker/__init__.py`:

```python
"""AdNihilator local worker."""
```

Create `tests/worker/__init__.py`:

```python
"""Worker tests."""
```

**Step 2: Write the failing test**

Create `tests/worker/test_client.py`:

```python
"""Tests for the worker API client."""

import pytest
from unittest.mock import MagicMock, patch

from worker.client import WorkerClient, EpisodeJob


@pytest.fixture
def mock_httpx():
    """Mock httpx client."""
    with patch("worker.client.httpx") as mock:
        yield mock


def test_claim_returns_job(mock_httpx):
    """Test claiming returns an EpisodeJob."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "ep-123",
        "podcast_id": "pod-456",
        "guid": "guid-789",
        "title": "Test Episode",
        "original_audio_url": "https://example.com/ep.mp3",
        "status": "processing",
    }
    mock_httpx.post.return_value = mock_response

    client = WorkerClient("https://api.example.com", "test-key")
    job = client.claim()

    assert job is not None
    assert job.id == "ep-123"
    assert job.original_audio_url == "https://example.com/ep.mp3"


def test_claim_returns_none_when_empty(mock_httpx):
    """Test claiming returns None when queue is empty."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = None
    mock_httpx.post.return_value = mock_response

    client = WorkerClient("https://api.example.com", "test-key")
    job = client.claim()

    assert job is None


def test_complete_sends_metadata(mock_httpx):
    """Test completing sends the right metadata."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "complete"}
    mock_httpx.post.return_value = mock_response

    client = WorkerClient("https://api.example.com", "test-key")
    client.complete(
        episode_id="ep-123",
        audio_key="pod/ep.mp3",
        duration=3600.0,
        ads_removed=120.5,
    )

    mock_httpx.post.assert_called_once()
    call_args = mock_httpx.post.call_args
    assert "ep-123" in call_args[0][0]
    assert call_args[1]["json"]["processed_audio_key"] == "pod/ep.mp3"


def test_fail_sends_error(mock_httpx):
    """Test failing sends the error message."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "retrying"}
    mock_httpx.post.return_value = mock_response

    client = WorkerClient("https://api.example.com", "test-key")
    client.fail("ep-123", "Something went wrong")

    mock_httpx.post.assert_called_once()
    call_args = mock_httpx.post.call_args
    assert call_args[1]["json"]["error"] == "Something went wrong"
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/worker/test_client.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'worker.client'"

**Step 4: Create client.py**

Create `worker/client.py`:

```python
"""API client for communicating with the web service."""

from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class EpisodeJob:
    """An episode job to process."""

    id: str
    podcast_id: str
    guid: str
    title: Optional[str]
    original_audio_url: str


class WorkerClient:
    """Client for the worker API."""

    def __init__(self, api_url: str, api_key: str, timeout: float = 30.0):
        """Initialize the client.

        Args:
            api_url: Base URL of the API (e.g., https://feeds.example.com)
            api_key: Worker API key
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        """Get headers for API requests."""
        return {"X-Worker-API-Key": self.api_key}

    def claim(self) -> Optional[EpisodeJob]:
        """Claim the next pending episode.

        Returns:
            EpisodeJob if one was claimed, None if queue is empty.
        """
        response = httpx.post(
            f"{self.api_url}/api/queue/claim",
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        if data is None:
            return None

        return EpisodeJob(
            id=data["id"],
            podcast_id=data["podcast_id"],
            guid=data["guid"],
            title=data.get("title"),
            original_audio_url=data["original_audio_url"],
        )

    def complete(
        self,
        episode_id: str,
        audio_key: str,
        duration: float,
        ads_removed: float,
    ) -> None:
        """Mark an episode as complete.

        Args:
            episode_id: ID of the episode
            audio_key: R2 key where audio was uploaded
            duration: Duration of processed audio in seconds
            ads_removed: Seconds of ads removed
        """
        response = httpx.post(
            f"{self.api_url}/api/queue/{episode_id}/complete",
            headers=self._headers(),
            json={
                "processed_audio_key": audio_key,
                "processed_duration": duration,
                "ads_removed_seconds": ads_removed,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

    def fail(self, episode_id: str, error: str) -> None:
        """Mark an episode as failed.

        Args:
            episode_id: ID of the episode
            error: Error message describing the failure
        """
        response = httpx.post(
            f"{self.api_url}/api/queue/{episode_id}/fail",
            headers=self._headers(),
            json={"error": error},
            timeout=self.timeout,
        )
        response.raise_for_status()
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/worker/test_client.py -v`
Expected: All 4 tests PASS

**Step 6: Commit**

```bash
git add worker/ tests/worker/
git commit -m "feat: add worker API client"
```

---

### Task 9: Create R2 Upload Service

**Files:**
- Create: `worker/r2.py`
- Test: `tests/worker/test_r2.py`

**Step 1: Write the failing test**

Create `tests/worker/test_r2.py`:

```python
"""Tests for R2 upload service."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


@pytest.fixture
def mock_boto3():
    """Mock boto3 client."""
    with patch("worker.r2.boto3") as mock:
        yield mock


def test_upload_file(mock_boto3, tmp_path):
    """Test uploading a file to R2."""
    from worker.r2 import R2Client

    # Create a test file
    test_file = tmp_path / "test.mp3"
    test_file.write_bytes(b"fake audio content")

    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    r2 = R2Client(
        access_key="test-key",
        secret_key="test-secret",
        bucket="test-bucket",
        endpoint="https://test.r2.cloudflarestorage.com",
    )
    r2.upload_file(str(test_file), "podcast/episode.mp3")

    mock_client.upload_file.assert_called_once()


def test_get_file_size(mock_boto3):
    """Test getting file size from R2."""
    from worker.r2 import R2Client

    mock_client = MagicMock()
    mock_client.head_object.return_value = {"ContentLength": 12345}
    mock_boto3.client.return_value = mock_client

    r2 = R2Client(
        access_key="test-key",
        secret_key="test-secret",
        bucket="test-bucket",
        endpoint="https://test.r2.cloudflarestorage.com",
    )
    size = r2.get_file_size("podcast/episode.mp3")

    assert size == 12345


def test_delete_file(mock_boto3):
    """Test deleting a file from R2."""
    from worker.r2 import R2Client

    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    r2 = R2Client(
        access_key="test-key",
        secret_key="test-secret",
        bucket="test-bucket",
        endpoint="https://test.r2.cloudflarestorage.com",
    )
    r2.delete_file("podcast/episode.mp3")

    mock_client.delete_object.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/worker/test_r2.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'worker.r2'"

**Step 3: Create r2.py**

Create `worker/r2.py`:

```python
"""Cloudflare R2 upload service."""

import boto3
from botocore.exceptions import ClientError


class R2Client:
    """Client for uploading files to Cloudflare R2."""

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        bucket: str,
        endpoint: str,
    ):
        """Initialize the R2 client.

        Args:
            access_key: R2 access key ID
            secret_key: R2 secret access key
            bucket: R2 bucket name
            endpoint: R2 endpoint URL (e.g., https://<account>.r2.cloudflarestorage.com)
        """
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def upload_file(self, local_path: str, key: str) -> None:
        """Upload a file to R2.

        Args:
            local_path: Path to the local file
            key: Object key in R2
        """
        self.client.upload_file(
            local_path,
            self.bucket,
            key,
            ExtraArgs={"ContentType": "audio/mpeg"},
        )

    def get_file_size(self, key: str) -> int:
        """Get the size of a file in R2.

        Args:
            key: Object key in R2

        Returns:
            File size in bytes
        """
        response = self.client.head_object(Bucket=self.bucket, Key=key)
        return response["ContentLength"]

    def delete_file(self, key: str) -> None:
        """Delete a file from R2.

        Args:
            key: Object key to delete
        """
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def file_exists(self, key: str) -> bool:
        """Check if a file exists in R2.

        Args:
            key: Object key to check

        Returns:
            True if file exists, False otherwise
        """
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/worker/test_r2.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add worker/r2.py tests/worker/test_r2.py
git commit -m "feat: add R2 upload client"
```

---

### Task 10: Create Worker Daemon

**Files:**
- Create: `worker/daemon.py`
- Modify: `adnihilator/cli.py` (add worker command)

**Step 1: Create daemon.py**

Create `worker/daemon.py`:

```python
"""Worker daemon for processing podcast episodes."""

import os
import tempfile
import time
from pathlib import Path

import httpx

from adnihilator.ad_keywords import find_ad_candidates
from adnihilator.ad_llm import create_llm_client
from adnihilator.audio import get_duration
from adnihilator.config import load_config
from adnihilator.splice import splice_audio
from adnihilator.transcribe import transcribe_audio

from .client import WorkerClient, EpisodeJob
from .r2 import R2Client


class WorkerDaemon:
    """Daemon that processes podcast episodes."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        r2_access_key: str,
        r2_secret_key: str,
        r2_bucket: str,
        r2_endpoint: str,
        whisper_model: str = "small",
        device: str = "cpu",
    ):
        """Initialize the worker daemon."""
        self.api_client = WorkerClient(api_url, api_key)
        self.r2_client = R2Client(
            r2_access_key, r2_secret_key, r2_bucket, r2_endpoint
        )
        self.whisper_model = whisper_model
        self.device = device
        self.config = load_config()

    def process_job(self, job: EpisodeJob) -> None:
        """Process a single episode job.

        Args:
            job: The episode job to process
        """
        print(f"Processing: {job.title or job.guid}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Download audio
            print("  Downloading audio...")
            audio_path = tmpdir_path / "episode.mp3"
            self._download_audio(job.original_audio_url, audio_path)

            # Get duration
            duration = get_duration(str(audio_path))
            print(f"  Duration: {duration:.0f}s")

            # Transcribe
            print("  Transcribing...")
            segments = transcribe_audio(
                str(audio_path),
                model_name=self.whisper_model,
                device=self.device,
            )

            # Detect ads
            print("  Detecting ads...")
            candidates = find_ad_candidates(segments, duration)

            # Refine with LLM
            print("  Refining with LLM...")
            llm_client = create_llm_client(self.config)
            ad_spans = llm_client.refine_candidates(segments, candidates, self.config)

            # Splice
            print("  Splicing audio...")
            output_path = tmpdir_path / "processed.mp3"
            stats = splice_audio(
                str(audio_path),
                str(output_path),
                ad_spans,
                duration,
            )

            # Upload to R2
            print("  Uploading to R2...")
            audio_key = f"{job.podcast_id}/{job.id}.mp3"
            self.r2_client.upload_file(str(output_path), audio_key)

            # Verify upload
            uploaded_size = self.r2_client.get_file_size(audio_key)
            local_size = output_path.stat().st_size
            if uploaded_size != local_size:
                raise RuntimeError(
                    f"Upload verification failed: expected {local_size}, got {uploaded_size}"
                )

            # Report completion
            self.api_client.complete(
                episode_id=job.id,
                audio_key=audio_key,
                duration=stats["new_duration"],
                ads_removed=stats["time_removed"],
            )

            print(f"  Done! Removed {stats['time_removed']:.0f}s of ads")

    def _download_audio(self, url: str, dest: Path) -> None:
        """Download audio file from URL."""
        with httpx.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)

    def run_once(self) -> bool:
        """Process one job from the queue.

        Returns:
            True if a job was processed, False if queue was empty.
        """
        job = self.api_client.claim()
        if job is None:
            return False

        try:
            self.process_job(job)
            return True
        except Exception as e:
            print(f"  Error: {e}")
            self.api_client.fail(job.id, str(e)[:500])
            return True

    def run_daemon(self, interval: int = 300) -> None:
        """Run the daemon loop.

        Args:
            interval: Seconds to wait between queue checks
        """
        print(f"Starting worker daemon (interval: {interval}s)")

        while True:
            try:
                job_processed = self.run_once()
                if not job_processed:
                    print(f"Queue empty, sleeping for {interval}s...")
                    time.sleep(interval)
            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                time.sleep(60)  # Back off on errors
```

**Step 2: Add worker command to CLI**

Add to `adnihilator/cli.py` after the existing imports:

```python
from worker.daemon import WorkerDaemon
```

Add the worker command before `version`:

```python
@app.command()
def worker(
    once: Annotated[
        bool,
        typer.Option("--once", help="Process one job and exit"),
    ] = False,
    daemon: Annotated[
        bool,
        typer.Option("--daemon", help="Run as daemon"),
    ] = False,
    interval: Annotated[
        int,
        typer.Option("--interval", "-i", help="Seconds between queue checks"),
    ] = 300,
) -> None:
    """Run the local worker to process podcast episodes."""
    # Get configuration from environment
    api_url = os.environ.get("API_URL")
    if not api_url:
        typer.echo("Error: API_URL environment variable not set", err=True)
        raise typer.Exit(1)

    api_key = os.environ.get("WORKER_API_KEY")
    if not api_key:
        typer.echo("Error: WORKER_API_KEY environment variable not set", err=True)
        raise typer.Exit(1)

    r2_access_key = os.environ.get("R2_ACCESS_KEY")
    r2_secret_key = os.environ.get("R2_SECRET_KEY")
    r2_bucket = os.environ.get("R2_BUCKET")
    r2_endpoint = os.environ.get("R2_ENDPOINT")

    if not all([r2_access_key, r2_secret_key, r2_bucket, r2_endpoint]):
        typer.echo("Error: R2 environment variables not set", err=True)
        typer.echo("Required: R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET, R2_ENDPOINT", err=True)
        raise typer.Exit(1)

    worker_daemon = WorkerDaemon(
        api_url=api_url,
        api_key=api_key,
        r2_access_key=r2_access_key,
        r2_secret_key=r2_secret_key,
        r2_bucket=r2_bucket,
        r2_endpoint=r2_endpoint,
    )

    if once:
        processed = worker_daemon.run_once()
        if processed:
            typer.echo("Processed one job")
        else:
            typer.echo("No jobs in queue")
    elif daemon:
        worker_daemon.run_daemon(interval=interval)
    else:
        typer.echo("Specify --once or --daemon mode")
        raise typer.Exit(1)
```

**Step 3: Commit**

```bash
git add worker/daemon.py adnihilator/cli.py
git commit -m "feat: add worker daemon command"
```

---

## Phase 4: Deployment Files

### Task 11: Create Deployment Configuration

**Files:**
- Create: `deploy/adnihilator.service`
- Create: `deploy/nginx-site.conf`
- Create: `deploy/README.md`

**Step 1: Create systemd service file**

Create `deploy/adnihilator.service`:

```ini
[Unit]
Description=AdNihilator Web Service
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/adnihilator
EnvironmentFile=/etc/adnihilator/env
ExecStart=/opt/adnihilator/venv/bin/uvicorn web.app:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/adnihilator

[Install]
WantedBy=multi-user.target
```

**Step 2: Create nginx config**

Create `deploy/nginx-site.conf`:

```nginx
# AdNihilator Web Service
# Save to: /etc/nginx/sites-available/adnihilator

server {
    listen 80;
    server_name feeds.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name feeds.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/feeds.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/feeds.yourdomain.com/privkey.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for long-running requests
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

**Step 3: Create deployment README**

Create `deploy/README.md`:

```markdown
# Deployment Guide

## VPS Setup

### 1. Clone the repository

```bash
sudo mkdir -p /opt/adnihilator
sudo chown $USER:$USER /opt/adnihilator
git clone https://github.com/your-repo/adnihilator.git /opt/adnihilator
```

### 2. Create virtual environment

```bash
cd /opt/adnihilator
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### 3. Create data directory

```bash
sudo mkdir -p /var/lib/adnihilator
sudo chown www-data:www-data /var/lib/adnihilator
```

### 4. Configure environment

```bash
sudo mkdir -p /etc/adnihilator
sudo nano /etc/adnihilator/env
```

Add:
```
ADMIN_USERNAME=admin
ADMIN_PASSWORD=<generate-strong-password>
WORKER_API_KEY=<generate-random-key>
DATABASE_PATH=/var/lib/adnihilator/adnihilator.db
R2_PUBLIC_URL=https://your-bucket.r2.dev
```

Secure the file:
```bash
sudo chmod 600 /etc/adnihilator/env
sudo chown root:www-data /etc/adnihilator/env
```

### 5. Install systemd service

```bash
sudo cp deploy/adnihilator.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable adnihilator
sudo systemctl start adnihilator
```

### 6. Configure nginx

```bash
sudo cp deploy/nginx-site.conf /etc/nginx/sites-available/adnihilator
sudo ln -s /etc/nginx/sites-available/adnihilator /etc/nginx/sites-enabled/
# Edit the file to replace yourdomain.com with your actual domain
sudo nano /etc/nginx/sites-available/adnihilator
```

### 7. Get SSL certificate

```bash
sudo certbot --nginx -d feeds.yourdomain.com
```

### 8. Restart nginx

```bash
sudo nginx -t
sudo systemctl restart nginx
```

## Local Worker Setup

### 1. Set environment variables

Add to `~/.zshrc` or `~/.bashrc`:

```bash
export API_URL=https://feeds.yourdomain.com
export WORKER_API_KEY=<same-key-as-server>
export R2_ACCESS_KEY=<cloudflare-r2-key>
export R2_SECRET_KEY=<cloudflare-r2-secret>
export R2_BUCKET=adnihilator-audio
export R2_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
export OPENAI_API_KEY=<your-openai-key>
```

### 2. Run the worker

```bash
# One-shot mode (process one job)
adnihilator worker --once

# Daemon mode (continuous processing)
adnihilator worker --daemon --interval 300
```

### 3. (Optional) Configure launchd for auto-start

Create `~/Library/LaunchAgents/com.adnihilator.worker.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.adnihilator.worker</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/adnihilator</string>
        <string>worker</string>
        <string>--daemon</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>API_URL</key>
        <string>https://feeds.yourdomain.com</string>
        <!-- Add other env vars -->
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

Load it:
```bash
launchctl load ~/Library/LaunchAgents/com.adnihilator.worker.plist
```
```

**Step 4: Commit**

```bash
git add deploy/
git commit -m "feat: add deployment configuration"
```

---

## Phase 5: Integration & Polish

### Task 12: Update Index Template with Podcast List

**Files:**
- Modify: `web/templates/index.html`
- Modify: `web/app.py`

**Step 1: Update index route to pass podcasts**

Modify `web/app.py` to query podcasts:

```python
@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
):
    """Dashboard home page."""
    from .models import Podcast

    podcasts = db.query(Podcast).order_by(Podcast.created_at.desc()).all()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "Dashboard",
            "podcasts": podcasts,
        },
    )
```

**Step 2: Update index.html template**

Update `web/templates/index.html`:

```html
{% extends "base.html" %}

{% block content %}
<div class="card">
    <h2>Add Podcast</h2>
    <form action="/podcasts" method="post">
        <input type="url" name="rss_url" placeholder="https://example.com/feed.xml" required>
        <button type="submit">Add Podcast</button>
    </form>
</div>

<div class="card">
    <h2>Your Podcasts</h2>
    {% if podcasts %}
    <table>
        <thead>
            <tr>
                <th>Title</th>
                <th>Feed URL</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody>
            {% for podcast in podcasts %}
            <tr>
                <td>
                    <a href="/podcasts/{{ podcast.id }}/episodes">
                        {{ podcast.title or "Untitled" }}
                    </a>
                </td>
                <td>
                    <code>/feed/{{ podcast.feed_token }}.xml</code>
                </td>
                <td>
                    <button onclick="deletePodcast('{{ podcast.id }}')" class="btn-danger">Delete</button>
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% else %}
    <p>No podcasts yet. Add one above!</p>
    {% endif %}
</div>

<script>
async function deletePodcast(id) {
    if (!confirm('Delete this podcast and all its episodes?')) return;

    const response = await fetch(`/podcasts/${id}`, { method: 'DELETE' });
    if (response.ok) {
        window.location.reload();
    } else {
        alert('Failed to delete podcast');
    }
}
</script>

<style>
.btn-danger {
    background: #dc3545;
}
.btn-danger:hover {
    background: #c82333;
}
</style>
{% endblock %}
```

**Step 3: Commit**

```bash
git add web/templates/index.html web/app.py
git commit -m "feat: add podcast list to dashboard"
```

---

### Task 13: Add Episode Management Page

**Files:**
- Create: `web/templates/episodes.html`
- Modify: `web/routes/podcasts.py`

**Step 1: Add episodes route**

Add to `web/routes/podcasts.py`:

```python
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


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
        "episodes.html",
        {
            "request": request,
            "title": podcast.title or "Episodes",
            "podcast": podcast,
            "episodes": episodes,
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

    episode.status = EpisodeStatus.PENDING.value
    episode.retry_count = 0
    episode.error_message = None
    db.commit()

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

    episode.status = EpisodeStatus.SKIPPED.value
    db.commit()

    return {"status": "skipped"}
```

**Step 2: Create episodes.html template**

Create `web/templates/episodes.html`:

```html
{% extends "base.html" %}

{% block content %}
<p><a href="/">&larr; Back to Dashboard</a></p>

<div class="card">
    <h2>{{ podcast.title or "Untitled Podcast" }}</h2>
    <p>
        <strong>Source:</strong> {{ podcast.source_rss_url }}<br>
        <strong>Feed URL:</strong> <code>/feed/{{ podcast.feed_token }}.xml</code>
    </p>
</div>

<div class="card">
    <h2>Episodes</h2>
    {% if episodes %}
    <table>
        <thead>
            <tr>
                <th>Title</th>
                <th>Status</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody>
            {% for episode in episodes %}
            <tr>
                <td>{{ episode.title or episode.guid }}</td>
                <td class="status-{{ episode.status }}">
                    {{ episode.status }}
                    {% if episode.error_message %}
                    <br><small>{{ episode.error_message[:50] }}...</small>
                    {% endif %}
                </td>
                <td>
                    {% if episode.status in ['pending', 'processing'] %}
                    <!-- No action needed -->
                    {% elif episode.status == 'complete' %}
                    <span>✓ Done</span>
                    {% elif episode.status in ['failed', 'skipped'] %}
                    <button onclick="queueEpisode('{{ podcast.id }}', '{{ episode.id }}')">Retry</button>
                    {% endif %}

                    {% if episode.status != 'skipped' %}
                    <button onclick="skipEpisode('{{ podcast.id }}', '{{ episode.id }}')" class="btn-secondary">Skip</button>
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% else %}
    <p>No episodes found. The feed will be synced automatically.</p>
    {% endif %}
</div>

<script>
async function queueEpisode(podcastId, episodeId) {
    const response = await fetch(`/podcasts/${podcastId}/episodes/${episodeId}/queue`, { method: 'POST' });
    if (response.ok) {
        window.location.reload();
    } else {
        alert('Failed to queue episode');
    }
}

async function skipEpisode(podcastId, episodeId) {
    const response = await fetch(`/podcasts/${podcastId}/episodes/${episodeId}/skip`, { method: 'POST' });
    if (response.ok) {
        window.location.reload();
    } else {
        alert('Failed to skip episode');
    }
}
</script>

<style>
.btn-secondary {
    background: #6c757d;
}
.btn-secondary:hover {
    background: #5a6268;
}
.status-expired { color: #6c757d; }
</style>
{% endblock %}
```

**Step 3: Commit**

```bash
git add web/templates/episodes.html web/routes/podcasts.py
git commit -m "feat: add episode management page"
```

---

### Task 14: Add Feed Sync Service

**Files:**
- Create: `web/services/feed_sync.py`
- Test: `tests/web/test_feed_sync.py`

**Step 1: Write the failing test**

Create `tests/web/test_feed_sync.py`:

```python
"""Tests for feed sync service."""

import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from web.database import Base
from web.models import Episode, Podcast


@pytest.fixture
def db_session():
    """Create test database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def mock_feedparser():
    """Mock feedparser."""
    with patch("web.services.feed_sync.feedparser") as mock:
        yield mock


def test_sync_podcast_adds_new_episodes(db_session, mock_feedparser):
    """Test syncing adds new episodes."""
    from web.services.feed_sync import sync_podcast

    # Create podcast
    podcast = Podcast(
        source_rss_url="https://example.com/feed.xml",
        title="Test Podcast",
    )
    db_session.add(podcast)
    db_session.commit()

    # Mock feed response
    mock_feedparser.parse.return_value = MagicMock(
        feed=MagicMock(title="Test Podcast"),
        entries=[
            MagicMock(
                id="ep-1",
                title="Episode 1",
                enclosures=[MagicMock(href="https://example.com/ep1.mp3")],
                published_parsed=(2024, 1, 1, 0, 0, 0, 0, 0, 0),
            ),
            MagicMock(
                id="ep-2",
                title="Episode 2",
                enclosures=[MagicMock(href="https://example.com/ep2.mp3")],
                published_parsed=(2024, 1, 2, 0, 0, 0, 0, 0, 0),
            ),
        ],
    )

    # Sync
    added = sync_podcast(db_session, podcast)

    assert added == 2
    episodes = db_session.query(Episode).filter_by(podcast_id=podcast.id).all()
    assert len(episodes) == 2


def test_sync_podcast_skips_existing(db_session, mock_feedparser):
    """Test syncing skips already-known episodes."""
    from web.services.feed_sync import sync_podcast

    # Create podcast with existing episode
    podcast = Podcast(
        source_rss_url="https://example.com/feed.xml",
        title="Test Podcast",
    )
    db_session.add(podcast)
    db_session.commit()

    episode = Episode(
        podcast_id=podcast.id,
        guid="ep-1",
        title="Episode 1",
    )
    db_session.add(episode)
    db_session.commit()

    # Mock feed response with same episode
    mock_feedparser.parse.return_value = MagicMock(
        feed=MagicMock(title="Test Podcast"),
        entries=[
            MagicMock(
                id="ep-1",
                title="Episode 1",
                enclosures=[MagicMock(href="https://example.com/ep1.mp3")],
                published_parsed=(2024, 1, 1, 0, 0, 0, 0, 0, 0),
            ),
        ],
    )

    # Sync
    added = sync_podcast(db_session, podcast)

    assert added == 0
    episodes = db_session.query(Episode).filter_by(podcast_id=podcast.id).all()
    assert len(episodes) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/web/test_feed_sync.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create feed_sync.py**

Create `web/services/feed_sync.py`:

```python
"""Feed synchronization service."""

from datetime import datetime
from time import mktime

import feedparser
from sqlalchemy.orm import Session

from ..models import Episode, EpisodeStatus, Podcast


def sync_podcast(db: Session, podcast: Podcast) -> int:
    """Sync episodes from the source RSS feed.

    Args:
        db: Database session
        podcast: Podcast to sync

    Returns:
        Number of new episodes added
    """
    feed = feedparser.parse(podcast.source_rss_url)

    # Update podcast title if not set
    if not podcast.title and hasattr(feed.feed, "title"):
        podcast.title = feed.feed.title

    # Get existing episode GUIDs
    existing_guids = set(
        guid for (guid,) in db.query(Episode.guid).filter_by(podcast_id=podcast.id).all()
    )

    added = 0
    for entry in feed.entries:
        guid = getattr(entry, "id", None) or getattr(entry, "link", None)
        if not guid or guid in existing_guids:
            continue

        # Extract audio URL from enclosures
        audio_url = None
        if hasattr(entry, "enclosures") and entry.enclosures:
            for enc in entry.enclosures:
                if hasattr(enc, "href"):
                    audio_url = enc.href
                    break

        # Parse published date
        published_at = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                published_at = datetime.fromtimestamp(mktime(entry.published_parsed))
            except (ValueError, OverflowError):
                pass

        episode = Episode(
            podcast_id=podcast.id,
            guid=guid,
            title=getattr(entry, "title", None),
            original_audio_url=audio_url,
            published_at=published_at,
            status=EpisodeStatus.PENDING.value if podcast.auto_process else EpisodeStatus.SKIPPED.value,
        )
        db.add(episode)
        added += 1

    db.commit()
    return added


def sync_all_podcasts(db: Session) -> dict:
    """Sync all podcasts.

    Returns:
        Dict mapping podcast IDs to number of new episodes
    """
    results = {}
    podcasts = db.query(Podcast).all()

    for podcast in podcasts:
        try:
            added = sync_podcast(db, podcast)
            results[podcast.id] = {"added": added, "error": None}
        except Exception as e:
            results[podcast.id] = {"added": 0, "error": str(e)}

    return results
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/web/test_feed_sync.py -v`
Expected: All 2 tests PASS

**Step 5: Commit**

```bash
git add web/services/feed_sync.py tests/web/test_feed_sync.py
git commit -m "feat: add feed sync service"
```

---

### Task 15: Run All Tests and Final Commit

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 2: Final commit**

```bash
git add -A
git commit -m "feat: complete AdNihilator web service implementation"
```

---

## Summary

This implementation plan covers:

1. **Phase 1: Foundation** - Dependencies, database models, auth, FastAPI shell
2. **Phase 2: Podcast Management** - CRUD routes, worker API, RSS feeds
3. **Phase 3: Local Worker** - API client, R2 uploads, daemon
4. **Phase 4: Deployment** - systemd, nginx, documentation
5. **Phase 5: Integration** - Dashboard, episode management, feed sync

Total: 15 tasks, each with TDD approach (test first, then implement).
