"""FastAPI dependencies."""

from fastapi import Depends
from sqlalchemy.orm import Session

# Database session factory (initialized at startup)
SessionFactory = None


def get_db():
    """Dependency to get a database session."""
    if SessionFactory is None:
        raise RuntimeError("Database not initialized")
    session = SessionFactory()
    try:
        yield session
    finally:
        session.close()


def require_transcripts(db: Session = Depends(get_db)) -> None:
    """Unavailable feature returns a safe 503 without altering legacy services."""
    from fastapi import HTTPException
    from .transcript_migration import ready
    if not ready(db.get_bind()):
        raise HTTPException(503, "Transcript schema requires explicit owner migration")
