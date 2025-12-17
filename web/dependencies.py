"""FastAPI dependencies."""

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
