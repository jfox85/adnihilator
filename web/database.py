"""Database configuration and session management."""

import os
from pathlib import Path

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


def get_database_path():
    """Get the database path from environment or default."""
    return os.environ.get(
        "DATABASE_PATH",
        str(Path.home() / ".adnihilator" / "adnihilator.db")
    )


def get_engine(database_url: str | None = None):
    """Create database engine with WAL mode for SQLite."""
    if database_url is None:
        # Get database path (can be overridden by environment)
        database_path = get_database_path()

        # Ensure directory exists (unless it's :memory:)
        if database_path != ":memory:":
            db_path = Path(database_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

        database_url = f"sqlite:///{database_path}"

    # For in-memory databases, disable same-thread check for testing
    connect_args = {}
    if ":memory:" in database_url:
        connect_args["check_same_thread"] = False

    engine = create_engine(database_url, echo=False, connect_args=connect_args)

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
    from .transcript_migration import TABLES, upgrade
    owned = {table.name for table in TABLES}
    Base.metadata.create_all(engine, tables=[table for table in Base.metadata.sorted_tables if table.name not in owned])
    _add_missing_nullable_columns(engine)
    # Existing reviewed schema stays usable. Normal startup never migrates it.
    if os.getenv("TRANSCRIPT_SCHEMA_OPT_IN", "false").lower() == "true":
        upgrade(engine)
    return engine


# Additive, nullable columns that create_all() cannot add to existing tables.
# Only plain nullable additions belong here; anything else needs a migration.
_ADDITIVE_COLUMNS = {
    "episodes": {"processed_description": "TEXT"},
}


def _add_missing_nullable_columns(engine) -> None:
    """Add known nullable columns to pre-existing tables (idempotent)."""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table, columns in _ADDITIVE_COLUMNS.items():
            if table not in tables:
                continue
            existing = {c["name"] for c in inspector.get_columns(table)}
            for name, sql_type in columns.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}"))
