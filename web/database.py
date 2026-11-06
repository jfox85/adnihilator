"""Database configuration and session management."""

import os
from pathlib import Path

from sqlalchemy import create_engine, event, inspect
from sqlalchemy.engine import Engine
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


def init_db(engine: Engine | None = None) -> Engine:
    """Initialize the database, creating all tables."""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    # SQLite has no ALTER TABLE IF NOT EXISTS. Serialize startup upgrades so
    # concurrent web processes cannot race while adding this nullable column.
    with engine.connect() as connection:
        if engine.dialect.name == "sqlite":
            connection.exec_driver_sql("BEGIN IMMEDIATE")
        columns = {column["name"] for column in inspect(connection).get_columns("podcasts")}
        if "last_sync_error" not in columns:
            connection.exec_driver_sql("ALTER TABLE podcasts ADD COLUMN last_sync_error TEXT")
        connection.commit()
    return engine
