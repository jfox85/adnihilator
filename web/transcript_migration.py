"""Additive transcript migration. Rollback drops only transcript-owned tables."""

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from .models import EpisodeTranscript, TranscriptJob, TranscriptVersion


TABLES = [EpisodeTranscript.__table__, TranscriptVersion.__table__, TranscriptJob.__table__]


def upgrade(engine: Engine) -> None:
    """Idempotent upgrade; existing episode/audio tables and columns are untouched."""
    for table in TABLES:
        table.create(engine, checkfirst=True)
    with engine.begin() as connection:
        connection.execute(text("""CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts
            USING fts5(text, episode_id UNINDEXED, checksum UNINDEXED, ordinal UNINDEXED)"""))
        connection.execute(text("""CREATE TRIGGER IF NOT EXISTS transcript_episode_delete
            AFTER DELETE ON episodes BEGIN
                DELETE FROM episode_transcripts WHERE episode_id=old.id;
                DELETE FROM transcript_versions WHERE episode_id=old.id;
                DELETE FROM transcript_jobs WHERE episode_id=old.id;
            END"""))
        connection.execute(text("""CREATE TRIGGER IF NOT EXISTS transcript_fts_episode_delete
            AFTER DELETE ON episodes BEGIN DELETE FROM transcript_fts WHERE episode_id=old.id; END"""))


def downgrade(engine: Engine) -> None:
    """Owner-only rollback; transcripts must be exported/backed up first."""
    with engine.begin() as connection:
        connection.execute(text("DROP TRIGGER IF EXISTS transcript_episode_delete"))
        connection.execute(text("DROP TRIGGER IF EXISTS transcript_fts_episode_delete"))
        connection.execute(text("DROP TABLE IF EXISTS transcript_fts"))
    for table in reversed(TABLES):
        table.drop(engine, checkfirst=True)


def ready(engine: Engine) -> bool:
    """Read-only check: explicit migration/opt-in must have created all objects."""
    if not {table.name for table in TABLES} | {"transcript_fts"} <= set(inspect(engine).get_table_names()):
        return False
    with engine.connect() as connection:
        triggers = set(connection.scalars(text("SELECT name FROM sqlite_master WHERE type='trigger'")))
    return {"transcript_episode_delete", "transcript_fts_episode_delete"} <= triggers
