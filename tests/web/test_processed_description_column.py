"""Startup adds the processed_description column to existing databases."""

import sqlite3

from sqlalchemy import inspect

from web.database import get_engine, init_db


def test_init_db_adds_processed_description_to_existing_episodes_table(tmp_path):
    """A pre-existing production schema gains the nullable column, data intact."""
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE podcasts (id VARCHAR(36) PRIMARY KEY, feed_token VARCHAR(64), source_rss_url TEXT)"
    )
    conn.execute(
        "CREATE TABLE episodes (id VARCHAR(36) PRIMARY KEY, podcast_id VARCHAR(36), guid TEXT, "
        "description TEXT, status VARCHAR(20))"
    )
    conn.execute("INSERT INTO episodes (id, podcast_id, guid, description, status) VALUES ('e1', 'p1', 'g', 'd', 'complete')")
    conn.commit()
    conn.close()

    engine = init_db(get_engine(f"sqlite:///{db_path}"))
    init_db(engine)  # idempotent

    columns = {c["name"] for c in inspect(engine).get_columns("episodes")}
    assert "processed_description" in columns
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT description, processed_description FROM episodes").fetchone() == ("d", None)
