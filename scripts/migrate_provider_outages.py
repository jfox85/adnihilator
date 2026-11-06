#!/usr/bin/env python3
"""Add durable provider-outage tables and upgrade existing R1 hold schemas.

Usage: python scripts/migrate_provider_outages.py --db /explicit/existing.sqlite
No environment/default database is selected. All DDL is additive and atomic.
"""

import argparse
import sqlite3
import sys
from pathlib import Path


_OUTAGES_SQL = """CREATE TABLE IF NOT EXISTS provider_outages (
    provider VARCHAR(20) NOT NULL PRIMARY KEY,
    category VARCHAR(30) NOT NULL,
    message TEXT NOT NULL,
    active BOOLEAN NOT NULL,
    manual_pause BOOLEAN NOT NULL,
    failure_count INTEGER NOT NULL,
    retry_after DATETIME,
    probe_episode_id VARCHAR(36),
    probe_token VARCHAR(64),
    last_probe_error TEXT,
    updated_at DATETIME NOT NULL
)"""
_HOLDS_SQL = """CREATE TABLE IF NOT EXISTS episode_provider_holds (
    episode_id VARCHAR(36) NOT NULL,
    provider VARCHAR(20) NOT NULL,
    created_at DATETIME NOT NULL,
    last_probe_at DATETIME,
    PRIMARY KEY (episode_id, provider)
)"""
_REQUIRED_COLUMNS = {
    "provider_outages": {"provider", "category", "message", "active", "manual_pause",
                         "failure_count", "retry_after", "probe_episode_id", "probe_token",
                         "last_probe_error", "updated_at"},
    "episode_provider_holds": {"episode_id", "provider", "created_at"},
}


def migrate(db_path: str | Path) -> bool:
    """Serialize additive DDL; preserve all existing state and roll back errors.

    Returns whether the legacy hold table needed its nullable probe timestamp.
    The caller must select an existing SQLite file explicitly. No tables, rows,
    episode/transcript constraints, retry budgets or leases are dropped/reset.
    """
    path = Path(db_path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("An existing SQLite file is required")
    # mode=rw also prevents accidental creation if the file vanishes after check.
    connection = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=30,
                                 isolation_level=None)
    try:
        # Take the write reservation before inspecting schema, so concurrent
        # migration invocations recheck the committed result instead of racing.
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(_OUTAGES_SQL)
        connection.execute(_HOLDS_SQL)
        columns = {}
        for table, required in _REQUIRED_COLUMNS.items():
            columns[table] = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            missing = required - columns[table]
            if missing:
                raise RuntimeError(f"Existing {table} schema lacks required columns: {', '.join(sorted(missing))}")
        added = "last_probe_at" not in columns["episode_provider_holds"]
        if added:
            connection.execute("ALTER TABLE episode_provider_holds ADD COLUMN last_probe_at DATETIME")
        connection.commit()
        return added
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.close()


def main() -> None:
    """Run the explicit migration without starting the app or worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path, metavar="EXISTING_SQLITE_FILE")
    args = parser.parse_args()
    if not args.db.is_file():
        parser.error("--db must name an existing SQLite file")
    try:
        added = migrate(args.db)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        print(f"ERROR: migration rolled back: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    print("Provider outage schema ready; last_probe_at " + ("added." if added else "already present."))


if __name__ == "__main__":
    main()
