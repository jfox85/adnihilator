#!/usr/bin/env python3
"""Add LLM tracking fields to episodes table.

This migration adds the following columns:
- llm_provider: LLM provider name (e.g., "openai", "gemini")
- llm_model: Model name (e.g., "gpt-4o-mini", "gemini-2.0-flash-exp")
- llm_input_tokens: Number of input tokens used
- llm_output_tokens: Number of output tokens used
- llm_total_tokens: Total tokens used
- llm_cost_usd: Total cost in USD (stored as REAL for SQLite compatibility)
- detection_source: Source of ad detection ("timestamps", "external", "gemini", "whisper")

Usage:
    python scripts/migrate_add_llm_fields.py [db_path]

If no db_path is provided, defaults to data/adnihilator.db
"""

import sqlite3
import sys
from pathlib import Path


def migrate(db_path: str) -> None:
    """Add LLM fields to existing database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Define columns to add
    columns = [
        ("llm_provider", "TEXT"),
        ("llm_model", "TEXT"),
        ("llm_input_tokens", "INTEGER"),
        ("llm_output_tokens", "INTEGER"),
        ("llm_total_tokens", "INTEGER"),
        ("llm_cost_usd", "REAL"),  # SQLite uses REAL for decimals
        ("detection_source", "TEXT"),
    ]

    added = 0
    skipped = 0

    for col_name, col_type in columns:
        try:
            cursor.execute(f"ALTER TABLE episodes ADD COLUMN {col_name} {col_type}")
            print(f"  Added column: {col_name} ({col_type})")
            added += 1
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                print(f"  Skipped (exists): {col_name}")
                skipped += 1
            else:
                raise

    conn.commit()
    conn.close()

    print()
    print(f"Migration complete: {added} added, {skipped} skipped")


def main():
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        # Default to local database
        db_path = "data/adnihilator.db"

    db_file = Path(db_path)
    if not db_file.exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    print(f"Migrating database: {db_path}")
    print()
    migrate(db_path)


if __name__ == "__main__":
    main()
