"""Explicit additive migration/rebuild; requires an explicit database path."""

import argparse
from pathlib import Path

from sqlalchemy import select

from web.database import get_engine, get_session_factory
from web.models import EpisodeTranscript
from web.services.transcripts import rebuild_episode
from web.transcript_migration import downgrade, upgrade


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--rollback", action="store_true", help="Drops transcript data; back it up first")
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()
    if not args.db.exists():
        parser.error("Database must already exist")
    engine = get_engine("sqlite:///" + str(args.db.resolve()))
    if args.rollback:
        downgrade(engine)
    else:
        upgrade(engine)
        if args.rebuild_index:
            with get_session_factory(engine)() as db:
                ids = list(db.scalars(select(EpisodeTranscript.episode_id).where(EpisodeTranscript.active_checksum.isnot(None))))
                for episode_id in ids:
                    rebuild_episode(db, episode_id)
    print("Transcript migration operation completed")


if __name__ == "__main__":
    main()
