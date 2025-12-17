"""Background scheduler for periodic tasks."""

import asyncio
import logging
import os
import random
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class FeedSyncScheduler:
    """Background scheduler that periodically syncs podcast feeds.

    Runs feed sync at configurable intervals with a random jitter
    to avoid looking automated (e.g., not exactly on the hour).
    """

    def __init__(
        self,
        session_factory,
        interval_seconds: int = 3600,  # Default: 1 hour
        jitter_seconds: int = 300,     # Default: +/- 5 minutes
    ):
        """Initialize the scheduler.

        Args:
            session_factory: SQLAlchemy session factory
            interval_seconds: Base interval between syncs
            jitter_seconds: Random jitter added to interval (0 to jitter_seconds)
        """
        self.session_factory = session_factory
        self.interval_seconds = interval_seconds
        self.jitter_seconds = jitter_seconds
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def _get_next_interval(self) -> int:
        """Get the next sleep interval with random jitter."""
        jitter = random.randint(0, self.jitter_seconds)
        return self.interval_seconds + jitter

    async def _sync_all_feeds(self) -> None:
        """Sync all podcast feeds."""
        from .feed_sync import sync_all_podcasts

        db = self.session_factory()
        try:
            logger.info("Starting scheduled feed sync...")
            results = sync_all_podcasts(db)

            total_added = sum(r["added"] for r in results.values())
            errors = sum(1 for r in results.values() if r["error"])

            logger.info(
                f"Feed sync complete: {len(results)} podcasts, "
                f"{total_added} new episodes, {errors} errors"
            )
        except Exception as e:
            logger.error(f"Feed sync failed: {e}")
        finally:
            db.close()

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        # Initial delay with jitter so we don't sync immediately on startup
        initial_delay = random.randint(60, 300)  # 1-5 minutes after startup
        logger.info(f"Feed sync scheduler started, first sync in {initial_delay}s")
        await asyncio.sleep(initial_delay)

        while self._running:
            try:
                await self._sync_all_feeds()
            except Exception as e:
                logger.error(f"Unexpected error in feed sync: {e}")

            # Wait for next interval
            interval = self._get_next_interval()
            next_sync = datetime.now().timestamp() + interval
            logger.info(
                f"Next feed sync in {interval}s "
                f"(at {datetime.fromtimestamp(next_sync).strftime('%H:%M:%S')})"
            )
            await asyncio.sleep(interval)

    def start(self) -> None:
        """Start the scheduler background task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Feed sync scheduler initialized")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Feed sync scheduler stopped")


def create_scheduler(session_factory) -> FeedSyncScheduler:
    """Create a scheduler with settings from environment variables.

    Environment variables:
        FEED_SYNC_INTERVAL: Sync interval in seconds (default: 3600)
        FEED_SYNC_JITTER: Random jitter in seconds (default: 300)
        FEED_SYNC_ENABLED: Set to "false" to disable (default: true)
    """
    enabled = os.getenv("FEED_SYNC_ENABLED", "true").lower() != "false"
    if not enabled:
        logger.info("Feed sync scheduler disabled via FEED_SYNC_ENABLED=false")
        return None

    interval = int(os.getenv("FEED_SYNC_INTERVAL", "3600"))
    jitter = int(os.getenv("FEED_SYNC_JITTER", "300"))

    logger.info(f"Feed sync scheduler configured: interval={interval}s, jitter={jitter}s")

    return FeedSyncScheduler(
        session_factory=session_factory,
        interval_seconds=interval,
        jitter_seconds=jitter,
    )
