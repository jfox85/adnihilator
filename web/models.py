"""SQLAlchemy models for the web service."""

import secrets
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from decimal import Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class EpisodeStatus(str, Enum):
    """Status of an episode in the processing pipeline."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"
    EXPIRED = "expired"


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid4())


def generate_feed_token() -> str:
    """Generate a secure random feed token (32+ chars)."""
    return secrets.token_urlsafe(32)


class Podcast(Base):
    """A podcast subscription."""

    __tablename__ = "podcasts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    feed_token: Mapped[str] = mapped_column(
        String(64), unique=True, default=generate_feed_token
    )
    source_rss_url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    artwork_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    auto_process: Mapped[bool] = mapped_column(Boolean, default=False)
    last_sync_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_synced_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    episodes: Mapped[list["Episode"]] = relationship(
        "Episode", back_populates="podcast", cascade="all, delete-orphan"
    )


class Episode(Base):
    """An episode from a podcast."""

    __tablename__ = "episodes"
    __table_args__ = (
        UniqueConstraint("podcast_id", "guid", name="uq_podcast_episode_guid"),
        CheckConstraint(
            "status IN ('pending', 'processing', 'complete', 'failed', 'skipped', 'expired')",
            name="ck_episode_status",
        ),
        # Claim path: select oldest pending episode.
        Index("ix_episodes_status_created_at", "status", "created_at"),
        # Stuck-job recovery: find stale processing episodes by claimed_at.
        Index("ix_episodes_status_claimed_at", "status", "claimed_at"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    podcast_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("podcasts.id"), nullable=False
    )
    guid: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    duration: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # seconds
    original_audio_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Episode page URL for transcripts
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), default=EpisodeStatus.PENDING.value
    )
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    claimed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    progress_step: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    progress_percent: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    processed_audio_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    processed_duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ads_removed_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    detection_result_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # LLM Usage Tracking
    llm_provider: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    llm_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    llm_input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    llm_output_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    llm_total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    llm_cost_usd: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6), nullable=True)

    # Detection Source Tracking
    detection_source: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # 'timestamps', 'external', 'gemini', 'whisper'

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    podcast: Mapped["Podcast"] = relationship("Podcast", back_populates="episodes")

    @property
    def status_enum(self) -> EpisodeStatus:
        """Get status as enum."""
        return EpisodeStatus(self.status)

    @status_enum.setter
    def status_enum(self, value: EpisodeStatus):
        """Set status from enum."""
        self.status = value.value


class EpisodeTranscript(Base):
    """Coverage and atomic pointer to the last complete immutable document."""

    __tablename__ = "episode_transcripts"
    __table_args__ = (CheckConstraint("state IN ('available','missing','queued','processing','failed')"),)
    episode_id: Mapped[str] = mapped_column(ForeignKey("episodes.id", ondelete="CASCADE"), primary_key=True)
    state: Mapped[str] = mapped_column(String(20), default="missing")
    active_checksum: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    indexed_checksum: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    downloaded_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TranscriptVersion(Base):
    """Private canonical text, immutable and separate from its rebuildable index."""

    __tablename__ = "transcript_versions"
    __table_args__ = (UniqueConstraint("episode_id", "checksum"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    episode_id: Mapped[str] = mapped_column(ForeignKey("episodes.id", ondelete="CASCADE"), index=True)
    checksum: Mapped[str] = mapped_column(String(64))
    canonical_json: Mapped[str] = mapped_column(Text)
    episode_metadata_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TranscriptJob(Base):
    """Explicit transcript requests share worker polling but never audio status."""

    __tablename__ = "transcript_jobs"
    __table_args__ = (CheckConstraint("state IN ('queued','processing','complete','failed')"),)
    episode_id: Mapped[str] = mapped_column(ForeignKey("episodes.id", ondelete="CASCADE"), primary_key=True)
    state: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    claim_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    claimed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
