"""Versioned canonical original-audio transcripts, independent of search indexes."""

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_DOCUMENT_BYTES = 12 * 1024 * 1024


class Segment(BaseModel):
    """Preserve source indices/word metadata; never invent a speaker."""

    model_config = ConfigDict(extra="allow")
    index: int = Field(ge=0)
    start: float = Field(ge=0, allow_inf_nan=False)
    end: float = Field(ge=0, allow_inf_nan=False)
    text: str = Field(max_length=10000)
    words: list[dict[str, Any]] | None = Field(default=None, max_length=5000)

    @model_validator(mode="after")
    def check_end(self) -> "Segment":
        if self.end < self.start:
            raise ValueError("Segment end precedes start")
        return self


class Span(BaseModel):
    """Ad spans tag original text without removing it; retain detection metadata."""

    model_config = ConfigDict(extra="allow")
    start: float = Field(ge=0, allow_inf_nan=False)
    end: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_end(self) -> "Span":
        if self.end <= self.start:
            raise ValueError("Empty ad span")
        return self


class TranscriptDocument(BaseModel):
    """An immutable content version; service validates identity against Episode."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    episode_id: str = Field(min_length=1, max_length=100)
    podcast_id: str = Field(min_length=1, max_length=100)
    guid: str = Field(min_length=1, max_length=2000)
    original_audio_url: str = Field(max_length=4000)
    source: str = Field(min_length=1, max_length=100)
    model: str | None = Field(default=None, max_length=200)
    source_artifact_id: str | None = Field(default=None, max_length=100)
    source_artifact_checksum: str | None = Field(default=None, pattern="^[a-f0-9]{64}$")
    producer_version: str = Field(default="adnihilator-0.1/transcript-1", max_length=100)
    duration: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    timestamp_basis: Literal["original_audio", "external_source"] = "original_audio"
    segments: list[Segment] = Field(min_length=1, max_length=30000)
    ad_spans: list[Span] = Field(default_factory=list, max_length=1000)

    @model_validator(mode="after")
    def validate_content(self) -> "TranscriptDocument":
        if len({s.index for s in self.segments}) != len(self.segments):
            raise ValueError("Duplicate source segment indices")
        if self.duration and max(s.end for s in self.segments) > self.duration + 2:
            raise ValueError("Transcript exceeds original duration")
        if self.duration and any(span.end > self.duration + 2 for span in self.ad_spans):
            raise ValueError("Ad span exceeds original duration")
        if len(self.canonical_bytes()) > MAX_DOCUMENT_BYTES:
            raise ValueError("Transcript document too large")
        return self

    def canonical_bytes(self) -> bytes:
        """Deterministic JSON retains every canonical segment and extra source field."""
        return json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()

    def checksum(self) -> str:
        """Content and provenance changes create a new revision."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def audio_identity(url: str) -> str:
    """Stable Megaphone enclosure identity; other providers must match exactly."""
    match = re.search(r"(?:^|/)([A-Z]+\d+\.mp3)(?:[?&#]|$)", url)
    return match.group(1) if match else url
