"""Pydantic data models for AdNihilator."""

from typing import Any, Literal, Optional

from pydantic import BaseModel


class WordTimestamp(BaseModel):
    """A single word with its timestamp."""

    word: str
    start: float
    end: float
    probability: float


class TranscriptSegment(BaseModel):
    """A single segment from the transcript."""

    index: int
    start: float
    end: float
    text: str
    words: Optional[list[WordTimestamp]] = None


class AdCandidate(BaseModel):
    """A candidate advertisement segment identified by heuristics."""

    start: float
    end: float
    segment_indices: list[int]
    trigger_keywords: list[str]
    heuristic_score: float
    sponsors_found: list[str] = []
    sponsors_missing: list[str] = []


class AdSpan(BaseModel):
    """A refined advertisement span after LLM processing."""

    start: float
    end: float
    confidence: float
    reason: str
    candidate_indices: list[int]


class Sponsor(BaseModel):
    """A sponsor extracted from episode show notes."""

    name: str
    url: Optional[str] = None
    code: Optional[str] = None


class SponsorInfo(BaseModel):
    """Sponsor information extracted from episode description."""

    sponsors: list[Sponsor]
    extraction_method: Literal["patterns", "llm", "none"]


class DetectionResult(BaseModel):
    """Complete result of ad detection on an audio file."""

    audio_path: str
    duration: float
    segments: list[TranscriptSegment]
    candidates: list[AdCandidate]
    ad_spans: list[AdSpan]
    model_info: dict[str, Any]
