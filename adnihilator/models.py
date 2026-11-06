"""Pydantic data models for AdNihilator."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProviderFailureInfo(BaseModel):
    """Safe provider failure details and the existing throttle retry policy."""

    category: Literal["billing", "authentication", "model", "rate_limit", "transient", "unknown"]
    status_code: int | None = None
    global_failure: bool = False
    retryable: bool = False
    message: str


class GeminiAdDetection(BaseModel):
    """A structurally valid provider detection before timestamp localization."""

    model_config = ConfigDict(strict=True, allow_inf_nan=False, str_strip_whitespace=True)

    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    sponsor: str = Field(min_length=1)
    confidence: float = Field(default=0.9, ge=0, le=1)
    ad_type: str | None = None
    language: str = "unknown"
    reason: str = ""

    @model_validator(mode="after")
    def validate_time_order(self) -> "GeminiAdDetection":
        """Reject reversed and empty ranges instead of silently losing coverage."""
        if self.end_time <= self.start_time:
            raise ValueError("Detection end_time must be after start_time")
        return self


class GeminiDetectionResponse(BaseModel):
    """Require an explicit list, including an explicit empty no-ad result."""

    model_config = ConfigDict(strict=True)
    ads: list[GeminiAdDetection]


class MergedAdDetection(BaseModel):
    """A valid required merge decision before clamping to episode bounds."""

    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    start: float
    end: float
    confidence: float = Field(default=0.9, ge=0, le=1)
    reason: str = "LLM merged"
    sources: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_time_order(self) -> "MergedAdDetection":
        """Reject malformed ranges instead of treating them as rejected ads."""
        if self.end <= self.start:
            raise ValueError("Merged detection end must be after start")
        return self


class MergedDetectionResponse(BaseModel):
    """A required merger must explicitly return its complete ad decision."""

    model_config = ConfigDict(strict=True)
    ads: list[MergedAdDetection]


class TranscriptAdDetection(BaseModel):
    """A complete line-index or legacy timestamp transcript decision."""

    model_config = ConfigDict(strict=True, allow_inf_nan=False)
    start_line: int | None = Field(default=None, ge=0)
    end_line: int | None = Field(default=None, ge=0)
    ad_start: float | None = None
    ad_end: float | None = None
    confidence: float = Field(default=0.7, ge=0, le=1)
    reason: str = "LLM analysis"

    @model_validator(mode="after")
    def validate_bounds(self) -> "TranscriptAdDetection":
        """Never silently skip missing, reversed or partial boundary pairs."""
        has_lines = self.start_line is not None or self.end_line is not None
        has_times = self.ad_start is not None or self.ad_end is not None
        if has_lines:
            if self.start_line is None or self.end_line is None or self.end_line < self.start_line:
                raise ValueError("Transcript line bounds must be a complete ordered pair")
        if has_times and (self.ad_start is None or self.ad_end is None or self.ad_end <= self.ad_start):
            raise ValueError("Transcript timestamps must be a complete increasing pair")
        if not has_lines and not has_times:
            raise ValueError("Transcript decision must provide line bounds or timestamps")
        return self


class TranscriptRefinementResponse(BaseModel):
    """A required transcript refinement must explicitly decide every ad record."""

    model_config = ConfigDict(strict=True)
    ads: list[TranscriptAdDetection]


class FinalReviewRejection(BaseModel):
    """An explicit rejection of one supplied advertisement index."""

    model_config = ConfigDict(strict=True)
    index: int = Field(ge=0)
    reason: str = "unknown"


class FinalReviewMerge(BaseModel):
    """A merge referring to distinct original input advertisement indices."""

    model_config = ConfigDict(strict=True)
    indices: list[int] = Field(min_length=2)
    reason: str = ""

    @model_validator(mode="after")
    def validate_indices(self) -> "FinalReviewMerge":
        """Reject negative and repeated merge indices."""
        if any(index < 0 for index in self.indices) or len(set(self.indices)) != len(self.indices):
            raise ValueError("Merge indices must be distinct nonnegative integers")
        return self


class FinalAdReview(BaseModel):
    """A final review must explicitly authorize the cuts it approves."""

    model_config = ConfigDict(strict=True, allow_inf_nan=False)
    approved_ads: list[int]
    rejected_ads: list[FinalReviewRejection] = Field(default_factory=list)
    merge_suggestions: list[FinalReviewMerge] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    overall_confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_approvals(self) -> "FinalAdReview":
        """Never interpret negative indices using Python's reverse indexing."""
        if any(index < 0 for index in self.approved_ads):
            raise ValueError("Approval indices must be nonnegative integers")
        return self


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
    candidate_indices: list[int] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)  # ["gemini", "keywords", "llm_new"]
    ad_type: Optional[str] = None  # "dynamic_insertion", "host_read", "network_bumper"


class Sponsor(BaseModel):
    """A sponsor extracted from episode show notes."""

    name: str
    url: Optional[str] = None
    code: Optional[str] = None


class SponsorInfo(BaseModel):
    """Sponsor information extracted from episode description."""

    sponsors: list[Sponsor]
    extraction_method: Literal["patterns", "llm", "none"]


class AdTimestamp(BaseModel):
    """Ad timestamp extracted from episode description."""

    start: float  # seconds
    end: float  # seconds
    label: Optional[str] = None  # e.g., "Sponsors", "Ad Break"
    confidence: float  # 0-1, based on pattern match quality
    extraction_method: str  # e.g., "chapter_marker", "time_range"


class DetectionResult(BaseModel):
    """Complete result of ad detection on an audio file."""

    audio_path: str
    duration: float
    segments: list[TranscriptSegment]
    candidates: list[AdCandidate]
    ad_spans: list[AdSpan]
    model_info: dict[str, Any]
