"""Tests for Pydantic data models."""

import json

import pytest

from adnihilator.models import (
    AdCandidate,
    AdSpan,
    DetectionResult,
    TranscriptSegment,
)


class TestTranscriptSegment:
    def test_create_segment(self):
        segment = TranscriptSegment(
            index=0,
            start=0.0,
            end=5.5,
            text="Hello world",
        )
        assert segment.index == 0
        assert segment.start == 0.0
        assert segment.end == 5.5
        assert segment.text == "Hello world"

    def test_segment_json_serialization(self):
        segment = TranscriptSegment(
            index=1,
            start=10.5,
            end=15.0,
            text="Test segment",
        )
        json_str = segment.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["index"] == 1
        assert parsed["start"] == 10.5
        assert parsed["end"] == 15.0
        assert parsed["text"] == "Test segment"

    def test_segment_from_dict(self):
        data = {
            "index": 2,
            "start": 20.0,
            "end": 25.5,
            "text": "From dict",
        }
        segment = TranscriptSegment(**data)
        assert segment.index == 2
        assert segment.text == "From dict"


class TestAdCandidate:
    def test_create_candidate(self):
        candidate = AdCandidate(
            start=60.0,
            end=120.0,
            segment_indices=[10, 11, 12],
            trigger_keywords=["sponsor", "promo code"],
            heuristic_score=0.75,
        )
        assert candidate.start == 60.0
        assert candidate.end == 120.0
        assert len(candidate.segment_indices) == 3
        assert "sponsor" in candidate.trigger_keywords
        assert candidate.heuristic_score == 0.75

    def test_candidate_json_roundtrip(self):
        original = AdCandidate(
            start=100.0,
            end=150.0,
            segment_indices=[5, 6, 7, 8],
            trigger_keywords=["brought to you by"],
            heuristic_score=0.9,
        )
        json_str = original.model_dump_json()
        restored = AdCandidate.model_validate_json(json_str)

        assert restored.start == original.start
        assert restored.end == original.end
        assert restored.segment_indices == original.segment_indices
        assert restored.trigger_keywords == original.trigger_keywords
        assert restored.heuristic_score == original.heuristic_score


class TestAdSpan:
    def test_create_span(self):
        span = AdSpan(
            start=60.0,
            end=120.0,
            confidence=0.85,
            reason="Sponsor language detected",
            candidate_indices=[0],
        )
        assert span.start == 60.0
        assert span.end == 120.0
        assert span.confidence == 0.85
        assert span.reason == "Sponsor language detected"
        assert span.candidate_indices == [0]

    def test_span_json_serialization(self):
        span = AdSpan(
            start=200.0,
            end=260.0,
            confidence=0.7,
            reason="heuristic_only",
            candidate_indices=[1, 2],
        )
        json_str = span.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["confidence"] == 0.7
        assert parsed["reason"] == "heuristic_only"


class TestDetectionResult:
    def test_create_result(self):
        segments = [
            TranscriptSegment(index=0, start=0.0, end=5.0, text="Hello"),
            TranscriptSegment(index=1, start=5.0, end=10.0, text="World"),
        ]
        candidates = [
            AdCandidate(
                start=5.0,
                end=10.0,
                segment_indices=[1],
                trigger_keywords=["sponsor"],
                heuristic_score=0.5,
            )
        ]
        ad_spans = [
            AdSpan(
                start=5.0,
                end=10.0,
                confidence=0.4,
                reason="heuristic_only",
                candidate_indices=[0],
            )
        ]

        result = DetectionResult(
            audio_path="/path/to/audio.mp3",
            duration=600.0,
            segments=segments,
            candidates=candidates,
            ad_spans=ad_spans,
            model_info={"whisper_model": "small", "llm_provider": "none"},
        )

        assert result.audio_path == "/path/to/audio.mp3"
        assert result.duration == 600.0
        assert len(result.segments) == 2
        assert len(result.candidates) == 1
        assert len(result.ad_spans) == 1
        assert result.model_info["whisper_model"] == "small"

    def test_result_json_roundtrip(self):
        segments = [
            TranscriptSegment(index=0, start=0.0, end=5.0, text="Test"),
        ]
        original = DetectionResult(
            audio_path="/test.mp3",
            duration=300.0,
            segments=segments,
            candidates=[],
            ad_spans=[],
            model_info={"whisper_model": "tiny"},
        )

        json_str = original.model_dump_json()
        restored = DetectionResult.model_validate_json(json_str)

        assert restored.audio_path == original.audio_path
        assert restored.duration == original.duration
        assert len(restored.segments) == len(original.segments)


class TestSponsorModels:
    def test_sponsor_with_all_fields(self):
        """Sponsor with name, url, and code."""
        from adnihilator.models import Sponsor

        sponsor = Sponsor(name="ExpressVPN", url="expressvpn.com/twit", code="TWIT")
        assert sponsor.name == "ExpressVPN"
        assert sponsor.url == "expressvpn.com/twit"
        assert sponsor.code == "TWIT"

    def test_sponsor_name_only(self):
        """Sponsor with just a name."""
        from adnihilator.models import Sponsor

        sponsor = Sponsor(name="Miro")
        assert sponsor.name == "Miro"
        assert sponsor.url is None
        assert sponsor.code is None

    def test_sponsor_info(self):
        """SponsorInfo container."""
        from adnihilator.models import Sponsor, SponsorInfo

        info = SponsorInfo(
            sponsors=[Sponsor(name="Shopify", url="shopify.com/lex")],
            extraction_method="patterns"
        )
        assert len(info.sponsors) == 1
        assert info.extraction_method == "patterns"

    def test_sponsor_info_empty(self):
        """SponsorInfo with no sponsors."""
        from adnihilator.models import SponsorInfo

        info = SponsorInfo(sponsors=[], extraction_method="none")
        assert info.sponsors == []
