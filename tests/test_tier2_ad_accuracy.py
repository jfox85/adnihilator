"""Regression tests for Tier 2 ad accuracy improvements.

Tier 2 adds two post-processing steps:
- ``_clamp_spans_to_duration``: keep every span inside ``[0, duration]``.
- ``_refine_mega_spans``: targeted LLM refine for implausibly long spans, with
  a Gemini host-read guardrail so genuine long reads are never shortened.
"""

import json
import types

import pytest

from adnihilator.models import AdSpan, Sponsor, SponsorInfo, TranscriptSegment
from worker.daemon import WorkerDaemon


def _seg(index: int, start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(index=index, start=start, end=end, text=text)


# --- _clamp_spans_to_duration ------------------------------------------------


def test_clamp_trims_span_past_end() -> None:
    spans = [AdSpan(start=540.0, end=742.0, confidence=1.0, reason="gemini")]
    result = WorkerDaemon._clamp_spans_to_duration(spans, 600.0)
    assert len(result) == 1
    assert result[0].start == 540.0
    assert result[0].end == 600.0


def test_clamp_drops_span_starting_after_end() -> None:
    spans = [AdSpan(start=700.0, end=900.0, confidence=1.0, reason="gemini")]
    result = WorkerDaemon._clamp_spans_to_duration(spans, 600.0)
    assert result == []


def test_clamp_floors_negative_start() -> None:
    spans = [AdSpan(start=-5.0, end=30.0, confidence=1.0, reason="x")]
    result = WorkerDaemon._clamp_spans_to_duration(spans, 600.0)
    assert result[0].start == 0.0
    assert result[0].end == 30.0


def test_clamp_leaves_in_range_spans_untouched() -> None:
    span = AdSpan(start=10.0, end=40.0, confidence=0.9, reason="x", sources=["keywords"])
    result = WorkerDaemon._clamp_spans_to_duration([span], 600.0)
    assert result[0] is span  # unchanged objects pass through by identity


def test_clamp_noop_when_duration_unknown() -> None:
    spans = [AdSpan(start=10.0, end=9999.0, confidence=1.0, reason="x")]
    assert WorkerDaemon._clamp_spans_to_duration(spans, 0.0) == spans


# --- _refine_mega_spans ------------------------------------------------------


class _FakeOpenAIClient:
    """Stands in for ad_llm.OpenAIClient (isinstance check + creds)."""

    api_key = "test"
    base_url = None


def _install_fake_llm(monkeypatch, response_payload, capture=None):
    """Patch create_llm_client + OpenAIClient + openai.OpenAI in worker.daemon."""
    import worker.daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "create_llm_client", lambda config: _FakeOpenAIClient())
    monkeypatch.setattr(daemon_mod, "OpenAIClient", _FakeOpenAIClient)

    class _Resp:
        def __init__(self, content):
            self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
            self.usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5)

    class _Completions:
        def create(self, **kwargs):
            if capture is not None:
                capture.append(kwargs)
            return _Resp(json.dumps(response_payload))

    class _Chat:
        completions = _Completions()

    class _FakeOpenAI:
        def __init__(self, *a, **k):
            self.chat = _Chat()

    import openai
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)


def _daemon_with_config():
    d = WorkerDaemon.__new__(WorkerDaemon)
    d.config = types.SimpleNamespace(llm=types.SimpleNamespace(model="gpt-4o-mini"))
    return d


def test_mega_span_short_spans_are_untouched(monkeypatch) -> None:
    """No span exceeds the mega threshold -> no LLM call, returned unchanged."""
    d = _daemon_with_config()
    spans = [AdSpan(start=0.0, end=120.0, confidence=0.9, reason="ad")]
    segments = [_seg(0, 0.0, 120.0, "brought to you by Acme")]
    sponsors = SponsorInfo(sponsors=[Sponsor(name="Acme")], extraction_method="patterns")
    called = []
    _install_fake_llm(monkeypatch, {"ad_ranges": []}, capture=called)

    result, usage = d._refine_mega_spans(spans, segments, sponsors)

    assert result == spans
    assert usage is None
    assert called == [], "no LLM call for sub-threshold spans"


def test_mega_span_refined_into_subranges(monkeypatch) -> None:
    """A 20-min keyword span is trimmed to the LLM-identified ad windows."""
    d = _daemon_with_config()
    spans = [AdSpan(start=0.0, end=1200.0, confidence=0.9, reason="Keywords")]
    segments = [
        _seg(0, 0.0, 60.0, "brought to you by Acme, use code SAVE"),
        _seg(1, 60.0, 1100.0, "long normal interview about many topics"),
        _seg(2, 1100.0, 1200.0, "our sponsor Globex, check them out"),
    ]
    sponsors = SponsorInfo(
        sponsors=[Sponsor(name="Acme"), Sponsor(name="Globex")],
        extraction_method="patterns",
    )
    payload = {"ad_ranges": [
        {"start": 0.0, "end": 60.0, "reason": "Acme read"},
        {"start": 1100.0, "end": 1200.0, "reason": "Globex read"},
    ]}
    _install_fake_llm(monkeypatch, payload)

    result, usage = d._refine_mega_spans(spans, segments, sponsors)

    assert len(result) == 2
    assert result[0].start == 0.0 and result[0].end == 60.0
    assert result[1].start == 1100.0 and result[1].end == 1200.0
    # Refined spans freed the ~1000s of normal content in the middle.
    assert sum(s.end - s.start for s in result) < 200.0
    assert usage is not None and usage["input_tokens"] > 0


def test_mega_span_gemini_host_read_is_not_shortened(monkeypatch) -> None:
    """A long span vouched for by a matching Gemini host_read is left intact."""
    d = _daemon_with_config()
    spans = [AdSpan(start=100.0, end=800.0, confidence=1.0, reason="host read")]
    segments = [_seg(0, 100.0, 800.0, "Meter sponsor read continuous")]
    sponsors = SponsorInfo(sponsors=[Sponsor(name="Meter")], extraction_method="patterns")
    gemini = [{"start": 95.0, "end": 805.0, "ad_type": "host_read"}]
    called = []
    # If the LLM were called it would try to shrink; assert it is NOT called.
    _install_fake_llm(monkeypatch, {"ad_ranges": [{"start": 100.0, "end": 200.0}]}, capture=called)

    result, usage = d._refine_mega_spans(spans, segments, sponsors, gemini)

    assert result == spans
    assert usage is None
    assert called == [], "Gemini-vouched long host read must not be refined"


def test_mega_span_empty_response_keeps_original(monkeypatch) -> None:
    """Empty ad_ranges = no clear ad found -> keep original (conservative)."""
    d = _daemon_with_config()
    spans = [AdSpan(start=0.0, end=900.0, confidence=0.9, reason="Keywords")]
    segments = [_seg(0, 0.0, 900.0, "brought to you by Acme then lots of talk")]
    sponsors = SponsorInfo(sponsors=[Sponsor(name="Acme")], extraction_method="patterns")
    _install_fake_llm(monkeypatch, {"ad_ranges": []})

    result, _ = d._refine_mega_spans(spans, segments, sponsors)

    assert len(result) == 1
    assert result[0].start == 0.0 and result[0].end == 900.0


def test_mega_span_malformed_response_keeps_original(monkeypatch) -> None:
    """Missing ad_ranges key -> keep original rather than guess."""
    d = _daemon_with_config()
    spans = [AdSpan(start=0.0, end=900.0, confidence=0.9, reason="Keywords")]
    segments = [_seg(0, 0.0, 900.0, "brought to you by Acme then lots of talk")]
    sponsors = SponsorInfo(sponsors=[Sponsor(name="Acme")], extraction_method="patterns")
    _install_fake_llm(monkeypatch, {"unexpected": True})

    result, _ = d._refine_mega_spans(spans, segments, sponsors)

    assert len(result) == 1
    assert result[0].end == 900.0


def test_mega_span_subranges_clamped_within_span(monkeypatch) -> None:
    """LLM ranges that exceed the span are clamped to its bounds."""
    d = _daemon_with_config()
    spans = [AdSpan(start=200.0, end=1000.0, confidence=0.9, reason="Keywords")]
    segments = [_seg(0, 200.0, 1000.0, "Acme sponsor read here and there")]
    sponsors = SponsorInfo(sponsors=[Sponsor(name="Acme")], extraction_method="patterns")
    payload = {"ad_ranges": [{"start": 0.0, "end": 5000.0, "reason": "whole"}]}
    _install_fake_llm(monkeypatch, payload)

    result, _ = d._refine_mega_spans(spans, segments, sponsors)

    assert len(result) == 1
    assert result[0].start == 200.0
    assert result[0].end == 1000.0
