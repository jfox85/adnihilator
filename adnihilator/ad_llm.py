"""LLM-based advertisement refinement."""

import json
import math
from abc import ABC, abstractmethod
from typing import Any

from .config import Config
from .models import (
    AdCandidate,
    AdSpan,
    MergedDetectionResponse,
    SponsorInfo,
    TranscriptAdDetection,
    TranscriptRefinementResponse,
    TranscriptSegment,
)
from .provider_responses import completed_openai_text
from .provider_errors import InvalidProviderDecision, ProviderOutageError, classify_provider_error
from .redaction import redact_secrets

# Context window: how many seconds before/after candidate to include
CONTEXT_BEFORE = 30.0  # 30 seconds before
CONTEXT_AFTER = 120.0  # 2 minutes after (ads often have trailing CTAs and back-to-back sponsors)

# Merge candidates within this many seconds of each other
MERGE_THRESHOLD = 300.0  # 5 minutes

SYSTEM_PROMPT = """You are an expert at identifying PAID SPONSOR ADVERTISEMENTS in podcast transcripts.

Your task is to identify ONLY scripted sponsor reads and HOUSE ADS. This includes:
1. Paid sponsor advertisements - where the host reads an ad for a paying sponsor
2. House ads / network promos - where the host promotes OTHER shows or content (not the current podcast)

Do NOT flag genuine content recommendations or gift suggestions about products.

CRITICAL DISTINCTION - Ads vs Content:

THIS IS AN AD (flag it):
- "This episode is brought to you by [Sponsor]"
- "Thanks to [Sponsor] for sponsoring this episode"
- "Today's sponsor is [Sponsor]"
- Scripted product pitches with promo codes like "use code PODCAST for 20% off"
- Sponsor URLs like "visit allthehacks.com/sponsor" or "go to sponsor.com/podcast"
- HOUSE ADS: Promotions for OTHER podcasts, shows, or network content (e.g., "check out my other show", "available on the Ringer Podcast Network", "subscribe to [Other Show]")

THIS IS NOT AN AD (do NOT flag):
- Personal product recommendations ("I love these sunglasses", "my favorite bag is...")
- Gift guide suggestions ("here are some gift ideas")
- Genuine reviews or opinions about products the host actually uses
- Mentions of affiliate links without a scripted sponsor read
- Product comparisons or recommendations that are part of the episode's actual content
- Shopping tips or deal-finding advice (even if products are mentioned)
- Introductions or descriptions of THIS podcast itself (e.g., "welcome to [Current Podcast]")

The key test: Does the segment start with a clear sponsor introduction phrase like "brought to you by" or "today's sponsor is"? If not, it's probably content, not an ad. For house ads, is the host promoting OTHER content (other shows, network, etc.) rather than THIS podcast's content?

Signs the ad block has ENDED:
- Topic change back to the main discussion
- Phrases like "anyway", "so", "alright", "now let's get back to"
- Return to interview/conversation with a guest
- Discussion of the episode's main topic

BOUNDARY RULES:
1. START: Must begin with a clear sponsor introduction phrase - never flag content before it
2. END: Include all back-to-back sponsors as one block - end AFTER the final URL/CTA
3. Back-to-back sponsors (Sponsor A -> Sponsor B) are ONE ad block
4. Be CONSERVATIVE - when in doubt, it's probably content, not an ad"""


class InvalidRefinementResponse(InvalidProviderDecision):
    """A required refinement did not produce a complete authoritative decision."""


class AdLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def refine_candidates(
        self,
        segments: list[TranscriptSegment],
        candidates: list[AdCandidate],
        config: Config,
        sponsors: SponsorInfo | None = None,
        podcast_title: str | None = None,
    ) -> list[AdSpan]:
        """Refine ad candidates using LLM analysis.

        Args:
            segments: Full transcript segments.
            candidates: Heuristic ad candidates.
            config: Configuration object.
            sponsors: Optional sponsor information from show notes.
            podcast_title: Optional podcast title for context in pre-roll detection.

        Returns:
            Refined list of AdSpan objects.
        """
        pass


class HeuristicOnlyClient(AdLLMClient):
    """Fallback client that converts candidates to spans without LLM."""

    def refine_candidates(
        self,
        segments: list[TranscriptSegment],
        candidates: list[AdCandidate],
        config: Config,
        sponsors: SponsorInfo | None = None,
        podcast_title: str | None = None,
    ) -> list[AdSpan]:
        """Convert candidates directly to spans with reduced confidence."""
        spans: list[AdSpan] = []
        for idx, candidate in enumerate(candidates):
            spans.append(
                AdSpan(
                    start=candidate.start,
                    end=candidate.end,
                    confidence=candidate.heuristic_score * 0.8,
                    reason="heuristic_only",
                    candidate_indices=[idx],
                )
            )
        return spans


def merge_nearby_candidates(
    candidates: list[AdCandidate], threshold: float = MERGE_THRESHOLD
) -> list[tuple[list[int], AdCandidate]]:
    """Merge candidates that are close together.

    Returns list of (original_indices, merged_candidate) tuples.
    """
    if not candidates:
        return []

    # Sort by start time
    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda x: x[1].start)

    merged: list[tuple[list[int], AdCandidate]] = []
    current_indices = [indexed[0][0]]
    current = indexed[0][1]

    for orig_idx, candidate in indexed[1:]:
        # Check if this candidate is close to the current merged one
        if candidate.start <= current.end + threshold:
            # Merge them
            current_indices.append(orig_idx)
            combined_indices = list(set(current.segment_indices + candidate.segment_indices))
            combined_indices.sort()
            combined_triggers = list(dict.fromkeys(current.trigger_keywords + candidate.trigger_keywords))

            # Combine sponsor tracking from merged candidates
            combined_sponsors_found = list(dict.fromkeys(
                current.sponsors_found + candidate.sponsors_found
            ))
            combined_sponsors_missing = list(dict.fromkeys(
                current.sponsors_missing + candidate.sponsors_missing
            ))

            current = AdCandidate(
                start=current.start,
                end=max(current.end, candidate.end),
                segment_indices=combined_indices,
                trigger_keywords=combined_triggers,
                heuristic_score=max(current.heuristic_score, candidate.heuristic_score),
                sponsors_found=combined_sponsors_found,
                sponsors_missing=combined_sponsors_missing,
            )
        else:
            merged.append((current_indices, current))
            current_indices = [orig_idx]
            current = candidate

    merged.append((current_indices, current))
    return merged


class OpenAIClient(AdLLMClient):
    """OpenAI-based LLM client for ad refinement."""

    def __init__(
        self, api_key: str, model: str = "gpt-4.1-mini", base_url: str | None = None,
        *, require_success: bool = False,
    ) -> None:
        """Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key.
            model: Model to use.
            base_url: Optional custom base URL for API.
            require_success: Reject API errors instead of optional heuristic fallback.
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.require_success = require_success
        self.completed_decisions = 0

    def refine_candidates(
        self,
        segments: list[TranscriptSegment],
        candidates: list[AdCandidate],
        config: Config,
        sponsors: SponsorInfo | None = None,
        podcast_title: str | None = None,
    ) -> list[AdSpan]:
        """Refine candidates using OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is required. Run: pip install openai")

        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as error:
            safe_error = redact_secrets(str(error), extra_secrets=[self.api_key])
            raise InvalidRefinementResponse(f"OpenAI refinement client unavailable: {safe_error}") from None

        # Merge nearby candidates first
        merged_candidates = merge_nearby_candidates(candidates, MERGE_THRESHOLD)

        spans: list[AdSpan] = []

        for candidate_indices, merged_candidate in merged_candidates:
            # Get segments with wider context window
            context_start = max(0, merged_candidate.start - CONTEXT_BEFORE)
            context_end = merged_candidate.end + CONTEXT_AFTER

            context_segments = [
                s for s in segments
                if s.end >= context_start and s.start <= context_end
            ]
            context_segments.sort(key=lambda s: s.start)

            # Build prompt with the candidate's transcript and context
            prompt = self._build_prompt(
                context_segments, merged_candidate, context_start, context_end,
                sponsors=sponsors, podcast_title=podcast_title
            )

            try:
                # Build API call params - some models have different requirements
                api_params = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "max_completion_tokens": 800,
                }
                # GPT-5 models are reasoning models - need reasoning_effort, no temperature
                if self.model.startswith("gpt-5"):
                    api_params["reasoning_effort"] = "low"
                else:
                    api_params["temperature"] = 0.1
                response = client.chat.completions.create(**api_params)

                try:
                    result = TranscriptRefinementResponse.model_validate(
                        self._parse_response(completed_openai_text(response))
                    )
                    if not context_segments:
                        raise ValueError("No transcript context for refinement")
                    for ad in result.ads:
                        if ad.start_line is not None:
                            if ad.start_line >= len(context_segments) or ad.end_line >= len(context_segments):
                                raise ValueError("Transcript line index is outside the supplied context")
                            ad_start = context_segments[ad.start_line].start
                            ad_end = context_segments[ad.end_line].end
                        else:
                            # Preserve legacy timestamps while requiring overlap
                            # with actual supplied transcript evidence.
                            ad_start = max(context_segments[0].start, ad.ad_start)
                            ad_end = min(max(segment.end for segment in context_segments), ad.ad_end)
                        if not math.isfinite(ad_start) or not math.isfinite(ad_end) or ad_start >= ad_end:
                            raise ValueError("Transcript detection does not overlap valid transcript context")
                        spans.append(AdSpan(
                            start=ad_start, end=ad_end, confidence=ad.confidence,
                            reason=ad.reason, candidate_indices=candidate_indices,
                        ))
                    self.completed_decisions += 1
                except Exception as error:
                    safe_error = redact_secrets(str(error), extra_secrets=[self.api_key])
                    raise InvalidRefinementResponse(f"Invalid OpenAI transcript decision: {safe_error}") from None

            except InvalidRefinementResponse:
                raise
            except Exception as e:
                safe_error = redact_secrets(str(e), extra_secrets=[self.api_key])
                if self.require_success:
                    failure = classify_provider_error(e, (self.api_key,))
                    if failure.global_failure:
                        raise ProviderOutageError("openai", failure) from None
                    raise InvalidRefinementResponse(f"Required OpenAI refinement failed: {safe_error}") from None
                # Fall back to heuristic for this candidate
                spans.append(
                    AdSpan(
                        start=merged_candidate.start,
                        end=merged_candidate.end,
                        confidence=merged_candidate.heuristic_score * 0.6,
                        reason=f"heuristic_fallback (LLM error: {safe_error[:50]})",
                        candidate_indices=candidate_indices,
                    )
                )

        # Hunt mode: search for missing sponsors in gaps
        if sponsors and candidates:
            # Collect missing sponsors from all candidates
            missing = set()
            for c in candidates:
                missing.update(c.sponsors_missing)

            if missing:
                hunt_spans = self._hunt_missing_sponsors(
                    segments, spans, sponsors, missing, config, client
                )
                spans.extend(hunt_spans)

        # Merge overlapping spans
        return self._merge_overlapping_spans(spans)

    def _build_prompt(
        self,
        context_segments: list[TranscriptSegment],
        candidate: AdCandidate,
        context_start: float,
        context_end: float,
        sponsors: SponsorInfo | None = None,
        podcast_title: str | None = None,
    ) -> str:
        """Build the prompt for LLM analysis.

        Uses LINE NUMBERS instead of timestamps to reduce prompt size.
        The LLM returns line numbers which we map back to timestamps.
        """
        # Format transcript with LINE numbers (much smaller than timestamps)
        transcript_lines = []
        for i, s in enumerate(context_segments):
            # Mark segments that are in the flagged region
            in_flagged = s.start >= candidate.start and s.end <= candidate.end
            marker = "*" if in_flagged else " "
            transcript_lines.append(f"{marker}[LINE {i}] {s.text}")

        transcript_text = "\n".join(transcript_lines)

        # Build podcast context if available
        podcast_context = ""
        if podcast_title:
            podcast_context = f"""
Current podcast: {podcast_title}
(Use this to distinguish THIS podcast's content from OTHER shows or network promos)
"""

        # Build sponsor context if available
        sponsor_context = ""
        if sponsors and sponsors.sponsors:
            sponsor_lines = []
            for s in sponsors.sponsors:
                parts = [s.name]
                if s.url:
                    parts.append(f"({s.url})")
                if s.code:
                    parts.append(f"code: {s.code}")
                sponsor_lines.append(f"- {' '.join(parts)}")

            sponsor_context = f"""
Known sponsors from show notes:
{chr(10).join(sponsor_lines)}
"""

        # Check if this is a pre-roll or outro region (special regions)
        is_pre_roll_region = "pre_roll_region" in candidate.trigger_keywords
        is_outro_region = "outro_region" in candidate.trigger_keywords

        special_instruction = ""
        if is_pre_roll_region:
            special_instruction = """
NOTE: This is the PRE-ROLL REGION at the start of the episode. Look for:
- HOUSE ADS: Promotions for OTHER shows, network content, or sister podcasts
- Example patterns: "check out my other show", "host of [Other Show]", "on the Ringer Podcast Network"
- Introductions to OTHER content before the main episode begins
- IGNORE: Introductions to THIS podcast itself (welcome messages, episode descriptions)
- COMPARE: Is this promoting THIS podcast or SOMETHING ELSE? Only flag if promoting OTHER content.
"""
        elif is_outro_region:
            special_instruction = """
NOTE: This is the OUTRO REGION at the end of the episode. Look for:
- Dynamically inserted ads that appear AFTER the hosts sign off
- Product pitches without explicit "brought to you by" introductions
- Abrupt topic changes from show content to promotional material
- Multiple back-to-back ads are common here
"""

        return f"""Analyze this transcript for PAID SPONSOR ADVERTISEMENTS.

IMPORTANT: Identify ads by LINE NUMBER, not timestamp. I will map line numbers to timestamps.
{podcast_context}
Transcript (lines marked * were flagged by heuristics):
{transcript_text}
{sponsor_context}{special_instruction}
INSTRUCTIONS:
1. Find scripted sponsor reads - look for:
   - Explicit intros: "brought to you by", "sponsored by", "today's sponsor is"
   - Dynamically inserted ads: product pitches with URLs, promo codes, or calls to action
   - Post-roll ads: promotional content appearing after hosts say goodbye
2. Report the LINE NUMBER where each ad starts and ends
3. Include the first/last few words as verification
4. If multiple sponsors are back-to-back, report them as separate ads

Respond with JSON:
{{
  "ads": [
    {{
      "start_line": <LINE number where ad starts>,
      "start_phrase": "<first few words of the ad>",
      "end_line": <LINE number where ad ends>,
      "end_phrase": "<last few words of the ad>",
      "sponsors": ["<sponsor name>"],
      "confidence": 0.0-1.0,
      "reason": "<brief explanation>"
    }}
  ]
}}

If NO ads found: {{"ads": [], "reason": "<explanation>"}}"""

    def _parse_response(self, content: str) -> dict:
        """Parse LLM response JSON."""
        content = content.strip()

        # Retain the conventional complete JSON fence, but never salvage a
        # partial object from an unfinished response or surrounding content.
        if content.startswith("```") and content.endswith("```"):
            content = content[3:-3].strip()
            if content.startswith("json"):
                content = content[4:].strip()

        result = json.loads(content)
        if not isinstance(result, dict):
            raise ValueError("OpenAI refinement must return a JSON object")
        return result

    def _hunt_missing_sponsors(
        self,
        segments: list[TranscriptSegment],
        existing_spans: list[AdSpan],
        sponsors: SponsorInfo,
        missing: set[str],
        config: Config,
        client: Any,
    ) -> list[AdSpan]:
        """Search gaps in coverage for missing sponsors."""
        if not segments:
            return []

        duration = segments[-1].end

        # Find gaps > 10 minutes between ads
        gaps: list[tuple[float, float]] = []
        sorted_spans = sorted(existing_spans, key=lambda s: s.start)

        prev_end = 0.0
        for span in sorted_spans:
            if span.start - prev_end > 600:  # 10 minute gap
                gaps.append((prev_end, span.start))
            prev_end = max(prev_end, span.end)

        # Check final gap
        if duration - prev_end > 600:
            gaps.append((prev_end, duration))

        # Search up to 3 gaps
        hunt_spans: list[AdSpan] = []
        for gap_start, gap_end in gaps[:3]:
            midpoint = (gap_start + gap_end) / 2
            window_start = max(0, midpoint - 120)  # 2 min before
            window_end = min(duration, midpoint + 120)  # 2 min after

            window_segments = [
                s for s in segments
                if s.end >= window_start and s.start <= window_end
            ]

            if not window_segments:
                continue

            # Build hunt prompt
            prompt = self._build_hunt_prompt(
                window_segments, list(missing), window_start, window_end
            )

            try:
                # Build API call params - some models have different requirements
                hunt_params = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "max_completion_tokens": 500,
                }
                # GPT-5 models are reasoning models - need reasoning_effort, no temperature
                if self.model.startswith("gpt-5"):
                    hunt_params["reasoning_effort"] = "low"
                else:
                    hunt_params["temperature"] = 0.1
                response = client.chat.completions.create(**hunt_params)

                result = self._parse_response(completed_openai_text(response))

                for ad in result.get("ads", []):
                    if ad.get("is_ad") is True:
                        validated = TranscriptAdDetection.model_validate(ad)
                        start = max(window_start, validated.ad_start)
                        end = min(window_end, validated.ad_end)
                        if start >= end:
                            raise ValueError("Hunt detection does not overlap its transcript window")
                        hunt_spans.append(AdSpan(
                            start=start,
                            end=end,
                            confidence=validated.confidence if "confidence" in validated.model_fields_set else 0.6,
                            reason=f"hunt_mode: {validated.reason if 'reason' in validated.model_fields_set else 'found missing sponsor'}",
                            candidate_indices=[],
                        ))
            except Exception:
                continue

        return hunt_spans

    def _build_hunt_prompt(
        self,
        segments: list[TranscriptSegment],
        missing_sponsors: list[str],
        window_start: float,
        window_end: float,
    ) -> str:
        """Build prompt for hunting missing sponsors."""
        transcript_lines = []
        for s in segments:
            mins = int(s.start // 60)
            secs = s.start % 60
            transcript_lines.append(f"[{mins:02d}:{secs:05.2f}] {s.text}")

        return f'''We're looking for ads that our initial scan may have missed.

Missing sponsors (from show notes): {', '.join(missing_sponsors)}

Search window: {window_start:.0f}s to {window_end:.0f}s

Transcript:
{chr(10).join(transcript_lines)}

Look carefully for any mention of these sponsors or their products. They may be subtle.

Respond with JSON:
{{
  "ads": [
    {{"is_ad": true, "ad_start": <seconds>, "ad_end": <seconds>, "confidence": 0.0-1.0, "reason": "found [sponsor name]"}}
  ]
}}

If no ads found: {{"ads": [], "reason": "no sponsor mentions found"}}'''

    def _merge_overlapping_spans(self, spans: list[AdSpan]) -> list[AdSpan]:
        """Merge overlapping ad spans."""
        if not spans:
            return []

        # Sort by start time
        sorted_spans = sorted(spans, key=lambda s: s.start)
        merged: list[AdSpan] = [sorted_spans[0]]

        for span in sorted_spans[1:]:
            last = merged[-1]
            # Check for overlap (with small tolerance)
            if span.start <= last.end + 2.0:
                # Merge spans
                merged[-1] = AdSpan(
                    start=last.start,
                    end=max(last.end, span.end),
                    confidence=max(last.confidence, span.confidence),
                    reason=f"{last.reason}; {span.reason}",
                    candidate_indices=list(set(last.candidate_indices + span.candidate_indices)),
                )
            else:
                merged.append(span)

        return merged

    def merge_and_refine(
        self,
        prompt_data: dict,
        config: Config,
    ) -> tuple[list[AdSpan], dict]:
        """Merge candidates from multiple detection methods using LLM.

        Used by parallel detection to intelligently merge Gemini and keyword candidates.

        Args:
            prompt_data: Dict with gemini_candidates, keyword_candidates, transcript_context,
                        sponsors, and duration.
            config: Configuration object.

        Returns:
            Tuple of (ad_spans, usage_dict) for cost tracking.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is required. Run: pip install openai")

        duration = prompt_data.get("duration", 0)

        system_prompt = """You are refining ad detection candidates from two sources:

1. Keyword candidates: Detected via transcript keyword matching. These are GROUNDED in the
   transcript text - the keywords triggered at these specific timestamps. TRUST these timestamps
   as they come directly from verified transcript text.

2. Gemini candidates: Detected via audio analysis. These MAY have incorrect timestamps -
   Gemini sometimes hallucinates timestamps, especially for long episodes. You MUST VERIFY
   each Gemini candidate against the transcript context before including it.

VERIFICATION RULES:
- For EACH Gemini candidate, check if the transcript context around that timestamp contains
  sponsor names, ad keywords, or promotional language
- If a Gemini timestamp has NO supporting evidence in the transcript, REJECT it
- When Gemini and Keyword candidates conflict, PREFER the Keyword candidate's timestamps
  (since keywords are grounded in actual transcript text)

Your tasks:
1. VERIFY each Gemini candidate has transcript evidence (reject if not)
2. MERGE overlapping or adjacent ad candidates (keep ad blocks together)
3. REFINE boundaries using transcript context (find exact transition points)
4. If you discover an ad in the transcript NOT covered by candidates, you MAY add it
5. OUTPUT final ad spans

PRE-ROLL AD DETECTION (CRITICAL):
The first 0-120 seconds often contain DYNAMICALLY INSERTED pre-roll ads. These are:
- Multiple back-to-back sponsor ads (Peloton, Corona, PayPal, etc.) inserted by the ad network
- Often followed by a network bumper ("Podcasts you love", "This is TWiT", etc.)
- End when the ACTUAL SHOW starts (host greetings, episode intro, guest introduction)

For pre-roll regions:
- Keep ALL back-to-back ads as ONE block (do NOT split by sponsor)
- Find the END by looking for: "Episode [number]", "I'm [host name]", "Hello and welcome", etc.
- The show typically starts after all ads AND network bumpers complete
- If keyword candidate says 0-90s, EXTEND to find the actual show start, don't truncate

BOUNDARY RULES:
- TRUST keyword candidate boundaries as minimum range
- EXTEND (don't shrink) when you find more ad content beyond the candidate
- Look for actual show content to determine where ads END
- All timestamps must be within [0, {duration}] seconds
- Each ad must have start < end
- Output ads sorted by start time, non-overlapping

Output JSON:
{{
    "ads": [
        {{
            "start": <seconds>,
            "end": <seconds>,
            "confidence": 0.0-1.0,
            "reason": "<brief explanation>",
            "sources": ["gemini", "keywords"]  // or ["llm_new"] if discovered
        }}
    ]
}}""".format(duration=duration)

        # Build efficient context
        transcript_context = prompt_data.get("transcript_context", "")

        # Truncate context if too long (avoid hitting token limits)
        MAX_CONTEXT_CHARS = 15000
        if len(transcript_context) > MAX_CONTEXT_CHARS:
            transcript_context = transcript_context[:MAX_CONTEXT_CHARS] + "\n[...truncated...]"

        user_prompt = f"""Episode duration: {duration:.0f} seconds

GEMINI CANDIDATES (from audio analysis):
{json.dumps(prompt_data.get("gemini_candidates", []), indent=2)}

KEYWORD CANDIDATES (from transcript matching):
{json.dumps(prompt_data.get("keyword_candidates", []), indent=2)}

TRANSCRIPT CONTEXT:
{transcript_context}

KNOWN SPONSORS: {prompt_data.get("sponsors", [])}

Analyze the candidates and transcript to produce refined ad boundaries."""

        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.chat.completions.create(
                model=config.llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            result = MergedDetectionResponse.model_validate_json(
                completed_openai_text(response)
            )

            # Extract usage for cost tracking
            usage = {
                "provider": "openai",
                "model": config.llm.model,
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            # Validate and filter output
            validated_ads = []
            for ad in result.ads:
                # Clamp to valid range
                start = max(0, min(ad.start, duration))
                end = max(0, min(ad.end, duration))

                # Skip invalid spans
                if start >= end:
                    raise ValueError("Merged detection does not overlap the episode")

                # Default sources to ["llm_new"] if LLM omitted them
                sources = ad.sources
                if not sources:
                    sources = ["llm_new"]

                validated_ads.append(AdSpan(
                    start=start,
                    end=end,
                    confidence=ad.confidence,
                    reason=ad.reason,
                    sources=sources,
                    candidate_indices=[],
                ))

            # Sort by start time
            validated_ads.sort(key=lambda x: x.start)

            # Enforce non-overlapping: resolve any overlapping spans
            if len(validated_ads) > 1:
                validated_ads = self._enforce_non_overlapping(validated_ads)

            return validated_ads, usage

        except Exception as e:
            failure = classify_provider_error(e, (self.api_key,))
            if failure.global_failure:
                raise ProviderOutageError("openai", failure) from None
            message = redact_secrets(str(e), extra_secrets=[self.api_key])
            print(f"LLM merge failed: {message}")
            # The required worker stage fails closed; suppress raw SDK/schema
            # exception context, which can contain credentials or input excerpts.
            raise ValueError(f"LLM merge failed: {message}") from None

    def _enforce_non_overlapping(self, ads: list[AdSpan]) -> list[AdSpan]:
        """Post-process to resolve any overlapping spans (safety net for LLM output).

        Strategy: TRIM instead of MERGE to preserve distinct sponsors.
        - If two spans overlap, trim the later one's start to the earlier one's end
        - If trimming would make the span too short (<5s), drop the lower-confidence one
        """
        if not ads:
            return ads

        MIN_SPAN_DURATION = 5.0  # Minimum ad duration after trimming

        result = [ads[0]]
        for ad in ads[1:]:
            prev = result[-1]
            if ad.start < prev.end:  # Overlap detected
                overlap = prev.end - ad.start
                remaining_duration = ad.end - prev.end

                if remaining_duration >= MIN_SPAN_DURATION:
                    # Trim: move later span's start to after previous span's end
                    trimmed = AdSpan(
                        start=prev.end,
                        end=ad.end,
                        confidence=ad.confidence,
                        reason=f"{ad.reason} (trimmed {overlap:.1f}s overlap)",
                        sources=ad.sources,
                        candidate_indices=[],
                    )
                    result.append(trimmed)
                else:
                    # Trimming would make span too short - keep higher confidence
                    if ad.confidence > prev.confidence:
                        # Replace previous with current (trim current's end)
                        trimmed = AdSpan(
                            start=ad.start,
                            end=ad.end,
                            confidence=ad.confidence,
                            reason=ad.reason,
                            sources=ad.sources,
                            candidate_indices=[],
                        )
                        # Trim previous span to not overlap
                        prev_trimmed = AdSpan(
                            start=prev.start,
                            end=ad.start,
                            confidence=prev.confidence,
                            reason=f"{prev.reason} (trimmed)",
                            sources=prev.sources,
                            candidate_indices=[],
                        )
                        if prev_trimmed.end - prev_trimmed.start >= MIN_SPAN_DURATION:
                            result[-1] = prev_trimmed
                            result.append(trimmed)
                        else:
                            # Previous too short after trim, drop it and use current
                            result[-1] = trimmed
                    # else: keep previous, drop current (lower confidence)
            else:
                result.append(ad)
        return result


def create_llm_client(config: Config, *, require_success: bool = False) -> AdLLMClient:
    """Create an LLM client based on configuration.

    Args:
        config: Configuration object.
        require_success: Disable optional API-error fallback for required worker decisions.

    Returns:
        An AdLLMClient implementation.
    """
    provider = config.llm.provider.lower()

    if provider == "none":
        return HeuristicOnlyClient()

    if provider == "openai":
        api_key = config.llm.api_key
        if not api_key:
            raise ValueError(
                f"OpenAI API key not found. Set the {config.llm.api_key_env} environment variable."
            )
        return OpenAIClient(
            api_key=api_key,
            model=config.llm.model,
            base_url=config.llm.base_url,
            require_success=require_success,
        )

    raise ValueError(f"Unknown LLM provider: {provider}. Supported: none, openai")
