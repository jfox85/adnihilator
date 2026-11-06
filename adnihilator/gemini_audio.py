"""Gemini audio-based ad detection client.

Uses Gemini (default: Gemini 3.5 Flash-Lite) to detect advertisements
directly from audio via the google-genai SDK's generateContent API,
without requiring transcription first.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from .gemini_common import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_THINKING_LEVEL,
    build_generate_config,
    create_genai_client,
    extract_usage,
    file_state_name,
    validate_detections,
)
from .models import AdSpan, SponsorInfo
from .provider_errors import classify_provider_error, describe_provider_failure, redact_secrets

logger = logging.getLogger(__name__)


class GeminiAudioClient:
    """Client for Gemini audio-based ad detection."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        thinking_level: str = DEFAULT_THINKING_LEVEL,
    ):
        """Initialize the Gemini client.

        Args:
            api_key: Gemini API key
            model: Model name (default: gemini-3.5-flash-lite)
            thinking_level: Gemini 3+ thinking level (ignored for older models)
        """
        self.api_key = api_key
        self.model = model
        self.thinking_level = thinking_level
        self._client: Any = None  # Lazy SDK client

    def _get_client(self) -> Any:
        """Lazily create the google-genai client."""
        if self._client is None:
            self._client = create_genai_client(self.api_key)
        return self._client

    def detect_ads(
        self,
        audio_path: Path,
        podcast_title: Optional[str] = None,
        duration: Optional[float] = None,
        sponsors: Optional[SponsorInfo] = None,
    ) -> tuple[list[AdSpan], dict]:
        """Detect ads in audio file using Gemini.

        Args:
            audio_path: Path to audio file (MP3, WAV, etc.)
            podcast_title: Optional podcast title for context
            duration: Audio duration in seconds (for cost calculation)
            sponsors: Optional sponsor info extracted from episode description

        Returns:
            Tuple of (list of AdSpan objects, usage stats dict)

        Raises:
            ValueError: If response parsing fails or API error occurs
        """
        logger.info(f"Uploading audio to Gemini: {audio_path}")
        upload_start = time.time()

        try:
            client = self._get_client()

            # Upload audio to Gemini Files API
            audio_file = client.files.upload(file=str(audio_path))
            upload_time = time.time() - upload_start
            logger.info(f"Upload complete in {upload_time:.2f}s")

            # Wait for file to be processed
            max_wait = 120  # seconds
            start = time.time()
            while file_state_name(audio_file) == "PROCESSING":
                if time.time() - start > max_wait:
                    raise ValueError("Gemini file upload timeout after 120s")
                time.sleep(2)
                audio_file = client.files.get(name=audio_file.name)
                logger.debug(f"File state: {file_state_name(audio_file)}")

            if file_state_name(audio_file) == "FAILED":
                raise ValueError(f"Gemini file upload failed: {audio_file.state}")

            # Build prompt
            prompt = self._build_detection_prompt(podcast_title, sponsors)

            # Log sponsor info for debugging
            if sponsors and sponsors.sponsors:
                sponsor_names = [s.name for s in sponsors.sponsors]
                logger.info(f"Passing {len(sponsor_names)} sponsors to Gemini: {sponsor_names}")
            else:
                logger.info("No sponsors to pass to Gemini")

            # Call Gemini
            logger.info(f"Sending audio to Gemini {self.model} for analysis...")
            analysis_start = time.time()
            # Single user turn (prompt + audio); no prefilled model turn and
            # no deprecated sampling parameters.
            response = client.models.generate_content(
                model=self.model,
                contents=[prompt, audio_file],
                config=build_generate_config(self.model, self.thinking_level),
            )
            analysis_time = time.time() - analysis_start
            logger.info(f"Analysis complete in {analysis_time:.2f}s")

            # Validate response
            if not response:
                raise ValueError("Empty response from Gemini API")

            response_text = getattr(response, "text", None)
            if not response_text:
                # Check for safety block / non-STOP finish
                candidates = getattr(response, "candidates", None)
                if candidates:
                    finish_reason = getattr(candidates[0], "finish_reason", None)
                    if finish_reason is not None:
                        raise ValueError(f"Gemini response blocked: {finish_reason}")
                raise ValueError("Empty text in Gemini response")

            # Parse JSON response
            result = self._parse_response(response_text)

            # Extract usage stats
            if getattr(response, "usage_metadata", None) is None:
                logger.warning("Missing usage_metadata in Gemini response")
            usage = {
                "provider": "gemini",
                "model": self.model,
                **extract_usage(response),
                "audio_duration_seconds": duration or 0,
            }

            # Convert to AdSpan objects
            ad_spans = []
            for ad in validate_detections(result):
                start_time = ad.start_time
                end_time = ad.end_time
                sponsor = ad.sponsor
                ad_type = ad.ad_type
                reason = ad.reason

                ad_spans.append(
                    AdSpan(
                        start=start_time,
                        end=end_time,
                        confidence=ad.confidence if "confidence" in ad.model_fields_set else 0.95,
                        reason=f"Gemini: {sponsor} ({ad_type or 'unknown'}) - {reason}",
                        candidate_indices=[],  # Not applicable for Gemini
                        ad_type=ad_type,
                    )
                )

            logger.info(f"Detected {len(ad_spans)} ads via Gemini")
            return ad_spans, usage

        except Exception as e:
            # Re-raise with context
            failure = classify_provider_error(e, (self.api_key,))
            raise ValueError(f"Gemini detection failed: {describe_provider_failure(failure)}") from None

    def _build_detection_prompt(
        self,
        podcast_title: Optional[str] = None,
        sponsors: Optional[SponsorInfo] = None,
    ) -> str:
        """Build the ad detection prompt.

        Args:
            podcast_title: Optional podcast title for context
            sponsors: Optional sponsor info extracted from episode description

        Returns:
            Prompt string for Gemini
        """
        title_context = ""
        if podcast_title:
            title_context = f'Podcast: "{podcast_title}"\n'

        # Build sponsor section if we have sponsor info
        sponsor_section = ""
        if sponsors and sponsors.sponsors:
            sponsor_lines = []
            for s in sponsors.sponsors:
                if s.url:
                    sponsor_lines.append(f"- {s.name} ({s.url})")
                elif s.code:
                    sponsor_lines.append(f"- {s.name} (code: {s.code})")
                else:
                    sponsor_lines.append(f"- {s.name}")

            sponsor_section = f"""
KNOWN SPONSORS FOR THIS EPISODE:
{chr(10).join(sponsor_lines)}

You MUST find the ad read for EACH sponsor listed above.
If you cannot find an ad for a listed sponsor, note which sponsors were not found.
"""

        return f"""You are analyzing a podcast episode to identify ALL advertisement segments.
{title_context}
{sponsor_section}
TYPES OF ADS TO DETECT:

1. DYNAMICALLY INSERTED ADS (easy to detect by audio):
   - Different voices from podcast hosts (professional announcers)
   - Different audio production quality (music, sound effects, studio polish)
   - Standard 15/30/60 second spot lengths
   - MAY BE IN A DIFFERENT LANGUAGE than the podcast (Spanish, etc.) - detect these!
   - Often appear at BEGINNING (pre-roll), MIDDLE (mid-roll), or END (post-roll)
   - Mark these as ad_type: "dynamic_insertion"

2. HOST-READ SPONSOR SEGMENTS:
   - Podcast host reads an ad script
   - Phrases like "brought to you by", "thanks to our sponsor"
   - Mentions website URLs, promo codes, special offers
   - Mark these as ad_type: "host_read"

3. NETWORK BUMPERS:
   - Short audio stings like "This is TWiT", "Podcasts you love"
   - Mark these as ad_type: "network_bumper"

AD LOCATIONS - CHECK ALL OF THESE:
- PRE-ROLL: First 0-120 seconds (dynamic ads before show starts)
- MID-ROLL: Throughout the middle - dynamic ads can appear ANYWHERE, often around 25-35%, 50%, 75% marks
- POST-ROLL: Final 2-3 minutes (dynamic ads after show ends)

CRITICAL: Dynamic mid-roll ads are common! Listen to the ENTIRE episode, not just beginning/end.

FINDING BOUNDARIES:
- START: When the ad audio/voice begins
- END: After the final URL/promo code mention or call-to-action
- Include the FULL ad through the transition back to content

For each ad found, provide:
- start_time: timestamp in seconds when the ad begins
- end_time: timestamp in seconds when the ad ends
- sponsor: name of the sponsor/brand (or "Network Promo" for bumpers)
- ad_type: "dynamic_insertion", "host_read", or "network_bumper"
- confidence: 0.0-1.0 how confident you are this is an ad
- reason: brief explanation

Respond with JSON:
{{
  "ads": [
    {{
      "start_time": <seconds>,
      "end_time": <seconds>,
      "sponsor": "<brand name>",
      "ad_type": "<type>",
      "confidence": 0.0-1.0,
      "reason": "<explanation>"
    }}
  ],
  "sponsors_not_found": ["<sponsor name if not detected>"]
}}

If no ads found: {{"ads": [], "reason": "<explanation>"}}"""

    def _parse_response(self, text: str) -> dict:
        """Parse Gemini response text as JSON.

        Args:
            text: Response text from Gemini

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If JSON parsing fails
        """
        content = text.strip()

        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            # Generic code block
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1].strip()
                # Remove language identifier if present (e.g., "json\n{...")
                if content.startswith("json"):
                    content = content[4:].strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            logger.error(f"Response text: {redact_secrets(text, (self.api_key,))[:500]}...")
            raise ValueError(f"Failed to parse Gemini response as JSON: {e}")


def calculate_gemini_cost(
    input_tokens: int,
    output_tokens: int,
    audio_duration_seconds: float,
) -> float:
    """Calculate legacy Gemini API cost estimate in USD.

    Legacy helper (not used by the worker, which prices per model in
    ``WorkerDaemon._calculate_llm_cost``). Pricing (as of Dec 2024):
    - Text input: $0.075 per 1M tokens
    - Text output: $0.30 per 1M tokens
    - Audio: $0.000025 per second

    Args:
        input_tokens: Number of input tokens (text prompt)
        output_tokens: Number of output tokens (response)
        audio_duration_seconds: Duration of audio in seconds

    Returns:
        Total cost in USD
    """
    # Token costs
    input_cost = input_tokens * 0.075 / 1_000_000
    output_cost = output_tokens * 0.30 / 1_000_000

    # Audio cost
    audio_cost = audio_duration_seconds * 0.000025

    return input_cost + output_cost + audio_cost
