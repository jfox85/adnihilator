"""Gemini audio-based ad detection client.

Uses Gemini 2.0 Flash to detect advertisements directly from audio,
without requiring transcription first. This is faster and more accurate
than the Whisper + LLM pipeline, but costs more per episode (~$0.087).
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

from .models import AdSpan

logger = logging.getLogger(__name__)


class GeminiAudioClient:
    """Client for Gemini audio-based ad detection."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        """Initialize the Gemini client.

        Args:
            api_key: Gemini API key
            model: Model name (default: gemini-2.0-flash-exp)
        """
        self.api_key = api_key
        self.model = model
        self._genai = None  # Lazy import

    def _get_genai(self):
        """Lazily import and configure google.generativeai."""
        if self._genai is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self._genai = genai
            except ImportError:
                raise ImportError(
                    "google-generativeai is required for Gemini detection. "
                    "Run: pip install google-generativeai"
                )
        return self._genai

    def detect_ads(
        self,
        audio_path: Path,
        podcast_title: Optional[str] = None,
        duration: Optional[float] = None,
    ) -> tuple[list[AdSpan], dict]:
        """Detect ads in audio file using Gemini.

        Args:
            audio_path: Path to audio file (MP3, WAV, etc.)
            podcast_title: Optional podcast title for context
            duration: Audio duration in seconds (for cost calculation)

        Returns:
            Tuple of (list of AdSpan objects, usage stats dict)

        Raises:
            ValueError: If response parsing fails or API error occurs
        """
        genai = self._get_genai()

        logger.info(f"Uploading audio to Gemini: {audio_path}")
        upload_start = time.time()

        try:
            # Upload audio to Gemini Files API
            audio_file = genai.upload_file(path=str(audio_path))
            upload_time = time.time() - upload_start
            logger.info(f"Upload complete in {upload_time:.2f}s, URI: {audio_file.uri}")

            # Wait for file to be processed
            max_wait = 120  # seconds
            start = time.time()
            while audio_file.state.name == "PROCESSING":
                if time.time() - start > max_wait:
                    raise ValueError("Gemini file upload timeout after 120s")
                time.sleep(2)
                audio_file = genai.get_file(audio_file.name)
                logger.debug(f"File state: {audio_file.state.name}")

            if audio_file.state.name == "FAILED":
                raise ValueError(f"Gemini file upload failed: {audio_file.state}")

            # Build prompt
            prompt = self._build_detection_prompt(podcast_title)

            # Call Gemini
            logger.info(f"Sending audio to Gemini {self.model} for analysis...")
            analysis_start = time.time()
            model = genai.GenerativeModel(self.model)
            response = model.generate_content([prompt, audio_file])
            analysis_time = time.time() - analysis_start
            logger.info(f"Analysis complete in {analysis_time:.2f}s")

            # Validate response
            if not response:
                raise ValueError("Empty response from Gemini API")

            if not hasattr(response, "text") or not response.text:
                # Check for safety block
                if hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, "finish_reason"):
                        raise ValueError(
                            f"Gemini response blocked: {candidate.finish_reason}"
                        )
                raise ValueError("Empty text in Gemini response")

            # Parse JSON response
            result = self._parse_response(response.text)

            # Extract usage stats
            if not hasattr(response, "usage_metadata"):
                logger.warning("Missing usage_metadata in Gemini response")
                usage = {
                    "provider": "gemini",
                    "model": self.model,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "audio_duration_seconds": duration or 0,
                }
            else:
                usage = {
                    "provider": "gemini",
                    "model": self.model,
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                    "audio_duration_seconds": duration or 0,
                }

            # Convert to AdSpan objects
            ad_spans = []
            for ad in result.get("ads", []):
                start_time = ad.get("start_time")
                end_time = ad.get("end_time")

                # Validate timestamps
                if start_time is None or end_time is None:
                    logger.warning(f"Skipping ad with missing timestamps: {ad}")
                    continue

                if start_time >= end_time:
                    logger.warning(f"Skipping ad with invalid time range: {ad}")
                    continue

                sponsor = ad.get("sponsor", "Unknown")
                language = ad.get("language", "Unknown")
                reason = ad.get("reason", "")

                ad_spans.append(
                    AdSpan(
                        start=start_time,
                        end=end_time,
                        confidence=ad.get("confidence", 0.95),
                        reason=f"Gemini: {sponsor} ({language}) - {reason}",
                        candidate_indices=[],  # Not applicable for Gemini
                    )
                )

            logger.info(f"Detected {len(ad_spans)} ads via Gemini")
            return ad_spans, usage

        except Exception as e:
            # Re-raise with context
            raise ValueError(f"Gemini detection failed: {e}") from e

    def _build_detection_prompt(self, podcast_title: Optional[str] = None) -> str:
        """Build the ad detection prompt.

        Args:
            podcast_title: Optional podcast title for context

        Returns:
            Prompt string for Gemini
        """
        title_context = ""
        if podcast_title:
            title_context = f'\nThe podcast is "{podcast_title}".\n'

        return f"""You are analyzing a podcast episode to identify all advertisement segments.
{title_context}
Look for ANY of the following:
- Sponsor reads (e.g., "This episode is brought to you by...")
- Product promotions or advertisements
- Service advertisements (therapy apps, payment apps, grocery stores, etc.)
- Mid-roll ads inserted into the content
- Pre-roll or post-roll advertisements
- Ads in ANY LANGUAGE (English, Spanish, etc.)
- Host reads that promote a product/service

IMPORTANT:
- Listen for ads in BOTH English AND Spanish (and other languages)
- Ads can appear anywhere: beginning, middle, or end of episode
- Some ads may be very short (15-30 seconds)
- Pay attention to audio cues: music changes, tone shifts, different speakers
- Watch for sponsor names like grocery stores, apps, services, etc.

Analyze the ENTIRE audio file and identify ALL advertisement segments.

For each ad found, provide:
- start_time: timestamp in seconds when the ad begins
- end_time: timestamp in seconds when the ad ends
- sponsor: name of the sponsor/brand
- language: language of the ad (English, Spanish, etc.)
- confidence: 0.0-1.0 how confident you are this is an ad
- reason: brief explanation of why this is an ad (include audio cues if relevant)

Respond with JSON:
{{
  "ads": [
    {{
      "start_time": <seconds>,
      "end_time": <seconds>,
      "sponsor": "<brand name>",
      "language": "<language>",
      "confidence": 0.0-1.0,
      "reason": "<explanation>"
    }}
  ]
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
            logger.error(f"Response text: {text[:500]}...")
            raise ValueError(f"Failed to parse Gemini response as JSON: {e}")


def calculate_gemini_cost(
    input_tokens: int,
    output_tokens: int,
    audio_duration_seconds: float,
) -> float:
    """Calculate Gemini API cost in USD.

    Pricing (as of Dec 2024):
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
