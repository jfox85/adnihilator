"""Chunked Gemini audio-based ad detection.

Splits long podcasts into chunks for more reliable detection,
then merges results back together.
"""

import concurrent.futures
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .models import AdSpan, SponsorInfo

logger = logging.getLogger(__name__)

# Chunk settings
CHUNK_DURATION = 300  # 5 minutes per chunk
MAX_PARALLEL_CHUNKS = 2  # Limit parallel Gemini calls (avoid rate limits)


def split_audio_into_chunks(
    audio_path: Path,
    chunk_duration: int = CHUNK_DURATION,
    output_dir: Optional[Path] = None,
) -> list[tuple[Path, float, float]]:
    """Split audio file into chunks.

    Args:
        audio_path: Path to the audio file
        chunk_duration: Duration of each chunk in seconds
        output_dir: Directory for chunk files (uses temp if not provided)

    Returns:
        List of (chunk_path, start_time, end_time) tuples
    """
    # Get total duration
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError(
            f"ffprobe failed on {audio_path}: {result.stderr.strip()}"
        )
    total_duration = float(result.stdout.strip())

    if output_dir is None:
        # Caller owns cleanup when no output_dir is provided
        output_dir = Path(tempfile.mkdtemp(prefix="gemini_chunks_"))
        logger.debug(f"Created temp dir for chunks (caller must clean up): {output_dir}")

    chunks = []
    start = 0.0
    chunk_num = 0

    while start < total_duration:
        end = min(start + chunk_duration, total_duration)
        chunk_path = output_dir / f"chunk_{chunk_num:03d}.mp3"

        # Extract chunk with ffmpeg
        ffmpeg_result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path),
             "-ss", str(start), "-t", str(end - start),
             "-acodec", "libmp3lame", "-q:a", "4",
             str(chunk_path)],
            capture_output=True
        )
        if ffmpeg_result.returncode != 0:
            raise ValueError(
                f"ffmpeg failed on chunk {chunk_num} ({start}-{end}s): "
                f"{ffmpeg_result.stderr.decode(errors='replace').strip()}"
            )

        chunks.append((chunk_path, start, end))
        start = end
        chunk_num += 1

    logger.info(f"Split audio into {len(chunks)} chunks of {chunk_duration}s each")
    return chunks


def detect_ads_in_chunk(
    genai,
    chunk_path: Path,
    chunk_start: float,
    chunk_end: float,
    model_name: str,
    podcast_title: Optional[str] = None,
    max_retries: int = 3,
) -> list[dict]:
    """Detect ads in a single chunk.

    Args:
        genai: Configured google.generativeai module
        chunk_path: Path to the chunk audio file
        chunk_start: Start time of this chunk in the original audio
        chunk_end: End time of this chunk in the original audio
        model_name: Gemini model name
        podcast_title: Optional podcast title for context
        max_retries: Maximum retries on rate limit errors

    Returns:
        List of ad detections with absolute timestamps
    """
    import json
    import time

    # Upload chunk with retry
    for attempt in range(max_retries):
        try:
            audio_file = genai.upload_file(path=str(chunk_path))
            break
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = 30 * (attempt + 1)
                logger.warning(f"Rate limited on upload, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

    # Wait for processing
    max_wait = 60
    start_wait = time.time()
    while audio_file.state.name == "PROCESSING":
        if time.time() - start_wait > max_wait:
            raise ValueError(f"Chunk upload timeout: {chunk_path}")
        time.sleep(2)
        audio_file = genai.get_file(audio_file.name)

    if audio_file.state.name == "FAILED":
        raise ValueError(f"Chunk upload failed: {chunk_path}")

    # Build prompt
    chunk_duration = chunk_end - chunk_start
    title_ctx = f'Podcast: "{podcast_title}"\n' if podcast_title else ""

    prompt = f"""{title_ctx}Analyze this {chunk_duration:.0f}-second audio clip for advertisements.

DETECT ONLY ACTUAL ADS - not regular podcast discussion!

AD TYPES:
1. DYNAMIC ADS - Different voice than hosts, professional production, music/effects
   - May be in ANY LANGUAGE (Spanish, English, etc.)
   - Typical length: 15-60 seconds (rarely over 2 minutes)
   - Brands like insurance, cars, tech, beverages

2. HOST-READ ADS - Host explicitly promotes a sponsor
   - MUST have sponsorship language: "brought to you by", "sponsored by", "thanks to our sponsor"
   - MUST have call-to-action: website URL, promo code, "sign up", "visit", "use code"
   - Typical length: 30-90 seconds
   - This is NOT: hosts casually mentioning products they use or discussing companies

3. NETWORK BUMPERS - Short jingles like "This is TWiT", "Podcasts you love"
   - Typical length: 5-15 seconds

CRITICAL - DO NOT FLAG THESE AS ADS:
- Hosts discussing products/services they personally use (NOT sponsored)
- Natural conversation mentioning brands without promotional intent
- Product recommendations that lack sponsorship language or call-to-action
- Educational content about companies or products

Example NOT an ad: "I've been taking vitamin D from Now Foods, it's $8 for a year supply"
Example IS an ad: "Thanks to our sponsor Now Foods! Visit nowfoods.com/podcast for 20% off"

IMPORTANT:
- Most ads are under 2 minutes. An ad over 3 minutes is very rare.
- If unsure, do NOT include it. Only flag clear advertisements.
- Every ad MUST have a specific sponsor name (not "Unknown")

Return JSON with start_time and end_time in SECONDS from clip start:
{{
  "ads": [
    {{"start_time": <seconds>, "end_time": <seconds>, "sponsor": "<brand name>", "ad_type": "dynamic_insertion|host_read|network_bumper", "confidence": 0.0-1.0, "language": "en|es|etc"}}
  ]
}}

If no ads found: {{"ads": []}}"""

    # Call Gemini with retry on rate limit
    model = genai.GenerativeModel(model_name)
    response = None
    for attempt in range(max_retries):
        try:
            response = model.generate_content([prompt, audio_file])
            break
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = 30 * (attempt + 1)
                logger.warning(f"Rate limited on generate, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

    if not response or not hasattr(response, "text") or not response.text:
        raise ValueError(f"Empty Gemini response for chunk {chunk_path}")

    # Parse response
    text = response.text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Gemini chunk response: {e}") from e

    # Convert to absolute timestamps
    ads = []
    for ad in result.get("ads", []):
        start_time = ad.get("start_time")
        end_time = ad.get("end_time")

        if start_time is None or end_time is None:
            continue

        # Convert chunk-relative to absolute timestamps
        abs_start = chunk_start + start_time
        abs_end = chunk_start + end_time

        # Clamp to chunk boundaries
        abs_start = max(chunk_start, abs_start)
        abs_end = min(chunk_end, abs_end)

        if abs_start >= abs_end:
            continue

        ads.append({
            "start": abs_start,
            "end": abs_end,
            "sponsor": ad.get("sponsor", "Unknown"),
            "ad_type": ad.get("ad_type", "dynamic_insertion"),
            "confidence": ad.get("confidence", 0.9),
            "language": ad.get("language", "unknown"),
            "chunk_start": chunk_start,  # For debugging
        })

    return ads


def merge_chunk_detections(
    all_detections: list[dict],
    merge_gap: float = 5.0,
    min_duration: float = 8.0,
) -> list[AdSpan]:
    """Merge ad detections from multiple chunks.

    Handles ads that span chunk boundaries by merging adjacent detections.

    Args:
        all_detections: List of all detections from all chunks
        merge_gap: Maximum gap (seconds) between ads to merge them
        min_duration: Minimum ad duration to keep (filters false positives)

    Returns:
        List of merged AdSpan objects
    """
    if not all_detections:
        return []

    # Sort by start time
    sorted_ads = sorted(all_detections, key=lambda x: x["start"])

    # Merge overlapping/adjacent ads
    merged = []
    current = sorted_ads[0].copy()

    for ad in sorted_ads[1:]:
        # Check if this ad should be merged with current
        # Merge if: overlapping OR within merge_gap seconds
        if ad["start"] <= current["end"] + merge_gap:
            # Extend current ad
            current["end"] = max(current["end"], ad["end"])
            current["confidence"] = max(current["confidence"], ad["confidence"])

            # Combine sponsor names if different
            if ad["sponsor"] != current["sponsor"] and ad["sponsor"] != "Unknown":
                if current["sponsor"] == "Unknown":
                    current["sponsor"] = ad["sponsor"]
                else:
                    current["sponsor"] = f"{current['sponsor']}/{ad['sponsor']}"
        else:
            # Save current and start new
            merged.append(current)
            current = ad.copy()

    merged.append(current)

    # Filter by minimum duration (removes false positives like brief mentions)
    filtered = [ad for ad in merged if (ad["end"] - ad["start"]) >= min_duration]
    logger.info(f"Filtered {len(merged) - len(filtered)} ads below {min_duration}s minimum")

    # Filter out invalid ad_types (Gemini sometimes returns things like "regular podcast discussion")
    VALID_AD_TYPES = {"dynamic_insertion", "host_read", "network_bumper"}
    before_count = len(filtered)
    filtered = [ad for ad in filtered if ad.get("ad_type") in VALID_AD_TYPES]
    if len(filtered) < before_count:
        logger.info(f"Filtered {before_count - len(filtered)} ads with invalid ad_type")

    # Filter out detections with "Unknown" sponsor - if Gemini can't identify the sponsor,
    # it's likely not a real ad (per prompt: "Every ad MUST have a specific sponsor name")
    before_count = len(filtered)
    filtered = [ad for ad in filtered if ad.get("sponsor") != "Unknown"]
    if len(filtered) < before_count:
        logger.info(f"Filtered {before_count - len(filtered)} ads with Unknown sponsor")

    # Flag suspiciously long detections (>= 4 min) for visibility in logs
    MAX_REASONABLE_AD = 240.0  # 4 minutes
    suspicious = [ad for ad in filtered if (ad["end"] - ad["start"]) >= MAX_REASONABLE_AD]
    if suspicious:
        logger.warning(f"Found {len(suspicious)} suspiciously long detections (>= {MAX_REASONABLE_AD}s)")
        for ad in suspicious:
            logger.warning(f"  {ad['start']:.0f}-{ad['end']:.0f}s ({ad['end']-ad['start']:.0f}s): {ad['sponsor']}")

    # Convert to AdSpan objects
    return [
        AdSpan(
            start=ad["start"],
            end=ad["end"],
            confidence=ad["confidence"],
            reason=f"Gemini chunked: {ad['sponsor']} ({ad.get('language', 'unknown')}) - {ad['ad_type']}",
            candidate_indices=[],
            sources=["gemini_chunked"],
            ad_type=ad["ad_type"],
        )
        for ad in filtered
    ]


class GeminiChunkedClient:
    """Client for chunked Gemini audio-based ad detection."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        self._genai = None

    def _get_genai(self):
        """Lazily import and configure google.generativeai."""
        if self._genai is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._genai = genai
            except ImportError:
                raise ImportError(
                    "google-generativeai required. Run: pip install google-generativeai"
                )
        return self._genai

    def detect_ads(
        self,
        audio_path: Path,
        podcast_title: Optional[str] = None,
        duration: Optional[float] = None,
        sponsors: Optional[SponsorInfo] = None,
        chunk_duration: int = CHUNK_DURATION,
        max_parallel: int = MAX_PARALLEL_CHUNKS,
    ) -> tuple[list[AdSpan], dict]:
        """Detect ads by processing audio in chunks.

        Args:
            audio_path: Path to audio file
            podcast_title: Optional podcast title
            duration: Audio duration (calculated if not provided)
            sponsors: Optional sponsor info (not used in chunked mode yet)
            chunk_duration: Duration of each chunk in seconds
            max_parallel: Maximum parallel Gemini calls

        Returns:
            Tuple of (list of AdSpan, usage stats dict)
        """
        import time

        genai = self._get_genai()

        logger.info(f"Starting chunked detection for {audio_path}")
        start_time = time.time()

        # Split into chunks
        with tempfile.TemporaryDirectory(prefix="gemini_chunks_") as tmpdir:
            chunks = split_audio_into_chunks(
                audio_path,
                chunk_duration=chunk_duration,
                output_dir=Path(tmpdir)
            )

            logger.info(f"Processing {len(chunks)} chunks with max {max_parallel} parallel")

            # Process chunks in parallel
            all_detections = []
            errors = []

            def process_chunk(chunk_info):
                chunk_path, chunk_start, chunk_end = chunk_info
                return detect_ads_in_chunk(
                    genai, chunk_path, chunk_start, chunk_end,
                    self.model, podcast_title
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
                futures = {executor.submit(process_chunk, c): c for c in chunks}

                for future in concurrent.futures.as_completed(futures):
                    chunk_info = futures[future]
                    try:
                        detections = future.result()
                        all_detections.extend(detections)
                        logger.info(f"Chunk {chunk_info[1]:.0f}-{chunk_info[2]:.0f}s: {len(detections)} ads")
                    except Exception as e:
                        logger.error(f"Chunk {chunk_info[1]:.0f}-{chunk_info[2]:.0f}s failed: {e}")
                        errors.append(str(e))

        total_time = time.time() - start_time
        logger.info(f"Chunked detection complete in {total_time:.1f}s, found {len(all_detections)} raw detections")

        if errors:
            raise RuntimeError(
                f"Gemini chunked detection failed for {len(errors)} chunk(s); "
                "refusing to return incomplete ad coverage"
            )

        # Merge detections
        ad_spans = merge_chunk_detections(all_detections)
        logger.info(f"After merging: {len(ad_spans)} ads")

        # Build usage stats
        usage = {
            "provider": "gemini_chunked",
            "model": self.model,
            "chunks_processed": len(chunks),
            "chunks_failed": len(errors),
            "raw_detections": len(all_detections),
            "merged_detections": len(ad_spans),
            "processing_time": total_time,
            "audio_duration_seconds": duration or 0,
        }

        return ad_spans, usage
