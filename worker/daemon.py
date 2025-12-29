"""Worker daemon for processing podcast episodes."""

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import httpx

from adnihilator.ad_keywords import find_ad_candidates
from adnihilator.ad_llm import create_llm_client, OpenAIClient
from adnihilator.ad_timestamps import extract_ad_timestamps
from adnihilator.audio import get_duration
from adnihilator.config import load_config
from adnihilator.external_transcript import fetch_external_transcript
from adnihilator.models import AdSpan, DetectionResult
from adnihilator.splice import splice_audio
from adnihilator.sponsors import extract_sponsors
from adnihilator.transcribe import transcribe_audio
from adnihilator.two_pass import two_pass_detect

from .client import WorkerClient, EpisodeJob
from .r2 import R2Client


class WorkerDaemon:
    """Daemon that processes podcast episodes."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        r2_access_key: str,
        r2_secret_key: str,
        r2_bucket: str,
        r2_endpoint: str,
        whisper_model: str = "small",
        device: str = "cpu",
        artifacts_dir: Optional[str] = None,
    ):
        """Initialize the worker daemon."""
        self.api_client = WorkerClient(api_url, api_key)
        self.r2_client = R2Client(
            r2_access_key, r2_secret_key, r2_bucket, r2_endpoint
        )
        self.whisper_model = whisper_model
        self.device = device

        # Load config from adnihilator.toml if it exists, otherwise use defaults
        config_path = Path("adnihilator.toml")
        self.config = load_config(str(config_path)) if config_path.exists() else load_config()

        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None

    def _get_openai_client(self):
        """Get OpenAI client for sponsor extraction LLM fallback."""
        try:
            from openai import OpenAI
            api_key = self.config.llm.api_key
            if api_key:
                return OpenAI(api_key=api_key)
        except ImportError:
            pass
        return None

    def _get_gemini_client(self):
        """Get Gemini client for audio-based ad detection.

        Returns None if Gemini is not enabled or library not installed.
        """
        if not self.config.gemini.enabled:
            return None

        api_key = self.config.gemini.api_key
        if not api_key:
            print("  Warning: Gemini enabled but GEMINI_API_KEY not set")
            return None

        try:
            from adnihilator.gemini_audio import GeminiAudioClient
            return GeminiAudioClient(api_key=api_key, model=self.config.gemini.model)
        except ImportError:
            print("  Warning: google-generativeai not installed, Gemini disabled")
            return None

    def _calculate_llm_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        audio_duration_seconds: float = 0.0,
    ) -> float:
        """Calculate LLM cost in USD based on provider pricing.

        Args:
            provider: LLM provider (openai, gemini)
            model: Model name
            input_tokens: Text input tokens
            output_tokens: Text output tokens
            audio_duration_seconds: Audio duration for Gemini audio models

        Returns:
            Total cost in USD
        """
        # Pricing table (as of Dec 2024)
        PRICING = {
            "openai": {
                "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
                "gpt-4.1-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
                "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
            },
            "gemini": {
                "gemini-2.0-flash-exp": {
                    "input": 0.075 / 1_000_000,
                    "output": 0.30 / 1_000_000,
                    "audio": 0.000025,  # Per second of audio
                },
            },
        }

        pricing = PRICING.get(provider, {}).get(model)
        if not pricing:
            return 0.0

        # Text token costs
        input_cost = input_tokens * pricing.get("input", 0)
        output_cost = output_tokens * pricing.get("output", 0)

        # Audio costs (Gemini only)
        audio_cost = 0.0
        if "audio" in pricing and audio_duration_seconds > 0:
            audio_cost = audio_duration_seconds * pricing["audio"]

        return input_cost + output_cost + audio_cost

    def _validate_gemini_candidates(
        self,
        gemini_candidates: list[dict],
        segments: list,
        sponsors: list[str],
        buffer_seconds: float = 30.0,
    ) -> tuple[list[dict], list[dict]]:
        """Validate Gemini candidates against transcript.

        Gemini sometimes hallucinates timestamps. This method checks if the
        transcript around each Gemini timestamp actually contains sponsor
        keywords or ad-related phrases.

        Args:
            gemini_candidates: List of Gemini candidate dicts
            segments: Transcript segments
            sponsors: List of sponsor names
            buffer_seconds: Buffer around candidate to search

        Returns:
            Tuple of (validated_candidates, rejected_candidates)
        """
        if not gemini_candidates or not segments:
            return gemini_candidates, []

        # Build keyword set for validation
        ad_keywords = {
            "sponsor", "sponsored", "brought to you", "thanks to",
            "promo code", "use code", "discount", "% off", "percent off",
            ".com/", "check out", "sign up", "free trial",
        }
        sponsor_keywords = {s.lower() for s in sponsors}

        validated = []
        rejected = []

        for candidate in gemini_candidates:
            start = candidate.get("start", 0)
            end = candidate.get("end", 0)

            # Get transcript text around this candidate
            context_start = max(0, start - buffer_seconds)
            context_end = end + buffer_seconds

            context_text = ""
            for seg in segments:
                if seg.end >= context_start and seg.start <= context_end:
                    context_text += " " + seg.text

            context_lower = context_text.lower()

            # Check for any sponsor name in context
            has_sponsor = any(s in context_lower for s in sponsor_keywords)

            # Check for ad keywords in context
            has_ad_keyword = any(kw in context_lower for kw in ad_keywords)

            if has_sponsor or has_ad_keyword:
                validated.append(candidate)
            else:
                rejected.append(candidate)
                print(f"    Rejected Gemini candidate {start:.0f}-{end:.0f}s: no ad evidence in transcript")

        return validated, rejected

    def _get_combined_transcript_context(
        self,
        segments: list,
        candidates: list[dict],
        buffer_seconds: float = 30.0,
        max_chars: int = 15000,
    ) -> str:
        """Extract single contiguous transcript block around all candidates.

        More efficient than per-candidate extraction - sends one block to LLM.
        """
        if not candidates or not segments:
            return ""

        # Find min/max across all candidates
        min_start = min(c["start"] for c in candidates)
        max_end = max(c["end"] for c in candidates)

        # Add buffer
        context_start = max(0, min_start - buffer_seconds)
        context_end = max_end + buffer_seconds

        # Extract text with timestamps
        parts = []
        for seg in segments:
            if seg.end >= context_start and seg.start <= context_end:
                parts.append(f"[{seg.start:.0f}s] {seg.text}")

        result = "\n".join(parts)

        # Truncate if too long
        if len(result) > max_chars:
            result = result[:max_chars] + "\n[...truncated...]"

        return result

    def _simple_merge_candidates(
        self,
        gemini_candidates: list[dict],
        keyword_candidates: list[dict],
    ) -> list[AdSpan]:
        """Simple merge fallback: union of candidates, merge overlapping.

        Preserves source tracking.
        """
        all_candidates = gemini_candidates + keyword_candidates
        if not all_candidates:
            return []

        # Sort by start time
        sorted_candidates = sorted(all_candidates, key=lambda c: c["start"])

        merged = []
        current = sorted_candidates[0].copy()
        current["sources"] = [current.get("source", "unknown")]

        for c in sorted_candidates[1:]:
            if c["start"] <= current["end"] + 5:  # Merge if within 5 seconds
                current["end"] = max(current["end"], c["end"])
                current["confidence"] = max(current["confidence"], c["confidence"])
                # Track all sources
                source = c.get("source", "unknown")
                if source not in current["sources"]:
                    current["sources"].append(source)
                # Combine reasons
                current["reason"] = f"{current.get('reason', '')}; {c.get('reason', '')}"
            else:
                merged.append(current)
                current = c.copy()
                current["sources"] = [current.get("source", "unknown")]

        merged.append(current)

        return [
            AdSpan(
                start=c["start"],
                end=c["end"],
                confidence=c["confidence"],
                reason=c["reason"][:200],  # Truncate long reasons
                candidate_indices=[],
                sources=c["sources"],
            )
            for c in merged
        ]

    def _llm_merge_candidates(
        self,
        gemini_candidates: list[dict],
        keyword_candidates: list[dict],
        segments: list,
        sponsors,
        duration: float,
    ) -> tuple[list[AdSpan], dict | None]:
        """Use LLM to merge candidates from both detection methods.

        Returns:
            Tuple of (ad_spans, llm_usage_dict or None)
            Always returns usage if LLM was called (even on error/empty result)
        """
        all_candidates = gemini_candidates + keyword_candidates

        # Short-circuit if no candidates
        if not all_candidates:
            return [], None

        # Build efficient context
        transcript_context = self._get_combined_transcript_context(segments, all_candidates)

        prompt_data = {
            "gemini_candidates": gemini_candidates,
            "keyword_candidates": keyword_candidates,
            "transcript_context": transcript_context,
            "sponsors": [s.name for s in sponsors.sponsors] if sponsors and sponsors.sponsors else [],
            "duration": duration,
        }

        llm_client = create_llm_client(self.config)
        usage = None

        if isinstance(llm_client, OpenAIClient):
            try:
                spans, usage = llm_client.merge_and_refine(prompt_data, self.config)
                # Empty result = LLM says "no ads" - this is valid, NOT a failure
                # Only fallback on actual exception (parse error, API error, etc.)
                return spans, usage
            except Exception as e:
                print(f"  Warning: LLM merge failed ({e}), using simple merge fallback")
                # Fallback to simple merge on actual error (still return usage if available)
                return self._simple_merge_candidates(gemini_candidates, keyword_candidates), usage

        # No LLM client available - use simple merge
        return self._simple_merge_candidates(gemini_candidates, keyword_candidates), None

    def process_job(self, job: EpisodeJob) -> None:
        """Process a single episode job.

        Args:
            job: The episode job to process
        """
        print(f"Processing: {job.title or job.guid}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Track LLM usage for this job
            llm_usage = None  # Will be populated if LLM is used
            detection_source = None  # Track which method detected ads
            raw_gemini_candidates = []  # Store raw Gemini candidates for debugging

            # Extract sponsors from description
            print("  Extracting sponsors...")
            sponsor_info = extract_sponsors(
                job.description or "",
                llm_client=self._get_openai_client() if self.config.llm.provider == "openai" else None,
            )
            if sponsor_info.sponsors:
                print(f"    Found {len(sponsor_info.sponsors)} sponsors via {sponsor_info.extraction_method}")

            # TIER 1: Check for ad timestamps in description (FREE)
            ad_timestamps = extract_ad_timestamps(job.description or "")
            high_confidence_timestamps = [t for t in ad_timestamps if t.confidence >= 0.85]

            if high_confidence_timestamps:
                print(f"  Found {len(high_confidence_timestamps)} high-confidence ad timestamps in description")
                detection_source = "timestamps"

                # Convert timestamps to AdSpan format
                ad_spans = [
                    AdSpan(
                        start=t.start,
                        end=t.end,
                        confidence=t.confidence,
                        reason=f"From description: {t.label or 'Ad'} ({t.extraction_method})",
                        candidate_indices=[],
                    )
                    for t in high_confidence_timestamps
                ]

                # Download audio for splicing (still need it)
                print("  Downloading audio...")
                self.api_client.update_progress(job.id, "downloading")
                audio_path = tmpdir_path / "episode.mp3"
                self._download_audio(job.original_audio_url, audio_path)

                # Get duration
                duration = get_duration(str(audio_path))
                print(f"  Duration: {duration:.0f}s")

                # Skip directly to splicing (no transcription needed)
                segments = []
                candidates = []
                transcript_source = "none"

            else:
                # No timestamp-based detection, continue with other methods

                # Download audio
                print("  Downloading audio...")
                self.api_client.update_progress(job.id, "downloading")
                audio_path = tmpdir_path / "episode.mp3"
                self._download_audio(job.original_audio_url, audio_path)

                # Get duration
                duration = get_duration(str(audio_path))
                print(f"  Duration: {duration:.0f}s")

                # Initialize detection state
                segments = None
                ad_spans = []
                candidates = []
                transcript_source = "whisper"

                # TIER 2a: Try external transcript first (FREE - if source_url available)
                if job.source_url:
                    print("  Checking for external transcript...")
                    segments = fetch_external_transcript(job.source_url)
                    if segments:
                        # Determine source based on URL
                        if 'lexfridman.com' in job.source_url:
                            transcript_source = "lexfridman"
                        elif 'substack' in job.source_url:
                            transcript_source = "substack"
                        else:
                            transcript_source = "external"
                        detection_source = "external"
                        print(f"    Found external transcript ({transcript_source}): {len(segments)} segments")

                # Check if parallel mode enabled
                if self.config.detection.parallel_enabled:
                    # PARALLEL DETECTION MODE
                    import concurrent.futures

                    gemini_candidates = []
                    keyword_candidates = []
                    gemini_usage = None
                    gemini_error = None
                    whisper_error = None
                    gemini_attempted = False  # Track if Gemini ran (vs failed to run)

                    def run_gemini():
                        nonlocal gemini_candidates, gemini_usage, gemini_error, gemini_attempted
                        try:
                            gemini_client = self._get_gemini_client()
                            if gemini_client:
                                gemini_attempted = True  # Mark as attempted BEFORE call
                                spans, usage = gemini_client.detect_ads(
                                    audio_path,
                                    podcast_title=job.podcast_title,
                                    duration=duration,
                                    sponsors=sponsor_info,
                                )
                                gemini_candidates = [
                                    {
                                        "start": s.start,
                                        "end": s.end,
                                        "confidence": s.confidence,
                                        "reason": s.reason,
                                        "source": "gemini",
                                    }
                                    for s in spans
                                ]
                                gemini_usage = usage
                        except Exception as e:
                            gemini_error = str(e)

                    def run_transcript_keywords():
                        nonlocal keyword_candidates, segments, whisper_error, transcript_source, candidates
                        try:
                            # Use existing external transcript if available
                            if segments is None:
                                # Need to transcribe
                                print("    Transcribing with Whisper...")

                                def on_transcribe_progress(percent: int) -> None:
                                    self.api_client.update_progress(job.id, "transcribing", percent)

                                segments = transcribe_audio(
                                    str(audio_path),
                                    model_name=self.whisper_model,
                                    device=self.device,
                                    duration=duration,
                                    progress_callback=on_transcribe_progress,
                                )
                                transcript_source = "whisper"

                            # Find keyword candidates
                            cands = find_ad_candidates(
                                segments, duration, sponsors=sponsor_info, podcast_name=job.podcast_title
                            )
                            keyword_candidates = [
                                {
                                    "start": c.start,
                                    "end": c.end,
                                    "confidence": c.heuristic_score,
                                    "reason": f"Keywords: {', '.join(c.trigger_keywords[:3])}",
                                    "matched_keywords": c.trigger_keywords,
                                    "source": "keywords",
                                }
                                for c in cands
                            ]
                            # Store original candidates for artifact
                            candidates = cands
                        except Exception as e:
                            whisper_error = str(e)

                    print("  Running parallel detection (Gemini + Transcript/Keywords)...")
                    # Start with "transcribing" since that's the slow part - progress updates will follow
                    self.api_client.update_progress(job.id, "transcribing", 0)

                    # Run Gemini in background thread (I/O bound - safe in thread)
                    # Run Whisper in main thread (uses multiprocessing internally - unsafe in thread)
                    # This avoids the leaked semaphore issue from running multiprocessing inside threads
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        gemini_future = executor.submit(run_gemini)

                        # Run Whisper/keywords in main thread
                        run_transcript_keywords()

                        # Wait for Gemini to complete
                        gemini_future.result()

                    # Update status after parallel work completes
                    self.api_client.update_progress(job.id, "detecting")

                    if gemini_error:
                        print(f"  Warning: Gemini failed: {gemini_error}")
                    if whisper_error:
                        print(f"  Warning: Whisper/keywords failed: {whisper_error}")

                    print(f"    Gemini: {len(gemini_candidates)} candidates")
                    print(f"    Keywords: {len(keyword_candidates)} candidates")

                    # Store raw Gemini candidates for debugging (before validation)
                    raw_gemini_candidates = gemini_candidates.copy()

                    # Validate Gemini candidates against transcript
                    # This catches hallucinated timestamps that don't contain ad content
                    if gemini_candidates and segments:
                        sponsor_names = [s.name for s in sponsor_info.sponsors] if sponsor_info and sponsor_info.sponsors else []
                        gemini_candidates, rejected = self._validate_gemini_candidates(
                            gemini_candidates, segments, sponsor_names
                        )
                        if rejected:
                            print(f"    Validated: {len(gemini_candidates)} Gemini candidates ({len(rejected)} rejected)")

                    # Track all usages for accurate cost reporting
                    all_usages = []
                    total_cost = 0.0

                    if gemini_usage:
                        gemini_cost = self._calculate_llm_cost(
                            provider=gemini_usage.get("provider", "gemini"),
                            model=gemini_usage.get("model", self.config.gemini.model),
                            input_tokens=gemini_usage.get("input_tokens", 0),
                            output_tokens=gemini_usage.get("output_tokens", 0),
                            audio_duration_seconds=duration,
                        )
                        gemini_usage["cost"] = gemini_cost
                        total_cost += gemini_cost
                        all_usages.append(gemini_usage)
                        print(f"    Gemini cost: ${gemini_cost:.4f}")

                    if gemini_candidates or keyword_candidates:
                        if segments:
                            print("  Merging with LLM...")
                            self.api_client.update_progress(job.id, "refining")
                            ad_spans, merge_usage = self._llm_merge_candidates(
                                gemini_candidates,
                                keyword_candidates,
                                segments,
                                sponsor_info,
                                duration,
                            )
                            detection_source = "parallel"

                            # Track merge LLM cost (even if empty result or error)
                            if merge_usage:
                                if "input_tokens" in merge_usage:
                                    merge_cost = self._calculate_llm_cost(
                                        provider=merge_usage.get("provider", "openai"),
                                        model=merge_usage.get("model", self.config.llm.model),
                                        input_tokens=merge_usage.get("input_tokens", 0),
                                        output_tokens=merge_usage.get("output_tokens", 0),
                                    )
                                    merge_usage["cost"] = merge_cost
                                    total_cost += merge_cost
                                    print(f"    LLM merge cost: ${merge_cost:.4f}")
                                all_usages.append(merge_usage)
                        else:
                            # No transcript (Whisper failed) - use Gemini directly
                            print("  No transcript, using Gemini candidates directly")
                            ad_spans = [
                                AdSpan(
                                    start=c["start"],
                                    end=c["end"],
                                    confidence=c["confidence"],
                                    reason=c["reason"],
                                    candidate_indices=[],
                                    sources=["gemini"],
                                )
                                for c in gemini_candidates
                            ]
                            detection_source = "gemini_only"
                            transcript_source = "none"
                    elif gemini_attempted or segments:
                        # Valid "no ads" outcome: either Gemini ran and found nothing,
                        # or we have transcript but no keyword matches
                        print("  No candidates found from either method (valid no-ads result)")
                        ad_spans = []
                        detection_source = "parallel_empty"
                    else:
                        # Both actually failed to run
                        raise RuntimeError("Both Gemini and Whisper detection failed")

                    # Build combined llm_usage for reporting
                    if all_usages:
                        # Use first provider's info but aggregate cost
                        llm_usage = all_usages[0].copy()
                        llm_usage["cost"] = total_cost
                        if len(all_usages) > 1:
                            llm_usage["providers"] = [u.get("provider") for u in all_usages]

                    print(f"    Total cost: ${total_cost:.4f}")

                # TIER 2b: Try Gemini audio detection (if no external transcript) - TIERED MODE
                elif segments is None:
                    gemini_client = self._get_gemini_client()
                    if gemini_client:
                        print("  Using Gemini audio detection...")
                        self.api_client.update_progress(job.id, "detecting")
                        try:
                            ad_spans, llm_usage = gemini_client.detect_ads(
                                audio_path,
                                podcast_title=job.podcast_title,
                                duration=duration,
                                sponsors=sponsor_info,
                            )
                            detection_source = "gemini"
                            transcript_source = "gemini_audio"
                            segments = []  # No transcription segments from Gemini
                            print(f"    Gemini detected {len(ad_spans)} ads")

                            # Calculate cost
                            if llm_usage:
                                llm_usage["cost"] = self._calculate_llm_cost(
                                    provider=llm_usage.get("provider", "gemini"),
                                    model=llm_usage.get("model", self.config.gemini.model),
                                    input_tokens=llm_usage.get("input_tokens", 0),
                                    output_tokens=llm_usage.get("output_tokens", 0),
                                    audio_duration_seconds=llm_usage.get("audio_duration_seconds", duration),
                                )
                                print(f"    Gemini cost: ${llm_usage['cost']:.4f}")

                        except Exception as e:
                            print(f"  Warning: Gemini detection failed: {e}")
                            print("  Falling back to Whisper...")
                            gemini_client = None  # Fall through to Whisper

                # TIER 3: Fall back to Whisper if no external transcript and Gemini failed/disabled
                if segments is None and not ad_spans:
                    detection_source = "whisper"

                    # Create LLM client to determine transcription strategy
                    llm_client = create_llm_client(self.config)

                    # Use two-pass mode if LLM is OpenAI (fast segment-level + targeted word-level)
                    if isinstance(llm_client, OpenAIClient):
                        print("  Transcribing with two-pass mode...")
                        self.api_client.update_progress(job.id, "transcribing", 0)

                        def on_transcribe_progress(percent: int) -> None:
                            # Map progress to steps: 0-99 = Pass 1, 100 = detecting, 101-199 = Pass 2, 200 = complete
                            if percent <= 99:
                                self.api_client.update_progress(job.id, "transcribing", percent)
                            elif percent == 100:
                                self.api_client.update_progress(job.id, "detecting", 0)
                            elif percent <= 199:
                                # Pass 2: map 100-199 to 0-100%
                                self.api_client.update_progress(job.id, "pass2", percent - 100)
                            else:
                                self.api_client.update_progress(job.id, "refining", 100)

                        segments, ad_spans = two_pass_detect(
                            str(audio_path),
                            llm_client,
                            self.config,
                            duration,
                            model_name=self.whisper_model,
                            device=self.device,
                            progress_callback=on_transcribe_progress,
                            sponsors=sponsor_info,
                            podcast_name=job.podcast_title,
                        )
                        print(f"    Two-pass complete: {len(segments)} segments, {len(ad_spans)} ads detected")
                    else:
                        # Single-pass mode (full word-level transcription)
                        print("  Transcribing with Whisper...")
                        self.api_client.update_progress(job.id, "transcribing", 0)

                        def on_transcribe_progress(percent: int) -> None:
                            self.api_client.update_progress(job.id, "transcribing", percent)

                        segments = transcribe_audio(
                            str(audio_path),
                            model_name=self.whisper_model,
                            device=self.device,
                            duration=duration,
                            progress_callback=on_transcribe_progress,
                        )

                # Save transcript if we have one (before any further processing)
                if segments and self.artifacts_dir:
                    artifact_dir = self.artifacts_dir / job.podcast_id
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    transcript_path = artifact_dir / f"{job.id}_transcript.txt"
                    with open(transcript_path, "w") as f:
                        for seg in segments:
                            f.write(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}\n")
                    print(f"  Saved transcript to {transcript_path}")

                # If we don't have ad_spans yet (external transcript or single-pass mode),
                # run heuristic detection + LLM refinement
                # Skip if parallel detection already ran (detection_source will be set)
                if not ad_spans and segments and detection_source is None:
                    # Detect ads
                    print("  Detecting ads...")
                    self.api_client.update_progress(job.id, "detecting")
                    candidates = find_ad_candidates(
                        segments, duration, sponsors=sponsor_info, podcast_name=job.podcast_title
                    )

                    # Refine with LLM
                    print("  Refining with LLM...")
                    self.api_client.update_progress(job.id, "refining")
                    llm_client = create_llm_client(self.config)
                    ad_spans = llm_client.refine_candidates(
                        segments, candidates, self.config,
                        sponsors=sponsor_info, podcast_title=job.podcast_title
                    )

            # Save detection result artifact if artifacts_dir is set
            detection_result_path = None
            if self.artifacts_dir:
                print("  Saving detection result...")
                model_info = {
                    "transcript_source": transcript_source,
                    "llm_provider": self.config.llm.provider,
                    "detection_source": detection_source,
                }
                # Only include whisper details if we used whisper
                if transcript_source == "whisper":
                    model_info["whisper_model"] = self.whisper_model
                    model_info["device"] = self.device
                # Store raw Gemini candidates for debugging (before validation)
                if raw_gemini_candidates:
                    model_info["gemini_candidates_raw"] = raw_gemini_candidates

                detection_result = DetectionResult(
                    audio_path=job.original_audio_url,
                    duration=duration,
                    segments=segments,
                    candidates=candidates,
                    ad_spans=ad_spans,
                    model_info=model_info,
                )
                artifact_dir = self.artifacts_dir / job.podcast_id
                artifact_dir.mkdir(parents=True, exist_ok=True)
                artifact_path = artifact_dir / f"{job.id}.json"
                artifact_path.write_text(detection_result.model_dump_json(indent=2))
                detection_result_path = str(artifact_path)

            # Save sponsor artifact
            if self.artifacts_dir and sponsor_info.sponsors:
                artifact_dir = self.artifacts_dir / job.podcast_id
                artifact_dir.mkdir(parents=True, exist_ok=True)
                sponsor_path = artifact_dir / f"{job.id}_sponsors.json"
                sponsor_path.write_text(sponsor_info.model_dump_json(indent=2))
                print(f"    Saved sponsors to {sponsor_path}")

            # Splice
            print("  Splicing audio...")
            self.api_client.update_progress(job.id, "splicing")
            output_path = tmpdir_path / "processed.mp3"
            stats = splice_audio(
                str(audio_path),
                str(output_path),
                ad_spans,
                duration,
            )

            # Upload to R2
            print("  Uploading to R2...")
            self.api_client.update_progress(job.id, "uploading")
            audio_key = f"{job.podcast_id}/{job.id}.mp3"
            self.r2_client.upload_file(str(output_path), audio_key)

            # Verify upload
            uploaded_size = self.r2_client.get_file_size(audio_key)
            local_size = output_path.stat().st_size
            if uploaded_size != local_size:
                raise RuntimeError(
                    f"Upload verification failed: expected {local_size}, got {uploaded_size}"
                )

            # Report completion with LLM tracking data
            completion_kwargs = {
                "episode_id": job.id,
                "audio_key": audio_key,
                "duration": stats["new_duration"],
                "ads_removed": stats["time_removed"],
                "detection_result_path": detection_result_path,
                "detection_source": detection_source,
            }

            # Add LLM usage data if available
            if llm_usage:
                completion_kwargs.update({
                    "llm_provider": llm_usage.get("provider"),
                    "llm_model": llm_usage.get("model"),
                    "llm_input_tokens": llm_usage.get("input_tokens"),
                    "llm_output_tokens": llm_usage.get("output_tokens"),
                    "llm_total_tokens": llm_usage.get("total_tokens"),
                    "llm_cost_usd": llm_usage.get("cost"),
                })

            self.api_client.complete(**completion_kwargs)

            print(f"  Done! Removed {stats['time_removed']:.0f}s of ads")

    def _resolve_with_google_dns(self, hostname: str) -> str | None:
        """Resolve hostname using Google DNS (8.8.8.8) as fallback.

        Returns IP address or None if resolution fails.
        """
        try:
            result = subprocess.run(
                ["dig", "+short", hostname, "@8.8.8.8"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Get the first IP (dig may return multiple)
                for line in result.stdout.strip().split("\n"):
                    # Skip CNAME records, only return A records (IPs)
                    if line and not line.endswith("."):
                        return line
        except Exception:
            pass
        return None

    def _download_audio(self, url: str, dest: Path) -> None:
        """Download audio file from URL.

        Uses curl subprocess with fallback DNS resolution via Google DNS
        for domains that may fail with local DNS resolvers.
        """
        # Build curl command with User-Agent (required by some podcast CDNs for proper SSL SNI)
        curl_cmd = [
            "curl", "-L", "-o", str(dest), "-f", "--silent", "--show-error",
            "-A", "AdNihilator/1.0 (Podcast Downloader)"
        ]

        # Try download, retry with resolved IPs if DNS fails
        result = subprocess.run(
            curl_cmd + [url],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return

        # Check if it's a DNS error
        if "Could not resolve host" in result.stderr:
            # Extract the hostname that failed from the error
            match = re.search(r"Could not resolve host: (\S+)", result.stderr)
            if match:
                failed_host = match.group(1)
                ip = self._resolve_with_google_dns(failed_host)
                if ip:
                    # Retry with --resolve to bypass local DNS
                    curl_cmd_with_resolve = curl_cmd + [
                        "--resolve", f"{failed_host}:443:{ip}",
                        "--resolve", f"{failed_host}:80:{ip}",
                        url,
                    ]
                    result = subprocess.run(
                        curl_cmd_with_resolve,
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        return

        raise RuntimeError(f"curl download failed: {result.stderr}")

    def run_once(self) -> bool:
        """Process one job from the queue.

        Returns:
            True if a job was processed, False if queue was empty.
        """
        job = self.api_client.claim()
        if job is None:
            return False

        try:
            self.process_job(job)
            return True
        except Exception as e:
            print(f"  Error: {e}")
            self.api_client.fail(job.id, str(e)[:500])
            return True

    def run_daemon(self, interval: int = 300, watchdog_hours: int = 12) -> None:
        """Run the daemon loop with health monitoring.

        Args:
            interval: Seconds to wait between queue checks
            watchdog_hours: Exit if no job processed in this many hours (0 to disable)
        """
        print(f"Starting worker daemon (interval: {interval}s, watchdog: {watchdog_hours}h)")

        last_job_time = time.time()
        empty_queue_count = 0

        while True:
            try:
                job_processed = self.run_once()
                if job_processed:
                    last_job_time = time.time()
                    empty_queue_count = 0
                else:
                    empty_queue_count += 1
                    # Log health check every 10 empty checks (10 * interval seconds)
                    if empty_queue_count % 10 == 0:
                        hours_idle = (time.time() - last_job_time) / 3600
                        print(f"Health check: {hours_idle:.1f}h since last job, {empty_queue_count} empty checks")

                    print(f"Queue empty, sleeping for {interval}s...")
                    time.sleep(interval)

                    # Watchdog: exit if stuck too long without processing
                    if watchdog_hours > 0:
                        hours_since_job = (time.time() - last_job_time) / 3600
                        if hours_since_job > watchdog_hours:
                            print(f"WATCHDOG: No job processed in {hours_since_job:.1f}h, exiting for restart...")
                            self.api_client.close()
                            break

            except KeyboardInterrupt:
                print("\nShutting down...")
                self.api_client.close()
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                time.sleep(60)  # Back off on errors
