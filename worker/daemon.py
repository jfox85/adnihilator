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
from adnihilator.audio import get_duration
from adnihilator.config import load_config
from adnihilator.external_transcript import fetch_external_transcript
from adnihilator.models import DetectionResult
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

    def process_job(self, job: EpisodeJob) -> None:
        """Process a single episode job.

        Args:
            job: The episode job to process
        """
        print(f"Processing: {job.title or job.guid}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Extract sponsors from description
            print("  Extracting sponsors...")
            sponsor_info = extract_sponsors(
                job.description or "",
                llm_client=self._get_openai_client() if self.config.llm.provider == "openai" else None,
            )
            if sponsor_info.sponsors:
                print(f"    Found {len(sponsor_info.sponsors)} sponsors via {sponsor_info.extraction_method}")

            # Download audio
            print("  Downloading audio...")
            self.api_client.update_progress(job.id, "downloading")
            audio_path = tmpdir_path / "episode.mp3"
            self._download_audio(job.original_audio_url, audio_path)

            # Get duration
            duration = get_duration(str(audio_path))
            print(f"  Duration: {duration:.0f}s")

            # Try external transcript first (if source_url available)
            segments = None
            ad_spans = []
            candidates = []
            transcript_source = "whisper"
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
                    print(f"    Found external transcript ({transcript_source}): {len(segments)} segments")

            # Fall back to Whisper if no external transcript
            if segments is None:
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

            # Save transcript immediately after transcription (before any further processing)
            # This ensures we don't lose the expensive transcription work if later steps fail
            if self.artifacts_dir:
                artifact_dir = self.artifacts_dir / job.podcast_id
                artifact_dir.mkdir(parents=True, exist_ok=True)
                transcript_path = artifact_dir / f"{job.id}_transcript.txt"
                with open(transcript_path, "w") as f:
                    for seg in segments:
                        f.write(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}\n")
                print(f"  Saved transcript to {transcript_path}")

            # If we don't have ad_spans yet (external transcript or single-pass mode),
            # run heuristic detection + LLM refinement
            if not ad_spans:
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
                }
                # Only include whisper details if we used whisper
                if transcript_source == "whisper":
                    model_info["whisper_model"] = self.whisper_model
                    model_info["device"] = self.device

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

            # Report completion
            self.api_client.complete(
                episode_id=job.id,
                audio_key=audio_key,
                duration=stats["new_duration"],
                ads_removed=stats["time_removed"],
                detection_result_path=detection_result_path,
            )

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
