#!/usr/bin/env python3
"""Reprocess episodes using existing transcripts.

This script re-runs ad detection, splicing, and upload without re-transcribing.
Useful when improving detection algorithms.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from adnihilator.ad_keywords import find_ad_candidates
from adnihilator.ad_llm import create_llm_client
from adnihilator.config import load_config
from adnihilator.models import DetectionResult, TranscriptSegment
from adnihilator.splice import splice_audio
from adnihilator.sponsors import extract_sponsors

from worker.r2 import R2Client


def load_artifact(artifact_path: Path) -> DetectionResult:
    """Load existing detection result from artifact file."""
    with open(artifact_path) as f:
        data = json.load(f)
    return DetectionResult(**data)


def reprocess_episode(
    artifact_path: Path,
    podcast_title: str,
    episode_description: str,
    r2_client: R2Client,
    config,
    output_dir: Path | None = None,
) -> dict:
    """Reprocess an episode using existing transcript.

    Args:
        artifact_path: Path to existing detection result JSON
        podcast_title: Name of the podcast (for promo code filtering)
        episode_description: Episode description (for sponsor extraction)
        r2_client: R2 client for uploading
        config: AdNihilator config
        output_dir: Optional directory to save new artifacts

    Returns:
        Dict with processing stats
    """
    print(f"Loading artifact from {artifact_path}")
    existing = load_artifact(artifact_path)

    # Extract episode and podcast IDs from path
    # Path format: .../artifacts/{podcast_id}/{episode_id}.json
    episode_id = artifact_path.stem
    podcast_id = artifact_path.parent.name

    print(f"  Episode: {episode_id}")
    print(f"  Podcast: {podcast_id} ({podcast_title})")
    print(f"  Duration: {existing.duration:.0f}s")
    print(f"  Segments: {len(existing.segments)}")

    # Re-extract sponsors
    print("  Extracting sponsors...")
    llm_client_for_sponsors = None
    if config.llm.provider == "openai" and config.llm.api_key:
        from openai import OpenAI
        llm_client_for_sponsors = OpenAI(api_key=config.llm.api_key)

    sponsor_info = extract_sponsors(episode_description, llm_client=llm_client_for_sponsors)
    if sponsor_info.sponsors:
        print(f"    Found {len(sponsor_info.sponsors)} sponsors via {sponsor_info.extraction_method}:")
        for s in sponsor_info.sponsors:
            print(f"      - {s.name}")

    # Re-run ad detection with improved keywords
    print("  Detecting ads (with improved keywords)...")
    candidates = find_ad_candidates(
        existing.segments,
        existing.duration,
        sponsors=sponsor_info,
        podcast_name=podcast_title,
    )
    print(f"    Found {len(candidates)} candidates")

    # Refine with LLM
    print("  Refining with LLM...")
    llm_client = create_llm_client(config)
    ad_spans = llm_client.refine_candidates(
        existing.segments, candidates, config, sponsors=sponsor_info
    )
    print(f"    Refined to {len(ad_spans)} ad spans")

    for span in ad_spans:
        print(f"      [{span.start:.1f}s - {span.end:.1f}s] ({span.end - span.start:.0f}s) - {span.reason[:50]}")

    # Compare with old detection
    old_ad_time = sum(span.end - span.start for span in existing.ad_spans)
    new_ad_time = sum(span.end - span.start for span in ad_spans)
    print(f"  Old detection: {old_ad_time:.0f}s of ads")
    print(f"  New detection: {new_ad_time:.0f}s of ads")
    print(f"  Difference: {new_ad_time - old_ad_time:+.0f}s")

    # Download original audio
    print("  Downloading original audio...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        audio_path = tmpdir_path / "episode.mp3"

        # Download with redirects
        response = httpx.get(existing.audio_path, follow_redirects=True, timeout=120.0)
        response.raise_for_status()
        audio_path.write_bytes(response.content)
        print(f"    Downloaded {audio_path.stat().st_size / 1024 / 1024:.1f} MB")

        # Splice
        print("  Splicing audio...")
        output_path = tmpdir_path / "processed.mp3"
        stats = splice_audio(
            str(audio_path),
            str(output_path),
            ad_spans,
            existing.duration,
        )
        print(f"    Removed {stats['time_removed']:.0f}s, new duration: {stats['new_duration']:.0f}s")

        # Upload to R2
        print("  Uploading to R2...")
        audio_key = f"{podcast_id}/{episode_id}.mp3"
        r2_client.upload_file(str(output_path), audio_key)

        # Verify upload
        uploaded_size = r2_client.get_file_size(audio_key)
        local_size = output_path.stat().st_size
        if uploaded_size != local_size:
            raise RuntimeError(f"Upload verification failed: expected {local_size}, got {uploaded_size}")
        print(f"    Uploaded to {audio_key}")

        # Save new artifact if output_dir specified
        if output_dir:
            new_result = DetectionResult(
                audio_path=existing.audio_path,
                duration=existing.duration,
                segments=existing.segments,
                candidates=candidates,
                ad_spans=ad_spans,
                model_info={
                    **existing.model_info,
                    "reprocessed": True,
                    "sponsor_extraction": sponsor_info.extraction_method,
                },
            )
            artifact_dir = output_dir / podcast_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            new_artifact_path = artifact_dir / f"{episode_id}.json"
            new_artifact_path.write_text(new_result.model_dump_json(indent=2))
            print(f"    Saved new artifact to {new_artifact_path}")

    return {
        "episode_id": episode_id,
        "podcast_id": podcast_id,
        "old_ad_time": old_ad_time,
        "new_ad_time": new_ad_time,
        "time_removed": stats["time_removed"],
        "new_duration": stats["new_duration"],
    }


def main():
    parser = argparse.ArgumentParser(description="Reprocess episodes using existing transcripts")
    parser.add_argument("artifact", type=Path, help="Path to existing detection result JSON")
    parser.add_argument("--podcast-title", required=True, help="Podcast title for promo code filtering")
    parser.add_argument("--description", default="", help="Episode description for sponsor extraction")
    parser.add_argument("--description-file", type=Path, help="File containing episode description")
    parser.add_argument("--output-dir", type=Path, help="Directory to save new artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Run detection without uploading")

    args = parser.parse_args()

    # Load description
    description = args.description
    if args.description_file:
        description = args.description_file.read_text()

    # Load config
    config = load_config()

    # Create R2 client
    r2_client = R2Client(
        access_key=os.environ.get("R2_ACCESS_KEY", ""),
        secret_key=os.environ.get("R2_SECRET_KEY", ""),
        bucket=os.environ.get("R2_BUCKET", ""),
        endpoint=os.environ.get("R2_ENDPOINT", ""),
    )

    # Reprocess
    stats = reprocess_episode(
        artifact_path=args.artifact,
        podcast_title=args.podcast_title,
        episode_description=description,
        r2_client=r2_client,
        config=config,
        output_dir=args.output_dir,
    )

    print("\n=== Summary ===")
    print(f"Old ad time: {stats['old_ad_time']:.0f}s")
    print(f"New ad time: {stats['new_ad_time']:.0f}s")
    print(f"Difference: {stats['new_ad_time'] - stats['old_ad_time']:+.0f}s")


if __name__ == "__main__":
    main()
