#!/usr/bin/env python3
"""Test full transcript LLM processing on Decoding TV episode."""

import json
import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from adnihilator.ad_llm import OpenAIClient
from adnihilator.config import load_config
from adnihilator.models import TranscriptSegment


def load_transcript(json_path: str) -> list[TranscriptSegment]:
    """Load transcript from detection results JSON."""
    with open(json_path) as f:
        data = json.load(f)

    segments = []
    for seg_data in data.get("segments", []):
        segments.append(
            TranscriptSegment(
                index=seg_data["index"],
                start=seg_data["start"],
                end=seg_data["end"],
                text=seg_data["text"],
            )
        )

    return segments


def format_transcript_for_llm(segments: list[TranscriptSegment]) -> str:
    """Format transcript segments for LLM processing."""
    lines = []
    for seg in segments:
        timestamp = f"[{int(seg.start // 60):02d}:{int(seg.start % 60):02d}]"
        lines.append(f"{timestamp} {seg.text}")
    return "\n".join(lines)


def main():
    # Load config
    config = load_config()

    # Path to the Decoding TV episode
    episode_id = "33fd523b-d8b0-4b61-878b-506036c75d00"
    podcast_id = "65b629f5-3ce7-47db-aa5a-120420d83499"
    json_path = f"data/artifacts/{podcast_id}/{episode_id}.json"

    print(f"Loading transcript from {json_path}...")
    segments = load_transcript(json_path)

    print(f"Loaded {len(segments)} segments")
    if segments:
        print(f"Duration: {segments[-1].end:.1f} seconds ({segments[-1].end/60:.1f} minutes)")
    else:
        print("Duration: 0.0 seconds (no segments loaded)")
    print()

    # Format full transcript
    full_text = format_transcript_for_llm(segments)

    # Calculate token estimate
    char_count = len(full_text)
    token_estimate = char_count / 4
    print(f"Full transcript:")
    print(f"  Characters: {char_count:,}")
    print(f"  Estimated tokens: {int(token_estimate):,}")
    print()

    # Create OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    client = OpenAIClient(api_key=api_key, model=config.llm.model)

    # Build prompt for full transcript analysis
    prompt = f"""You are analyzing a podcast transcript to identify all advertisement segments.

The podcast is "Decoding TV" - a show about television and streaming content.

Look for ANY of the following:
- Sponsor reads (e.g., "This episode is brought to you by...")
- Product promotions or advertisements
- Service advertisements (therapy apps, payment apps, grocery stores, etc.)
- Mid-roll ads inserted into the content
- Pre-roll or post-roll advertisements
- Ads in ANY LANGUAGE (English, Spanish, etc.)
- Host reads that promote a product/service

IMPORTANT:
- Look for ads in BOTH English AND Spanish
- Watch for sponsor names like: Fred Meyer, Earnin, Grow Therapy, ACAST
- Ads can appear anywhere: beginning, middle, or end of episode
- Some ads may be very short (15-30 seconds)

Here's the full transcript with timestamps:

{full_text}

Analyze the ENTIRE transcript and identify ALL advertisement segments.

For each ad found, provide:
- start_time: timestamp in seconds when the ad begins
- end_time: timestamp in seconds when the ad ends
- sponsor: name of the sponsor/brand
- language: language of the ad (English, Spanish, etc.)
- confidence: 0.0-1.0 how confident you are this is an ad
- reason: brief explanation of why this is an ad

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

    print("Sending full transcript to LLM for analysis...")
    print(f"Using model: {config.llm.model}")
    print()

    try:
        # Import OpenAI
        from openai import OpenAI

        openai_client = OpenAI(api_key=api_key, base_url=client.base_url)

        # Make API call
        response = openai_client.chat.completions.create(
            model=client.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at identifying advertisement segments in podcast transcripts. You can detect ads in multiple languages including English and Spanish.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_completion_tokens=2000,
        )

        # Parse response (strip markdown fences if present)
        content = response.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)

        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print()

        ads = result.get("ads", [])

        if not ads:
            print(f"No ads found. Reason: {result.get('reason', 'Unknown')}")
        else:
            print(f"Found {len(ads)} advertisement segments:")
            print()

            for i, ad in enumerate(ads, 1):
                start = ad.get("start_time", 0)
                end = ad.get("end_time", 0)
                duration = end - start
                sponsor = ad.get("sponsor", "Unknown")
                language = ad.get("language", "Unknown")
                confidence = ad.get("confidence", 0)
                reason = ad.get("reason", "")

                print(f"Ad #{i}:")
                print(f"  Time: {start:.1f}s - {end:.1f}s ({duration:.1f}s)")
                print(f"  Sponsor: {sponsor}")
                print(f"  Language: {language}")
                print(f"  Confidence: {confidence:.2f}")
                print(f"  Reason: {reason}")
                print()

        # Show token usage
        usage = response.usage
        print("=" * 70)
        print("USAGE")
        print("=" * 70)
        print(f"Input tokens: {usage.prompt_tokens:,}")
        print(f"Output tokens: {usage.completion_tokens:,}")
        print(f"Total tokens: {usage.total_tokens:,}")

        # Calculate cost (GPT-4o-mini pricing)
        input_cost = usage.prompt_tokens * 0.15 / 1_000_000
        output_cost = usage.completion_tokens * 0.60 / 1_000_000
        total_cost = input_cost + output_cost

        print()
        print(f"Cost breakdown:")
        print(f"  Input: ${input_cost:.6f}")
        print(f"  Output: ${output_cost:.6f}")
        print(f"  Total: ${total_cost:.6f}")

        # Save results
        output_file = "test_full_transcript_results.json"
        with open(output_file, "w") as f:
            json.dump(
                {
                    "ads": ads,
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    },
                    "cost": total_cost,
                },
                f,
                indent=2,
            )

        print()
        print(f"Results saved to: {output_file}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
