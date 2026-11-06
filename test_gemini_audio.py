#!/usr/bin/env python3
"""Test Gemini 2.0 Flash audio processing on Decoding TV episode."""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import requests


def download_audio(url: str, output_path: Path) -> None:
    """Download audio file from URL."""
    print(f"Downloading audio from {url}...")

    # Add User-Agent header for Acast compatibility
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    file_size = output_path.stat().st_size
    print(f"Downloaded {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")


def upload_to_gemini(audio_path: Path, api_key: str) -> tuple:
    """Upload audio file to Gemini Files API and return URI and upload time."""
    print(f"Uploading audio to Gemini Files API...")

    import google.generativeai as genai
    genai.configure(api_key=api_key)

    # Upload file
    start_time = time.time()
    audio_file = genai.upload_file(path=str(audio_path))
    upload_time = time.time() - start_time

    print(f"Uploaded file URI: {audio_file.uri}")
    print(f"Upload took: {upload_time:.2f}s")

    return audio_file, upload_time


def analyze_with_gemini(audio_file, api_key: str) -> tuple:
    """Send audio to Gemini 2.0 Flash for ad detection. Returns (result, usage, analysis_time)."""
    print("Analyzing audio with Gemini 2.0 Flash...")

    import google.generativeai as genai
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-2.0-flash-exp")

    prompt = """You are analyzing a podcast episode to identify all advertisement segments.

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
- Listen for ads in BOTH English AND Spanish
- Watch for sponsor names like: Fred Meyer, Earnin, Grow Therapy, ACAST
- Ads can appear anywhere: beginning, middle, or end of episode
- Some ads may be very short (15-30 seconds)
- Pay attention to audio cues: music changes, tone shifts, different speakers

Analyze the ENTIRE audio file and identify ALL advertisement segments.

For each ad found, provide:
- start_time: timestamp in seconds when the ad begins
- end_time: timestamp in seconds when the ad ends
- sponsor: name of the sponsor/brand
- language: language of the ad (English, Spanish, etc.)
- confidence: 0.0-1.0 how confident you are this is an ad
- reason: brief explanation of why this is an ad (include audio cues if relevant)

Respond with JSON:
{
  "ads": [
    {
      "start_time": <seconds>,
      "end_time": <seconds>,
      "sponsor": "<brand name>",
      "language": "<language>",
      "confidence": 0.0-1.0,
      "reason": "<explanation>"
    }
  ]
}

If no ads found: {"ads": [], "reason": "<explanation>"}"""

    start_time = time.time()
    response = model.generate_content([prompt, audio_file])
    analysis_time = time.time() - start_time

    print(f"Analysis took: {analysis_time:.2f}s")

    # Extract JSON from response
    content = response.text

    # Try to find JSON in response
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    result = json.loads(content)

    # Get usage metadata
    usage = {
        "prompt_tokens": response.usage_metadata.prompt_token_count,
        "completion_tokens": response.usage_metadata.candidates_token_count,
        "total_tokens": response.usage_metadata.total_token_count,
    }

    return result, usage, analysis_time


def main():
    # Audio URL from Acast
    audio_url = "https://open.acast.com/public/streams/62d997ae8fb68e00135146a4/episodes/694536f0e13e237fde94bc05.mp3"

    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)

    # Create temp file for audio
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Download audio
        download_start = time.time()
        download_audio(audio_url, tmp_path)
        download_time = time.time() - download_start
        print(f"Download took: {download_time:.2f}s")
        print()

        # Upload to Gemini
        audio_file, upload_time = upload_to_gemini(tmp_path, api_key)
        print()

        # Analyze with Gemini
        result, usage, analysis_time = analyze_with_gemini(audio_file, api_key)

        print()
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
        print("=" * 70)
        print("USAGE")
        print("=" * 70)
        print(f"Input tokens: {usage['prompt_tokens']:,}")
        print(f"Output tokens: {usage['completion_tokens']:,}")
        print(f"Total tokens: {usage['total_tokens']:,}")

        # Calculate cost (Gemini 2.0 Flash pricing)
        # Audio: $0.000025/second, Text input: $0.075/1M tokens, Output: $0.30/1M tokens

        # Get audio duration (roughly estimate from file size)
        file_size = tmp_path.stat().st_size
        # Typical MP3 bitrate: 128kbps = 16KB/s
        estimated_duration = file_size / (16 * 1024)

        audio_cost = estimated_duration * 0.000025
        input_cost = usage['prompt_tokens'] * 0.075 / 1_000_000
        output_cost = usage['completion_tokens'] * 0.30 / 1_000_000
        total_cost = audio_cost + input_cost + output_cost

        print()
        print(f"Cost breakdown:")
        print(f"  Audio: ${audio_cost:.6f} (~{estimated_duration:.0f}s)")
        print(f"  Input text: ${input_cost:.6f}")
        print(f"  Output: ${output_cost:.6f}")
        print(f"  Total: ${total_cost:.6f}")

        # Calculate total time
        total_time = download_time + upload_time + analysis_time

        print()
        print("=" * 70)
        print("TIMING")
        print("=" * 70)
        print(f"Download: {download_time:.2f}s")
        print(f"Upload: {upload_time:.2f}s")
        print(f"Analysis: {analysis_time:.2f}s")
        print(f"Total: {total_time:.2f}s ({total_time/60:.2f} minutes)")

        # Save results
        output_file = "test_gemini_audio_results.json"
        with open(output_file, "w") as f:
            json.dump(
                {
                    "ads": ads,
                    "usage": usage,
                    "cost": total_cost,
                    "estimated_duration": estimated_duration,
                    "timing": {
                        "download": download_time,
                        "upload": upload_time,
                        "analysis": analysis_time,
                        "total": total_time,
                    },
                },
                f,
                indent=2,
            )

        print()
        print(f"Results saved to: {output_file}")

    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    main()
