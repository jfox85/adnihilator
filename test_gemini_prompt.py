#!/usr/bin/env python3
"""Test script for iterating on Gemini audio ad detection prompts.

Usage:
    python test_gemini_prompt.py [audio_file]
"""

import json
import os
import sys
import time
from pathlib import Path

# Load API key from environment
def get_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    # Try reading from .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    raise ValueError("Set GEMINI_API_KEY env var or add to .env file")


# The prompt we're iterating on
DETECTION_PROMPT = """You are analyzing a podcast episode to identify ALL advertisement segments.

LISTEN CAREFULLY TO THE ENTIRE EPISODE. Ads can appear:
- PRE-ROLL: First 0-120 seconds (dynamically inserted ads before show starts)
- MID-ROLL: In the middle of the episode (often around 30-50% through)
- POST-ROLL: Final 2-3 minutes (dynamically inserted ads after show ends)

TYPES OF ADS TO DETECT:

1. DYNAMICALLY INSERTED ADS (easy to detect):
   - Different voices from podcast hosts (professional announcers)
   - Different audio production quality (music, sound effects, studio polish)
   - Standard 15/30/60 second spot lengths
   - National brands (cars, insurance, beverages, tech, etc.)
   - Often appear back-to-back in ad "pods"

2. HOST-READ SPONSOR SEGMENTS:
   - Podcast host reads an ad script
   - Phrases like "brought to you by", "thanks to our sponsor", "our partner"
   - Mentions website URLs, promo codes, special offers
   - Usually 60-120 seconds long

3. NETWORK BUMPERS (include these):
   - Short audio stings like "This is TWiT", "Podcasts you love"
   - These are promotional and should be included in the ad block

IMPORTANT INSTRUCTIONS:
- Listen to the ENTIRE episode, not just the beginning
- Find ALL ads at the start, middle, AND end of the episode
- Include network bumpers in your ad timestamps
- Be precise with timestamps (within 1-2 seconds)

For EACH ad segment found, provide:
- start_time: seconds when the ad begins
- end_time: seconds when the ad ends
- sponsor: brand/company name (or "Network Promo" for bumpers)
- ad_type: "dynamic_insertion", "host_read", or "network_bumper"
- confidence: 0.0-1.0
- reason: brief explanation

Respond with JSON:
{
  "ads": [
    {
      "start_time": <seconds>,
      "end_time": <seconds>,
      "sponsor": "<brand>",
      "ad_type": "<type>",
      "confidence": 0.0-1.0,
      "reason": "<explanation>"
    }
  ],
  "pre_roll_end": <seconds when actual podcast content begins>,
  "post_roll_start": <seconds when post-roll ads begin, or null if none>,
  "total_ad_count": <number of distinct ads found>,
  "notes": "<any observations about ad structure>"
}"""


def run_gemini(audio_path: str, prompt: str = DETECTION_PROMPT):
    """Run Gemini detection on an audio file and print results."""
    import google.generativeai as genai

    api_key = get_api_key()
    genai.configure(api_key=api_key)

    print(f"Uploading audio: {audio_path}")
    start = time.time()
    audio_file = genai.upload_file(path=audio_path)
    print(f"Upload complete in {time.time() - start:.1f}s")

    # Wait for processing
    while audio_file.state.name == "PROCESSING":
        time.sleep(2)
        audio_file = genai.get_file(audio_file.name)
        print(f"  Processing... {audio_file.state.name}")

    if audio_file.state.name == "FAILED":
        print(f"ERROR: Upload failed - {audio_file.state}")
        return None

    print(f"\nSending to Gemini (gemini-2.0-flash-exp)...")
    start = time.time()
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    response = model.generate_content([prompt, audio_file])
    print(f"Analysis complete in {time.time() - start:.1f}s")

    # Print raw response
    print("\n" + "="*60)
    print("RAW RESPONSE:")
    print("="*60)
    print(response.text)

    # Parse and pretty-print JSON
    try:
        text = response.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        result = json.loads(text)

        print("\n" + "="*60)
        print("PARSED RESULTS:")
        print("="*60)

        ads = result.get("ads", [])
        print(f"\nFound {len(ads)} ads:")
        total_duration = 0
        for ad in ads:
            start_t = ad.get("start_time", 0)
            end_t = ad.get("end_time", 0)
            dur = end_t - start_t
            total_duration += dur
            print(f"\n  {start_t:.0f}s - {end_t:.0f}s ({dur:.0f}s)")
            print(f"    Sponsor: {ad.get('sponsor', 'Unknown')}")
            print(f"    Type: {ad.get('ad_type', 'unknown')}")
            print(f"    Confidence: {ad.get('confidence', 0):.2f}")
            print(f"    Reason: {ad.get('reason', '')}")

        print(f"\nTotal ad time: {total_duration:.0f}s")

        if "pre_roll_end" in result:
            print(f"\nPre-roll ends at: {result['pre_roll_end']}s")

        if "notes" in result:
            print(f"\nNotes: {result['notes']}")

        return result

    except json.JSONDecodeError as e:
        print(f"\nFailed to parse JSON: {e}")
        return None


if __name__ == "__main__":
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "data/test_audio/sn1058.mp3"
    run_gemini(audio_file)
