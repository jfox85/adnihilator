# Ad Removal Pipeline

This document explains how AdNihilator detects and removes podcast advertisements, the techniques used at each stage, and lessons learned during development.

## Pipeline Overview

```
Audio File
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 1: Transcription                                        │
│   - Try external transcript (Substack CloudFront JSON)       │
│   - Fall back to Whisper (faster-whisper, x86_64/Rosetta)    │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 2: Sponsor Extraction                                   │
│   - Parse show notes HTML with regex patterns                 │
│   - LLM fallback for unstructured descriptions               │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 3: Heuristic Candidate Detection                        │
│   - Keyword pattern matching with category weights            │
│   - Sponsor name matching with smart word splitting           │
│   - Positional boosts (intro/midroll/outro)                  │
│   - Automatic outro region detection (last 3 min)            │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 4: LLM Refinement                                       │
│   - Merge nearby candidates (within 5 min)                   │
│   - Send transcript context to GPT-4.1-mini                  │
│   - Line-number based output for efficiency                  │
│   - "Hunt mode" searches gaps for missing sponsors           │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 5: Splicing                                             │
│   - Filter by confidence threshold (≥0.35)                   │
│   - ffmpeg filter_complex for seamless concatenation         │
│   - Fallback: segment extraction + concat demuxer            │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
Processed Audio (ads removed)
```

## Stage 1: Transcription

### External Transcripts (Preferred)

For podcasts that provide their own transcripts, we fetch them directly rather than using Whisper:

**File:** `adnihilator/external_transcript.py`

**Supported platforms:**

1. **Substack podcasts** - Extract CloudFront signed URL from episode page HTML, parse JSON with word-level timestamps
2. **Lex Fridman podcasts** - Parse HTML transcript from lexfridman.com with segment-level timestamps (no word-level)

**Why this matters:** External transcripts are more accurate for proper nouns, brand names, and promo codes that Whisper often mishears. They're also much faster than running Whisper.

### Whisper Fallback

**File:** `adnihilator/transcribe.py`

Uses `faster-whisper` library with the following considerations:

- Runs under Rosetta on Apple Silicon (x86_64 binary)
- Default model: "small" for balance of speed/accuracy
- Progress callback reports percentage during long transcriptions

**Lesson learned:** Whisper can split compound brand names ("Bitwarden" → "bit warden", "ExpressVPN" → "express VPN"). We handle this in sponsor keyword generation.

## Stage 2: Sponsor Extraction

**File:** `adnihilator/sponsors.py`

Before detecting ads in audio, we extract known sponsors from show notes to inform detection.

### Regex Patterns

Multiple pattern types for different podcast formats:

1. **HTML lists** - `<strong>SPONSORS:</strong>` followed by `<li><a href="...">` items
2. **Lex Fridman style** - `<b>Sponsor Name:</b> description <a href="url">`
3. **Plain text** - `Partner Deals:\nName: description with code CODE`
4. **Simple lists** - `Sponsors: Name1, Name2, Name3`

### Promo Code Extraction

Captures codes like "use code SAVE20" from sponsor links and descriptions.

### LLM Fallback

When patterns fail but sponsor keywords are present, uses GPT-4o-mini to extract sponsor information.

### Smart Keyword Generation

**Key function:** `generate_sponsor_keywords()`

Handles Whisper transcription quirks:

| Sponsor Name | Generated Keywords |
|--------------|-------------------|
| ExpressVPN | `expressvpn`, `express vpn` |
| Bitwarden | `bitwarden`, `bit warden` |
| LMNT | `lmnt`, `l m n t` |
| shopify.com/lex | `shopify`, `shopify dot com slash lex` |

**Podcast name filtering:** Promo codes matching the podcast name are excluded to prevent false positives (e.g., "TWIT" code on "This Week in Tech" podcast).

## Stage 3: Heuristic Detection

**File:** `adnihilator/ad_keywords.py`

### Keyword Categories and Weights

| Category | Weight | Example Patterns |
|----------|--------|------------------|
| intro_sponsor | 0.5 | "brought to you by", "today's sponsor" |
| promo_code | 0.3 | "promo code", "use code", "discount code" |
| offer | 0.25 | "free trial", "percent off", "money back" |
| cta | 0.15 | "go to", "check out", "sign up at" |
| url | 0.1 | ".com/", ".io/", ".org/" |

### Strong Ad Indicators

These patterns trigger span extension (full 60-second forward capture):

- "brought to you by"
- "our sponsor today"
- "this episode is sponsored"
- "word from our sponsor"

### Positional Boosts

Ads are more likely at certain positions:

| Position | Boost | Rationale |
|----------|-------|-----------|
| Intro (first 90s) | +0.2 | Pre-roll sponsor reads |
| Midroll (40-60% of episode) | +0.15 | Natural break points |
| Outro (last 90s) | +0.1 | Post-roll insertions |

### Outro Region Detection

**Constants:**
- `OUTRO_REGION_DURATION = 180.0` (last 3 minutes)
- `OUTRO_MIN_EPISODE_LENGTH = 300.0` (only for episodes > 5 min)

Automatically adds a low-score (0.1) candidate for the outro region since dynamically inserted post-roll ads often lack keyword triggers. The LLM decides whether it's actually an ad.

### Sliding Window Context

Keyword matching uses a 3-segment sliding window to catch phrases split across transcript boundaries:

```
"brought to you" | "by ExpressVPN"  →  detected as single phrase
```

## Stage 4: LLM Refinement

**File:** `adnihilator/ad_llm.py`

### Candidate Merging

Candidates within 5 minutes are merged before sending to LLM:

- Reduces API calls
- Captures back-to-back sponsor reads as logical units

### Context Window

| Direction | Duration | Rationale |
|-----------|----------|-----------|
| Before | 30s | Capture ad lead-in |
| After | 120s | Capture trailing CTAs, back-to-back sponsors |

### Prompt Engineering

**Key distinctions in system prompt:**

- "brought to you by" = AD
- Personal recommendations = NOT an ad
- Gift guides without sponsor intro = NOT an ad
- Promo codes with explicit sponsor read = AD

**Line number output:** The LLM returns line numbers instead of timestamps, which we map back. This reduces token usage and avoids timestamp parsing errors.

### Hunt Mode

When sponsors from show notes aren't found in initial candidates:

1. Find gaps > 10 minutes between detected ads
2. Search 4-minute windows around gap midpoints
3. Look specifically for missing sponsor names
4. Add any found ads with "hunt_mode" reason

This catches ads that lack obvious keyword triggers but mention specific sponsors.

## Stage 5: Splicing

**File:** `adnihilator/splice.py`

### Confidence Threshold

**Current threshold: 0.35**

| Confidence | Treatment |
|------------|-----------|
| ≥ 0.35 | Removed from audio |
| < 0.35 | Kept in audio |

**Lesson learned:** Initially we used 0.5, but this filtered out intro sponsor reads (typically 0.40 confidence). After analysis:
- False positives (product recommendations): ~0.24 confidence
- Intro ads: ~0.40 confidence
- Mid-roll ads: ~0.50+ confidence
- Outro regions (pre-LLM): ~0.08 confidence

The 0.35 threshold catches intro ads while filtering false positives.

### ffmpeg Implementation

**Primary method:** `filter_complex` with `atrim` and `concat`

```
[0:a]atrim=0:45,asetpts=PTS-STARTPTS[a0];
[0:a]atrim=120:180,asetpts=PTS-STARTPTS[a1];
[a0][a1]concat=n=2:v=0:a=1[out]
```

**Fallback method:** Segment extraction + concat demuxer (if filter_complex fails)

## Key Lessons Learned

### 1. Confidence Calibration

Don't assume all ads have similar confidence scores. Intro sponsor reads are often softer ("Speaking of productivity, I want to tell you about...") and score lower than explicit mid-roll ads.

### 2. Whisper Transcription Quirks

Brand names get mangled:
- "Bitwarden" → "bit warden"
- "ExpressVPN" → "express VPN"
- "LMNT" → "element" or "l m n t"

Solution: Generate multiple keyword variants including word-boundary splits.

### 3. Podcast Name as Promo Code

Some podcasts use their name as a promo code ("use code TWIT"). This causes false positives since the podcast name appears throughout the episode.

Solution: Filter out promo codes matching the podcast name.

### 4. Outro Region Special Handling

Dynamically inserted post-roll ads often appear after the hosts sign off, without any sponsor introduction phrases. They need special detection.

Solution: Always flag the last 3 minutes for LLM review with a dedicated "outro_region" candidate.

### 5. External Transcripts Save Time and Improve Accuracy

When available, podcast-provided transcripts are:
- Faster (no Whisper processing)
- More accurate for proper nouns
- Already have timestamped segments

Currently supported:
- Substack podcasts (CloudFront signed JSON, word-level timestamps)
- Lex Fridman podcasts (HTML transcript pages, segment-level timestamps)

### 6. LLM Prompt Efficiency

Using line numbers instead of timestamps in LLM prompts:
- Reduces token count
- Avoids timestamp formatting/parsing issues
- The mapping back to timestamps is trivial

### 7. Content vs Ad Distinction

The hardest cases are personal recommendations that sound like ads. Key tests:
- Does it have an explicit sponsor intro? ("brought to you by")
- Is there a promo code specific to this podcast?
- Is it integrated into content or a clear break?

When in doubt, we err on the side of NOT removing content.

## Configuration

**File:** `adnihilator.toml`

```toml
[llm]
provider = "openai"
api_key_env = "OPENAI_API_KEY"
model = "gpt-4.1-mini"

[audio]
whisper_model = "small"
device = "cpu"
```

## Future Improvements

Areas for potential enhancement:

1. **Adaptive confidence thresholds** - Learn from user feedback on false positives/negatives
2. **Speaker diarization** - Distinguish host reads from guest speech
3. **Ad network fingerprinting** - Detect known dynamically-inserted ads by audio signature
4. **More external transcript sources** - Support additional podcast platforms (e.g., Spotify, Apple Podcasts)
