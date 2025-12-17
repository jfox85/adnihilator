# External Transcript Fetching

## Overview

Skip Whisper transcription when external transcripts are available. Currently supports:

1. **Substack podcasts** (like Lenny's Podcast) - JSON transcripts with word-level timestamps
2. **Lex Fridman podcasts** - HTML transcripts with segment-level timestamps

## Data Flow

1. **Check for transcript URL** - Fetch episode's source page (from RSS `<link>`)
2. **Extract JSON URL** - Parse HTML for `transcription.json` CloudFront URL
3. **Fetch transcript** - Download the JSON transcript
4. **Convert to internal format** - Map to `TranscriptSegment` model
5. **Skip Whisper** - Proceed directly to ad detection
6. **Fallback** - If any step fails, fall back to Whisper

**Expected savings:** 95%+ of transcription time when transcript available.

## Implementation

### New module: `adnihilator/external_transcript.py`

```python
def fetch_external_transcript(episode_page_url: str) -> list[TranscriptSegment] | None:
    """
    Attempt to fetch transcript from any supported platform.
    Tries Lex Fridman, then Substack.
    Returns None if not available (triggers Whisper fallback).
    """
```

**Functions:**

**Substack support:**
- `extract_transcript_url(html: str) -> str | None` - Regex for `transcription.json` URLs
- `parse_substack_transcript(data: list[dict]) -> list[TranscriptSegment]` - Format conversion
- `fetch_substack_transcript(episode_page_url: str)` - Fetch Substack transcript

**Lex Fridman support:**
- `parse_lexfridman_transcript(html: str) -> list[TranscriptSegment] | None` - Parse HTML transcript
- `fetch_lexfridman_transcript(episode_page_url: str)` - Fetch Lex transcript from `-transcript` page

**Unified interface:**
- `fetch_external_transcript(episode_page_url: str)` - Orchestrator that tries all platforms

### Integration in worker daemon

```python
# In worker/daemon.py process_job()
transcript_segments = fetch_external_transcript(episode.source_url)
if transcript_segments is None:
    transcript_segments = transcribe_audio(audio_path)
```

The function automatically detects the platform and uses the appropriate parser.

## Format Mapping

### Substack Format

**Substack JSON format:**
```json
{
  "start": 0.06,
  "end": 5.331,
  "text": "You guys hit a billion...",
  "words": [{"word": "You", "start": 0.06, "end": 0.16, "score": 0.906}],
  "speaker": "SPEAKER_0"
}
```

**Our TranscriptSegment:**
- `start`, `end`, `text` → direct copy
- `words` → map to `WordTiming`, filter missing timestamps
- `index` → assign sequentially
- `speaker` → ignore (not used in ad detection)

### Lex Fridman Format

**Lex Fridman HTML format:**
- Transcript available at `{episode_url}-transcript`
- Each segment has: speaker name + `[HH:MM:SS]` timestamp + text
- Timestamps link to YouTube with `t=SECONDS` parameter
- Segment-level timestamps only (no word-level)

**Our TranscriptSegment:**
- `start` → extracted from YouTube `t=` parameter (converted to seconds)
- `end` → computed from next segment's start (last segment gets +60s buffer)
- `text` → full segment text with timestamp removed
- `words` → None (no word-level data available)
- `index` → assign sequentially

## Error Handling

All failures trigger Whisper fallback:
- No source URL in RSS
- Page fetch fails (5s timeout)
- No transcript URL in page HTML
- Signed URL expired
- JSON parse error
- Empty transcript

## Metadata Tracking

```python
model_info = {
    "transcript_source": "substack" | "lexfridman" | "whisper",
    "whisper_model": "small",  # only if whisper used
}
```

The worker automatically detects the source and sets the appropriate `transcript_source` value.
