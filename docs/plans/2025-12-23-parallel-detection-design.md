# Parallel Detection Pipeline Design

## Problem

Gemini audio detection is non-deterministic - the same episode can yield different results on different runs. It excels at finding dynamically inserted ads but sometimes misses host-read ads. Whisper + keyword detection is reliable for host-read ads but less effective at detecting inserted ads.

## Solution

Run both detection methods in parallel, merge candidates, then use LLM refinement to produce precise final boundaries.

## Pipeline Flow

```
1. Download audio
2. Extract sponsors from description

3. PARALLEL:
   ├─ 3a. Gemini audio detection (with sponsors + title)
   │       → gemini_candidates[]
   │
   └─ 3b. Whisper transcription
          → Run keyword detection
          → keyword_candidates[]

4. Merge candidates from both methods
   → all_candidates[]

5. LLM refinement on merged candidates
   → Uses transcript to refine ALL boundaries
   → final_ads[]

6. Splice & upload

7. Save metadata (which methods flagged each ad, confidence scores)
```

## Merge Logic

### Overlap Handling

If two spans overlap or are within 5 seconds of each other, merge into one:

```
Gemini:   [====100s-200s====]
Keywords:      [====150s-250s====]
Result:   [========100s-250s========]
```

Take earliest start, latest end.

### Confidence Scoring

- Ads found by **both** methods → confidence 0.95
- Ads found by **Gemini only** → confidence 0.90
- Ads found by **keywords only** → confidence 0.75

### Source Tracking

Each candidate tracks its detection source(s):

```python
AdSpan(
    start=100, end=250,
    confidence=0.95,
    reason="Merged: Gemini + Keywords",
    sources=["gemini", "keywords"]
)
```

## LLM Refinement Context

Each candidate passed to the LLM includes detection context:

```python
{
    "start": 1930,
    "end": 2037,
    "source": "gemini",
    "context": "Detected via Gemini audio analysis. Identified as BetterHelp sponsor read.",
    "confidence": 0.98
}

{
    "start": 1925,
    "end": 2100,
    "source": "keywords",
    "context": "Keyword matches: 'sponsored by', 'betterhelp.com', 'use code'.",
    "confidence": 0.75
}
```

LLM prompt guidance:
- Gemini audio detection provides good ad identification but timestamps may be slightly off
- Keyword detection identifies regions with ad language - use transcript for precise boundaries
- Merge overlapping candidates intelligently

## Implementation

### Files to Modify

1. **`worker/daemon.py`**
   - Restructure to run Gemini and Whisper in parallel
   - Call merge function before LLM refinement
   - Pass detection context to LLM

2. **`adnihilator/models.py`**
   - Add `sources: list[str]` field to `AdSpan`

3. **`adnihilator/ad_merge.py`** (new file)
   - `merge_ad_spans(gemini_ads, keyword_ads) -> list[AdSpan]`
   - Overlap detection, merging, confidence scoring
   - ~50-80 lines

4. **`adnihilator/ad_llm.py`**
   - Update prompt to include detection source context
   - Accept candidates with source metadata

### Estimated Scope

~150-200 lines modified/added across 4 files.

## Cost & Performance

| Metric | Before | After |
|--------|--------|-------|
| Gemini API | $0.09-0.18 | $0.09-0.18 |
| LLM refinement | $0.004 | $0.004 |
| Whisper | Free (local) | Free (local) |
| **Total cost** | ~$0.10-0.19 | ~$0.10-0.19 |
| Processing time | ~1-2 min | ~2-3 min |

Parallel execution keeps time increase minimal despite running both methods.

## Fallback Behavior

- If Gemini fails → Whisper + keywords + LLM still works
- If Whisper fails → Gemini results still available
- Both methods failing → Episode marked as failed (existing behavior)

## Success Criteria

1. Host-read ads detected reliably (keyword strength)
2. Dynamically inserted ads detected (Gemini strength)
3. Processing cost stays under $0.20/episode
4. Processing time under 5 minutes for typical episodes
