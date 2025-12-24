# Parallel Detection Pipeline Design

## Problem

Gemini audio detection is non-deterministic - the same episode can yield different results on different runs. It excels at finding dynamically inserted ads but sometimes misses host-read ads. Whisper + keyword detection is reliable for host-read ads but less effective at detecting inserted ads.

## Solution

Run both detection methods in parallel, pass all candidates to LLM for intelligent merging and boundary refinement.

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

4. LLM refinement (handles merging + boundary refinement)
   → Receives BOTH candidate lists + transcript context
   → Merges overlapping candidates intelligently
   → Refines boundaries using transcript
   → final_ads[]

5. Splice & upload

6. Save metadata (which methods flagged each ad, confidence scores)
```

## LLM-Based Merging

Instead of pre-merging candidates before LLM refinement, we pass both candidate lists directly to the LLM. This allows the LLM to:

- **Merge overlapping detections** from both methods intelligently
- **Split incorrectly merged spans** if Gemini grouped distinct ads
- **Choose best boundaries** using transcript evidence
- **Validate candidates** against transcript content

### Why LLM Merge is Better

| Pre-merge | LLM Merge |
|-----------|-----------|
| Fixed 5-second gap rule | Context-aware decisions |
| May collapse back-to-back ads | Can keep distinct ads separate |
| Loses per-source boundaries | Sees all evidence |
| Can't split over-merged spans | Can split if needed |

## LLM Refinement Context

Each candidate passed to the LLM includes:

1. **Detection source and rationale**
2. **Transcript snippet** (30 seconds before/after the candidate)
3. **Original confidence score**

### Example Input to LLM

```python
{
    "gemini_candidates": [
        {
            "start": 1930,
            "end": 2037,
            "reason": "Identified as BetterHelp sponsor read via audio analysis",
            "confidence": 0.98
        }
    ],
    "keyword_candidates": [
        {
            "start": 1925,
            "end": 2100,
            "reason": "Keyword matches: 'sponsored by', 'betterhelp.com', 'use code'",
            "confidence": 0.75,
            "matched_keywords": ["sponsored by", "betterhelp.com", "use code"]
        }
    ],
    "transcript_context": {
        "1900-1930": "...and that's why I think the movie worked so well.",
        "1930-1960": "Speaking of things that work, this episode is sponsored by BetterHelp...",
        "1960-1990": "...online therapy that matches you with a licensed therapist...",
        "1990-2020": "...visit betterhelp.com/filmcast for 10% off your first month...",
        "2020-2050": "...thanks to BetterHelp for sponsoring. Now back to our review...",
        "2050-2080": "So as I was saying about the cinematography..."
    },
    "sponsors": ["BetterHelp"]
}
```

### LLM Prompt Guidance

```
You are refining ad detection candidates. You receive:
- Gemini candidates: Detected via audio analysis. Good at finding ads but timestamps may be slightly off.
- Keyword candidates: Detected via transcript keyword matching. May have imprecise boundaries.
- Transcript context: Use this to find precise ad start/end boundaries.

Your task:
1. Merge overlapping candidates if they refer to the same ad
2. Split candidates if they contain multiple distinct ads
3. Refine boundaries using transcript (find exact transition points)
4. Output final ad spans with confidence scores

An ad starts when the host transitions to promotional content and ends when they return to regular content.
```

## Fallback Behavior

### If Whisper Fails (No Transcript)

```
1. Log error: "Whisper transcription failed, using Gemini-only detection"
2. Use Gemini's ad spans directly (no LLM refinement)
3. Mark detection_source as "gemini_only"
4. Continue with splice & upload
```

Rationale: Gemini provides usable ad boundaries even without refinement. Better to remove some ads imperfectly than fail the entire episode.

### If Gemini Fails

```
1. Log error: "Gemini detection failed, using keyword detection only"
2. Continue with Whisper + keywords + LLM refinement (existing flow)
3. Mark detection_source as "keywords_only"
```

### If Both Fail

```
1. Mark episode as failed
2. Log error with details
3. Existing retry behavior applies
```

## Implementation

### Files to Modify

1. **`worker/daemon.py`**
   - Restructure to run Gemini and Whisper in parallel
   - Handle fallback cases (Whisper fails, Gemini fails)
   - Pass both candidate lists to LLM

2. **`adnihilator/models.py`**
   - Add `sources: list[str]` field to `AdSpan`
   - Add `detection_source` to track which path was used

3. **`adnihilator/ad_llm.py`**
   - Update prompt to handle both candidate lists
   - Include transcript context in prompt
   - Support merge/split operations

4. **Remove: `adnihilator/ad_merge.py`**
   - No longer needed - LLM handles merging

### Estimated Scope

~150-200 lines modified across 3 files.

## Cost & Performance

| Metric | Before | After |
|--------|--------|-------|
| Gemini API | $0.09-0.18 | $0.09-0.18 |
| LLM refinement | $0.004 | ~$0.006 (slightly more context) |
| Whisper | Free (local) | Free (local) |
| **Total cost** | ~$0.10-0.19 | ~$0.10-0.19 |
| Processing time | ~1-2 min | ~2-3 min |

## Success Criteria

1. Host-read ads detected reliably (keyword strength)
2. Dynamically inserted ads detected (Gemini strength)
3. Processing cost stays under $0.20/episode
4. Processing time under 5 minutes for typical episodes
5. Graceful degradation when one detection method fails

## External Review Feedback (Incorporated)

Based on reviews from GPT-5.2 and Gemini 2.5 Pro:

- ✅ **LLM-merge instead of pre-merge** - More flexible, can split/merge intelligently
- ✅ **Transcript context added** - LLM can ground boundary decisions in actual text
- ✅ **No-transcript fallback defined** - Use Gemini-only if Whisper fails
- ✅ **Evidence-based confidence** - LLM assigns confidence based on evidence quality
