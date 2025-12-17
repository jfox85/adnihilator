# Sponsor Extraction from Show Notes

## Problem

Podcast show notes often contain sponsor information (names, URLs, promo codes) that could improve ad detection accuracy. Currently we rely solely on transcript analysis, missing this valuable signal.

## Solution

Extract sponsor info from episode descriptions and use it to:
1. Add episode-specific keywords to the heuristic detector
2. Provide context to the LLM for better boundary detection
3. Hunt for missing ads when not all sponsors are found

## Data Model

```python
class Sponsor(BaseModel):
    name: str
    url: str | None = None
    code: str | None = None

class SponsorInfo(BaseModel):
    sponsors: list[Sponsor]
    extraction_method: str  # "patterns", "llm", or "none"
```

## Extraction Strategy (Hybrid)

### Step 1: Pattern Matching
Try regex patterns for common formats:
- HTML: `<b>Sponsors:</b>` or `<strong>Sponsors:</strong>` sections with `<li>` items
- Plain text: "Sponsor Deals", "Partner Deals" headers followed by lists
- Simple: "Sponsors: Name1, Name2, Name3"

### Step 2: LLM Fallback
If patterns find nothing but description contains trigger words ("sponsor", "partner", "code", "deal", "discount"), call LLM to extract sponsors.

### Output
Save as artifact: `{episode_id}_sponsors.json` for debugging.

## Keyword Detection Integration

### Dynamic Keywords
Generate variations from sponsor data:
- Name: `"ExpressVPN"` → `["expressvpn", "express vpn"]`
- CamelCase split: `"CodeRabbit"` → `["coderabbit", "code rabbit"]`
- URL slugs: `"shopify.com/lex"` → `["shopify", "shopify dot com slash lex"]`
- Promo codes: `"code TWIT"` as high-weight trigger

### Tracking
Add to `AdCandidate`:
- `sponsors_found: list[str]` - sponsors matched in transcript
- `sponsors_missing: list[str]` - sponsors from notes not found

## LLM Integration

### Normal Refinement
Pass sponsor list to LLM prompt:
```
Known sponsors from show notes:
- Shopify (shopify.com/lex)
- CodeRabbit (coderabbit.ai/lex)
- LMNT (drinkLMNT.com/lex)
```

### Hunt Mode
When `sponsors_missing` is not empty after keyword detection:

1. Identify gaps > 10 minutes between detected ads
2. Send ~3-4 minute windows around gap midpoints to LLM
3. Prompt: "We expect an ad for [Sponsor] but haven't found it. Check this segment."
4. Cap at 2-3 hunt attempts per missing sponsor

## Pipeline Changes

### worker/daemon.py
```python
def process_job(self, job: EpisodeJob):
    # Extract sponsors (new step)
    sponsor_info = extract_sponsors(job.description)

    # Pass to keyword detection
    candidates = find_ad_candidates(segments, duration, sponsors=sponsor_info)

    # Pass to LLM (enables hunt mode)
    ad_spans = llm_client.refine_candidates(
        segments, candidates, self.config, sponsors=sponsor_info
    )

    # Save artifact
    if self.artifacts_dir:
        (artifact_dir / f"{job.id}_sponsors.json").write_text(
            sponsor_info.model_dump_json(indent=2)
        )
```

### worker/client.py
Add `description: str | None` to `EpisodeJob`.

### web/routes/api.py
Include episode description in `/api/queue/claim` response.

## Files to Create/Modify

| File | Change |
|------|--------|
| `adnihilator/sponsors.py` | NEW - extraction module |
| `adnihilator/models.py` | Add `Sponsor`, `SponsorInfo`, update `AdCandidate` |
| `adnihilator/ad_keywords.py` | Accept `sponsors` param, generate dynamic keywords |
| `adnihilator/ad_llm.py` | Accept `sponsors` param, implement hunt mode |
| `worker/daemon.py` | Integrate sponsor extraction into pipeline |
| `worker/client.py` | Add `description` to `EpisodeJob` |
| `web/routes/api.py` | Return description in claim response |

## Future Enhancement

Timestamp-based ad detection: Some podcasts (e.g., Lex Fridman) include chapter timestamps like:
```
(00:29) – Sponsors, Comments, and Reflections
(10:09) – Biological intelligence
```

This could allow skipping transcription entirely for well-structured podcasts. Deferred to follow-up work.
