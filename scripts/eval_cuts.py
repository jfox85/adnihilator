"""Offline evaluation of ad-detection cut quality from debug artifacts.

Reads DetectionResult JSON artifacts (segments + ad_spans + model_info) and
computes heuristics that approximate detection/splice accuracy WITHOUT any
ground-truth labels, so we can spot likely false positives / boundary errors.

Usage:
    python scripts/eval_cuts.py [artifacts_dir]
"""

import glob
import json
import os
import sys
from statistics import mean, median

from adnihilator.models import AdSpan, TranscriptSegment
from adnihilator.splice import _snap_span_to_transcript

MIN_CONF = 0.35  # splice threshold (adnihilator/splice.py)


def words_in_span(segments, start, end):
    """Return concatenated transcript text whose words fall within [start, end]."""
    out = []
    for seg in segments:
        if seg["end"] < start or seg["start"] > end:
            continue
        out.append(seg["text"])
    return " ".join(out).strip()


def boundary_cut_word(segments, t):
    """How far (s) is time t from the nearest segment boundary? Big = mid-sentence cut."""
    best = None
    for seg in segments:
        for edge in (seg["start"], seg["end"]):
            d = abs(edge - t)
            if best is None or d < best:
                best = d
    return best if best is not None else 0.0


def analyze(path):
    d = json.load(open(path))
    segs = d.get("segments", [])
    model_segments = [TranscriptSegment(**s) for s in segs]
    spans = d.get("ad_spans", [])
    dur = d.get("duration", 0) or 0
    kept = [s for s in spans if s.get("confidence", 1) >= MIN_CONF]

    r = {
        "ep": os.path.basename(path),
        "dur": dur,
        "n_spans": len(spans),
        "n_kept": len(kept),
        "removed_s": sum(s["end"] - s["start"] for s in kept),
        "flags": [],
    }
    if dur:
        r["removed_pct"] = round(100 * r["removed_s"] / dur, 1)

    # Heuristic red flags
    for s in kept:
        length = s["end"] - s["start"]
        text = words_in_span(segs, s["start"], s["end"]) if segs else ""
        # Very long single cut (>4 min) — risk of cutting real content
        if length > 240:
            r["flags"].append(f"LONG_CUT {length:.0f}s @ {s['start']:.0f} ({s.get('reason','')[:40]})")
        # Mid-sentence boundary (>1.5s from any segment edge) when we have transcript
        if segs:
            snapped = _snap_span_to_transcript(AdSpan(**s), model_segments, dur)
            for edge_t, name in ((s["start"], "start"), (s["end"], "end")):
                off = boundary_cut_word(segs, edge_t)
                if off > 1.5:
                    r["flags"].append(f"MIDSENTENCE {name} off={off:.1f}s @ {edge_t:.0f}")
            r.setdefault("raw_boundaries", 0)
            r.setdefault("snapped_boundaries_changed", 0)
            r["raw_boundaries"] += 2
            if abs(snapped.start - s["start"]) > 0.01:
                r["snapped_boundaries_changed"] += 1
            if abs(snapped.end - s["end"]) > 0.01:
                r["snapped_boundaries_changed"] += 1
        # Keyword-only low-confidence near the very start (classic FP: intro 'sponsor' mention)
        if s["start"] < 5 and s.get("sources") == ["keywords"] and s.get("confidence", 1) < 0.45:
            r["flags"].append(f"START_FP conf={s.get('confidence')} len={length:.0f}s")

    # Overlap between kept spans (double-counting / merge failure)
    sk = sorted(kept, key=lambda x: x["start"])
    for a, b in zip(sk, sk[1:]):
        if b["start"] < a["end"] - 0.5:
            r["flags"].append(f"OVERLAP {a['end']:.0f}>{b['start']:.0f}")

    return r


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "data/artifacts"
    files = [f for f in glob.glob(os.path.join(base, "*/*.json")) if not f.endswith("sponsors.json")]
    rows = []
    for f in files:
        try:
            rows.append(analyze(f))
        except Exception as e:
            print(f"skip {f}: {e}")

    rows = [r for r in rows if r.get("dur")]
    print(f"episodes analyzed: {len(rows)}")
    pcts = [r["removed_pct"] for r in rows if "removed_pct" in r]
    print(f"removed% — median {median(pcts):.1f}  mean {mean(pcts):.1f}  max {max(pcts):.1f}")

    # Distribution buckets
    hi = [r for r in rows if r.get("removed_pct", 0) > 25]
    lo = [r for r in rows if r.get("removed_pct", 0) < 2]
    print(f"suspicious HIGH removal (>25%): {len(hi)}   suspicious LOW (<2%): {len(lo)}")

    # Flag frequency
    from collections import Counter
    fc = Counter()
    for r in rows:
        for fl in r["flags"]:
            fc[fl.split()[0]] += 1
    print("flag counts:", dict(fc))
    raw_boundaries = sum(r.get("raw_boundaries", 0) for r in rows)
    snapped_changed = sum(r.get("snapped_boundaries_changed", 0) for r in rows)
    if raw_boundaries:
        print(f"snap candidates: changed {snapped_changed}/{raw_boundaries} boundaries ({100*snapped_changed/raw_boundaries:.1f}%)")

    # Show worst offenders
    rows.sort(key=lambda r: len(r["flags"]), reverse=True)
    print("\n=== top 8 episodes by flag count ===")
    for r in rows[:8]:
        print(f"\n{r['ep']}  dur={r['dur']:.0f}s kept={r['n_kept']}/{r['n_spans']} removed={r.get('removed_pct','?')}%")
        for fl in r["flags"][:6]:
            print("   -", fl)


if __name__ == "__main__":
    main()
