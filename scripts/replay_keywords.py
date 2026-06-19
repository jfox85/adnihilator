"""Offline replay of the keyword detection pipeline against debug artifacts.

NO AI inference, NO splicing. Reconstructs the inputs to `find_ad_candidates`
from saved artifacts and runs it, so we can A/B a current vs. candidate version
of the keyword logic across the whole backlog and gate on regressions.

Guardrail cohort: TWiT-network shows do long *host-read* ads. We use the saved
Gemini `host_read` candidate regions as a proxy "must still cover" target and
report coverage so a rule change can't silently shred long host-reads.

Usage:
    python scripts/replay_keywords.py [artifacts_dir]
"""

import contextlib
import glob
import io
import json
import os
import sys
from statistics import mean, median

from adnihilator.ad_keywords import find_ad_candidates
from adnihilator.models import Sponsor, SponsorInfo, TranscriptSegment
from worker.daemon import WorkerDaemon


def load_episode(path):
    d = json.load(open(path))
    segs = [
        TranscriptSegment(
            index=s.get("index", i),
            start=s["start"],
            end=s["end"],
            text=s["text"],
            words=s.get("words") or [],
        )
        for i, s in enumerate(d.get("segments", []))
    ]
    dur = d.get("duration") or 0.0

    # Reconstruct SponsorInfo from sibling *_sponsors.json (best source) if present
    sponsors = None
    seg_id = os.path.splitext(os.path.basename(path))[0]
    sp_path = os.path.join(os.path.dirname(path), f"{seg_id}_sponsors.json")
    if os.path.exists(sp_path):
        try:
            sp = json.load(open(sp_path))
            sponsors = SponsorInfo(
                sponsors=[
                    Sponsor(name=x.get("name", ""), url=x.get("url"), code=x.get("code"))
                    for x in sp.get("sponsors", [])
                ],
                extraction_method=sp.get("extraction_method", "patterns") if sp.get("extraction_method") in {"patterns", "llm", "none"} else "patterns",
            )
        except Exception:
            sponsors = None

    # Host-read guardrail regions from saved Gemini candidates
    host_reads = [
        (g["start"], g["end"])
        for g in (d.get("model_info", {}).get("gemini_candidates_raw") or [])
        if g.get("ad_type") == "host_read" and g["end"] - g["start"] >= 60
    ]
    return d, segs, dur, sponsors, host_reads


def removed_seconds(spans, min_conf=0.0):
    """Total seconds covered by spans (union), to avoid double counting overlaps."""
    iv = sorted((s.start, s.end) for s in spans if getattr(s, "heuristic_score", 1) >= min_conf)
    total = 0.0
    cur_s = cur_e = None
    for a, b in iv:
        if cur_e is None or a > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = a, b
        else:
            cur_e = max(cur_e, b)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def coverage(spans, regions):
    """Fraction of each guardrail region covered by spans; returns mean over regions."""
    if not regions:
        return None
    covs = []
    for rs, re in regions:
        rlen = re - rs
        if rlen <= 0:
            continue
        covered = 0.0
        for s in sorted(spans, key=lambda x: x.start):
            lo, hi = max(rs, s.start), min(re, s.end)
            if hi > lo:
                covered += hi - lo
        covs.append(min(covered, rlen) / rlen)
    return mean(covs) if covs else None


def run_variant(segs, dur, sponsors, **kwargs):
    return find_ad_candidates(segs, dur, sponsors=sponsors, **kwargs)


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "data/artifacts"
    files = [f for f in glob.glob(os.path.join(base, "*/*.json")) if not f.endswith("sponsors.json")]

    twit_hint = ("twit", "security now", "macbreak", "intelligent machines", "this week in")
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    rows = []
    for f in files:
        try:
            d, segs, dur, sponsors, host_reads = load_episode(f)
        except Exception as e:
            print(f"skip {f}: {e}")
            continue
        if not segs or not dur:
            continue
        blob = json.dumps(d.get("model_info", {})).lower()
        is_twit = any(h in blob for h in twit_hint)

        cands = run_variant(segs, dur, sponsors)

        sponsor_names = [s.name for s in sponsors.sponsors] if sponsors and sponsors.sponsors else []
        raw_gemini = d.get("model_info", {}).get("gemini_candidates_raw") or []
        with contextlib.redirect_stdout(io.StringIO()):
            raw_validated, raw_rejected = daemon._validate_gemini_candidates(
                raw_gemini,
                segs,
                sponsor_names,
                duration=dur,
            )
        rescued = sum(1 for c in raw_validated if "timestamp rescued" in c.get("reason", ""))

        rows.append({
                         "f": os.path.basename(f)[:14],
            "dur": dur,
            "twit": is_twit,
            "n": len(cands),
            "removed_pct": 100 * removed_seconds(cands) / dur,
            "hostread_cov": coverage(cands, host_reads),
            "n_hostread": len(host_reads),
            "rescued_gemini": rescued,
            "rejected_gemini": len(raw_rejected),
        })

    def report(label, rs):
        if not rs:
            return
        pcts = [r["removed_pct"] for r in rs]
        covs = [r["hostread_cov"] for r in rs if r["hostread_cov"] is not None]
        over30 = sum(1 for p in pcts if p > 30)
        print(f"\n[{label}] n={len(rs)}")
        print(f"  removed%: median={median(pcts):.1f} mean={mean(pcts):.1f} max={max(pcts):.1f}  >30%:{over30}")
        if covs:
            print(f"  host-read coverage: median={median(covs):.2f} mean={mean(covs):.2f} "
                  f"min={min(covs):.2f}  episodes-with-hostread={len(covs)}")
        print(
            f"  gemini timestamp rescue: rescued={sum(r['rescued_gemini'] for r in rs)} "
            f"still-rejected={sum(r['rejected_gemini'] for r in rs)}"
        )

    report("ALL", rows)
    report("TWiT cohort", [r for r in rows if r["twit"]])
    report("non-TWiT", [r for r in rows if not r["twit"]])


if __name__ == "__main__":
    main()
