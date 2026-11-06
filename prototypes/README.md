> Historical standalone prototype. The integrated feature and current validation are documented in [docs/transcript-search.md](../docs/transcript-search.md).

# Local transcript-search proof of concept

`transcript_search.py` runs offline using only the Python standard library and
SQLite FTS5. It does not import the worker, call an API, transcribe, or change the
web schema. Inputs are read-only; index and reports live outside worker data.

## Reproduce

From `/Users/jfox/Documents/Codex/2026-10-02/task-2`:

```bash
python3 adnihilator-poc/prototypes/transcript_search.py ingest \
  --feed local-output/animal-spirits-feed.xml \
  --artifacts /Users/jfox/projects/adnihilator/data/artifacts \
  --podcast-id d32207be-371c-42db-9532-c6983577d6f3 \
  --limit 20 --db local-output/transcripts.sqlite \
  --inventory local-output/coverage.json

python3 adnihilator-poc/prototypes/transcript_search.py search \
  'What book were they talking about in my latest episode of Animal Spirits? The one guy really loved it.' \
  --podcast 'Animal Spirits Podcast' --latest \
  --context-seconds 60 --db local-output/transcripts.sqlite

python3 adnihilator-poc/prototypes/transcript_search.py search \
  'millionaire memoir McNally' --podcast 'Animal Spirits Podcast' --latest \
  --context-seconds 60 --db local-output/transcripts.sqlite

python3 adnihilator-poc/prototypes/transcript_search.py search \
  'book' --podcast 'Animal Spirits Podcast' --after 2026-09-01 \
  --before 2026-10-01 --db local-output/transcripts.sqlite

python3 adnihilator-poc/prototypes/transcript_search.py search \
  'xylophonemars' --podcast 'Animal Spirits Podcast' --latest \
  --db local-output/transcripts.sqlite

python3 -m unittest discover -s adnihilator-poc/tests -p test_transcript_search_poc.py -v
cd adnihilator-poc
/Users/jfox/projects/adnihilator/.venv/bin/python -m pytest -q
```

No new dependencies are needed. A SQLite build without FTS5 will fail explicitly.
Search opens SQLite in read-only mode. Ingestion refuses an existing database
without the prototype marker and refuses outputs under the input artifact root.

## Design and data availability

The isolated clone starts from local commit
`01f211a4c81e9442660fbb465ca8b75d8e1209f3`, branch
`codex/transcript-search-poc`. The original checkout has four untracked entries
and was left untouched. The local web database is an old snapshot containing two
podcasts and no Animal Spirits. The launchd runner uses the hosted service; its
CLI enables `data/artifacts` by default. No hosted admin state was read or changed.

`worker/daemon.py` writes timestamped transcripts before splicing and detection
JSON with segments. JSON is preferred to rounded text timestamps. Text is a
fallback when JSON has no segments. A matching artifact without either remains
`no_segments`; no matching artifact is `missing`. Multiple artifacts for a media
identity are `ambiguous_artifacts`, never silently selected.

The public RSS snapshot came from `https://feeds.megaphone.fm/TCP6464651487`.
Megaphone enclosure media ID matches identify actual local worker artifacts;
tracking wrappers and URL query changes do not change media identity. Stable
index identity is a hash of podcast ID plus RSS GUID. Original worker episode ID
and both source paths are retained. RSS publication dates are normalized to UTC,
not inferred from file modification times. The public feed cannot prove hosted
queue status or what Jon last played in his player.

845 transcript text files exist across 20 podcast artifact directories. The
Animal Spirits directory contains 43 detection JSON artifacts and 42 transcript
text files; all 43 media identities match the current feed. One older artifact,
“The Biggest Risk in 2026 (EP. 444),” has no segments. The index selects the latest
20 **feed episodes**, retaining nine missing episodes explicitly, and indexes 11
available transcripts (15,304 segments). The missing recent entries are “Talk
Your Book” episodes. They are not presumed transcribed or processed elsewhere.

The newest local matched artifact and newest RSS entry both identify “Will Higher
Rates Kill the Stock Market? (EP. 484),” September 30, 2026, 10:00 UTC. Its RSS GUID
is `b3acf696-ab19-11f1-8f22-bbbc7cf2098c`; worker episode ID is
`a4d3d35d-119f-4e5e-92c7-56915b397e73`. It has 1,319 Parakeet segments and an
original duration of 3,595.154278 seconds. No new transcription was needed.

## Retrieval behavior

The `episodes.canonical_json` column preserves the complete segment list,
original indices, word timestamps, full timestamped text when available,
episode identity, source URLs, artifact paths, and transcript model metadata.
`segments` stores source sequence and exact numeric start/end times; `passages`
provides FTS5 ranking. Re-ingestion replaces the episode atomically and removes
stale FTS rows. The index is a local snapshot; ingesting a narrower selection
does not purge previously ingested episodes. Use a new index for a different
strict selection window.

Queries are tokenized, quoted, and combined with lexical OR. Common question
words and a scoped podcast's name are omitted. This retrieves evidence, not a
semantic answer. “Loved” can also retrieve a film discussion; inspect context or
refine terms. Matches include surrounding conversation by source timestamps,
deduplicate overlapping contexts, and expose episode provenance. `--before` is
exclusive. `--latest` resolves episode ordering before coverage or date filters;
a missing latest transcript cannot quietly fall back to an older episode.

Original-audio timestamps are explicitly labeled. They do not refer to the
processed/ad-free enclosure, and dynamic ad insertion can affect a fresh audio
download. Existing small overlaps/out-of-order timings are retained, not repaired.
No speaker diarization is present in these artifacts.

## Real-query result and limits

Strongest candidate: **The Everywhere Millionaire: Who Is Really Rich in America
and How They Got There**, by **Owen Zidar and Eric Zwick**. At original-audio
57:24–58:07, the host calls it “the finance book of the year” (57:28.24), says
“It's so good,” and the title is named at 57:53.84. The author/title spelling is
verified against the [publisher](https://us.macmillan.com/books/9781250378507/theeverywheremillionaire/)
and [authors' site](https://www.everywheremillionaire.com/); author names are not
spoken in this passage. Confidence is high that this recommendation occurs in
the latest episode, and medium-high that it is the book Jon remembers.

Alternatives: Keith McNally's memoir at 47:50–48:37 (“I love memoirs,” “I just
finished Keith McNally,” “Anyway, great book”). Its title, *I Regret Almost
Everything*, is external bibliographic corroboration from
[AP](https://apnews.com/article/21d5881808440f6323cfe2f64f90d8f7), not a title
spoken here. Next, at 48:37–48:50, another praised book by Ari Emanuel is called
“awesome,” with Bob Odenkirk identified as narrator. Its title is not stated in
this passage. The speaker mentions a forthcoming Sylvester Stallone memoir,
but says he is looking forward to it rather than having finished it. No host
name is assigned to any of these recommendations without diarization/audio
verification. The memoir host addresses Ben, which suggests Michael but does
not supply reliable segment-level speaker identity.

## Validation and next integration steps

10 isolated prototype tests cover ordering/identity, missing latest, latest with
date bounds, repeat ingestion, replacement of stale search terms, timestamp and
word preservation, context, podcast/date filters, negative queries, ambiguous
media identities, text fallback, literal query syntax, and protection of unrelated
databases. Final repository suite: **191 passed, one existing dependency warning**.
There was an intermediate empty-filter regression during development; it was
fixed and both negative-filter and latest/date-bound tests now pass.

Saved evidence lives in `/Users/jfox/Documents/Codex/2026-10-02/task-2/local-output/`:
`coverage.json`, `real-query.json`, `book-candidates.json`, `negative-query.json`,
`prototype-tests.txt`, `pytest.txt`, `provenance.json`, `book-passage.json`, and
`transcript-search-report.html`. The database is `transcripts.sqlite`.

Not tested: audio listening/diarization, hosted subscription/queue state, automatic
feed refresh, multi-podcast scale, cross-provider enclosure identity, or processed
timestamp conversion. Public RSS lacks an episode-page link here. The latest
target is covered, so no recovery work is necessary. For a future missing target,
first look for its existing detection/text artifact or publisher transcript; price
and benchmark a single local transcription before starting it. No bulk processing.

Possible integration: associate transcripts with the existing Episode GUID/ID;
persist canonical original-audio segments independently of detection fast paths;
run a separate idempotent indexer; surface coverage warnings and provenance; add
an explicit original-to-processed time mapping only if ad-free playback needs it.
Those are proposals; this prototype changes no production models or worker code.
