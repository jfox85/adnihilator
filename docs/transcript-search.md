# Local transcript search feature

Implemented on isolated branch `codex/transcript-search-poc` in `/Users/jfox/Documents/Codex/2026-10-02/task-2/adnihilator-poc`. The original checkout, running worker and podcast artifacts remain unchanged. No push, deployment, production migration, credential creation/rotation, paid inference, or Library publication was performed.

## Grounded outcome

The latest Animal Spirits artifact matches the latest entry in the saved public RSS: **Will Higher Rates Kill the Stock Market? (EP. 484)**, published **2026-09-30 10:00 UTC**, GUID `b3acf696-ab19-11f1-8f22-bbbc7cf2098c`. Original worker episode ID: `a4d3d35d-119f-4e5e-92c7-56915b397e73`. Its 1,319 Parakeet segments cover 3,595.154278 seconds of original audio. No episode-page link appears in this feed item; the reader supplies the actual original enclosure and feed provenance.

Jon confirmed the leading candidate: **The Everywhere Millionaire: Who Is Really Rich in America and How They Got There**, by **Owen Zidar and Eric Zwick**. Original-audio discussion is approximately **57:24–58:07**:

- 57:28.24: “This is like the finance book of the year.”
- 57:30.16: “It's so good.”
- 57:53.84: “It's called Everywhere Millionaire.”
- 57:56.00: “I thoroughly enjoyed it.”

The first title recognition is misspelled “Everware”; the later explicit title and surrounding enthusiasm support high confidence. Authors/full subtitle are verified externally, not spoken in this excerpt: [publisher](https://us.macmillan.com/books/9781250378507/theeverywheremillionaire/) and [book site](https://www.everywheremillionaire.com/). There is no diarization, so no named speaker is asserted. Other positive book discussions exist: Keith McNally's memoir at 47:50–48:37 (title not spoken; externally identified as *I Regret Almost Everything*) and an Ari Emanuel audiobook at about 48:37–48:50 (title not established here). Search returns these alternatives rather than hard-coding a book answer.

Across existing worker data: 845 transcript text files in 20 podcast folders. Animal Spirits has 43 detection artifacts and 42 text transcripts; all 43 matched RSS IDs. One older artifact lacks segments. Among the latest 20 feed episodes, **11 have available canonical transcripts and 9 are missing**, mostly Talk Your Book episodes. Latest publication is resolved before transcript coverage or date filters; missing latest content produces a warning and never silently substitutes an older episode. Historical download times are unknown rather than inferred from file mtimes.

## Architecture and changes

`adnihilator/transcript_document.py` validates versioned canonical documents, exact source segment indices/timestamps, word timing dictionaries, ad spans and metadata. Canonical SHA256 identifies immutable revisions. Source artifact identity/checksum are retained. Malformed, oversized and invalid-timing inputs fail closed; overlap and source order remain intact. External supplied transcript timestamps are labeled `external_source` because original-audio alignment is unverified. No processed/ad-free seek timestamp is invented.

`web/models.py` adds coverage, immutable versions and separate transcript jobs. `web/transcript_migration.py` adds these tables plus a rebuildable FTS5 index. Canonical data and its active pointer commit atomically; indexing can fail independently while canonical content remains available with an index-pending state. Repeat ingestion deduplicates. An old revision retry cannot replace the active newer version.

`web/services/transcripts.py` is the shared bounded read/storage facade. Lexical search supports podcast/date/latest filters, literal tokens or phrases, surrounding conversation and stable revision/ordinal references. Responses carry episode identity, source URLs, UTC dates, timestamp basis and coverage. Reader pages preserve complete segments; search previews are explicitly bounded/cropped, with offsets and truncation flags. Transcript text is treated as untrusted data. UI escapes all text and inserts only controlled highlight markup.

The separate recovery queue supports one explicitly requested episode, claim fencing, heartbeat leases, bounded retries and clear missing/queued/processing/failed/available states. It does not change audio processing status or keys. The worker offers an explicit recovery opportunity after at most three audio jobs, and also when the audio queue is empty. It never preclaims and abandons an audio job. This is fairness by job count, not a wall-clock latency guarantee; a long current job still blocks the next opportunity. A background lease timer spans download, external fetch and ASR, supplementing stage progress heartbeats.

`worker/transcripts.py` writes fsync/atomic durable local canonical archives and pending upload markers. Temporary errors retry and move behind newer pending records, preventing a few failures from occupying every bounded upload cycle. HTTP 404/503 from an endpoint not yet deployed remain pending, preserving recoverable data. Permanent malformed/identity-conflict errors retain a rejected archive for inspection. Upload/disk failures cannot turn a successfully processed audio job into a failure. Fast paths without transcription remain uncovered; no fake empty transcript is published. Explicit transcript-only jobs use existing local engines/free external transcript sources and do not invoke the paid ad detector or publish new audio.

## Authenticated surfaces

UI and mutations retain existing admin Basic/session authentication. Transcript REST GETs and the read-only MCP adapter additionally accept an optional separate Bearer credential when the owner configures `TRANSCRIPT_READ_TOKEN_SHA256` to its lowercase SHA256 hash. This boundary is disabled by default; no credential/hash was created or configured outside synthetic tests. It grants transcript reads across all podcasts only, never admin UI, recovery requests, worker operations or other admin routes. Removing/changing the hash revokes it. There is no expiry, per-client/per-podcast ACL or OAuth discovery; owner secret-management and rotation policy are required. Do not distribute the full admin password as the agent configuration. RSS feed tokens and worker keys cannot read transcripts. Worker keys only authorize transcript publication/job operations. Responses use `Cache-Control: no-store`. Session-authenticated recovery writes require same-origin Origin; any supplied cross-origin Origin is rejected. Upload/MCP request sizes and read result sizes are capped.

- UI: `/transcripts`, `/transcripts/episodes/{episode_id}`; episode list links show coverage.
- REST: `GET /api/transcripts/episodes`, `/search`, `/episodes/{id}`, `/episodes/{id}/coverage`; `POST /episodes/{id}/request` queues a single explicit recovery.
- Worker: `PUT /api/worker/transcripts/{id}`; claim/fail/heartbeat under `/api/worker/transcript-jobs`.
- MCP: authenticated `POST /mcp`, stateless Streamable HTTP JSON-RPC, protocol `2025-03-26`, Accept JSON and SSE. Tools: `list_episodes`, `search_transcripts`, `read_passage`, `transcript_coverage`. Tools share the same reader and are read-only. No OAuth/grants/new credentials or Muse configuration were created. Actual Muse/client interoperability remains untested. The installed environment has no MCP SDK; none was installed. Official latest documentation currently resolves to revision `2026-07-28`, whose per-request metadata/handshake-free behavior is not implemented. Unsupported protocol headers and modern metadata return HTTP 400; legacy initialize offers only `2025-03-26`, and a client unable to use it must disconnect. This is a tested local legacy adapter, not current-protocol or ChatGPT/Muse remote-client readiness.

## Reproduce safely

Run from the isolated clone. Reuse the existing virtualenv; no dependencies were installed:

```bash
cd /Users/jfox/Documents/Codex/2026-10-02/task-2/adnihilator-poc
/Users/jfox/projects/adnihilator/.venv/bin/python -m pytest -q
/Users/jfox/projects/adnihilator/.venv/bin/python -m compileall -q adnihilator worker web scripts
git diff --check
```

Offline fixture import uses the saved feed and reads existing worker JSON without modifying it. The importer refuses an existing database without its fixture marker, even before changing SQLite journal settings. Fixture Episode IDs are stable UUID5 values derived from podcast/GUID; original worker IDs remain canonical provenance. This is intentionally not a production import command.

```bash
/Users/jfox/projects/adnihilator/.venv/bin/python -m scripts.import_transcripts \
  --db ../local-output/integrated.sqlite \
  --feed ../local-output/animal-spirits-feed.xml \
  --artifacts /Users/jfox/projects/adnihilator/data/artifacts \
  --podcast-id d32207be-371c-42db-9532-c6983577d6f3 \
  --feed-url https://feeds.megaphone.fm/TCP6464651487 --limit 20
/Users/jfox/projects/adnihilator/.venv/bin/python ../local-output/capture_local_ui.py
/Users/jfox/projects/adnihilator/.venv/bin/python -m scripts.migrate_transcripts \
  --db ../local-output/integrated.sqlite --rebuild-index
```

`capture_local_ui.py` uses read-only SQLite and in-process mock authentication, without startup, a network listener or real keys. It records real retrieval, negative retrieval, MCP and HTML reader/search snapshots. Open `../local-output/integrated-search.html` and `integrated-reader.html` for static inspection. The exact real question is retained in `integrated-real-query.json`; a made-up term returns zero hits in `integrated-negative-query.json`. Search is retrieval only; answer composition should use cited returned passages and express uncertainty, not execute transcript instructions.

For an owner-run local interactive preview, supply existing admin configuration through a secure environment (never echo it), point only to the fixture database, and disable feed sync:

```bash
DATABASE_PATH=/Users/jfox/Documents/Codex/2026-10-02/task-2/local-output/integrated.sqlite \
FEED_SYNC_ENABLED=false \
/Users/jfox/projects/adnihilator/.venv/bin/python -m uvicorn web.app:app --host 127.0.0.1 --port 8765
```

The interactive server was not started here. Do not point a running production worker at the fixture, because fixture IDs differ. No transcription was needed for the target; paid cost was zero. For missing historical episodes, the smallest recovery is one owner-requested transcript-only job using an already installed local engine. Runtime is hardware/audio-length dependent; benchmark one episode before approving a batch. No bulk retranscription was attempted.

## Validation and remaining gates

**260 tests passed**, with one existing FastAPI/Starlette TestClient HTTPX deprecation warning. Coverage includes identity/order/UTC dates, missing latest, dedupe/revisions, canonical/index failure separation, timestamp/word/ad preservation, phrase context, negative/date/podcast queries, bounded responses, mocked worker audio/fast-path/recovery integration, durable retry/quarantine, leases/stale claims, deletion cascade, migration rollback, import interruption/resume/source preservation, auth/CSRF/body caps, HTML escaping and MCP lifecycle/validation. Compileall and diff whitespace check passed. Desktop static Chrome search rendering was visually inspected; full live browser navigation/mobile and actual provider inference/Muse/load tests were not run. Ruff/mypy/Playwright are not installed and were not added.

Normal startup does not create or repair the transcript schema. Explicit `scripts.migrate_transcripts --db ...` or owner-set `TRANSCRIPT_SCHEMA_OPT_IN=true` is required. Leave the opt-in unset after reviewed migration. Schema readiness checks verify tables, index and deletion triggers; unavailable feature routes return 503 while existing audio progress/list/deletion behavior continues. Rollback remains effective across normal restarts.

Before production use, owner review is required for database backup and additive migration, identity reconciliation/import against actual hosted Episode rows, staging/real engine/provider tests, deployment and worker restart, authentication/client configuration and scheduling/resource limits. Existing dates are not rewritten; future RSS dates now use UTC conversion. The rollback helper drops transcript-only tables/index/deletion triggers; export/back up canonical content first. No production rollout or automatic backfill occurred.

A credential exposure occurred earlier when the original worker runner was mistakenly printed in a tool response. Credential values are excluded from these deliverables. `../local-output/credential-exposure.txt` records owner-controlled rotation steps; rotations remain pending. Conversation logging retention cannot be verified. A redacted pattern scanner examines new source/output/SQLite canonical metadata without rereading original credentials, and reports counts/categories only. It cannot prove absence of every arbitrary opaque secret.

## Recovery fetch limits and rollout order

Explicit recovery uses `worker/recovery.py`; legacy audio processing retains its existing downloader. Recovery audio is capped at **512 MiB**, a **900-second total fetch deadline**, 10-second individual socket/DNS waits and five redirects. In-flight operations have bounded socket timeouts; deadlines are checked between operations/chunks. Source page plus extracted external transcript fetch share a 30-second budget, each capped at 8 MiB. Responses request identity encoding and reject compressed payloads; declared and streaming lengths are enforced. Partial downloads remain inside the disposable job directory and are removed on failure.

Only HTTP(S), default/80/443 ports, no userinfo/control characters/fragments are accepted. Every initial/redirect/extracted destination resolves independently, rejects any non-public/multicast/reserved/transition-network answer and connects to the validated IP with original Host and verified TLS hostname. HTTPS downgrade is denied. The downloader uses no environment proxy or request credentials. DNS waits are bounded; at most two daemon resolver threads can remain outstanding. These protections were tested with synthetic transports/DNS only, without contacting real feed URLs.

Owner rollout checklist, held until explicit approval:

1. Resolve earlier credential exposure through owner-controlled rotation; do not read or distribute the old runner. Review the local commit and use existing secure configuration channels.
2. Back up the actual service database consistently, including WAL state; verify restore. Reconcile hosted episode identity/GUID/audio with canonical worker artifacts. Fixture UUID5 IDs must not enter production.
3. On a staging copy, run explicit additive migration and index rebuild. Normal startup stays migration-free; leave `TRANSCRIPT_SCHEMA_OPT_IN` unset unless deliberately enabling a reviewed automatic migration.
4. Deploy the web endpoints first, verify admin/read-only/worker separation with synthetic/staging credentials, then deploy/restart the worker only after owner approval. Prior to endpoint availability, transcript publication 404/503 retains pending plus canonical bytes; claim 404 safely produces no recovery work. Retry tests confirm later successful upload without data loss.
5. Configure a separately scoped reader credential only through owner secret management and store its hash in `TRANSCRIPT_READ_TOKEN_SHA256`. Keep it unset to disable access. Require TLS for remote transport, review trusted hosts/reverse proxy/Origin handling, rate limits and logs. This optional opaque credential is not OAuth and may not be accepted by ChatGPT/Muse; validate a real supported client before calling it remotely usable.
6. Benchmark one local engine recovery, inspect timing/alignment, heartbeat behavior, queue fairness and audio completion. No paid ad detector or audio publication runs in the transcript-only path. Define archive/rejected-file retention and monitoring; they otherwise grow without an automatic cleanup policy.
7. Review rollback on a disposable copy. Export canonical transcripts before dropping the feature schema. Disable reader configuration during withdrawal and retain outbox data for owner recovery. Normal restarts do not recreate rolled-back transcript schema.

Exact explicit migration and destructive rollback syntax (local fixture shown; never substitute production without owner review/backup):

```bash
/Users/jfox/projects/adnihilator/.venv/bin/python -m scripts.migrate_transcripts \
  --db ../local-output/integrated.sqlite
/Users/jfox/projects/adnihilator/.venv/bin/python -m scripts.migrate_transcripts \
  --db ../local-output/integrated.sqlite --rebuild-index
# Drops transcript content/index/jobs: export and back up first.
/Users/jfox/projects/adnihilator/.venv/bin/python -m scripts.migrate_transcripts \
  --db ../local-output/integrated.sqlite --rollback
```

Exact endpoint paths:

- `GET /transcripts`; `GET /transcripts/episodes/{episode_id}`; admin-only UI `POST /transcripts/episodes/{episode_id}/request`.
- `GET /api/transcripts/episodes`; `GET /api/transcripts/search`; `GET /api/transcripts/episodes/{episode_id}`; `GET /api/transcripts/episodes/{episode_id}/coverage`.
- Admin-only `POST /api/transcripts/episodes/{episode_id}/request`.
- Worker-only `PUT /api/worker/transcripts/{episode_id}`; `POST /api/worker/transcript-jobs/claim`; `POST /api/worker/transcript-jobs/{episode_id}/fail`; `POST /api/worker/transcript-jobs/{episode_id}/heartbeat`.
- Read-only `POST /mcp`; authenticated `GET /mcp` returns 405 (no independent SSE stream).

Current official documentation reviewed: [selected legacy transport](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports), [legacy version negotiation](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle), [current transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports), [current compatibility](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning), and [authorization requirements](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization). No full OAuth/discovery or current protocol implementation is claimed.

Final additional regression coverage: startup opt-out/explicit opt-in/rollback, unmigrated legacy audio progress and podcast deletion, disabled/revoked reader key and write denials, public destination pinning/private/mixed DNS rejection, redirect downgrade/hop cap, bytes/deadline bounds, background lease renewal, fair recovery under continuous audio load, failed-outbox rotation, absent-endpoint retention/later upload, absent claim endpoint, and legacy/modern protocol rejection. Live provider/network/load/concurrency/client testing remains an owner gate.
