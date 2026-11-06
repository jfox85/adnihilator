# Source reconciliation provenance

This branch reconciles the production checkout's unmerged source with `main` (`834b9c1a0e3de4fabc016fc2bb4fc16d6d57dc70`). It preserves the transcript search, reader, scoped read-token and MCP feature from `f9ee33c825064eeb3f2af9df396352f7b48666d1`, including its migration/import tools and recovery tests.

The intended dirty source adds the google-genai SDK migration, shared model/thinking helpers, client usage accounting, SDK configuration/tests, upload compatibility helper/tests and worker configured-secret error masking. The three guarded root-level manual experiment scripts are retained as source only. They make paid calls when explicitly invoked and are not execution instructions or test evidence.

The original checkout remains untouched. Its private `adnihilator.toml`, credential-bearing worker runner, environment, databases and media were not copied. The public tracked configuration here starts from main and applies only PR7's reviewed SDK3 model/thinking defaults; it does not copy the production private configuration.

## Existing fix overlap

PR7 (`db0eb557aaf47a62a8e01ef7eb71637f12db312d`) independently contains the SDK migration plus additional detection/provider safety changes. This reconciliation retains the production SDK subset and does not absorb the whole detection change. The SDK configuration and dependency files agree with PR7; Gemini/worker implementation chunks overlap and require deliberate composition with PR7's final safety behavior. Do not overwrite PR7 with these earlier production versions.

This feature also intersects the existing auth, feed, worker API, UI and diagnostic batches. Their final fixes remain dependencies for the eventual combined release. Source reconciliation does not approve those blocked dependencies or replace their reviews. Deploy only after root review of the final composition, migrations, CI and outstanding owner decisions.

## Validation boundary

Tests use scratch SQLite, synthetic transcripts and mocked providers. External sockets and DNS are denied. No production worker, paid model request, production migration/import, deployment or restart is performed. The independently hunted feature scope is transcript indexing/search, scoped reads, transcript jobs/recovery, MCP protocol/auth/input handling and transcript UI; confirmation precedes fixes.

## Confirmed transcript follow-up behavior

Native blank date controls mean no date bound. Transcript dates/times use the existing episode date convention and the byte-identical PR10 duration formatter. Navigation wraps on phones; transcript links sit outside the episode disclosure/title row, preserving future native-button composition.

Search keeps `match_ordinal` as the first segment of the existing three-segment indexed window. Every per-segment stable reference still identifies those exact bytes. New `focus_ordinal` is the segment where SQLite FTS reports its first match, including accent folding and cross-segment phrases; reader links start and anchor there. A phrase can continue in another segment. Canonical content and complete-segment pagination remain bounded and unchanged.

Existing transcript claim IDs are checked by conditional writes through canonical/job commit; failure/expiry update only the observed current lease. This adds no audio attempt-token protocol or migration. Concurrent initial requests converge through SQLite upsert, and an index writer refreshes the active pointer under a writer reservation before replacing FTS rows. MCP synchronous reads run off the event loop; malformed deep JSON and UTC-unrepresentable dates are rejected.

Unknown enclosure providers remain exact identities. Basename normalization is limited to `traffic.megaphone.fm` and explicit embedded Megaphone paths on `chrt.fm`, `chtbl.com`, `dts.podtrac.com`, and `pdst.fm`. Unknown tracking wrappers, including synthetic lookalike domains, are intentionally opaque; owners must reconcile unsupported historical wrappers rather than silently join unrelated media.

## Local outbox ordering and rollout

The newest locally staged document wins only within the same episode and existing claim scope (audio uses `None`). A per-episode filesystem lock serializes staging and publishing calls. Canonical checksum archives are retained. The durable latest-generation marker is the stage commit point: a crash after the archive but before that commit retains the archive and requires producer retry; a crash after marker commit reconstructs missing pending bytes and suppresses older pending work before any publish. Server checksum idempotency remains unchanged.

Audio pending filenames retain their existing checksum form. Recovery claim scopes have distinct pending filenames to avoid replacing another scope with identical canonical bytes. Retry mtime controls fairness only, never revision precedence. Multiple legacy unsequenced pending files in one scope are retained and held for explicit restaging of the intended latest revision; mutable old mtimes cannot establish their order. Other episodes/scopes continue. Back up the complete outbox, including hidden markers, before the owner deploys/resumes a worker. No production outbox was read or migrated here.

This is a local producer guarantee, not global precedence across independent import/recovery/audio producers or delayed remote writes after network timeouts. Canonical archive retention and operator reconciliation remain owner responsibilities.

The UI5 classification received two independent checks: a deleted-podcast citation becomes a raw 404 just like main's stale episode bookmark; five invalid date/checksum/ordinal cases are manual-only. The global inherited error-page gap is PR11's dependency, not a duplicated transcript-specific handler. Ordinary blank-date form failures are fixed locally.

The owner states historical credentials are inactive (A4); this statement was not independently verified. Current read-only evidence found worker PID85605 ARM64 and successful public certificate handshakes; it does not prove the cause or absence of historical TLS/architecture failures.
