# Current Deployment

This documents the live deployment. It was written in December 2024 and
has had only partial updates since then. Anything marked **(as of Dec 2024)**
has not been checked against the live hosts recently. Confirm it before you
rely on it.

> **No secrets in this file.** This is a public repo. Credentials and account
> identifiers stay on the machines that use them. See
> [Where secrets live](#where-secrets-live) below.

## Infrastructure Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     pods.jonefox.com (VPS)                      │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │   nginx     │───▶│   uvicorn    │───▶│  SQLite DB        │  │
│  │  (reverse   │    │  (port 8001) │    │  adnihilator.db   │  │
│  │   proxy)    │    │  2 workers   │    └───────────────────┘  │
│  └─────────────┘    └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
         ▲                    │
         │                    │ Worker API
         │                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Local Mac (jfox)                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Worker Daemon (launchd)                                  │  │
│  │  - Claims jobs from API                                   │  │
│  │  - Transcribes with Whisper (x86_64 via Rosetta)         │  │
│  │  - Detects ads, splices audio                            │  │
│  │  - Uploads to Cloudflare R2                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  data/artifacts/                                          │  │
│  │  Detection results saved locally per podcast/episode      │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Cloudflare R2 (adnihilator bucket)                 │
│  - Processed audio files: {podcast_id}/{episode_id}.mp3        │
│  - Public URL: value of R2_PUBLIC_URL (see below)              │
└─────────────────────────────────────────────────────────────────┘
```

## Where secrets live

| Value | Used by | Where the real value lives |
|-------|---------|----------------------------|
| `ADMIN_USERNAME`, `ADMIN_PASSWORD` | web service | `/etc/adnihilator/env` on the VPS (systemd `EnvironmentFile`) |
| `WORKER_API_KEY` | web service **and** worker (must match) | `/etc/adnihilator/env` on the VPS; `scripts/run-worker.sh` on the Mac |
| `R2_PUBLIC_URL` | web service | `/etc/adnihilator/env` on the VPS |
| `DATABASE_PATH` | web service | `/etc/adnihilator/env` on the VPS |
| `API_URL` | worker | `scripts/run-worker.sh` on the Mac |
| `R2_ACCESS_KEY`, `R2_SECRET_KEY` | worker | `scripts/run-worker.sh` on the Mac; issued in Cloudflare dashboard → R2 → Manage API tokens |
| `R2_BUCKET`, `R2_ENDPOINT` | worker | `scripts/run-worker.sh` on the Mac |
| `OPENAI_API_KEY` | worker | `scripts/run-worker.sh` on the Mac |
| `GEMINI_API_KEY` | worker (only when Gemini detection is enabled in `adnihilator.toml`) | `scripts/run-worker.sh` on the Mac |

`scripts/run-worker.sh` is gitignored. Start from
`scripts/run-worker.sh.example`, which holds placeholders only. The launchd
plist (`*.plist`) is gitignored too. Never commit real values, and never paste
them into this file.

## Web Service (VPS)

**Host:** `root@jonefox.com` (resolves to pods.jonefox.com)
**URL:** https://pods.jonefox.com
**Code location:** `/opt/adnihilator/`
**Virtual env:** `/opt/adnihilator/.venv/`
**Database:** `/opt/adnihilator/data/adnihilator.db` (SQLite)

**Systemd service:** `adnihilator.service`
```bash
systemctl status adnihilator
systemctl restart adnihilator
journalctl -u adnihilator -f
```

**Environment file:** `/etc/adnihilator/env` (all web-service secrets; see table above)

**nginx:** Reverse proxy on port 443, forwards to uvicorn on 127.0.0.1:8001

> **(as of Dec 2024)** The live layout above (`.venv/`, port 8001, DB under
> `/opt/adnihilator/data/`) differs from the templates in `deploy/`
> (`venv/`, port 8000, DB under `/var/lib/adnihilator/`). Check
> `systemctl cat adnihilator` on the VPS to see which one is live.

## Worker Daemon (Local Mac)

**Location:** `/Users/jfox/projects/adnihilator/`
**Runner script:** `scripts/run-worker.sh`
**Artifacts:** `data/artifacts/{podcast_id}/{episode_id}.json`
**Logs:** `data/worker.log`

**launchd plist:** `~/Library/LaunchAgents/com.adnihilator.worker.plist`
```bash
launchctl unload ~/Library/LaunchAgents/com.adnihilator.worker.plist
launchctl load ~/Library/LaunchAgents/com.adnihilator.worker.plist
tail -f /Users/jfox/projects/adnihilator/data/worker.log
```

**(as of Dec 2024)** The worker runs under Rosetta (`arch -x86_64`) because
faster-whisper's pydantic_core is x86_64. The DevX dev environment runs native
arm64 now (`.devx/python.sh`), so the production worker may be able to drop
Rosetta. That hasn't been tested.

## Cloudflare R2

**Bucket:** `adnihilator`
**Endpoint:** `https://<CLOUDFLARE_ACCOUNT_ID>.r2.cloudflarestorage.com`.
The real value is `R2_ENDPOINT` in `scripts/run-worker.sh` on the Mac. The
account ID is also shown in Cloudflare dashboard → R2 → Overview.
**Public URL:** `https://pub-<BUCKET_PUBLIC_ID>.r2.dev`. The real value is
`R2_PUBLIC_URL` in `/etc/adnihilator/env` on the VPS. It's also shown in
Cloudflare dashboard → R2 → `adnihilator` → Settings → Public access.
**Credentials:** `R2_ACCESS_KEY` / `R2_SECRET_KEY` live only in
`scripts/run-worker.sh` on the Mac.

Processed audio stored as: `{podcast_id}/{episode_id}.mp3`

## Database Schema

### podcasts
| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR(36) | UUID primary key |
| feed_token | VARCHAR(64) | Unique token for ad-free RSS feed URL |
| source_rss_url | TEXT | Original podcast RSS feed |
| title | TEXT | Podcast title (from feed) |
| description | TEXT | Podcast description |
| artwork_url | TEXT | Cover art URL |
| auto_process | BOOLEAN | Auto-queue new episodes |
| last_synced_at | DATETIME | Last feed sync time |
| created_at | DATETIME | |
| updated_at | DATETIME | |

### episodes
| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR(36) | UUID primary key |
| podcast_id | VARCHAR(36) | FK to podcasts |
| guid | TEXT | Episode GUID from RSS |
| title | TEXT | Episode title |
| description | TEXT | Episode description (HTML) |
| duration | INTEGER | Duration in seconds |
| original_audio_url | TEXT | Original audio URL |
| source_url | TEXT | Episode page URL (for external transcripts) |
| published_at | DATETIME | Publication date |
| status | VARCHAR(20) | pending/processing/complete/failed/skipped/expired |
| retry_count | INTEGER | Number of processing retries |
| error_message | TEXT | Last error if failed |
| claimed_at | DATETIME | When worker claimed the job |
| progress_step | VARCHAR(50) | Current step (downloading/transcribing/etc) |
| progress_percent | INTEGER | Progress within current step |
| processed_audio_key | TEXT | R2 key for processed audio |
| processed_duration | FLOAT | Duration after ad removal |
| ads_removed_seconds | FLOAT | Total seconds of ads removed |
| detection_result_path | TEXT | Local path to detection JSON |
| llm_provider | VARCHAR(50) | LLM provider used for detection |
| llm_model | VARCHAR(100) | LLM model used |
| llm_input_tokens | INTEGER | Prompt tokens |
| llm_output_tokens | INTEGER | Completion tokens |
| llm_total_tokens | INTEGER | Total tokens |
| llm_cost_usd | NUMERIC(10,6) | Estimated LLM cost |
| detection_source | VARCHAR(50) | timestamps/external/gemini/whisper |
| created_at | DATETIME | |
| updated_at | DATETIME | |

**Unique constraint:** (podcast_id, guid)
**Check constraint:** status in the values listed above
**Indexes:** (status, created_at), (status, claimed_at)

The authoritative schema is `web/models.py`. If this table and that file
disagree, the code is correct.

## Episode Status Flow

```
[New from RSS] ──▶ pending ──▶ processing ──▶ complete
                     │              │
                     │              ▼
                     │           failed (retry_count < 2)
                     │              │
                     ▼              ▼
                  skipped       failed (permanent)
```

## Deploying Changes

**(as of Dec 2024)** Deploys are a manual file copy. The VPS doesn't pull from git.

```bash
# Deploy to VPS
scp adnihilator/*.py root@jonefox.com:/opt/adnihilator/adnihilator/
scp worker/*.py root@jonefox.com:/opt/adnihilator/worker/
scp web/*.py root@jonefox.com:/opt/adnihilator/web/
scp web/routes/*.py root@jonefox.com:/opt/adnihilator/web/routes/
scp web/services/*.py root@jonefox.com:/opt/adnihilator/web/services/
scp web/templates/*.html root@jonefox.com:/opt/adnihilator/web/templates/
# If pyproject.toml dependencies changed, also copy it and reinstall:
#   scp pyproject.toml root@jonefox.com:/opt/adnihilator/
#   ssh root@jonefox.com "/opt/adnihilator/.venv/bin/pip install -e /opt/adnihilator"

# Restart web service
ssh root@jonefox.com "systemctl restart adnihilator"

# Restart local worker
launchctl unload ~/Library/LaunchAgents/com.adnihilator.worker.plist
launchctl load ~/Library/LaunchAgents/com.adnihilator.worker.plist
```

## Useful Queries

```bash
# Check episode status
ssh root@jonefox.com "sqlite3 /opt/adnihilator/data/adnihilator.db \
  'SELECT title, status, ads_removed_seconds FROM episodes ORDER BY created_at DESC LIMIT 10'"

# Reset an episode to pending
ssh root@jonefox.com "sqlite3 /opt/adnihilator/data/adnihilator.db \
  \"UPDATE episodes SET status='pending', claimed_at=NULL WHERE id='...';\""

# Check queue
ssh root@jonefox.com "sqlite3 /opt/adnihilator/data/adnihilator.db \
  'SELECT COUNT(*), status FROM episodes GROUP BY status'"
```

## Stuck Job Recovery

Episodes are claimed by setting `status='processing'`. If a worker dies mid-job
(e.g. the 12h watchdog restart), the episode is orphaned in `processing` and the
queue looks empty even though work is waiting.

The web service self-heals this: on every `/api/queue/claim`, episodes stuck in
`processing` whose `claimed_at` is older than the timeout are recovered.

- `claimed_at` is treated as a renewable lease: every worker progress update
  refreshes it, so a job is only considered stuck once it stops reporting
  progress for longer than the timeout. Healthy long-running jobs are not
  reclaimed.
- Recovery consumes the retry budget like a failure: each stuck episode's
  `retry_count` is incremented and it returns to `pending` while under
  `MAX_RETRIES`, or is marked permanently `failed` once exhausted. This stops
  a poison episode from being requeued forever.
- Default timeout: 2 hours.
- Override with the `WORKER_STUCK_TIMEOUT_SECONDS` env var on the web service
  (set in `/etc/adnihilator/env`, then `systemctl restart adnihilator`).
  Values are clamped to 1s..30d; invalid values fall back to the default.

Manual reset is still available via the "Reset an episode to pending" query above.

### One-time index migration

This change adds two composite indexes used by the claim and recovery queries.
New databases get them automatically; apply them to the existing prod DB once:

```bash
ssh root@jonefox.com "sqlite3 /opt/adnihilator/data/adnihilator.db \
  'CREATE INDEX IF NOT EXISTS ix_episodes_status_created_at ON episodes(status, created_at); \
   CREATE INDEX IF NOT EXISTS ix_episodes_status_claimed_at ON episodes(status, claimed_at);'"
```
