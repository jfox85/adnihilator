# Current Deployment

This documents the live deployment as of December 2024.

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
│  - Public URL: pub-f19bf1c482d94b23bc7071f9e60c4a13.r2.dev     │
└─────────────────────────────────────────────────────────────────┘
```

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

**nginx:** Reverse proxy on port 443, forwards to uvicorn on 127.0.0.1:8001

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

The worker runs under Rosetta (`arch -x86_64`) because faster-whisper's pydantic_core is x86_64.

## Cloudflare R2

**Bucket:** `adnihilator`
**Endpoint:** `https://eac56d33df4b0f3eba62c14b7a0323a7.r2.cloudflarestorage.com`
**Public URL:** `https://pub-f19bf1c482d94b23bc7071f9e60c4a13.r2.dev`

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
| created_at | DATETIME | |
| updated_at | DATETIME | |

**Unique constraint:** (podcast_id, guid)

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

```bash
# Deploy to VPS
scp adnihilator/*.py root@jonefox.com:/opt/adnihilator/adnihilator/
scp worker/*.py root@jonefox.com:/opt/adnihilator/worker/
scp web/*.py root@jonefox.com:/opt/adnihilator/web/
scp web/routes/*.py root@jonefox.com:/opt/adnihilator/web/routes/
scp web/services/*.py root@jonefox.com:/opt/adnihilator/web/services/

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
