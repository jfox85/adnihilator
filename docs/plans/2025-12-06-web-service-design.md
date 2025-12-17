# AdNihilator Web Service Design

## Overview

A web service that provides RSS feed management and serving for ad-free podcast episodes. Processing happens locally on your Mac, while a lightweight web app on an existing VPS handles feed management and serves clean RSS feeds to podcast apps.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Existing DigitalOcean VPS                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  nginx                                               │   │
│  │  ├── yourdomain.com → WordPress                      │   │
│  │  └── feeds.yourdomain.com → adnihilator (port 8000) │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │  FastAPI Web App │  │  SQLite Database │                │
│  │  (systemd)       │  │                  │                │
│  └──────────────────┘  └──────────────────┘                │
└─────────────────────────────────────────────────────────────┘
          ▲                           │
          │ Worker API                │ RSS Feeds
          │                           ▼
┌──────────────────┐         ┌──────────────────┐
│  Your Mac        │         │  Podcast App     │
│  (Local Worker)  │         │                  │
└──────────────────┘         └──────────────────┘
          │
          │ Upload processed audio
          ▼
┌──────────────────┐
│  Cloudflare R2   │
│  (Audio Storage) │
└──────────────────┘
```

## Components

### Web App (VPS)

FastAPI application serving:
- Management UI for adding/removing podcast subscriptions
- Worker API for the local processor to claim and complete jobs
- RSS feed generation for podcast apps

Runs as a systemd service behind existing nginx.

### Local Worker (Mac)

Daemon that:
- Polls VPS for queued episodes
- Downloads original audio from podcast CDN
- Runs adnihilator pipeline (transcribe → detect → splice)
- Uploads processed MP3 to R2
- Reports completion to VPS

### Storage (Cloudflare R2)

Stores processed audio files. Chosen over S3 for zero egress fees.

## Data Model

```sql
podcasts
├── id              UUID PRIMARY KEY
├── feed_token      TEXT UNIQUE      -- random string for obscure URL (32+ chars)
├── source_rss_url  TEXT NOT NULL    -- original feed URL
├── title           TEXT             -- from feed metadata
├── auto_process    BOOLEAN DEFAULT TRUE
├── created_at      TIMESTAMP
└── updated_at      TIMESTAMP

episodes
├── id                   UUID PRIMARY KEY
├── podcast_id           UUID REFERENCES podcasts
├── guid                 TEXT             -- from original feed
├── title                TEXT
├── original_audio_url   TEXT
├── published_at         TIMESTAMP
├── status               TEXT             -- pending|processing|complete|failed|skipped|expired
├── retry_count          INTEGER DEFAULT 0
├── error_message        TEXT
├── claimed_at           TIMESTAMP        -- when worker claimed job (for timeout detection)
├── processed_audio_key  TEXT             -- R2 path when complete
├── processed_duration   FLOAT
├── ads_removed_seconds  FLOAT
├── created_at           TIMESTAMP
└── updated_at           TIMESTAMP

UNIQUE(podcast_id, guid)
CHECK(status IN ('pending', 'processing', 'complete', 'failed', 'skipped', 'expired'))
```

**Database Configuration:**
- Enable WAL mode for SQLite: `PRAGMA journal_mode=WAL;`
- This improves write concurrency for the web app

## API Endpoints

### Management UI

```
GET  /                              HTML dashboard
POST /podcasts                      Add podcast (RSS URL)
GET  /podcasts                      List podcasts
DELETE /podcasts/{id}               Remove podcast

GET  /podcasts/{id}/episodes        List episodes
POST /podcasts/{id}/episodes/{ep}/queue   Queue for processing
POST /podcasts/{id}/episodes/{ep}/skip    Mark skipped
POST /podcasts/{id}/episodes/{ep}/retry   Reset failed episode
```

### Worker API (authenticated via WORKER_API_KEY header)

```
POST /api/queue/claim               Atomically claim next pending episode
POST /api/queue/{ep}/complete       Mark complete with metadata (idempotent)
POST /api/queue/{ep}/fail           Mark failed with error (idempotent)
```

**Atomic claiming:** The `/api/queue/claim` endpoint performs a single atomic transaction:
1. Find next episode with `status='pending'` (ordered by priority/created_at)
2. Set `status='processing'` and `claimed_at=now()`
3. Return the episode data

This eliminates race conditions when multiple workers are running.

### RSS Feeds (public via obscurity)

```
GET  /feed/{feed_token}.xml         Generated RSS feed
```

## RSS Feed Generation

The generated feed:
- Mirrors original feed metadata (title, description, artwork)
- Only includes episodes with `status=complete`
- Audio URLs point to R2: `https://{r2-domain}/{podcast_id}/{episode_id}.mp3`
- Preserves original episode metadata (title, description, publish date)

## Local Worker Flow

```
loop:
    episode = POST /api/queue/claim  # atomic claim
    if not episode:
        sleep(interval)
        continue

    try:
        audio = download(episode.original_audio_url)
        result = adnihilator.detect(audio)
        processed = adnihilator.splice(audio, result)
        r2.upload(processed, key=f"{podcast_id}/{episode_id}.mp3")

        # Verify upload integrity
        verify_upload(key, expected_size=len(processed))

        POST /api/queue/{episode.id}/complete
    except Exception as e:
        POST /api/queue/{episode.id}/fail (error=str(e))
```

### Retry Logic

On failure:
- If `retry_count < 2`: increment count, reset to pending
- If `retry_count >= 2`: mark as permanently failed

Failed episodes (after retries) show in UI with error message and manual retry button.

### Stale Job Recovery

A scheduled task on the VPS runs every 30 minutes to detect abandoned jobs:

```sql
UPDATE episodes
SET status = 'pending', claimed_at = NULL, retry_count = retry_count + 1
WHERE status = 'processing'
  AND claimed_at < NOW() - INTERVAL '2 hours'
  AND retry_count < 2;
```

Jobs stuck in `processing` for over 2 hours are automatically re-queued.

### Running the Worker

```bash
# One-shot mode
adnihilator worker --once

# Daemon mode (poll every 5 minutes)
adnihilator worker --daemon --interval 300
```

Or configure via launchd for automatic startup.

## Authentication

- **RSS feeds**: Security through obscurity (32+ char random tokens in URLs)
- **Worker API**: Shared secret via `X-Worker-API-Key` header
- **Management UI**: HTTP Basic Auth (username/password configured via environment variables)

```python
# FastAPI Basic Auth for management UI
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

security = HTTPBasic()

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, headers={"WWW-Authenticate": "Basic"})
```

## Project Structure

```
adnihilator/
├── adnihilator/           # Existing CLI/core
│   ├── cli.py
│   ├── transcribe.py
│   ├── ad_llm.py
│   ├── splice.py
│   └── ...
├── web/                   # New web service
│   ├── __init__.py
│   ├── app.py             # FastAPI application
│   ├── models.py          # SQLAlchemy models
│   ├── database.py        # DB session management
│   ├── routes/
│   │   ├── ui.py          # HTML pages
│   │   ├── api.py         # Worker API
│   │   └── feeds.py       # RSS generation
│   ├── templates/         # Jinja2 templates
│   │   ├── base.html
│   │   ├── index.html
│   │   └── episodes.html
│   └── services/
│       ├── feed_parser.py # RSS parsing
│       └── r2.py          # R2 client
├── worker/                # Local worker
│   ├── __init__.py
│   ├── daemon.py          # Main worker loop
│   └── client.py          # API client for VPS
├── deploy/
│   ├── adnihilator.service   # systemd unit file
│   └── nginx-site.conf       # nginx config snippet
└── pyproject.toml
```

## Dependencies

New dependencies for web service:
- `fastapi`
- `uvicorn`
- `sqlalchemy`
- `jinja2`
- `feedgen`
- `feedparser`
- `httpx`
- `boto3` (for R2 S3-compatible API)

## Deployment

### VPS Setup

1. Clone repo to `/opt/adnihilator`
2. Create virtualenv and install dependencies
3. Create systemd service
4. Add nginx site config
5. Set environment variables

### Environment Variables

**VPS:**
```
WORKER_API_KEY=<random-secret>
ADMIN_USERNAME=admin
ADMIN_PASSWORD=<strong-password>
DATABASE_PATH=/var/lib/adnihilator/adnihilator.db
R2_PUBLIC_URL=https://your-bucket.r2.dev
```

**Local Worker:**
```
API_URL=https://feeds.yourdomain.com
WORKER_API_KEY=<same-secret>
R2_ACCESS_KEY=<cloudflare-key>
R2_SECRET_KEY=<cloudflare-secret>
R2_BUCKET=adnihilator-audio
R2_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
OPENAI_API_KEY=<for-llm-refinement>
```

### systemd Unit

```ini
[Unit]
Description=AdNihilator Web Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/adnihilator
Environment=DATABASE_PATH=/var/lib/adnihilator/adnihilator.db
EnvironmentFile=/etc/adnihilator/env
ExecStart=/opt/adnihilator/venv/bin/uvicorn web.app:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

### nginx Config

```nginx
server {
    listen 80;
    server_name feeds.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name feeds.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/feeds.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/feeds.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Cost Estimate

| Item | Monthly Cost |
|------|--------------|
| VPS | $0 (existing) |
| R2 storage (~30GB) | ~$0.45 |
| R2 egress | $0 |
| OpenAI API (~30 eps) | ~$0.60 |
| **Total** | **~$1/month** |

## Cleanup Policy

Processed episodes are deleted from R2 after 14 days. The VPS runs a daily cleanup job to:
1. Find episodes with `status=complete` and `updated_at < 14 days ago`
2. Delete audio from R2 (ignore "not found" errors for idempotency)
3. Set `status=expired` so they're excluded from RSS feeds

This ensures podcast apps won't see broken links to deleted audio files.

## Future Considerations (Not MVP)

- Processing stats dashboard
- Per-podcast LLM prompt tuning
- Notification on processing failures (email/webhook)
- Signed R2 URLs via proxy endpoint for better access control
- Database migrations with Alembic
