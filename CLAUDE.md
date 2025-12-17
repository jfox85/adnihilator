# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AdNihilator is a podcast ad detection and removal system with three components:

1. **CLI tool** (`adnihilator/cli.py`) - Standalone ad detection on audio files
2. **Web service** (`web/`) - FastAPI app for managing podcast RSS feeds
3. **Worker daemon** (`worker/`) - Runs locally, claims jobs from web service, processes audio, uploads to R2

## Common Commands

```bash
# Run tests
pytest
pytest tests/test_sponsors.py -v              # Single test file
pytest -k "test_extract" -v                   # Tests matching pattern

# Run CLI commands
adnihilator detect podcast.mp3 --out results.json
adnihilator download-model small              # Download Whisper model
adnihilator splice results.json --out clean.mp3

# Run web service locally
uvicorn web.app:app --reload

# Run worker daemon
export API_URL="https://your-server.com"
export WORKER_API_KEY="..."
export R2_ACCESS_KEY="..." R2_SECRET_KEY="..." R2_BUCKET="..." R2_ENDPOINT="..."
python -m adnihilator.cli worker --daemon --interval 60

# Deploy to remote
scp adnihilator/*.py root@your-server.com:/opt/adnihilator/adnihilator/
ssh root@your-server.com "systemctl restart adnihilator"

# Restart local worker (macOS)
launchctl unload ~/Library/LaunchAgents/com.adnihilator.worker.plist
launchctl load ~/Library/LaunchAgents/com.adnihilator.worker.plist
```

## Architecture

### Ad Detection Pipeline

1. **Transcription**: faster-whisper transcribes audio to timestamped segments
2. **Sponsor extraction** (`sponsors.py`): Extracts sponsor names from episode description HTML
3. **Keyword detection** (`ad_keywords.py`): Scores segments using sponsor names + ad phrase patterns
4. **LLM refinement** (`ad_llm.py`): OpenAI refines heuristic candidates into precise ad boundaries
5. **Splicing** (`splice.py`): ffmpeg removes ad segments, filtered by confidence threshold (default 0.35)

### Web/Worker Split Architecture

The web service (VPS) handles:
- RSS feed management and syncing (`web/services/feed_sync.py`)
- Episode queue with status: pending → processing → complete/failed
- Admin UI with HTTP basic auth
- Worker API endpoints (`/api/queue/claim`, `/api/queue/{id}/complete`)
- Ad-free RSS feed generation (`web/routes/feeds.py`)

The worker (runs locally on Mac with GPU) handles:
- Claiming pending episodes from API
- Downloading audio, transcribing, detecting ads
- Uploading processed audio to Cloudflare R2
- Reporting completion back to API

### Key Data Models

- `TranscriptSegment`: Timestamped text from Whisper
- `AdCandidate`: Heuristic detection result with score
- `AdSpan`: Final ad boundary with confidence (0-1)
- `DetectionResult`: Full detection output saved as JSON artifact
- `Episode`/`Podcast`: SQLAlchemy models in web service

### External Transcript Support

For Substack podcasts (e.g., Lenny's Newsletter), `external_transcript.py` fetches CloudFront-hosted transcripts instead of running Whisper. The worker checks `source_url` and tries external transcript first.

## Environment Variables

Worker requires: `API_URL`, `WORKER_API_KEY`, `R2_ACCESS_KEY`, `R2_SECRET_KEY`, `R2_BUCKET`, `R2_ENDPOINT`, `OPENAI_API_KEY`

Web service requires: `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `WORKER_API_KEY`, `DATABASE_PATH`, `R2_PUBLIC_URL`

## OpenMP Fix

macOS requires `KMP_DUPLICATE_LIB_OK=TRUE` before importing faster-whisper (set in cli.py).
