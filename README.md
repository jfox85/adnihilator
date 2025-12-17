# AdNihilator

Automatically detect and remove advertisements from podcast audio files using Whisper transcription and LLM analysis.

## What is AdNihilator?

AdNihilator is a podcast ad detection and removal system that:
- Transcribes podcast audio using OpenAI's Whisper model
- Detects sponsor segments using keyword heuristics and LLM refinement
- Removes ads with frame-accurate ffmpeg splicing
- Provides both a standalone CLI and a web service architecture

### How It Works

1. **Transcription**: Whisper transcribes audio to timestamped text segments
2. **Sponsor Extraction**: Analyzes episode descriptions to find sponsor names
3. **Keyword Detection**: Scores segments using sponsor names and ad phrase patterns
4. **LLM Refinement**: OpenAI GPT refines candidates into precise ad boundaries
5. **Splicing**: ffmpeg removes ad segments with frame-accurate cuts

### Two-Pass Mode (~2.5x Faster)

When using OpenAI LLM refinement, AdNihilator uses an optimized two-pass approach:
- **Pass 1**: Fast segment-level transcription of the full episode
- **Pass 2**: High-quality word-level transcription of only detected ad regions

This provides ~2.5x speedup compared to transcribing the entire episode with word timestamps.

### Pre-Roll & Outro Detection

AdNihilator specifically looks for ads in common placement zones:
- **Pre-roll (first 2 minutes)**: House ads, network promos, sponsor reads
- **Outro (last 2 minutes)**: Post-roll ads, dynamically inserted sponsors

## Features

### CLI Mode
- Process individual audio files locally
- Output JSON with ad timestamps and confidence scores
- Splice out ads to create clean audio files

### Web Service Mode
- Subscribe to podcast RSS feeds
- Auto-process new episodes with a worker daemon
- Generate ad-free RSS feeds for podcast apps
- Store processed audio on Cloudflare R2

### Detection Capabilities
- **External transcripts**: Fast processing using podcast-provided transcripts (Substack, Lex Fridman)
- **Sponsor-aware**: Extracts sponsor names from episode descriptions for better accuracy
- **Confidence scoring**: Each ad gets a 0-1 confidence score for filtering
- **Multi-region detection**: Pre-roll, mid-roll, and outro ad placement

## Installation

### Prerequisites

- **Python 3.11+**
- **ffmpeg**: Required for audio processing
  ```bash
  # macOS
  brew install ffmpeg

  # Ubuntu/Debian
  apt-get install ffmpeg
  ```

### Install AdNihilator

```bash
git clone https://github.com/yourusername/adnihilator.git
cd adnihilator
pip install -e .
```

### Download Whisper Model

Before first use, download a Whisper model:

```bash
adnihilator download-model small
```

Available models: `tiny`, `base`, `small`, `medium`, `large` (larger = more accurate but slower)

## Quick Start (CLI Mode)

### 1. Basic Ad Detection

```bash
adnihilator detect podcast.mp3 --out results.json
```

This runs heuristic detection only (no LLM). Fast but less accurate.

### 2. With OpenAI Refinement (Recommended)

```bash
export OPENAI_API_KEY="sk-..."
adnihilator detect podcast.mp3 --llm-provider openai --out results.json
```

Uses GPT-4.1-mini to refine ad boundaries. More accurate, requires OpenAI API key.

### 3. Splice Out Ads

```bash
adnihilator splice results.json --out clean.mp3
```

Creates a new audio file with all detected ads removed.

### 4. One-Step Detection + Splicing

```bash
adnihilator detect podcast.mp3 --llm-provider openai --splice --out clean.mp3
```

## Configuration

Create `adnihilator.toml` in your working directory:

```toml
[llm]
provider = "openai"
model = "gpt-4.1-mini"  # or "gpt-4.1"
api_key_env = "OPENAI_API_KEY"

[detect]
confidence_threshold = 0.35  # Filter out low-confidence detections
context_segments_before = 2  # Include 2 segments before ad for context
context_segments_after = 2   # Include 2 segments after ad for context

[transcribe]
model = "small"  # Whisper model size
device = "cpu"   # or "cuda" for GPU
```

## Web Service Setup (Optional)

For automated podcast processing, you can deploy the web service + worker architecture.

### Architecture

```
Web Service (VPS)           Worker Daemon (Local/GPU)
├── FastAPI app             ├── Claims jobs from API
├── SQLite database         ├── Transcribes audio
├── RSS feed management     ├── Detects & removes ads
└── Admin UI                └── Uploads to R2 storage
```

### 1. Deploy Web Service

On your VPS:

```bash
cd adnihilator
pip install -e .

# Set environment variables
export ADMIN_USERNAME="admin"
export ADMIN_PASSWORD="your-password"
export WORKER_API_KEY="your-worker-secret"
export DATABASE_PATH="data/adnihilator.db"
export R2_PUBLIC_URL="https://pub-xxxxx.r2.dev"

# Run with uvicorn
uvicorn web.app:app --host 0.0.0.0 --port 8001 --workers 2
```

See `deploy/adnihilator.service` for a systemd service example.

### 2. Run Worker Daemon

On your local machine (with GPU recommended):

```bash
# Copy and edit the example script
cp scripts/run-worker.sh.example scripts/run-worker.sh
# Edit run-worker.sh with your credentials

# Start the worker
./scripts/run-worker.sh
```

Or run directly:

```bash
export API_URL="https://your-server.com"
export WORKER_API_KEY="your-worker-secret"
export OPENAI_API_KEY="sk-..."
export R2_ACCESS_KEY="..."
export R2_SECRET_KEY="..."
export R2_BUCKET="adnihilator"
export R2_ENDPOINT="https://....r2.cloudflarestorage.com"

python -m adnihilator.cli worker --daemon --interval 60
```

### 3. Add Podcasts

Visit `https://your-server.com` and:
1. Login with admin credentials
2. Add podcast RSS feeds
3. Episodes are auto-queued for processing
4. Get your ad-free RSS feed URL

## CLI Reference

### `adnihilator detect`

Detect ads in an audio file.

```bash
adnihilator detect INPUT [OPTIONS]
```

**Options:**
- `--out FILE`: Output JSON file path
- `--whisper-model MODEL`, `-m`: Whisper model (tiny/base/small/medium/large)
- `--device DEVICE`, `-d`: Processing device (cpu/cuda)
- `--llm-provider PROVIDER`, `-l`: LLM provider (none/openai)
- `--config FILE`, `-c`: Config file path
- `--splice`: Automatically splice out ads after detection
- `--confidence-threshold FLOAT`: Minimum confidence to remove (default: 0.35)

### `adnihilator splice`

Remove detected ads from audio.

```bash
adnihilator splice DETECTION_JSON [OPTIONS]
```

**Options:**
- `--out FILE`: Output audio file path
- `--confidence-threshold FLOAT`: Minimum confidence to remove (default: 0.35)

### `adnihilator download-model`

Download a Whisper model.

```bash
adnihilator download-model MODEL
```

**Arguments:**
- `MODEL`: Model size (tiny/base/small/medium/large)

## Output Format

Detection results are saved as JSON:

```json
{
  "audio_path": "podcast.mp3",
  "duration": 3247.5,
  "segments": [
    {
      "index": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "Welcome to the podcast",
      "words": [...]
    }
  ],
  "candidates": [
    {
      "start": 120.5,
      "end": 165.3,
      "trigger_keywords": ["sponsor", "promo_code"],
      "heuristic_score": 0.75,
      "sponsors_found": ["BetterHelp"]
    }
  ],
  "ad_spans": [
    {
      "start": 122.1,
      "end": 163.8,
      "confidence": 0.95,
      "sponsor": "BetterHelp"
    }
  ],
  "model_info": {
    "whisper_model": "small",
    "llm_provider": "openai",
    "llm_model": "gpt-4.1-mini"
  }
}
```

## Development

### Running Tests

```bash
pytest
pytest tests/test_sponsors.py -v              # Single test file
pytest -k "test_extract" -v                   # Tests matching pattern
```

### Project Structure

```
adnihilator/
├── adnihilator/           # Core library
│   ├── cli.py             # CLI commands
│   ├── transcribe.py      # Whisper transcription
│   ├── external_transcript.py  # External transcript fetching
│   ├── sponsors.py        # Sponsor extraction from HTML
│   ├── ad_keywords.py     # Heuristic detection
│   ├── ad_llm.py          # LLM refinement
│   ├── two_pass.py        # Two-pass optimization
│   ├── splice.py          # Audio splicing
│   └── config.py          # Configuration
├── web/                   # Web service (optional)
│   ├── app.py             # FastAPI application
│   ├── routes/            # API endpoints
│   ├── services/          # RSS sync, R2 upload
│   └── templates/         # Admin UI
├── worker/                # Worker daemon (optional)
│   ├── daemon.py          # Job processing loop
│   ├── client.py          # API client
│   └── r2.py              # R2 upload
└── tests/                 # Test suite
```

## FAQ

### Why are some ads still present?

- **Low confidence**: Increase the threshold with `--confidence-threshold 0.5`
- **No LLM**: Heuristic-only mode is less accurate. Use `--llm-provider openai`
- **Dynamic ad insertion**: Some ads are inserted by podcast networks and may be harder to detect

### How accurate is ad detection?

With LLM refinement:
- **High confidence (>0.7)**: ~95% precision
- **Medium confidence (0.5-0.7)**: ~85% precision
- **Low confidence (<0.5)**: ~60-70% precision

Without LLM (heuristic only): ~70% precision overall

### Can I run this without OpenAI?

Yes! Use heuristic-only mode:
```bash
adnihilator detect podcast.mp3 --out results.json
```

This is faster and free, but less accurate than LLM refinement.

### What about privacy?

- **CLI mode**: Everything runs locally. Transcripts never leave your machine unless you enable LLM refinement.
- **LLM mode**: Transcripts are sent to OpenAI for ad detection. Do not use on sensitive content.
- **Web service**: Designed for self-hosting. You control all data.

### How much does this cost?

- **Whisper transcription**: Free (runs locally)
- **OpenAI LLM refinement**: ~$0.01-0.03 per hour of audio with GPT-4.1-mini
- **Cloudflare R2 storage**: ~$0.015/GB/month + minimal egress fees

A typical podcast (~1 hour) costs about 2 cents to process with LLM refinement.

## Supported Podcasts

AdNihilator works with any podcast, but has special optimizations for:

- **Substack podcasts**: Fast transcript fetching (no Whisper needed)
- **Lex Fridman Podcast**: Fast transcript fetching from lexfridman.com
- All other podcasts via Whisper transcription

## License

MIT

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Credits

Built with:
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Fast Whisper transcription
- [OpenAI API](https://openai.com) - LLM refinement
- [FastAPI](https://fastapi.tiangolo.com) - Web service
- [ffmpeg](https://ffmpeg.org) - Audio processing
