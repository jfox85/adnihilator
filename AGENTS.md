# AGENTS.md

Guidelines for AI coding agents working in the AdNihilator codebase.

## Build, Lint, and Test Commands

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_sponsors.py -v

# Run tests matching a pattern
pytest -k "test_extract" -v

# Run a specific test function
pytest tests/test_sponsors.py::TestPatternExtraction::test_twit_html_format -v

# Install in development mode
pip install -e ".[dev]"

# Run CLI
adnihilator detect podcast.mp3 --out results.json
adnihilator splice results.json --out clean.mp3

# Run web service locally
uvicorn web.app:app --reload

# Run worker daemon
python -m adnihilator.cli worker --daemon --interval 60
```

## Project Structure

```
adnihilator/          # Core library (Pydantic models, detection logic)
web/                  # FastAPI web service (SQLAlchemy models, routes)
worker/               # Worker daemon (claims jobs, processes audio)
tests/                # pytest test suite
  tests/web/          # Web service tests
  tests/worker/       # Worker tests
```

## Code Style

### Imports

Organize imports in this order, separated by blank lines:
1. Standard library (`os`, `json`, `pathlib`, `typing`)
2. Third-party (`typer`, `fastapi`, `pydantic`, `sqlalchemy`)
3. Local modules (`from .models import ...`, `from adnihilator.sponsors import ...`)

```python
import json
import os
from pathlib import Path
from typing import Annotated, Optional

import typer
from pydantic import BaseModel

from .models import DetectionResult
from .config import load_config
```

### Type Hints

**Always use type hints** for function arguments and return types:

```python
def format_duration(seconds: float) -> str:
    """Format duration in HH:MM:SS format."""
    ...

def fetch_external_transcript(url: str) -> list[TranscriptSegment] | None:
    """Fetch transcript from external source."""
    ...

def find_ad_candidates(
    segments: list[TranscriptSegment],
    duration: float,
    sponsors: SponsorInfo | None = None,
) -> list[AdCandidate]:
    ...
```

Use modern Python 3.10+ syntax:
- `list[str]` not `List[str]`
- `dict[str, int]` not `Dict[str, int]`
- `str | None` not `Optional[str]` (either is acceptable)

### Naming Conventions

| Element | Convention | Examples |
|---------|------------|----------|
| Functions | `snake_case` | `find_ad_candidates`, `process_job` |
| Classes | `PascalCase` | `WorkerDaemon`, `AdCandidate`, `EpisodeStatus` |
| Variables | `snake_case` | `api_url`, `heuristic_threshold` |
| Constants | `UPPER_SNAKE_CASE` | `SPONSOR_TRIGGER_KEYWORDS` |
| Private | Leading underscore | `_build_prompt`, `_extract_promo_code` |

### Data Models

Use **Pydantic** for core data structures in `adnihilator/models.py`:

```python
class AdSpan(BaseModel):
    """A refined advertisement span after LLM processing."""
    start: float
    end: float
    confidence: float
    reason: str
    candidate_indices: list[int] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    ad_type: Optional[str] = None
```

Use **SQLAlchemy 2.0** style for database models in `web/models.py`:

```python
class Episode(Base):
    __tablename__ = "episodes"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default=EpisodeStatus.PENDING.value)
```

### Error Handling

Define domain-specific exceptions and handle errors gracefully:

```python
class AudioError(Exception):
    """Raised when audio processing fails."""
    pass

# In CLI: print error and exit with code 1
try:
    validate_audio_file(str(input_mp3))
except AudioError as e:
    typer.echo(f"Error: {e}", err=True)
    raise typer.Exit(1)

# In worker: catch, log, and report failure
try:
    self.process_job(job)
except Exception as e:
    print(f"  Error: {e}")
    self.api_client.fail(job.id, str(e)[:500])
```

### Docstrings

Use triple-quoted docstrings for all public functions and classes:

```python
def extract_sponsors_with_patterns(description: str) -> list[Sponsor]:
    """Extract sponsors using regex patterns.

    Handles common formats:
    - HTML lists with <strong>Sponsors:</strong>
    - Plain text "Partner Deals" sections
    - Simple "Sponsors: Name1, Name2" lists

    Args:
        description: Episode description/show notes HTML or text.

    Returns:
        List of Sponsor objects found.
    """
```

## Testing Patterns

### Test Organization

- Group related tests in classes prefixed with `Test`
- Use descriptive test names that explain the scenario

```python
class TestPatternExtraction:
    def test_twit_html_format(self):
        """Extract sponsors from TWiT-style HTML."""
        ...

    def test_no_sponsors_returns_empty(self):
        """Description without sponsors returns empty list."""
        ...
```

### Fixtures

Use pytest fixtures for common setup:

```python
@pytest.fixture(scope="function")
def client(monkeypatch, tmp_path):
    """Create test client with isolated database."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DATABASE_PATH", db_path)
    
    from web.app import app
    with TestClient(app) as client:
        yield client
```

### Mocking

Use `unittest.mock` for external dependencies:

```python
from unittest.mock import Mock, patch

def test_llm_extraction_with_json_response(self):
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content='{"sponsors": [...]}'))]
    )
    
    sponsors = _extract_sponsors_with_llm(description, mock_client)
    assert len(sponsors) == 1
```

## Safety Rules

### Process Management

**NEVER** use broad `pkill` or `killall`. Find specific PIDs first:

```bash
# WRONG
pkill -f "worker"

# RIGHT
ps aux | grep "adnihilator.cli worker" | grep -v grep
kill <specific_pid>

# Or use launchctl for the worker
launchctl unload ~/Library/LaunchAgents/com.adnihilator.worker.plist
launchctl load ~/Library/LaunchAgents/com.adnihilator.worker.plist
```

### Environment Variables

Required for worker:
- `API_URL`, `WORKER_API_KEY`
- `R2_ACCESS_KEY`, `R2_SECRET_KEY`, `R2_BUCKET`, `R2_ENDPOINT`
- `OPENAI_API_KEY`

Required for web service:
- `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `WORKER_API_KEY`
- `DATABASE_PATH`, `R2_PUBLIC_URL`

### macOS OpenMP Fix

Set `KMP_DUPLICATE_LIB_OK=TRUE` before importing faster-whisper (already done in cli.py).
