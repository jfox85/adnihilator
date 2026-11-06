"""Durable atomic local outbox; best-effort publishing never invalidates audio."""

import json
import os
import time
from pathlib import Path
from typing import Any

from httpx import HTTPStatusError

from adnihilator.transcript_document import TranscriptDocument


def stage(root: Path, document: TranscriptDocument, claim_id: str | None = None,
          downloaded_at: str | None = None) -> Path:
    """Persist canonical bytes before upload. Fsync + replace survives interruption."""
    directory = root / document.episode_id
    # Episode IDs are server-generated, but reject traversal even for local callers.
    if Path(document.episode_id).name != document.episode_id or document.episode_id in (".", ".."):
        raise ValueError("Invalid episode ID")
    directory.mkdir(parents=True, exist_ok=True)
    checksum = document.checksum()
    canonical = directory / (checksum + ".json")
    pending = directory / (checksum + ".pending")
    for path, value in ((canonical, document.canonical_bytes()),
                        (pending, json.dumps(dict(document=document.model_dump(), claim_id=claim_id,
                                                  downloaded_at=downloaded_at)).encode())):
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return pending


def flush(root: Path, client: Any, limit: int = 5) -> dict[str, int]:
    """Bounded retry; keep canonical archive after successful upload."""
    succeeded = failed = rejected = 0
    # Retry scheduling uses mtime only, leaving canonical/pending content intact.
    # A failed old record moves behind fresh records for the next bounded cycle.
    pending_paths = sorted(root.glob("*/*.pending"), key=lambda p: p.stat().st_mtime)
    for path in pending_paths[:limit]:
        try:
            data = json.loads(path.read_text())
            document = TranscriptDocument.model_validate(data["document"])
            client.publish_transcript(document, data.get("claim_id"), data.get("downloaded_at"))
            path.unlink()
            succeeded += 1
        except (ValueError, HTTPStatusError) as exc:
            if isinstance(exc, ValueError) or exc.response.status_code in (400, 409, 413, 422):
                os.replace(path, path.with_suffix(".rejected"))
                rejected += 1
            else:
                os.utime(path, (time.time(), time.time()))
                failed += 1
        except Exception:
            os.utime(path, (time.time(), time.time()))
            failed += 1  # No body, URL, headers, secrets, or exception values logged.
    return dict(succeeded=succeeded, failed=failed, rejected=rejected)
