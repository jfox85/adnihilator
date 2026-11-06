"""Read existing worker artifacts without modifying them or trusting partial writes."""

import hashlib
import json
import re
from pathlib import Path

from .transcript_document import MAX_DOCUMENT_BYTES, TranscriptDocument


def read_artifact(path: Path, episode_id: str, podcast_id: str, guid: str,
                  original_audio_url: str) -> TranscriptDocument | None:
    """Preserve JSON precision, using timestamped text only if segments are absent."""
    before = path.stat()
    if before.st_size > MAX_DOCUMENT_BYTES:
        raise ValueError("Artifact too large")
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Artifact must be an object")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Artifact changed during import")
    segments = data.get("segments") or []
    if not segments:
        text_path = path.with_name(path.stem + "_transcript.txt")
        if text_path.exists():
            if text_path.stat().st_size > MAX_DOCUMENT_BYTES:
                raise ValueError("Artifact too large")
            text_before = text_path.stat()
            value = text_path.read_text()
            text_after = text_path.stat()
            if (text_before.st_size, text_before.st_mtime_ns) != (text_after.st_size, text_after.st_mtime_ns):
                raise ValueError("Text artifact changed during import")
            for line in value.splitlines():
                if not line.strip():
                    continue
                match = re.fullmatch(r"\[([\d.]+)s - ([\d.]+)s\] (.*)", line)
                if not match:
                    raise ValueError("Partial or unsupported text artifact")
                segments.append(dict(index=len(segments), start=float(match[1]), end=float(match[2]), text=match[3]))
    if not segments:
        return None
    metadata = data.get("model_info") or {}
    if not isinstance(metadata, dict) or any(
        value is not None and not isinstance(value, str)
        for value in (metadata.get(key) for key in ("transcript_source", "engine", "parakeet_model", "whisper_model"))
    ):
        raise ValueError("Invalid artifact model metadata")
    document = TranscriptDocument(episode_id=episode_id, podcast_id=podcast_id, guid=guid,
                              original_audio_url=data.get("audio_path") or original_audio_url,
                              source="artifact:" + (metadata.get("transcript_source") or metadata.get("engine") or "unknown"),
                              model=metadata.get("parakeet_model") or metadata.get("whisper_model"),
                              producer_version="artifact-import-v1", source_artifact_id=path.stem,
                              source_artifact_checksum=hashlib.sha256(raw).hexdigest(), duration=data.get("duration"),
                              timestamp_basis="external_source" if metadata.get("transcript_source") in ("external", "substack", "lexfridman") else "original_audio",
                              segments=segments, ad_spans=data.get("ad_spans", []))
    # Validate nested values before arithmetic so malformed artifacts stay
    # within the importer's per-episode ValueError boundary.
    duration = document.duration
    if duration and max(segment.end for segment in document.segments) < duration - max(120, duration * .1):
        raise ValueError("Artifact appears to be an incomplete prefix")
    return document
