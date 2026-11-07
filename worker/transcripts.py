"""Durable local transcript outbox with per-episode publication serialization."""

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from httpx import HTTPStatusError

from adnihilator.transcript_document import TranscriptDocument


def _scope(claim_id: str | None) -> str:
    """Keep the audio scope and every existing recovery claim independent."""
    return hashlib.sha256(json.dumps(claim_id).encode()).hexdigest()


def _sync(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic(path: Path, value: bytes) -> None:
    """Publish a complete file and durably record its directory entry."""
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _sync(path.parent)


def _encoded(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


@contextmanager
def _locked(directory: Path) -> Iterator[None]:
    """One process may stage or publish an episode at a time, including HTTP."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".outbox.lock").open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _marker_path(directory: Path, claim_id: str | None) -> Path:
    return directory / (".latest-" + _scope(claim_id) + ".json")


def _marker(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("generation"), int) or isinstance(value["generation"], bool) or value["generation"] < 1:
        raise ValueError("Invalid outbox generation")
    if not re.fullmatch(r"[a-f0-9]{64}", str(value.get("checksum", ""))) or value.get("state") not in ("pending", "delivered", "rejected"):
        raise ValueError("Invalid outbox marker")
    claim_id = value.get("claim_id")
    if claim_id is not None and not isinstance(claim_id, str):
        raise ValueError("Invalid outbox claim scope")
    if path.name != ".latest-" + _scope(claim_id) + ".json":
        raise ValueError("Outbox marker scope mismatch")
    return value


def _pending_path(directory: Path, marker: dict[str, Any]) -> Path:
    suffix = "" if marker["claim_id"] is None else "." + _scope(marker["claim_id"])
    return directory / (marker["checksum"] + suffix + ".pending")


def _envelope(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("document"), dict):
        raise ValueError("Invalid outbox envelope")
    if value.get("claim_id") is not None and not isinstance(value["claim_id"], str):
        raise ValueError("Invalid outbox claim scope")
    return value


def _body(document: TranscriptDocument, marker: dict[str, Any]) -> dict[str, Any]:
    return dict(document=document.model_dump(), claim_id=marker["claim_id"],
                downloaded_at=marker.get("downloaded_at"), outbox_generation=marker["generation"])


def _cleanup(directory: Path, marker: dict[str, Any]) -> None:
    """Retain older envelopes as superseded; never remove canonical archives."""
    current = _pending_path(directory, marker)
    for path in directory.glob("*.pending"):
        try:
            value = _envelope(path)
        except (ValueError, OSError):
            continue  # The normal flush parser will quarantine invalid files.
        if value.get("claim_id") != marker["claim_id"]:
            continue
        if path != current:
            os.replace(path, path.with_suffix(".superseded"))
        elif marker["state"] == "delivered":
            path.unlink()
        elif marker["state"] == "rejected":
            os.replace(path, path.with_suffix(".rejected"))
    _sync(directory)


def stage(root: Path, document: TranscriptDocument, claim_id: str | None = None,
          downloaded_at: str | None = None) -> Path:
    """Commit newest local staging order without inventing server precedence.

    Canonical archive precedes the atomic latest marker, the stage commit point.
    A committed marker can reconstruct its pending envelope after interruption.
    """
    if Path(document.episode_id).name != document.episode_id or document.episode_id in (".", ".."):
        raise ValueError("Invalid episode ID")
    directory = root / document.episode_id
    with _locked(directory):
        marker_path = _marker_path(directory, claim_id)
        previous = _marker(marker_path) if marker_path.exists() else None
        checksum = document.checksum()
        marker = dict(checksum=checksum, claim_id=claim_id, downloaded_at=downloaded_at,
                      generation=(previous["generation"] if previous else 0) + 1, state="pending")
        _atomic(directory / (checksum + ".json"), document.canonical_bytes())
        _atomic(marker_path, _encoded(marker))
        pending = _pending_path(directory, marker)
        _atomic(pending, _encoded(_body(document, marker)))
        _cleanup(directory, marker)
        return pending


def _recover(directory: Path, marker: dict[str, Any]) -> None:
    """A committed latest marker survives missing pending files or cleanup."""
    if marker["state"] == "pending":
        path = _pending_path(directory, marker)
        try:
            value = _envelope(path)
            ready = value.get("outbox_generation") == marker["generation"] and value.get("claim_id") == marker["claim_id"]
        except (ValueError, OSError):
            ready = False
        if not ready:
            document = TranscriptDocument.model_validate_json((directory / (marker["checksum"] + ".json")).read_bytes())
            if document.checksum() != marker["checksum"] or document.episode_id != directory.name:
                raise ValueError("Outbox archive identity mismatch")
            _atomic(path, _encoded(_body(document, marker)))
    _cleanup(directory, marker)


def _retry_age(directory: Path) -> float:
    """Retry fairness is independent of the immutable local revision marker."""
    paths = list(directory.glob("*.pending")) or list(directory.glob(".latest-*.json"))
    return min((path.stat().st_mtime for path in paths if path.exists()), default=float("inf"))


def flush(root: Path, client: Any, limit: int = 5) -> dict[str, int]:
    """Bound network attempts, recover latest markers, and retain legacy doubt.

    Multiple legacy pending files in one scope have no immutable ordering. Keep
    them for explicit producer restaging instead of inferring order from mtime.
    Other episodes/scopes remain eligible. No failure values are logged.
    """
    succeeded = failed = rejected = attempted = 0
    for directory in sorted((path for path in root.glob("*") if path.is_dir() and not path.is_symlink()), key=_retry_age):
        with _locked(directory):
            blocked = set()
            for marker_path in directory.glob(".latest-*.json"):
                try:
                    _recover(directory, _marker(marker_path))
                except (ValueError, OSError):
                    blocked.add(marker_path.name[len(".latest-"):-len(".json")])
                    failed += 1
            groups: dict[str, list[tuple[Path, dict[str, Any], TranscriptDocument]]] = defaultdict(list)
            for path in sorted(directory.glob("*.pending"), key=lambda entry: entry.stat().st_mtime):
                try:
                    value = _envelope(path)
                    document = TranscriptDocument.model_validate(value["document"])
                    if document.episode_id != directory.name:
                        raise ValueError("Outbox episode mismatch")
                    groups[_scope(value.get("claim_id"))].append((path, value, document))
                except ValueError:
                    os.replace(path, path.with_suffix(".rejected"))
                    rejected += 1
            for scope, entries in groups.items():
                if scope in blocked:
                    continue
                marker_path = _marker_path(directory, entries[0][1].get("claim_id"))
                marker = _marker(marker_path) if marker_path.exists() else None
                if marker is None and len(entries) > 1:
                    failed += len(entries)
                    continue
                for path, value, document in entries:
                    if marker and (marker["state"] != "pending" or marker["checksum"] != document.checksum() or value.get("outbox_generation") != marker["generation"]):
                        os.replace(path, path.with_suffix(".superseded"))
                        continue
                    if attempted >= max(0, limit):
                        continue
                    attempted += 1
                    try:
                        client.publish_transcript(document, value.get("claim_id"), value.get("downloaded_at"))
                        if marker:
                            marker["state"] = "delivered"
                            _atomic(marker_path, _encoded(marker))
                        path.unlink()
                        _sync(directory)
                        succeeded += 1
                    except (ValueError, HTTPStatusError) as exc:
                        if isinstance(exc, ValueError) or exc.response.status_code in (400, 409, 413, 422):
                            if marker:
                                marker["state"] = "rejected"
                                _atomic(marker_path, _encoded(marker))
                            os.replace(path, path.with_suffix(".rejected"))
                            rejected += 1
                        else:
                            os.utime(path, (time.time(), time.time()))
                            failed += 1
                    except Exception:
                        if path.exists():
                            os.utime(path, (time.time(), time.time()))
                        failed += 1  # Retain marker/envelope for idempotent retry.
    return dict(succeeded=succeeded, failed=failed, rejected=rejected)
