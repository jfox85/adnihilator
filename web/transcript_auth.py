"""Optional read-only agent boundary; disabled without an owner-supplied hash."""

import hashlib
import hmac
import os
import re

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBasicCredentials

from .auth import security, verify_admin


def verify_transcript_read(request: Request,
                           credentials: HTTPBasicCredentials | None = Depends(security)) -> None:
    """Bearer can only reach transcript GETs/MCP, never UI/admin/recovery writes."""
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        expected = os.getenv("TRANSCRIPT_READ_TOKEN_SHA256", "")
        token = authorization[7:]
        if not re.fullmatch(r"[a-f0-9]{64}", expected) or not 20 <= len(token) <= 4096:
            raise HTTPException(401, "Transcript read authentication denied")
        if not hmac.compare_digest(hashlib.sha256(token.encode()).hexdigest(), expected):
            raise HTTPException(401, "Transcript read authentication denied")
        return
    verify_admin(request, credentials)
