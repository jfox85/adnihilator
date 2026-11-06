"""Upload Gemini files without the legacy SDK's discovery endpoint."""

import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class GeminiFileClient:
    """Keep the legacy model interface while using authenticated REST uploads."""

    def __init__(self, legacy: Any, api_key: str) -> None:
        self._legacy = legacy
        self._api_key = api_key

    def __getattr__(self, name: str) -> Any:
        return getattr(self._legacy, name)

    def upload_file(self, path: str) -> Any:
        """Upload media using header authentication and return the SDK file type."""
        media = Path(path)
        mime = mimetypes.guess_type(media.name)[0] or "application/octet-stream"
        try:
            request = Request(
                "https://generativelanguage.googleapis.com/upload/v1beta/files",
                data=json.dumps({"file": {"display_name": media.name}}).encode(),
                headers={
                    "x-goog-api-key": self._api_key,
                    "X-Goog-Upload-Protocol": "resumable",
                    "X-Goog-Upload-Command": "start",
                    "X-Goog-Upload-Header-Content-Length": str(media.stat().st_size),
                    "X-Goog-Upload-Header-Content-Type": mime,
                    "Content-Type": "application/json",
                },
            )
            with urlopen(request, timeout=60) as response:
                upload_url = response.headers.get("X-Goog-Upload-URL", "")
            target = urlsplit(upload_url)
            if target.scheme != "https" or target.netloc != "generativelanguage.googleapis.com":
                raise ValueError("Unexpected Gemini upload destination")
            request = Request(
                upload_url,
                data=media.read_bytes(),
                headers={
                    "Content-Type": mime,
                    "X-Goog-Upload-Offset": "0",
                    "X-Goog-Upload-Command": "upload, finalize",
                },
            )
            with urlopen(request, timeout=300) as response:
                name = json.load(response)["file"]["name"]
            return self._legacy.get_file(name)
        except HTTPError as error:
            # Provider errors may include authentication or upload-session URLs.
            raise ValueError(f"Gemini file upload failed (HTTP {error.code})") from None
        except URLError:
            raise ValueError("Gemini file upload failed (network error)") from None
