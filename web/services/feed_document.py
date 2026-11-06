"""Preserve source enclosure attributes that feedparser normalizes away."""

import gzip
from http.client import HTTPResponse
from urllib.request import BaseHandler, Request
import xml.etree.ElementTree as ET
import zlib

from feedparser.encodings import convert_to_utf8


class FeedDocumentCapture(BaseHandler):
    """Observe the same HTTP body read by feedparser, without another fetch."""

    handler_order = 900  # Before urllib's response/error processor.

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._content_type = ""
        self._content_encoding = ""
        self.url: str | None = None

    def http_response(self, request: Request, response: HTTPResponse) -> HTTPResponse:
        """Retain only successful feed bytes and relevant encoding metadata."""
        if 200 <= response.status < 300:
            self._chunks = []
            self._content_type = response.headers.get("Content-Type", "")
            self._content_encoding = response.headers.get("Content-Encoding", "").lower()
            self.url = response.geturl()
            original_read = response.read

            def read(*args: object, **kwargs: object) -> bytes:
                data = original_read(*args, **kwargs)
                self._chunks.append(data)
                return data

            response.read = read
        return response

    https_response = http_response

    def atom_omitted_types(self) -> list[list[bool]]:
        """Return per-entry enclosure MIME omission from actual Atom attributes.

        Atom links default to text/html inside feedparser. The original XML is
        needed to distinguish an omitted type from an explicit non-audio type.
        Source order is retained; callers check alignment before applying it.
        """
        body = b"".join(self._chunks)
        if "gzip" in self._content_encoding:
            body = gzip.decompress(body)
        elif "deflate" in self._content_encoding:
            try:
                body = zlib.decompress(body)
            except zlib.error:
                body = zlib.decompress(body, -zlib.MAX_WBITS)
        body = convert_to_utf8({"content-type": self._content_type}, body, {})
        root = ET.fromstring(body)
        namespace = root.tag.split("}", 1)[0] + "}" if root.tag.startswith("{") else ""
        return [
            ["type" not in link.attrib for link in entry.findall(namespace + "link")
             if link.get("rel", "alternate") == "enclosure"]
            for entry in root.findall(namespace + "entry")
        ]
