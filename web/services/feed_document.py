"""Validate captured feed structure and retain attributes normalized by feedparser."""

import gzip
from html.entities import html5
from http.client import HTTPResponse
import re
from urllib.request import BaseHandler, Request
from xml.parsers import expat
import zlib

from feedparser.encodings import convert_to_utf8


# Only entity syntax is normalized, never tags, nesting or missing delimiters.
# Existing XML/numeric references retain their original XML interpretation.
_RECOVERABLE_AMPERSAND = re.compile(
    r"&(?!(?:amp|lt|gt|apos|quot);|#(?:[0-9]+|x[0-9a-fA-F]+);)"
    r"(?:([A-Za-z][A-Za-z0-9]*);)?"
)


def _validation_entity(match: re.Match[str]) -> str:
    """Make a loose-parser entity safe for structural validation only."""
    name = match.group(1)
    characters = html5.get(name + ";") if name else None
    if characters is not None:
        # Numeric references cannot inject markup into the structural parse.
        return "".join(f"&#{ord(character)};" for character in characters)
    return "&amp;" + match.group(0)[1:]


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

    def validate_structure(self) -> list[list[dict[str, str]]] | None:
        """Check complete RSS/Atom structure and return original Atom enclosures.

        feedparser recovers bare ampersands, HTML entities and undeclared
        namespace prefixes. Validate their structure without changing the data
        imported from feedparser: escape only entity tokens and disable namespace
        expansion, while still requiring closed, correctly nested XML tags.
        Original href/type/rel attributes are retained before URL normalization.
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
        validation_xml = _RECOVERABLE_AMPERSAND.sub(_validation_entity, body.decode("utf-8"))
        stack: list[str] = []
        root: str | None = None
        enclosures: list[list[dict[str, str]]] = []
        # Without namespace expansion Expat accepts an undeclared prefix, but
        # still enforces exact matching tags, proper nesting and end-of-document.
        parser = expat.ParserCreate()
        parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)

        def start(name: str, attributes: dict[str, str]) -> None:
            nonlocal root
            local = name.rsplit(":", 1)[-1].lower()
            if not stack:
                root = local
            if root == "feed" and stack == ["feed"] and local == "entry":
                enclosures.append([])
            elif root == "feed" and stack == ["feed", "entry"] and local == "link":
                attributes = {key.lower(): value for key, value in attributes.items()}
                if attributes.get("rel", "alternate").lower() == "enclosure":
                    enclosures[-1].append(attributes)
            stack.append(local)

        def end(name: str) -> None:
            stack.pop()  # Expat checks the original (case-sensitive) tag match.

        parser.StartElementHandler = start
        parser.EndElementHandler = end
        parser.Parse(validation_xml, True)
        if root not in {"rss", "rdf", "feed"}:
            raise ValueError("Not a recognized feed root")
        return enclosures if root == "feed" else None
