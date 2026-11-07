"""Validate captured feeds and retain source provenance before URL/MIME defaults."""

from dataclasses import dataclass, field
import gzip
from html.entities import html5
from http.client import HTTPResponse
import io
import re
from typing import Callable
from urllib.request import BaseHandler, Request
import xml.sax
from xml.parsers import expat
import zlib

import feedparser
from feedparser import api
from feedparser.encodings import convert_to_utf8
from feedparser.mixin import _FeedParserMixin
from feedparser.sanitizer import replace_doctype
from feedparser.urls import make_safe_absolute_uri


_RECOVERABLE_AMPERSAND = re.compile(
    r"&(?!(?:amp|lt|gt|apos|quot);|#(?:[0-9]+|x[0-9a-fA-F]+);)"
    r"(?:([A-Za-z][A-Za-z0-9]*);)?"
)
_BASE_NAMESPACES = {uri.lower() for uri, prefix in _FeedParserMixin.namespaces.items() if not prefix}


def _validation_entity(match: re.Match[str]) -> str:
    """Make a loose-parser entity safe for structural validation only."""
    name = match.group(1)
    characters = html5.get(name + ";") if name else None
    if characters is not None:
        return "".join(f"&#{ord(character)};" for character in characters)
    return "&amp;" + match.group(0)[1:]


def _feed_namespace(namespace: str) -> bool:
    """Recognize the base vocabularies handled as feed elements by feedparser."""
    return not namespace or namespace.lower() in _BASE_NAMESPACES or "backend.userland.com/rss" in namespace.lower()


def _href(attributes: dict[str, str]) -> str | None:
    """Use feedparser's URL/URI/href precedence, without repairing the value."""
    value = attributes.get("url", attributes.get("uri", attributes.get("href")))
    return value or attributes.get("href")


@dataclass(frozen=True)
class SourceEnclosure:
    """Attributes from one actual parser event, before URI repair or MIME defaults."""

    attributes: dict[str, str]
    base_url: str
    reference: dict[str, object]
    guard_attributes: dict[str, str] | None = None

    @property
    def href(self) -> str | None:
        """Return the parser's original, un-repaired reference."""
        return _href(self.attributes)

    @property
    def guard_href(self) -> str | None:
        """Keep D17's XML/entity-aware URL guard on recovered attribute syntax."""
        return _href(self.guard_attributes if self.guard_attributes is not None else self.attributes)


@dataclass
class SourceEntry:
    """Namespace identity and references belonging to one exact parsed entry."""

    actual_namespace: bool
    enclosures: list[SourceEnclosure] = field(default_factory=list)


class _SourceObserver:
    """Observe feedparser's own entry/context decisions, without changing its data."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.source_entries: list[SourceEntry] = []
        self.source_namespace = ""
        self.source_root_namespace: str | None = None

    def _start_item(self, attributes: dict[str, str]) -> None:
        super()._start_item(attributes)
        self.source_entries.append(SourceEntry(_feed_namespace(self.source_namespace)))

    _start_entry = _start_item

    def _guard_attributes(self, attributes: dict[str, str]) -> dict[str, str] | None:
        return None

    def _reference(self, attributes: dict[str, str], handler: Callable[[dict[str, str]], None]) -> None:
        original = dict(attributes)
        namespace = self.source_namespace
        base = self.baseuri
        context = self._get_context()
        before = len(context.get("links", []))
        handler(attributes)
        # This uses the library's exact context object, not guessed XML depth.
        # Source/notes/image links therefore cannot borrow an entry's provenance.
        if not self.entries or context is not self.entries[-1] or not _feed_namespace(namespace):
            return
        appended = context.get("links", [])[before:]
        for link in appended:
            if link.get("rel") == "enclosure":
                # Retain this exact reference object through later library
                # normalization; never rediscover it by list position or URL.
                self.source_entries[-1].enclosures.append(
                    SourceEnclosure(original, base, link, self._guard_attributes(original))
                )

    def _start_link(self, attributes: dict[str, str]) -> None:
        self._reference(attributes, super()._start_link)

    def _start_enclosure(self, attributes: dict[str, str]) -> None:
        self._reference(attributes, super()._start_enclosure)


class _SourceStrict(_SourceObserver, api.StrictFeedParser):
    def startElementNS(self, name: tuple[str | None, str], qname: str | None, attrs: object) -> None:
        """Retain the actual SAX namespace before feedparser remaps prefixes."""
        self.source_namespace = name[0] or ""
        if self.source_root_namespace is None:
            self.source_root_namespace = self.source_namespace
        super().startElementNS(name, qname, attrs)


class _SourceLoose(_SourceObserver, api.LooseFeedParser):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._source_scopes: list[dict[str, str]] = [{}]
        self._source_start_depth = 0
        self._source_end_depth = 0

    def unknown_starttag(self, tag: str, attrs: list[tuple[str, str]]) -> None:
        """Track lexical namespace scope during loose recovery."""
        # Namespace scopes here are independent of feedparser's global prefix
        # cache, which can retain a previous alias after a foreign re-declaration.
        outer = self._source_start_depth == 0
        if outer:
            scope = dict(self._source_scopes[-1])
            for key, value in attrs:
                if key.lower() == "xmlns":
                    scope[""] = value
                elif key.lower().startswith("xmlns:"):
                    scope[key[6:].lower()] = value
            prefix = tag.split(":", 1)[0].lower() if ":" in tag else ""
            self.source_namespace = scope.get(prefix, "" if not prefix else "urn:unbound:" + prefix)
            if self.source_root_namespace is None:
                self.source_root_namespace = self.source_namespace
            self._source_scopes.append(scope)
        self._source_start_depth += 1
        try:
            super().unknown_starttag(tag, attrs)
        finally:
            self._source_start_depth -= 1

    def unknown_endtag(self, tag: str) -> None:
        """Restore the enclosing namespace scope without changing parser data."""
        outer = self._source_end_depth == 0
        self._source_end_depth += 1
        try:
            super().unknown_endtag(tag)
        finally:
            self._source_end_depth -= 1
            if outer and len(self._source_scopes) > 1:
                self._source_scopes.pop()

    def _guard_attributes(self, attributes: dict[str, str]) -> dict[str, str]:
        # A loose parser can retain numeric/named entities in an attribute.
        # Decode the exact original start tag with the unchanged D17 validator,
        # but never use those validation-rewritten values as imported MIME/URLs.
        raw = self.get_starttag_text()
        if not raw:
            raise ValueError("Original enclosure tag unavailable")
        raw = raw.rstrip()
        if not raw.endswith("/>"):
            raw = raw[:-1] + "/>" if raw.endswith(">") else raw + "/>"
        result: dict[str, str] = {}
        parser = expat.ParserCreate()
        parser.StartElementHandler = lambda name, values: result.update({key.lower(): value for key, value in values.items()})
        parser.Parse(_RECOVERABLE_AMPERSAND.sub(_validation_entity, raw), True)
        return result


class _NoExternalEntities(xml.sax.handler.EntityResolver):
    def resolveEntity(self, publicId: str | None, systemId: str | None) -> xml.sax.xmlreader.InputSource:
        """Return empty bytes rather than opening an external XML resource."""
        source = xml.sax.xmlreader.InputSource()
        source.setByteStream(io.BytesIO(b""))
        return source


def _observe_sources(body: bytes, base: str, language: str | None, strict: bool) -> _SourceObserver:
    """Replay only captured bytes through per-call observers; never fetch again.

    These are the same strict/loose engines and doctype preprocessing used by
    feedparser.api.parse. No parser globals are patched. Their normalized entry
    data must equal the original parse before any observed index is trusted.
    """
    _, body, entities = replace_doctype(body)
    if strict and api._XML_AVAILABLE:
        observer = _SourceStrict(base, language, "utf-8")
        observer.resolve_relative_uris = feedparser.RESOLVE_RELATIVE_URIS
        observer.sanitize_html = feedparser.SANITIZE_HTML
        parser = xml.sax.make_parser(api.PREFERRED_XML_PARSERS)
        parser.setFeature(xml.sax.handler.feature_namespaces, True)
        for feature in [xml.sax.handler.feature_external_ges, xml.sax.handler.feature_external_pes]:
            try:
                parser.setFeature(feature, False)
            except (xml.sax.SAXNotSupportedException, xml.sax.SAXNotRecognizedException):
                pass
        parser.setEntityResolver(_NoExternalEntities())
        parser.setContentHandler(observer)
        parser.setErrorHandler(observer)
        source = xml.sax.xmlreader.InputSource()
        source.setByteStream(io.BytesIO(body))
        try:
            parser.parse(source)
            return observer
        except xml.sax.SAXException:
            pass
    observer = _SourceLoose(base, language, "utf-8", entities)
    observer.resolve_relative_uris = feedparser.RESOLVE_RELATIVE_URIS
    observer.sanitize_html = feedparser.SANITIZE_HTML
    observer.feed(body.decode("utf-8", "replace"))
    return observer


class FeedDocumentCapture(BaseHandler):
    """Observe the same HTTP body read by feedparser, without another fetch."""

    handler_order = 900

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._content_type = ""
        self._content_encoding = ""
        self._content_location = ""
        self._content_language: str | None = None
        self.url: str | None = None

    def http_response(self, request: Request, response: HTTPResponse) -> HTTPResponse:
        """Retain successful feed bytes and their original decoding/base headers."""
        if 200 <= response.status < 300:
            self._chunks = []
            self._content_type = response.headers.get("Content-Type", "")
            self._content_encoding = response.headers.get("Content-Encoding", "").lower()
            self._content_location = response.headers.get("Content-Location", "")
            self._content_language = response.headers.get("Content-Language")
            self.url = response.geturl()
            original_read = response.read

            def read(*args: object, **kwargs: object) -> bytes:
                data = original_read(*args, **kwargs)
                self._chunks.append(data)
                return data

            response.read = read
        return response

    https_response = http_response

    def validate_structure(self, entries: list[dict[str, object]]) -> list[SourceEntry] | None:
        """Keep D17 completeness checks, then observe exact namespace/context origin.

        Validation-only escaping is never used as imported enclosure data. Atom
        metadata comes from the original bytes before URL/MIME defaults, including
        RSS-style enclosures and author contexts. Foreign entry/link events are
        distinguished by their actual namespace, not local names or list lengths.
        """
        body = b"".join(self._chunks)
        if "gzip" in self._content_encoding:
            body = gzip.decompress(body)
        elif "deflate" in self._content_encoding:
            try:
                body = zlib.decompress(body)
            except zlib.error:
                body = zlib.decompress(body, -zlib.MAX_WBITS)
        conversion: dict[str, object] = {}
        body = convert_to_utf8({"content-type": self._content_type}, body, conversion)
        root: str | None = None
        parser = expat.ParserCreate()
        parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)

        def start(name: str, attributes: dict[str, str]) -> None:
            nonlocal root
            if root is None:
                root = name.rsplit(":", 1)[-1].lower()

        parser.StartElementHandler = start
        parser.Parse(_RECOVERABLE_AMPERSAND.sub(_validation_entity, body.decode("utf-8")), True)
        if root not in {"rss", "rdf", "feed"}:
            raise ValueError("Not a recognized feed root")
        if root != "feed":
            return None
        base = make_safe_absolute_uri(self.url or "", self._content_location) or make_safe_absolute_uri(self._content_location) or self.url or ""
        observer = _observe_sources(body, base, self._content_language, bool(conversion.get("encoding")))
        if observer.entries != entries or len(observer.source_entries) != len(entries):
            raise ValueError("Original entries and source provenance do not agree")
        if not _feed_namespace(observer.source_root_namespace or ""):
            raise ValueError("Not a recognized Atom namespace")
        return observer.source_entries
