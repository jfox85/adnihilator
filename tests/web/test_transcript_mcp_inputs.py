"""Actual MCP/REST requests for malformed inputs and event-loop liveness."""

from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any

import pytest
from sqlalchemy import event

from tests.web.test_transcript_ui_regressions import transcript_ui_client

MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def search_rpc(arguments: dict[str, Any]) -> dict[str, Any]:
    return dict(jsonrpc="2.0", id=1, method="tools/call", params=dict(name="search_transcripts", arguments=arguments))


class TestMalformedMCP:
    def test_under_cap_nested_json_returns_parse_error(self, transcript_ui_client: tuple) -> None:
        client, _ = transcript_ui_client
        response = client.post("/mcp", headers=MCP_HEADERS, content="[" * 20000 + "]" * 20000)
        assert response.status_code == 200
        assert response.json()["error"]["code"] == -32700
        control = client.post("/mcp", headers=MCP_HEADERS, json=dict(jsonrpc="2.0", id=1, method="ping"))
        assert control.json()["result"] == {}


class TestDateBoundaries:
    @pytest.mark.parametrize("bound,value", [
        ("after", "0001-01-01T00:00:00+05:00"),
        ("before", "9999-12-31T23:00:00-05:00"),
    ])
    def test_overflow_dates_are_rejected_on_every_read_surface(self, transcript_ui_client: tuple, bound: str, value: str) -> None:
        client, _ = transcript_ui_client
        arguments = {"query": "wombat", "podcast_id": "ui-podcast", bound: value}
        assert client.get("/api/transcripts/search", params=arguments).status_code == 422
        assert client.get("/transcripts", params=arguments).status_code == 422
        response = client.post("/mcp", headers=MCP_HEADERS, json=search_rpc(arguments))
        assert response.status_code == 200
        assert response.json()["error"]["code"] == -32602

    @pytest.mark.parametrize("bound,value", [("after", "0001-01-01T00:00:00Z"), ("before", "9999-12-31T23:00:00Z")])
    def test_valid_utc_limits_remain_searchable(self, transcript_ui_client: tuple, bound: str, value: str) -> None:
        client, _ = transcript_ui_client
        arguments = {"query": "wombat", "podcast_id": "ui-podcast", bound: value}
        assert client.get("/api/transcripts/search", params=arguments).json()["hits"]
        response = client.post("/mcp", headers=MCP_HEADERS, json=search_rpc(arguments))
        assert response.status_code == 200 and response.json()["result"]["isError"] is False


class TestMCPReadLiveness:
    @pytest.mark.parametrize("surface", ["mcp", "rest"])
    def test_slow_actual_sqlite_read_does_not_block_health(self, transcript_ui_client: tuple, surface: str) -> None:
        """Pause real FTS I/O, preserving real query results and HTTP handlers."""
        client, factory = transcript_ui_client
        started, release = threading.Event(), threading.Event()
        paused = False

        def wait_for_io(conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, many: bool) -> None:
            nonlocal paused
            if "transcript_fts match" in statement.lower() and not paused:
                paused = True
                started.set()
                assert release.wait(5)

        event.listen(factory.kw["bind"], "before_cursor_execute", wait_for_io)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                arguments = {"query": "wombat", "podcast_id": "ui-podcast"}
                reading = pool.submit(client.post, "/mcp", headers=MCP_HEADERS, json=search_rpc(arguments)) if surface == "mcp" else pool.submit(client.get, "/api/transcripts/search", params=arguments)
                assert started.wait(5)
                health = pool.submit(client.get, "/health")
                try:
                    assert health.result(timeout=0.5).status_code == 200
                finally:
                    release.set()
                    response = reading.result(timeout=5)
                assert response.status_code == 200
                assert "wombat" in response.text
        finally:
            release.set()
            event.remove(factory.kw["bind"], "before_cursor_execute", wait_for_io)
