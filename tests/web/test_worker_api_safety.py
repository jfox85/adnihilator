"""Real SQLite and HTTP regressions for worker callback safety."""

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event

from web.models import Episode, Podcast

HEADERS = {"X-Worker-API-Key": "synthetic-worker-key"}
COMPLETE = {"processed_audio_key": "podcast/episode.mp3", "processed_duration": 10, "ads_removed_seconds": 2}


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[tuple]:
    """Use only a temporary SQLite database, without feed or provider requests."""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api-safety.db"))
    monkeypatch.setenv("WORKER_API_KEY", HEADERS["X-Worker-API-Key"])
    from web.app import app
    from web import dependencies
    with TestClient(app, raise_server_exceptions=False) as client:
        with dependencies.SessionFactory() as db:
            podcast = Podcast(source_rss_url="https://fixture.invalid/feed", title="Synthetic")
            db.add(podcast)
            db.flush()
            episodes = [Episode(podcast_id=podcast.id, guid=f"episode-{index}", status="processing",
                                original_audio_url="https://fixture.invalid/audio.mp3",
                                created_at=datetime(2026, 1, 1) + timedelta(seconds=index)) for index in range(2)]
            db.add_all(episodes)
            db.commit()
            ids = [episode.id for episode in episodes]
        yield client, dependencies.SessionFactory, ids


class TestWorkerCallbackSafety:
    def test_simultaneous_http_claims_have_distinct_oldest_jobs(self, harness: tuple) -> None:
        """Two SQLite sessions reading the same oldest row cannot both claim it."""
        client, factory, ids = harness
        with factory() as db:
            db.query(Episode).update({Episode.status: "pending"})
            db.commit()
        barrier = threading.Barrier(2)
        seen = threading.local()
        engine = factory.kw["bind"]

        def align_first_selection(
            conn: Any, cursor: Any, statement: str, parameters: Any,
            context: Any, executemany: bool,
        ) -> None:
            if statement.lstrip().upper().startswith("SELECT") and "ORDER BY episodes.created_at ASC" in statement and not getattr(seen, "aligned", False):
                seen.aligned = True
                barrier.wait(timeout=10)

        event.listen(engine, "after_cursor_execute", align_first_selection)
        try:
            def claim(_: int) -> dict:
                # Separate portals allow actual simultaneous requests despite the
                # synchronous database work inside the async route.
                other = TestClient(client.app, raise_server_exceptions=False)
                try:
                    response = other.post("/api/queue/claim", headers=HEADERS)
                    assert response.status_code == 200, response.text
                    return response.json()
                finally:
                    other.close()
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(claim, range(2)))
        finally:
            event.remove(engine, "after_cursor_execute", align_first_selection)
        assert {result["id"] for result in results} == set(ids)
        with factory() as db:
            assert db.query(Episode).filter_by(status="processing").count() == 2

    @pytest.mark.parametrize("status", ["complete", "pending", "skipped", "failed"])
    def test_failure_callback_preserves_nonprocessing_episode(self, harness: tuple, status: str) -> None:
        """A late or duplicate failure cannot reset a finished or waiting job."""
        client, factory, ids = harness
        with factory() as db:
            episode = db.get(Episode, ids[0])
            episode.status = status
            episode.processed_audio_key = "preserved.mp3"
            episode.error_message = "preserved error"
            db.commit()
        response = client.post(f"/api/queue/{ids[0]}/fail", headers=HEADERS, json={"error": "late failure"})
        assert response.status_code == 200
        assert response.json()["status"] == ("already_failed" if status == "failed" else "ignored")
        with factory() as db:
            episode = db.get(Episode, ids[0])
            assert (episode.status, episode.retry_count, episode.error_message, episode.processed_audio_key) == (status, 0, "preserved error", "preserved.mp3")

    def test_duplicate_failure_spends_one_retry(self, harness: tuple) -> None:
        """Retrying the same HTTP delivery does not spend another attempt."""
        client, factory, ids = harness
        for _ in range(2):
            assert client.post(f"/api/queue/{ids[0]}/fail", headers=HEADERS, json={"error": "failed"}).status_code == 200
        with factory() as db:
            assert db.get(Episode, ids[0]).retry_count == 1

    def test_simultaneous_failure_deliveries_spend_one_retry(self, harness: tuple) -> None:
        """Concurrent retries of one callback cannot both change the attempt."""
        client, factory, ids = harness
        barrier = threading.Barrier(2)
        seen = threading.local()
        engine = factory.kw["bind"]

        def align_first_selection(
            conn: Any, cursor: Any, statement: str, parameters: Any,
            context: Any, executemany: bool,
        ) -> None:
            if statement.lstrip().upper().startswith("SELECT") and "WHERE episodes.id =" in statement and not getattr(seen, "aligned", False):
                seen.aligned = True
                barrier.wait(timeout=10)

        def fail(_: int) -> dict:
            other = TestClient(client.app, raise_server_exceptions=False)
            try:
                response = other.post(f"/api/queue/{ids[0]}/fail", headers=HEADERS, json={"error": "failed"})
                assert response.status_code == 200, response.text
                return response.json()
            finally:
                other.close()

        event.listen(engine, "after_cursor_execute", align_first_selection)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                responses = list(pool.map(fail, range(2)))
        finally:
            event.remove(engine, "after_cursor_execute", align_first_selection)
        assert {response["status"] for response in responses} == {"retrying", "ignored"}
        with factory() as db:
            episode = db.get(Episode, ids[0])
            assert (episode.status, episode.retry_count) == ("pending", 1)

    @pytest.mark.parametrize("field,value", [
        ("processed_duration", -1), ("ads_removed_seconds", -1),
        ("llm_input_tokens", -1), ("llm_output_tokens", 2**63), ("llm_total_tokens", 2**63),
        ("llm_cost_usd", -0.1), ("llm_cost_usd", 10000),
        ("processed_audio_key", ""), ("processed_audio_key", " \t\n"),
    ])
    def test_invalid_completion_is_422_without_database_mutation(self, harness: tuple, field: str, value: object) -> None:
        """Reject invalid persisted values before the episode becomes complete."""
        client, factory, ids = harness
        response = client.post(f"/api/queue/{ids[0]}/complete", headers=HEADERS, json={**COMPLETE, field: value})
        assert response.status_code == 422, response.text
        with factory() as db:
            assert db.get(Episode, ids[0]).status == "processing"

    @pytest.mark.parametrize("field", ["processed_duration", "ads_removed_seconds", "llm_cost_usd"])
    @pytest.mark.parametrize("number", ["1e309", "-1e309", "NaN"])
    def test_nonfinite_raw_http_returns_serializable_422(self, harness: tuple, field: str, number: str) -> None:
        """Nonfinite input must not turn the validation response into a 500."""
        client, factory, ids = harness
        import json
        body = json.dumps(COMPLETE)[:-1] + f', "{field}": {number}' + "}"
        response = client.post(f"/api/queue/{ids[0]}/complete", headers={**HEADERS, "Content-Type": "application/json"}, content=body)
        assert response.status_code == 422, response.text
        assert isinstance(response.json()["detail"], list)
        with factory() as db:
            assert db.get(Episode, ids[0]).status == "processing"

    def test_nested_nonfinite_validation_input_returns_422(self, harness: tuple) -> None:
        """An invalid body shape may put an entire array in the error input."""
        client, _, ids = harness
        response = client.post(
            f"/api/queue/{ids[0]}/complete",
            headers={**HEADERS, "Content-Type": "application/json"},
            content='[{"processed_duration": 1e309}]',
        )
        assert response.status_code == 422, response.text
        assert response.json()["detail"][0]["input"] == [{"processed_duration": "inf"}]

    @pytest.mark.parametrize("body", [{"step": "x", "percent": -1}, {"step": "x", "percent": 101}, {"step": "x" * 51}])
    def test_invalid_progress_is_422(self, harness: tuple, body: dict) -> None:
        """Progress stays inside the UI range and natural database column size."""
        client, factory, ids = harness
        response = client.post(f"/api/queue/{ids[0]}/progress", headers=HEADERS, json=body)
        assert response.status_code == 422, response.text
        with factory() as db:
            assert db.get(Episode, ids[0]).progress_step is None

    def test_valid_boundary_values_and_key_are_preserved(self, harness: tuple) -> None:
        """The guard accepts natural bounds without trimming a valid storage key."""
        client, factory, ids = harness
        key = " path/épisode.mp3 "
        response = client.post(f"/api/queue/{ids[0]}/complete", headers=HEADERS, json={**COMPLETE, "processed_audio_key": key, "llm_input_tokens": 2**63 - 1, "llm_cost_usd": 9999.999999})
        assert response.status_code == 200, response.text
        with factory() as db:
            assert db.get(Episode, ids[0]).processed_audio_key == key
