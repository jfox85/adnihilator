"""Required merge responses cannot silently erase detection coverage."""

import json
import socket
import traceback
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from adnihilator.ad_llm import OpenAIClient
from adnihilator.config import Config
from adnihilator.models import AdSpan, TranscriptSegment
from worker.daemon import DetectionSafetyError, WorkerDaemon


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forbid accidental paid provider requests in merge regressions."""
    def denied(*args: object, **kwargs: object) -> None:
        raise AssertionError("Network forbidden in required merge regressions")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


VALID_AD = {"start": 10, "end": 40, "confidence": 0.9, "reason": "Sponsor read", "sources": ["gemini"]}
INVALID_CONTENT = [
    None, "", "{}", "[]", '{"status":"error"}', '{"ads":null}', '{"ads":{}}',
    json.dumps({"ads": [None]}), json.dumps({"ads": [{"reason": "Missing bounds"}]}),
    *[json.dumps({"ads": [dict(VALID_AD, **change)]}) for change in [
        {"start": None}, {"start": "ten"}, {"start": "10"}, {"start": True},
        {"start": float("nan")}, {"end": float("inf")}, {"end": 10}, {"end": 5},
        {"confidence": float("nan")}, {"confidence": 1.5}, {"confidence": None},
        {"sources": "gemini"}, {"sources": [42]}, {"reason": {"error": "bad"}},
    ]],
    json.dumps({"ads": [VALID_AD, {"start": "bad", "end": 90}]}),
]


def sdk(content: str | None) -> Mock:
    """Construct a response without creating a real SDK transport."""
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))], usage=None,
    )
    return client


def merge(content: str | None) -> list[AdSpan]:
    """Exercise the real merger with only transport replaced."""
    client = OpenAIClient("synthetic-merge-key")
    with patch("openai.OpenAI", return_value=sdk(content)):
        result, _ = client.merge_and_refine({"duration": 100}, Config())
    return result


class TestRequiredMergeSafety:
    @pytest.mark.parametrize("content", INVALID_CONTENT)
    def test_malformed_response_is_not_a_no_ad_decision(self, content: str | None) -> None:
        """Every returned span is validated before any can become a cut."""
        with pytest.raises(ValueError):
            merge(content)

    @pytest.mark.parametrize("content", [None, "{}", '{"ads":[{"start":"bad","end":40}]}'])
    def test_worker_refuses_to_splice_after_invalid_required_merge(
        self, content: str | None, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An invalid response reaches the worker's fail-closed boundary."""
        monkeypatch.setenv("MERGE_TEST_KEY", "synthetic-merge-key")
        worker = WorkerDaemon.__new__(WorkerDaemon)
        worker.config = Config()
        worker.config.llm.provider = "openai"
        worker.config.llm.api_key_env = "MERGE_TEST_KEY"
        with patch("openai.OpenAI", return_value=sdk(content)):
            with pytest.raises(DetectionSafetyError, match="Required LLM merge failed"):
                worker._llm_merge_candidates(
                    [{"start": 10, "end": 40, "confidence": 0.9, "reason": "Sponsor"}], [],
                    [TranscriptSegment(index=0, start=0, end=100, text="Sponsor read")], None, 100,
                )

    def test_explicit_empty_list_is_still_a_valid_no_ad_decision(self) -> None:
        """A successful reviewer can explicitly reject all proposed cuts."""
        assert merge('{"ads":[]}') == []

    def test_valid_output_preserves_clamping_defaults_and_overlap_resolution(self) -> None:
        """Validation leaves existing valid boundary postprocessing intact."""
        ads = merge(json.dumps({"ads": [
            {"start": -5, "end": 40},
            {"start": 30, "end": 200, "confidence": 0.8, "sources": []},
        ]}))
        assert [(ad.start, ad.end) for ad in ads] == [(0, 40), (40, 100)]
        assert ads[0].confidence == 0.9
        assert ads[0].reason == "LLM merged"
        assert all(ad.sources == ["llm_new"] for ad in ads)

    def test_schema_error_does_not_leak_provider_credentials(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Pydantic input excerpts and exception context are sanitized."""
        content = json.dumps({"ads": [dict(VALID_AD, start="synthetic-merge-key")]})
        with pytest.raises(ValueError) as failure:
            merge(content)
        assert "synthetic-merge-key" not in str(failure.value)
        assert "synthetic-merge-key" not in "".join(traceback.format_exception(failure.value))
        assert "synthetic-merge-key" not in capsys.readouterr().out

    def test_provider_exception_is_sanitized_before_logging_or_reraising(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The same error boundary also masks raw SDK exception details."""
        provider = sdk(None)
        provider.chat.completions.create.side_effect = RuntimeError(
            "Request failed at https://provider.invalid/v1?key=synthetic-merge-key"
        )
        client = OpenAIClient("synthetic-merge-key")
        with patch("openai.OpenAI", return_value=provider):
            with pytest.raises(ValueError) as failure:
                client.merge_and_refine({"duration": 100}, Config())
        assert "synthetic-merge-key" not in "".join(traceback.format_exception(failure.value))
        assert "synthetic-merge-key" not in capsys.readouterr().out
