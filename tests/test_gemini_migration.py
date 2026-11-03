"""Provider-free tests for the Gemini 3.5 Flash-Lite / google-genai migration.

All SDK clients are mocked; no network calls or paid API usage.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from google.genai import types

from adnihilator import gemini_audio, gemini_chunked
from adnihilator.config import GeminiConfig, _parse_config, load_config
from adnihilator.gemini_audio import GeminiAudioClient
from adnihilator.gemini_chunked import GeminiChunkedClient, detect_ads_in_chunk
from adnihilator.gemini_common import (
    DEFAULT_GEMINI_MODEL,
    build_generate_config,
    extract_usage,
    supports_thinking_level,
)
from worker.daemon import WorkerDaemon

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPRECATED_FIELDS = ("temperature", "top_p", "top_k", "candidate_count")


def _usage(prompt: int = 1000, candidates: int = 50, thoughts: int = 20) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        thoughts_token_count=thoughts,
        total_token_count=prompt + candidates + thoughts,
    )


def _mock_client(response_text: str, usage: SimpleNamespace | None = None) -> Mock:
    client = Mock()
    client.files.upload.return_value = SimpleNamespace(
        name="files/abc",
        uri="https://example.invalid/files/abc",
        state=types.FileState.ACTIVE,
    )
    client.models.generate_content.return_value = SimpleNamespace(
        text=response_text,
        usage_metadata=usage if usage is not None else _usage(),
        candidates=[],
    )
    return client


def _assert_no_deprecated_params(config: types.GenerateContentConfig) -> None:
    for field in DEPRECATED_FIELDS:
        assert getattr(config, field) is None, f"{field} must not be sent to Gemini 3.5+"
    assert config.thinking_config.thinking_budget is None


class TestDefaults:
    def test_default_model_is_gemini_35_flash_lite(self):
        """Defaults move from gemini-2.5-flash to gemini-3.5-flash-lite."""
        assert DEFAULT_GEMINI_MODEL == "gemini-3.5-flash-lite"
        assert GeminiConfig().model == "gemini-3.5-flash-lite"
        assert GeminiConfig().thinking_level == "minimal"
        assert _parse_config({"gemini": {"enabled": True}}).gemini.model == "gemini-3.5-flash-lite"
        assert GeminiAudioClient(api_key="k").model == "gemini-3.5-flash-lite"
        assert GeminiChunkedClient(api_key="k").model == "gemini-3.5-flash-lite"

    def test_repo_config_uses_gemini_35_flash_lite(self):
        """The checked-in adnihilator.toml is migrated."""
        config = load_config(str(REPO_ROOT / "adnihilator.toml"))
        assert config.gemini.model == "gemini-3.5-flash-lite"
        assert config.gemini.thinking_level == "minimal"

    def test_explicit_model_override_preserved(self):
        """An explicitly configured supported model is not rewritten."""
        config = _parse_config({"gemini": {"model": "gemini-2.5-flash", "thinking_level": "HIGH"}})
        assert config.gemini.model == "gemini-2.5-flash"
        assert config.gemini.thinking_level == "high"

    def test_invalid_thinking_level_rejected(self):
        with pytest.raises(ValueError, match="thinking_level"):
            _parse_config({"gemini": {"thinking_level": "extreme"}})

    def test_unrelated_llm_provider_untouched(self):
        config = load_config(str(REPO_ROOT / "adnihilator.toml"))
        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4o-mini"


class TestRequestConfig:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gemini-3.5-flash-lite", True),
            ("models/gemini-3.5-flash-lite", True),
            ("gemini-3.6-flash", True),
            ("gemini-10-pro", True),
            ("gemini-2.5-flash", False),
            ("gemini-2.0-flash-exp", False),
            ("custom-model", False),
        ],
    )
    def test_supports_thinking_level(self, model: str, expected: bool):
        assert supports_thinking_level(model) is expected

    def test_gemini_35_config_uses_thinking_level_without_sampling(self):
        config = build_generate_config("gemini-3.5-flash-lite")
        assert isinstance(config, types.GenerateContentConfig)
        assert config.thinking_config.thinking_level == types.ThinkingLevel.MINIMAL
        _assert_no_deprecated_params(config)
        # Serialized request body carries only thinkingConfig.thinkingLevel
        dumped = config.model_dump(exclude_none=True)
        assert dumped == {"thinking_config": {"thinking_level": types.ThinkingLevel.MINIMAL}}

    def test_legacy_override_sends_no_config(self):
        """Pre-Gemini-3 overrides keep API defaults (no thinking_level)."""
        assert build_generate_config("gemini-2.5-flash", "high") is None

    def test_extract_usage_counts_thinking_as_output(self):
        usage = extract_usage(SimpleNamespace(usage_metadata=_usage(100, 10, 5)))
        assert usage == {
            "input_tokens": 100,
            "output_tokens": 15,
            "thinking_tokens": 5,
            "total_tokens": 115,
        }
        assert extract_usage(SimpleNamespace(usage_metadata=None))["output_tokens"] == 0


class TestSingleShotClient:
    def test_detect_ads_uses_genai_generate_content(self, tmp_path: Path):
        audio = tmp_path / "ep.mp3"
        audio.touch()
        payload = {"ads": [{"start_time": 10, "end_time": 40, "sponsor": "Acme",
                            "ad_type": "host_read", "confidence": 0.9, "reason": "promo"}]}
        client = _mock_client(json.dumps(payload))

        with patch.object(gemini_audio, "create_genai_client", return_value=client) as factory:
            spans, usage = GeminiAudioClient(api_key="test-key").detect_ads(audio, duration=120.0)

        factory.assert_called_once_with("test-key")
        client.files.upload.assert_called_once_with(file=str(audio))
        kwargs = client.models.generate_content.call_args.kwargs
        assert kwargs["model"] == "gemini-3.5-flash-lite"
        # Single user turn: [prompt text, uploaded file]; no prefilled model turn.
        contents = kwargs["contents"]
        assert len(contents) == 2 and isinstance(contents[0], str)
        assert contents[1] is client.files.upload.return_value
        assert not any(getattr(c, "role", None) == "model" for c in contents)
        _assert_no_deprecated_params(kwargs["config"])
        assert kwargs["config"].thinking_config.thinking_level == types.ThinkingLevel.MINIMAL

        assert [(s.start, s.end) for s in spans] == [(10, 40)]
        assert usage["model"] == "gemini-3.5-flash-lite"
        assert usage["input_tokens"] == 1000
        assert usage["output_tokens"] == 70

    def test_polls_processing_file_with_files_get(self, tmp_path: Path, monkeypatch):
        audio = tmp_path / "ep.mp3"
        audio.touch()
        client = _mock_client('{"ads": []}')
        client.files.upload.return_value = SimpleNamespace(
            name="files/abc", uri="u", state=types.FileState.PROCESSING
        )
        client.files.get.return_value = SimpleNamespace(
            name="files/abc", uri="u", state=types.FileState.ACTIVE
        )
        monkeypatch.setattr(gemini_audio.time, "sleep", lambda _s: None)

        with patch.object(gemini_audio, "create_genai_client", return_value=client):
            spans, _ = GeminiAudioClient(api_key="k").detect_ads(audio)

        client.files.get.assert_called_once_with(name="files/abc")
        assert spans == []

    def test_empty_text_with_finish_reason_fails_closed(self, tmp_path: Path):
        audio = tmp_path / "ep.mp3"
        audio.touch()
        client = _mock_client("")
        client.models.generate_content.return_value = SimpleNamespace(
            text=None, usage_metadata=None,
            candidates=[SimpleNamespace(finish_reason=types.FinishReason.SAFETY)],
        )
        with patch.object(gemini_audio, "create_genai_client", return_value=client):
            with pytest.raises(ValueError, match="blocked"):
                GeminiAudioClient(api_key="k").detect_ads(audio)

    def test_legacy_model_override_passes_through(self, tmp_path: Path):
        audio = tmp_path / "ep.mp3"
        audio.touch()
        client = _mock_client('{"ads": []}')
        with patch.object(gemini_audio, "create_genai_client", return_value=client):
            GeminiAudioClient(api_key="k", model="gemini-2.5-flash").detect_ads(audio)
        kwargs = client.models.generate_content.call_args.kwargs
        assert kwargs["model"] == "gemini-2.5-flash"
        assert kwargs["config"] is None


class TestChunkedClient:
    def test_chunk_request_shape_and_usage(self, tmp_path: Path):
        chunk = tmp_path / "chunk.mp3"
        chunk.touch()
        client = _mock_client(json.dumps({"ads": [{"start_time": 5, "end_time": 35,
                                                    "sponsor": "Acme", "ad_type": "host_read"}]}))
        totals: dict[str, int] = {}
        ads = detect_ads_in_chunk(client, chunk, 300.0, 600.0, "gemini-3.5-flash-lite",
                                  thinking_level="low", usage_totals=totals, max_retries=1)

        kwargs = client.models.generate_content.call_args.kwargs
        assert kwargs["model"] == "gemini-3.5-flash-lite"
        assert kwargs["contents"][1] is client.files.upload.return_value
        assert kwargs["config"].thinking_config.thinking_level == types.ThinkingLevel.LOW
        _assert_no_deprecated_params(kwargs["config"])
        assert ads[0]["start"] == 305.0 and ads[0]["end"] == 335.0
        assert totals["input_tokens"] == 1000 and totals["output_tokens"] == 70

    def test_chunked_client_aggregates_usage(self, tmp_path: Path, monkeypatch):
        audio = tmp_path / "ep.mp3"
        audio.touch()
        chunks = [(tmp_path / "c0.mp3", 0.0, 300.0), (tmp_path / "c1.mp3", 300.0, 600.0)]
        for path, _, _ in chunks:
            path.touch()
        monkeypatch.setattr(gemini_chunked, "split_audio_into_chunks",
                            lambda *_a, **_k: chunks)
        client = _mock_client('{"ads": []}')

        with patch.object(gemini_chunked, "create_genai_client", return_value=client):
            spans, usage = GeminiChunkedClient(api_key="k").detect_ads(audio, duration=600.0)

        assert spans == []
        assert client.models.generate_content.call_count == 2
        assert usage["model"] == "gemini-3.5-flash-lite"
        assert usage["input_tokens"] == 2000
        assert usage["output_tokens"] == 140
        assert usage["thinking_tokens"] == 40


class TestWorkerIntegration:
    def _daemon(self, **gemini_kwargs) -> WorkerDaemon:
        daemon = WorkerDaemon.__new__(WorkerDaemon)
        daemon.config = _parse_config({"gemini": {"enabled": True, **gemini_kwargs}})
        return daemon

    def test_worker_builds_clients_with_thinking_level(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        daemon = self._daemon(thinking_level="medium", chunked_threshold=1800)

        short = daemon._get_gemini_client(duration=600.0)
        long = daemon._get_gemini_client(duration=3600.0)

        assert isinstance(short, GeminiAudioClient)
        assert isinstance(long, GeminiChunkedClient)
        for client in (short, long):
            assert client.model == "gemini-3.5-flash-lite"
            assert client.thinking_level == "medium"

    def test_gemini_35_flash_lite_pricing(self):
        daemon = WorkerDaemon.__new__(WorkerDaemon)
        for provider in ("gemini", "gemini_chunked"):
            cost = daemon._calculate_llm_cost(
                provider=provider,
                model="gemini-3.5-flash-lite",
                input_tokens=1_000_000,
                output_tokens=1_000_000,
                audio_duration_seconds=3600.0,
            )
            # Audio is billed as input tokens; no per-second surcharge.
            assert cost == pytest.approx(0.30 + 2.50)
