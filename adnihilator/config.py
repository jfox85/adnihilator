"""Configuration management for AdNihilator."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LLMConfig:
    """Configuration for LLM-based refinement."""

    provider: str = "none"
    model: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None

    @property
    def api_key(self) -> str | None:
        """Get API key from environment variable."""
        return os.environ.get(self.api_key_env)


@dataclass
class GeminiConfig:
    """Configuration for Gemini audio-based detection."""

    enabled: bool = False  # Feature flag
    api_key_env: str = "GEMINI_API_KEY"
    model: str = "gemini-2.0-flash-exp"

    @property
    def api_key(self) -> str | None:
        """Get API key from environment variable."""
        return os.environ.get(self.api_key_env)


@dataclass
class DetectConfig:
    """Configuration for ad detection."""

    heuristic_threshold: float = 0.4
    context_segments_before: int = 2
    context_segments_after: int = 2


@dataclass
class DetectionConfig:
    """Detection strategy configuration."""

    parallel_enabled: bool = False  # Feature flag - default OFF for safety


@dataclass
class Config:
    """Main configuration for AdNihilator."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)


def load_config(path: str | None = None) -> Config:
    """Load configuration from a TOML file.

    Args:
        path: Path to the config file. If None, returns default config.

    Returns:
        A Config object with loaded or default values.
    """
    if path is None:
        return Config()

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        import tomli

        with open(config_path, "rb") as f:
            data = tomli.load(f)
    except ImportError:
        raise ImportError("tomli is required for config loading. Run: pip install tomli")

    return _parse_config(data)


def _parse_config(data: dict[str, Any]) -> Config:
    """Parse configuration dictionary into Config object.

    Args:
        data: Dictionary from TOML file.

    Returns:
        Parsed Config object.
    """
    llm_data = data.get("llm", {})
    gemini_data = data.get("gemini", {})
    detect_data = data.get("detect", {})
    detection_data = data.get("detection", {})

    llm_config = LLMConfig(
        provider=llm_data.get("provider", "none"),
        model=llm_data.get("model", "gpt-4.1-mini"),
        api_key_env=llm_data.get("api_key_env", "OPENAI_API_KEY"),
        base_url=llm_data.get("base_url"),
    )

    gemini_config = GeminiConfig(
        enabled=gemini_data.get("enabled", False),
        api_key_env=gemini_data.get("api_key_env", "GEMINI_API_KEY"),
        model=gemini_data.get("model", "gemini-2.0-flash-exp"),
    )

    detect_config = DetectConfig(
        heuristic_threshold=detect_data.get("heuristic_threshold", 0.4),
        context_segments_before=detect_data.get("context_segments_before", 2),
        context_segments_after=detect_data.get("context_segments_after", 2),
    )

    detection_config = DetectionConfig(
        parallel_enabled=detection_data.get("parallel_enabled", False),
    )

    return Config(
        llm=llm_config,
        gemini=gemini_config,
        detect=detect_config,
        detection=detection_config,
    )
