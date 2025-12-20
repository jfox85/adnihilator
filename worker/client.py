"""API client for communicating with the web service."""

import logging
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class EpisodeJob:
    """An episode job to process."""

    id: str
    podcast_id: str
    guid: str
    title: Optional[str]
    original_audio_url: str
    description: Optional[str] = None
    podcast_title: Optional[str] = None
    source_url: Optional[str] = None  # Episode page URL for external transcripts


class WorkerClient:
    """Client for the worker API with persistent connection pooling."""

    def __init__(self, api_url: str, api_key: str, timeout: float = 30.0):
        """Initialize the client.

        Args:
            api_url: Base URL of the API (e.g., https://feeds.example.com)
            api_key: Worker API key
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

        # Create persistent HTTP client with connection pooling
        # This prevents connection exhaustion and improves reliability
        self.client = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            transport=httpx.HTTPTransport(retries=3),
        )

    def _headers(self) -> dict:
        """Get headers for API requests."""
        return {"X-Worker-API-Key": self.api_key}

    def close(self) -> None:
        """Close the HTTP client and release connections."""
        self.client.close()

    def claim(self) -> Optional[EpisodeJob]:
        """Claim the next pending episode.

        Returns:
            EpisodeJob if one was claimed, None if queue is empty.
        """
        try:
            response = self.client.post(
                f"{self.api_url}/api/queue/claim",
                headers=self._headers(),
            )
            response.raise_for_status()

            data = response.json()
            if data is None:
                return None

            logger.info(f"Claimed episode: {data.get('title', data['id'])}")
            return EpisodeJob(
                id=data["id"],
                podcast_id=data["podcast_id"],
                guid=data["guid"],
                title=data.get("title"),
                original_audio_url=data["original_audio_url"],
                description=data.get("description"),
                podcast_title=data.get("podcast_title"),
                source_url=data.get("source_url"),
            )
        except httpx.TimeoutException as e:
            logger.error(f"⚠️  API TIMEOUT - Network issue or API is slow: {e}")
            logger.error("   This may indicate the API is down or unreachable")
            return None
        except httpx.ConnectError as e:
            logger.error(f"⚠️  API CONNECTION FAILED - Cannot reach {self.api_url}: {e}")
            logger.error("   Check network connectivity and API URL configuration")
            return None
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 401 or status == 403:
                logger.error(f"⚠️  API AUTHENTICATION FAILED - Check WORKER_API_KEY: {e}")
            elif status >= 500:
                logger.error(f"⚠️  API SERVER ERROR ({status}) - API may be down: {e}")
            else:
                logger.error(f"⚠️  API REQUEST FAILED ({status}): {e}")
            logger.error("   Worker will retry but this indicates an API problem")
            return None
        except httpx.HTTPError as e:
            logger.error(f"⚠️  API REQUEST FAILED - Unknown HTTP error: {e}")
            return None
        except Exception as e:
            logger.error(f"⚠️  UNEXPECTED ERROR claiming episode: {e}", exc_info=True)
            return None

    def complete(
        self,
        episode_id: str,
        audio_key: str,
        duration: float,
        ads_removed: float,
        detection_result_path: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_input_tokens: Optional[int] = None,
        llm_output_tokens: Optional[int] = None,
        llm_total_tokens: Optional[int] = None,
        llm_cost_usd: Optional[float] = None,
        detection_source: Optional[str] = None,
    ) -> None:
        """Mark an episode as complete.

        Args:
            episode_id: ID of the episode
            audio_key: R2 key where audio was uploaded
            duration: Duration of processed audio in seconds
            ads_removed: Seconds of ads removed
            detection_result_path: Local path to detection result JSON
            llm_provider: LLM provider used (e.g., "openai", "gemini")
            llm_model: Model name used
            llm_input_tokens: Number of input tokens
            llm_output_tokens: Number of output tokens
            llm_total_tokens: Total tokens used
            llm_cost_usd: Total cost in USD
            detection_source: Detection method ("timestamps", "gemini", "whisper", etc.)
        """
        payload = {
            "processed_audio_key": audio_key,
            "processed_duration": duration,
            "ads_removed_seconds": ads_removed,
        }
        if detection_result_path:
            payload["detection_result_path"] = detection_result_path

        # LLM tracking fields
        if llm_provider:
            payload["llm_provider"] = llm_provider
        if llm_model:
            payload["llm_model"] = llm_model
        if llm_input_tokens is not None:
            payload["llm_input_tokens"] = llm_input_tokens
        if llm_output_tokens is not None:
            payload["llm_output_tokens"] = llm_output_tokens
        if llm_total_tokens is not None:
            payload["llm_total_tokens"] = llm_total_tokens
        if llm_cost_usd is not None:
            payload["llm_cost_usd"] = llm_cost_usd
        if detection_source:
            payload["detection_source"] = detection_source

        try:
            response = self.client.post(
                f"{self.api_url}/api/queue/{episode_id}/complete",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            logger.info(f"Marked episode {episode_id} as complete")
        except httpx.HTTPError as e:
            logger.error(f"Failed to mark episode {episode_id} as complete: {e}")
            raise

    def fail(self, episode_id: str, error: str) -> None:
        """Mark an episode as failed.

        Args:
            episode_id: ID of the episode
            error: Error message describing the failure
        """
        try:
            response = self.client.post(
                f"{self.api_url}/api/queue/{episode_id}/fail",
                headers=self._headers(),
                json={"error": error},
            )
            response.raise_for_status()
            logger.info(f"Marked episode {episode_id} as failed: {error[:100]}")
        except httpx.HTTPError as e:
            logger.error(f"Failed to mark episode {episode_id} as failed: {e}")
            # Don't re-raise - we tried to report the failure

    def update_progress(
        self, episode_id: str, step: str, percent: Optional[int] = None
    ) -> None:
        """Update progress for a processing episode.

        Args:
            episode_id: ID of the episode
            step: Current processing step (e.g., "downloading", "transcribing")
            percent: Optional percentage complete for the current step
        """
        payload = {"step": step}
        if percent is not None:
            payload["percent"] = percent

        try:
            response = self.client.post(
                f"{self.api_url}/api/queue/{episode_id}/progress",
                headers=self._headers(),
                json=payload,
                timeout=5.0,  # Short timeout for progress updates
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            # Progress updates are non-critical, don't fail the job
            logger.debug(f"Failed to update progress for {episode_id}: {e}")
