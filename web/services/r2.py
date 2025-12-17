"""Cloudflare R2 integration."""

import os


def get_r2_public_url() -> str:
    """Get the public URL for R2 bucket."""
    url = os.environ.get("R2_PUBLIC_URL", "")
    return url.rstrip("/")


def get_audio_url(audio_key: str) -> str:
    """Get the full public URL for an audio file."""
    base_url = get_r2_public_url()
    return f"{base_url}/{audio_key}"
