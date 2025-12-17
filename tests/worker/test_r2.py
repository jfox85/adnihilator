"""Tests for R2 upload service."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


@pytest.fixture
def mock_boto3():
    """Mock boto3 client."""
    with patch("worker.r2.boto3") as mock:
        yield mock


def test_upload_file(mock_boto3, tmp_path):
    """Test uploading a file to R2."""
    from worker.r2 import R2Client

    # Create a test file
    test_file = tmp_path / "test.mp3"
    test_file.write_bytes(b"fake audio content")

    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    r2 = R2Client(
        access_key="test-key",
        secret_key="test-secret",
        bucket="test-bucket",
        endpoint="https://test.r2.cloudflarestorage.com",
    )
    r2.upload_file(str(test_file), "podcast/episode.mp3")

    mock_client.upload_file.assert_called_once()


def test_get_file_size(mock_boto3):
    """Test getting file size from R2."""
    from worker.r2 import R2Client

    mock_client = MagicMock()
    mock_client.head_object.return_value = {"ContentLength": 12345}
    mock_boto3.client.return_value = mock_client

    r2 = R2Client(
        access_key="test-key",
        secret_key="test-secret",
        bucket="test-bucket",
        endpoint="https://test.r2.cloudflarestorage.com",
    )
    size = r2.get_file_size("podcast/episode.mp3")

    assert size == 12345


def test_delete_file(mock_boto3):
    """Test deleting a file from R2."""
    from worker.r2 import R2Client

    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    r2 = R2Client(
        access_key="test-key",
        secret_key="test-secret",
        bucket="test-bucket",
        endpoint="https://test.r2.cloudflarestorage.com",
    )
    r2.delete_file("podcast/episode.mp3")

    mock_client.delete_object.assert_called_once()


def test_file_exists(mock_boto3):
    """Test checking if a file exists in R2."""
    from worker.r2 import R2Client

    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    r2 = R2Client(
        access_key="test-key",
        secret_key="test-secret",
        bucket="test-bucket",
        endpoint="https://test.r2.cloudflarestorage.com",
    )

    # File exists
    result = r2.file_exists("podcast/episode.mp3")
    assert result is True
    mock_client.head_object.assert_called_once()


def test_file_not_exists(mock_boto3):
    """Test checking if a file does not exist in R2."""
    from worker.r2 import R2Client
    from botocore.exceptions import ClientError

    mock_client = MagicMock()
    # Simulate 404 error
    error_response = {"Error": {"Code": "404"}}
    mock_client.head_object.side_effect = ClientError(error_response, "HeadObject")
    mock_boto3.client.return_value = mock_client

    r2 = R2Client(
        access_key="test-key",
        secret_key="test-secret",
        bucket="test-bucket",
        endpoint="https://test.r2.cloudflarestorage.com",
    )

    result = r2.file_exists("podcast/nonexistent.mp3")
    assert result is False
