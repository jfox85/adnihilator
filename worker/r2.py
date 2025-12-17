"""Cloudflare R2 upload service."""

import boto3
from botocore.exceptions import ClientError


class R2Client:
    """Client for uploading files to Cloudflare R2."""

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        bucket: str,
        endpoint: str,
    ):
        """Initialize the R2 client.

        Args:
            access_key: R2 access key ID
            secret_key: R2 secret access key
            bucket: R2 bucket name
            endpoint: R2 endpoint URL (e.g., https://<account>.r2.cloudflarestorage.com)
        """
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def upload_file(self, local_path: str, key: str) -> None:
        """Upload a file to R2.

        Args:
            local_path: Path to the local file
            key: Object key in R2
        """
        self.client.upload_file(
            local_path,
            self.bucket,
            key,
            ExtraArgs={"ContentType": "audio/mpeg"},
        )

    def get_file_size(self, key: str) -> int:
        """Get the size of a file in R2.

        Args:
            key: Object key in R2

        Returns:
            File size in bytes
        """
        response = self.client.head_object(Bucket=self.bucket, Key=key)
        return response["ContentLength"]

    def delete_file(self, key: str) -> None:
        """Delete a file from R2.

        Args:
            key: Object key to delete
        """
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def file_exists(self, key: str) -> bool:
        """Check if a file exists in R2.

        Args:
            key: Object key to check

        Returns:
            True if file exists, False otherwise
        """
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise
