import os
from time import time
from typing import Any, Optional

from lightning.data.streaming.constants import _BOTO3_AVAILABLE

if _BOTO3_AVAILABLE:
    import boto3
    import botocore
    from botocore.credentials import InstanceMetadataProvider
    from botocore.utils import InstanceMetadataFetcher


class S3Client:
    # TODO: Generalize to support more cloud providers.

    def __init__(self, refetch_interval: int = 3300) -> None:
        self._refetch_interval = refetch_interval
        self._last_time: Optional[float] = None
        self._has_cloud_space_id: bool = "LIGHTNING_CLOUD_SPACE_ID" in os.environ
        self._client: Optional[Any] = None

    @property
    def client(self) -> Any:
        if not self._has_cloud_space_id:
            if self._client is None:
                self._client = boto3.client(
                    "s3", config=botocore.config.Config(retries={"max_attempts": 1000, "mode": "adaptive"})
                )
            return self._client

        # Re-generate credentials for EC2
        if self._last_time is None or (time() - self._last_time) > self._refetch_interval:
            provider = InstanceMetadataProvider(iam_role_fetcher=InstanceMetadataFetcher(timeout=3600, num_attempts=5))
            credentials = provider.load()
            self._client = boto3.client(
                "s3",
                aws_access_key_id=credentials.access_key,
                aws_secret_access_key=credentials.secret_key,
                aws_session_token=credentials.token,
                config=botocore.config.Config(retries={"max_attempts": 1000, "mode": "adaptive"}),
            )
            self._last_time = time()

        return self._client
