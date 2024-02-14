import sys
from time import sleep, time
from unittest import mock

import pytest
from lightning.data.streaming import client


def test_s3_client_without_cloud_space_id(monkeypatch):
    boto3 = mock.MagicMock()
    monkeypatch.setattr(client, "boto3", boto3)

    botocore = mock.MagicMock()
    monkeypatch.setattr(client, "botocore", botocore)

    instance_metadata_provider = mock.MagicMock()
    monkeypatch.setattr(client, "InstanceMetadataProvider", instance_metadata_provider)

    instance_metadata_fetcher = mock.MagicMock()
    monkeypatch.setattr(client, "InstanceMetadataFetcher", instance_metadata_fetcher)

    s3 = client.S3Client(1)
    assert s3.client
    assert s3.client
    assert s3.client
    assert s3.client
    assert s3.client

    boto3.client.assert_called_once()


@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows")
@pytest.mark.parametrize("use_shared_credentials", [False, True])
def test_s3_client_with_cloud_space_id(use_shared_credentials, monkeypatch):
    boto3 = mock.MagicMock()
    monkeypatch.setattr(client, "boto3", boto3)

    botocore = mock.MagicMock()
    monkeypatch.setattr(client, "botocore", botocore)

    monkeypatch.setenv("LIGHTNING_CLOUD_SPACE_ID", "dummy")

    if use_shared_credentials:
        monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/.credentials/.aws_credentials")
        monkeypatch.setenv("AWS_CONFIG_FILE", "/.credentials/.aws_credentials")

    instance_metadata_provider = mock.MagicMock()
    monkeypatch.setattr(client, "InstanceMetadataProvider", instance_metadata_provider)

    instance_metadata_fetcher = mock.MagicMock()
    monkeypatch.setattr(client, "InstanceMetadataFetcher", instance_metadata_fetcher)

    s3 = client.S3Client(1)
    assert s3.client
    assert s3.client
    boto3.client.assert_called_once()
    sleep(1 - (time() - s3._last_time))
    assert s3.client
    assert s3.client
    assert len(boto3.client._mock_mock_calls) == 6
    sleep(1 - (time() - s3._last_time))
    assert s3.client
    assert s3.client
    assert len(boto3.client._mock_mock_calls) == 9

    assert instance_metadata_provider._mock_call_count == 0 if use_shared_credentials else 3
