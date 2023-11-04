from time import sleep, time
from unittest import mock

from lightning.data.streaming import client


def test_s3_client_without_cloud_space_id(monkeypatch):
    boto3 = mock.MagicMock()
    monkeypatch.setattr(client, "boto3", boto3)

    botocore = mock.MagicMock()
    monkeypatch.setattr(client, "botocore", botocore)

    s3 = client.S3Client(1)
    assert s3.client
    assert s3.client
    assert s3.client
    assert s3.client
    assert s3.client

    boto3.client.assert_called_once()


def test_s3_client_with_cloud_space_id(monkeypatch):
    boto3 = mock.MagicMock()
    monkeypatch.setattr(client, "boto3", boto3)

    botocore = mock.MagicMock()
    monkeypatch.setattr(client, "botocore", botocore)

    instance_metadata_provider = mock.MagicMock()
    monkeypatch.setattr(client, "InstanceMetadataProvider", instance_metadata_provider)

    instance_metadata_fetcher = mock.MagicMock()
    monkeypatch.setattr(client, "InstanceMetadataFetcher", instance_metadata_fetcher)

    monkeypatch.setenv("LIGHTNING_CLOUD_SPACE_ID", "dummy")

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
