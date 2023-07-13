import os
from collections import namedtuple
from typing import Mapping
from unittest import mock


def test_s3_dataset_backend_credentials_env_vars():
    from lightning.data.backends import S3DatasetBackend

    os.environ["AWS_ACCESS_KEY"] = "123"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "abc"

    assert S3DatasetBackend().credentials() == {"access_key": "123", "secret_key": "abc"}
    os.environ.pop("AWS_ACCESS_KEY")
    os.environ.pop("AWS_SECRET_ACCESS_KEY")


_Credentials = namedtuple("RefreshableCredentials", ("access_key", "secret_key", "token"))


@mock.patch("botocore.credentials.InstanceMetadataProvider.load", return_value=_Credentials("abc", "def", "ghi"))
def test_s3_dataset_backend_credentials_iam(patch1):
    from lightning.data.backends import S3DatasetBackend

    credentials = S3DatasetBackend().credentials()
    assert isinstance(credentials, Mapping)
    assert credentials == {"access_key": "abc", "secret_key": "def", "token": "ghi"}


def test_local_dataset_backend_credentials():
    from lightning.data.backends import LocalDatasetBackend

    assert LocalDatasetBackend().credentials() == {}
